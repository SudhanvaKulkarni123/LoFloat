#pragma once
#include "Matrix.h"
#include "Vector.h"
#include "cache_info.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lo_float;
namespace Lo_Gemm {

#ifdef _OPENMP
// Thread management functions
inline int get_num_threads() {
    return omp_get_max_threads();
}

inline void set_num_threads(int num_threads) {
    omp_set_num_threads(num_threads);
}

inline int get_thread_num() {
    return omp_get_thread_num();
}
#else
// Fallback for non-OpenMP builds
inline int get_num_threads() {
    return 1;
}

inline void set_num_threads(int num_threads) {
    // No-op without OpenMP
    (void)num_threads;
}

inline int get_thread_num() {
    return 0;
}
#endif

/*
 * Multi-Strategy BLIS-style GEMM Implementation
 * 
 * This GEMM uses adaptive strategy selection based on matrix sizes:
 * 
 * 1. NAIVE (m,n,k < 32):
 *    - Simple triple-loop implementation
 *    - Minimal overhead for tiny matrices
 *    - No blocking, no packing
 * 
 * 2. MICRO_KERNEL (32 <= m,n,k < 64):
 *    - Uses MR x NR micro-kernel blocking
 *    - Operates directly on original matrices (no packing)
 *    - Better cache behavior than naive for medium sizes
 *    - Avoids packing overhead
 * 
 * 3. BLOCKED_PACKED (m,n,k >= 64):
 *    - Full BLIS-style blocked GEMM with packing
 *    - 5-loop structure: NC -> KC -> MC -> NR -> MR
 *    - Packs A and B into contiguous buffers
 *    - Optimal for large matrices where packing cost is amortized
 *    - PARALLELIZED with OpenMP (when enabled)
 * 
 * Precision hierarchy:
 *   Input types (any) -> AccumType1 (micro-kernel) -> AccumType2 (C_accum) -> output_type (C)
 * 
 * Uses lo_float::Project() for quantization at each level.
 */

template<Float F1, Float F2, Float Fo>
using GemmMicroKernel =
    void (*)(const F1* A, const F2* B, Fo* C,
             std::size_t rs_c, std::size_t cs_c,
             std::size_t k_stride /* = KC */);

template<Float F1, Float F2, Float FS1, Float FS2, Float Fo>
using MX_GEMMMicroKernel =
    void (*)(const F1* A,const FS1* shared_exps_A, const F2* B, const FS2* shared_exp_B, Fo* C,
             std::size_t rs_c, std::size_t cs_c,
             std::size_t k_stride /* = KC */);

// Reference micro-kernel (naive), only adds loop reordering depending on formats of matrices
template<typename T_in, typename T_out, int MR, int NR, Layout layout>
static void ref_kernel(const T_in* A, const T_in* B, T_out* C,
                       std::size_t rs_c, std::size_t cs_c,
                       std::size_t K1,
                       std::size_t K2) noexcept
{
    if constexpr (layout == Layout::ColMajor) {
        for (std::size_t k = K1; k < K2; ++k)           // kc loop
            for (int j = 0; j < NR; ++j)                // loop over columns
                for (int i = 0; i < MR; ++i)            // loop over rows
                    C[i * rs_c + j * cs_c] +=
                        static_cast<T_out>(A[i + k * MR]) *
                        static_cast<T_out>(B[k * NR + j]);
    } else if constexpr (layout == Layout::RowMajor) {
        for (std::size_t k = K1; k < K2; ++k)           // kc loop
            for (int i = 0; i < MR; ++i)                // loop over rows
                for (int j = 0; j < NR; ++j)            // loop over columns
                    C[i * rs_c + j * cs_c] +=
                        static_cast<T_out>(A[i + k * MR]) *
                        static_cast<T_out>(B[k * NR + j]);
    }
}

// Default reference kernel that overwrites (for backward compatibility)
template<typename T_in, typename T_out, int MR, int NR, int KR>
static void ref_kernel_overwrite(
    const T_in* A,      // MR x K packed A 
    const T_in* B,      // K x NR sub-panel of B (strided)
    std::size_t ld_a,   // Leading dimension (row stride) of the A panel
    std::size_t ld_b,   // Leading dimension (column stride) of the B panel
    T_out* C,           // MR x NR output C (contiguous)
    std::size_t K_max   // Inner dimension
) noexcept {
    #pragma unroll(4)
    for (int i = 0; i < MR; ++i) {
        #pragma unroll(4)
        for (int j = 0; j < NR; ++j) {
            T_out accum = T_out{}; // Local accumulator for this single C(i,j) element
            for (std::size_t k = 0; k < KR; ++k) {
                accum += static_cast<T_out>(A[i * ld_a + k] * B[k * ld_b + j]);
            }
            C[i * NR + j] = accum;
        }
    }
}

// Generic vectorized microkernel using outer products
// Computes: C[MR x NR] = A[MR x KR] * B[KR x NR] (overwrites C)
// Uses mul_vec, fma_vec, and Project for arbitrary precision control
template<typename T1, typename T2, typename T_out, int MR, int NR, int KR,
         class arch = xsimd::default_arch> 
static void generic_microkernel(
    const T1* A,        // MR x KR packed, row-major
    const T2* B,        // KR x NR packed, row-major  
    T_out* C,           // MR x NR output (will be overwritten), row-major
    std::size_t ld_a,   // leading dimension of A (should be KR for packed)
    std::size_t ld_b,   // leading dimension of B (should be NR for packed)
    Rounding_Mode rm = Rounding_Mode::RoundToNearestEven,
    int stoch_len = 0
) noexcept {
    
    // Accumulate in T_out precision (user-specified micro-accumulator precision)
    T_out C_accum[MR * NR];
    
    // Zero initialize accumulator
    for (int i = 0; i < MR * NR; ++i) {
        C_accum[i] = T_out{};
    }
    
    // Outer product accumulation: C += A[:, k] * B[k, :]
    for (int k = 0; k < KR; ++k) {
        // Process each row of the output
        for (int i = 0; i < MR; ++i) {
            T_out a_ik = static_cast<T_out>(A[i * ld_a + k]);
            
            // Broadcast a_ik and prepare for multiplication
            T_out a_broadcast[NR];
            for (int j = 0; j < NR; ++j) {
                a_broadcast[j] = a_ik;
            }
            
            // Extract B[k, :] row
            T_out b_row[NR];
            for (int j = 0; j < NR; ++j) {
                b_row[j] = static_cast<T_out>(B[k * ld_b + j]);
            }
            
            // Temp buffer for multiplication result
            T_out temp[NR];
            
            // Compute: temp = a_ik * B[k, :] using mul_vec
            mul_vec<T_out, T_out, T_out, arch>(
                a_broadcast, b_row, temp, NR, rm, stoch_len
            );
            
            // Accumulate: C[i, :] += temp using add_vec
            add_vec<T_out, T_out, T_out, arch>(
                C_accum + i * NR, temp, C_accum + i * NR, NR, rm, stoch_len
            );
        }
    }
    
    // Project final result to output (handles rounding/quantization)
    lo_float::Project(C_accum, C, MR * NR, rm, stoch_len);
}

// Optimized version using FMA for better performance
template<typename T1, typename T2, typename T_out, int MR, int NR, int KR,
         class arch = xsimd::default_arch> 
static void generic_microkernel_fma(
    const T1* A,        // MR x KR packed, row-major
    const T2* B,        // KR x NR packed, row-major  
    T_out* C,           // MR x NR output (will be overwritten), row-major
    std::size_t ld_a,   // leading dimension of A
    std::size_t ld_b,   // leading dimension of B
    Rounding_Mode rm = Rounding_Mode::RoundToNearestEven,
    int stoch_len = 0
) noexcept {
    
    // Accumulate in T_out precision
    T_out C_accum[MR * NR];
    
    // Zero initialize accumulator
    for (int i = 0; i < MR * NR; ++i) {
        C_accum[i] = T_out{};
    }
    
    // Outer product accumulation using FMA: C += A[:, k] * B[k, :]
    for (int k = 0; k < KR; ++k) {
        for (int i = 0; i < MR; ++i) {
            T_out a_ik = static_cast<T_out>(A[i * ld_a + k]);
            
            // Broadcast a_ik for vectorization
            T_out a_broadcast[NR];
            for (int j = 0; j < NR; ++j) {
                a_broadcast[j] = a_ik;
            }
            
            // Extract B[k, :] row
            T_out b_row[NR];
            for (int j = 0; j < NR; ++j) {
                b_row[j] = static_cast<T_out>(B[k * ld_b + j]);
            }
            
            // FMA: C[i, :] = a_ik * B[k, :] + C[i, :]
            // fma_vec computes: out = x * y + z (where z is the accumulator)
            fma_vec<T_out, T_out, T_out, T_out, arch>(
                a_broadcast,           // x: broadcasted a_ik
                b_row,                 // y: B[k, :]
                C_accum + i * NR,      // z: current accumulator C[i, :]
                C_accum + i * NR,      // out: write back to C[i, :]
                NR,
                rm,
                stoch_len
            );
        }
    }
    
    // Project final result (handles rounding/quantization)
    lo_float::Project(C_accum, C, MR * NR, rm, stoch_len);
}

template<MatrixType MatrixA, MatrixType MatrixB, MatrixType MatrixC, 
         typename TypeAccum1, MatrixType MatrixAccum2, 
         std::size_t MR = 4, std::size_t NR = 4, std::size_t KR = 2>
class Gemm {
    using input_type_1 = typename MatrixA::scalar_type;
    using input_type_2 = typename MatrixB::scalar_type;
    using output_type = typename MatrixC::scalar_type;
    using AccumType1 = TypeAccum1;  // Micro-kernel accumulator precision
    using AccumType2 = typename MatrixAccum2::scalar_type;  // Global accumulator precision
    using MatrixAccum1 = Matrix<AccumType1, typename MatrixA::idx_type, MatrixA::layout>;
    using FloatMatrix = Matrix<float, typename MatrixA::idx_type, MatrixA::layout>;

    static_assert(MatrixA::layout == MatrixB::layout, "Matrix A and B must have the same layout");
    static_assert(MatrixA::layout == MatrixC::layout, "Matrix A and C must have the same layout");

    constexpr static Layout layout_a = MatrixA::layout;
    constexpr static Layout layout_b = MatrixB::layout;
    constexpr static Layout layout_c = MatrixC::layout;
    constexpr static Layout layout_accum = MatrixAccum2::layout;
    
    // Strategy thresholds
    static constexpr std::size_t SMALL_GEMM_THRESHOLD = 32;
    static constexpr std::size_t PACK_THRESHOLD_M = 64;
    static constexpr std::size_t PACK_THRESHOLD_N = 64;
    static constexpr std::size_t PACK_THRESHOLD_K = 64;
    
    // GEMM execution strategies
    enum class GemmStrategy {
        NAIVE,          // m,n,k < 32: simple triple loop
        MICRO_KERNEL,   // medium: use micro-kernel without packing
        BLOCKED_PACKED  // large: full BLIS with packing
    };

public:
    Gemm() = default;
    
    // Select strategy based on matrix dimensions
    GemmStrategy selectStrategy(std::size_t m, std::size_t n, std::size_t k) const {
        // Very small matrices: naive is fastest due to no overhead
        if (m <= SMALL_GEMM_THRESHOLD && 
            n <= SMALL_GEMM_THRESHOLD && 
            k <= SMALL_GEMM_THRESHOLD) {
            return GemmStrategy::NAIVE;
        }
        
        // Medium matrices: micro-kernel without packing
        if (m < PACK_THRESHOLD_M || 
            n < PACK_THRESHOLD_N || 
            k < PACK_THRESHOLD_K) {
            return GemmStrategy::MICRO_KERNEL;
        }
        
        // Large matrices: full blocking with packing
        return GemmStrategy::BLOCKED_PACKED;
    }

    void run(const MatrixA& A, const MatrixB& B, MatrixC& C, MatrixAccum2& C_accum,
             Rounding_Mode rm = Rounding_Mode::RoundToNearestEven,
             int stoch_len = 0) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        // Validate matrix dimensions
        assert(A.cols() == B.rows() && "Matrix A columns must match Matrix B rows");
        assert(C.rows() == m && C.cols() == n && "Matrix C must have the same dimensions as the result of A * B");
        assert(C_accum.rows() == m && C_accum.cols() == n && "C_accum must match C dimensions");

        // Select and execute appropriate strategy
        GemmStrategy strategy = selectStrategy(m, n, k);
        
        switch(strategy) {
            case GemmStrategy::NAIVE:
                run_naive(A, B, C, C_accum, rm, stoch_len);
                break;
                
            case GemmStrategy::MICRO_KERNEL:
                run_micro_kernel_unpacked(A, B, C, C_accum, rm, stoch_len);
                break;
                
            case GemmStrategy::BLOCKED_PACKED:
                run_blocked_with_packing(A, B, C, C_accum, rm, stoch_len);
                break;
        }
    }

private:
    // Strategy 1: Naive triple-loop GEMM for very small matrices
    void run_naive(const MatrixA& A, const MatrixB& B, 
                   MatrixC& C, MatrixAccum2& C_accum,
                   Rounding_Mode rm, int stoch_len) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        // Simple ijk loop - optimal for tiny matrices
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                AccumType1 sum = AccumType1{};
                for (std::size_t p = 0; p < k; ++p) {
                    sum += static_cast<AccumType1>(A(i, p)) * 
                           static_cast<AccumType1>(B(p, j));
                }
                C_accum(i, j) = static_cast<AccumType2>(sum);
                
                // Round to final output type
                output_type c_output[1];
                AccumType2 c_temp[1] = {C_accum(i, j)};
                lo_float::Project(c_temp, c_output, 1, rm, stoch_len);
                C(i, j) = c_output[0];
            }
        }
    }
    
    // Strategy 2: Micro-kernel without packing for medium matrices
    void run_micro_kernel_unpacked(const MatrixA& A, const MatrixB& B,
                                   MatrixC& C, MatrixAccum2& C_accum,
                                   Rounding_Mode rm, int stoch_len) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        // Use micro-kernel blocking but operate directly on original matrices
        for (std::size_t i = 0; i < m; i += MR) {
            std::size_t mr_eff = std::min(MR, m - i);
            
            for (std::size_t j = 0; j < n; j += NR) {
                std::size_t nr_eff = std::min(NR, n - j);
                
                AccumType1 C_micro[MR * NR];
                for (int idx = 0; idx < MR * NR; ++idx) {
                    C_micro[idx] = AccumType1{};
                }
                
                // Accumulate over k dimension in KR chunks
                for (std::size_t p = 0; p < k; p += KR) {
                    std::size_t kr_eff = std::min(KR, k - p);
                    
                    // Simple micro-kernel on original (strided) data
                    for (std::size_t kk = 0; kk < kr_eff; ++kk) {
                        for (std::size_t ii = 0; ii < mr_eff; ++ii) {
                            for (std::size_t jj = 0; jj < nr_eff; ++jj) {
                                C_micro[ii * NR + jj] += 
                                    static_cast<AccumType1>(A(i + ii, p + kk)) *
                                    static_cast<AccumType1>(B(p + kk, j + jj));
                            }
                        }
                    }
                }
                
                // Write back results
                AccumType2 C_accum_temp[MR * NR];
                lo_float::Project(C_micro, C_accum_temp, MR * NR, rm, stoch_len);
                
                for (std::size_t ii = 0; ii < mr_eff; ++ii) {
                    for (std::size_t jj = 0; jj < nr_eff; ++jj) {
                        C_accum(i + ii, j + jj) = C_accum_temp[ii * NR + jj];
                    }
                }
                
                // Project to final output
                output_type C_output_temp[MR * NR];
                lo_float::Project(C_accum_temp, C_output_temp, MR * NR, rm, stoch_len);
                
                for (std::size_t ii = 0; ii < mr_eff; ++ii) {
                    for (std::size_t jj = 0; jj < nr_eff; ++jj) {
                        C(i + ii, j + jj) = C_output_temp[ii * NR + jj];
                    }
                }
            }
        }
    }
    
    // Strategy 3: Full blocked GEMM with packing for large matrices
    void run_blocked_with_packing(const MatrixA& A, const MatrixB& B, 
                                  MatrixC& C, MatrixAccum2& C_accum,
                                  Rounding_Mode rm, int stoch_len) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        // Validate matrix dimensions
        assert(A.cols() == B.rows() && "Matrix A columns must match Matrix B rows");
        assert(C.rows() == m && C.cols() == n && "Matrix C must have the same dimensions as the result of A * B");
        assert(C_accum.rows() == m && C_accum.cols() == n && "C_accum must match C dimensions");

        if constexpr (layout_a == Layout::RowMajor && layout_b == Layout::RowMajor) {
            
            std::size_t NC = 1024;
            std::size_t KC = 512;
            std::size_t MC = 256;

#ifdef _OPENMP
            // Get number of threads for allocating thread-local buffers
            int num_threads = omp_get_max_threads();
            
            // Allocate thread-local packing buffers to avoid false sharing
            std::vector<std::vector<float>> A_pack_per_thread(num_threads);
            std::vector<std::vector<float>> B_pack_per_thread(num_threads);
            
            for (int t = 0; t < num_threads; ++t) {
                A_pack_per_thread[t].resize(MC * KC, 0.0f);
                B_pack_per_thread[t].resize(KC * NC, 0.0f);
            }
            
            // BLIS 5-loop structure with parallelization at the outer (jc) loop
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                
                // Each thread gets its own packing buffers
                float* A_pack = A_pack_per_thread[thread_id].data();
                float* B_pack = B_pack_per_thread[thread_id].data();
                FloatMatrix A_pack_matrix(A_pack, MC, KC, KC);
                FloatMatrix B_pack_matrix(B_pack, KC, NC, NC);
                
                // Loop 5: Parallelize over column panels (NC blocks)
                #pragma omp for schedule(dynamic)
                for (std::size_t jc = 0; jc < n; jc += NC) {
                    const std::size_t nc_eff = std::min(NC, n - jc);
                    
                    // Loop 4: over depth (KC blocks)
                    for (std::size_t pc = 0; pc < k; pc += KC) {
                        const std::size_t kc_eff = std::min(KC, k - pc);
                        
                        // Pack B panel (thread-local)
                        auto B_slice = slice(B, range{pc, pc + kc_eff}, range{(int)jc, (int)jc + (int)nc_eff});
                        pack(B_slice, B_pack_matrix);
                        
                        // Loop 3: over rows of C (MC blocks)
                        for (std::size_t ic = 0; ic < m; ic += MC) {
                            const std::size_t mc_eff = std::min(MC, m - ic);
                            
                            // Pack A panel (thread-local)
                            auto A_slice = slice(A, range{ic, ic + mc_eff}, range{(int)pc, (int)pc + (int)kc_eff});
                            pack(A_slice, A_pack_matrix);
                            
                            // Loop 2: over columns within NC (NR blocks)
                            for (std::size_t jr = 0; jr < nc_eff; jr += NR) {
                                const std::size_t nr_eff = std::min(NR, nc_eff - jr);
                                
                                // Loop 1: over rows within MC (MR blocks)
                                for (std::size_t ir = 0; ir < mc_eff; ir += MR) {
                                    const std::size_t mr_eff = std::min(MR, mc_eff - ir);
                                    
                                    // Micro-accumulator in AccumType1 precision
                                    AccumType1 C_micro_accum[MR * NR];
                                    
                                    // Initialize based on whether this is first k-iteration
                                    if (pc == 0) {
                                        // First k-iteration: zero initialize
                                        for (int idx = 0; idx < MR * NR; ++idx) {
                                            C_micro_accum[idx] = AccumType1{};
                                        }
                                    } else {
                                        // Subsequent k-iterations: load from C_accum
                                        for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                            for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                                C_micro_accum[i_micro * NR + j_micro] = 
                                                    static_cast<AccumType1>(C_accum(ic + ir + i_micro, jc + jr + j_micro));
                                            }
                                        }
                                    }
                                    
                                    // Accumulate over KR chunks using micro-kernel
                                    for (std::size_t kr = 0; kr < kc_eff; kr += KR) {
                                        const std::size_t kr_eff = std::min(KR, kc_eff - kr);
                                        
                                        // Temporary buffer for this KR chunk
                                        AccumType1 C_temp[MR * NR];
                                        
                                        // Call vectorized micro-kernel (overwrites C_temp)
                                        generic_microkernel_fma<float, float, AccumType1, MR, NR, KR>(
                                            A_pack + ir * A_pack_matrix.ld + kr,
                                            B_pack + kr * B_pack_matrix.ld + jr,
                                            C_temp,
                                            A_pack_matrix.ld,
                                            B_pack_matrix.ld,
                                            rm,
                                            stoch_len
                                        );
                                        
                                        // Accumulate C_temp into C_micro_accum
                                        for (int idx = 0; idx < MR * NR; ++idx) {
                                            C_micro_accum[idx] += C_temp[idx];
                                        }
                                    }
                                    
                                    // Project micro-accumulator to C_accum (AccumType2 precision)
                                    AccumType2 C_accum_temp[MR * NR];
                                    lo_float::Project(C_micro_accum, C_accum_temp, MR * NR, rm, stoch_len);
                                    
                                    // Store back to C_accum matrix
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_accum(ic + ir + i_micro, jc + jr + j_micro) = 
                                                C_accum_temp[i_micro * NR + j_micro];
                                        }
                                    }
                                    
                                    // On last k-iteration, project to final output C
                                    if (pc + kc_eff >= k) {
                                        output_type C_output_temp[MR * NR];
                                        lo_float::Project(C_accum_temp, C_output_temp, MR * NR, rm, stoch_len);
                                        
                                        for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                            for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                                C(ic + ir + i_micro, jc + jr + j_micro) = 
                                                    C_output_temp[i_micro * NR + j_micro];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } // end parallel region
#else
            // Single-threaded fallback (original code)
            std::vector<float> A_pack(MC * KC, 0.0f);
            std::vector<float> B_pack(KC * NC, 0.0f);
            FloatMatrix A_pack_matrix(A_pack.data(), MC, KC, KC);
            FloatMatrix B_pack_matrix(B_pack.data(), KC, NC, NC);

            // BLIS 5-loop structure (sequential)
            for (std::size_t jc = 0; jc < n; jc += NC) {
                const std::size_t nc_eff = std::min(NC, n - jc);
                
                // Loop 4: over depth (KC blocks)
                for (std::size_t pc = 0; pc < k; pc += KC) {
                    const std::size_t kc_eff = std::min(KC, k - pc);
                    
                    // Pack B panel
                    auto B_slice = slice(B, range{pc, pc + kc_eff}, range{(int)jc, (int)jc + (int)nc_eff});
                    pack(B_slice, B_pack_matrix);
                    
                    // Loop 3: over rows of C (MC blocks)
                    for (std::size_t ic = 0; ic < m; ic += MC) {
                        const std::size_t mc_eff = std::min(MC, m - ic);
                        
                        // Pack A panel
                        auto A_slice = slice(A, range{ic, ic + mc_eff}, range{(int)pc, (int)pc + (int)kc_eff});
                        pack(A_slice, A_pack_matrix);
                        
                        // Loop 2: over columns within NC (NR blocks)
                        for (std::size_t jr = 0; jr < nc_eff; jr += NR) {
                            const std::size_t nr_eff = std::min(NR, nc_eff - jr);
                            
                            // Loop 1: over rows within MC (MR blocks)
                            for (std::size_t ir = 0; ir < mc_eff; ir += MR) {
                                const std::size_t mr_eff = std::min(MR, mc_eff - ir);
                                
                                // Micro-accumulator in AccumType1 precision
                                AccumType1 C_micro_accum[MR * NR];
                                
                                // Initialize based on whether this is first k-iteration
                                if (pc == 0) {
                                    // First k-iteration: zero initialize
                                    for (int idx = 0; idx < MR * NR; ++idx) {
                                        C_micro_accum[idx] = AccumType1{};
                                    }
                                } else {
                                    // Subsequent k-iterations: load from C_accum
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_micro_accum[i_micro * NR + j_micro] = 
                                                static_cast<AccumType1>(C_accum(ic + ir + i_micro, jc + jr + j_micro));
                                        }
                                    }
                                }
                                
                                // Accumulate over KR chunks using micro-kernel
                                for (std::size_t kr = 0; kr < kc_eff; kr += KR) {
                                    const std::size_t kr_eff = std::min(KR, kc_eff - kr);
                                    
                                    // Temporary buffer for this KR chunk
                                    AccumType1 C_temp[MR * NR];
                                    
                                    // Call vectorized micro-kernel (overwrites C_temp)
                                    generic_microkernel_fma<float, float, AccumType1, MR, NR, KR>(
                                        A_pack.data() + ir * A_pack_matrix.ld + kr,
                                        B_pack.data() + kr * B_pack_matrix.ld + jr,
                                        C_temp,
                                        A_pack_matrix.ld,
                                        B_pack_matrix.ld,
                                        rm,
                                        stoch_len
                                    );
                                    
                                    // Accumulate C_temp into C_micro_accum
                                    for (int idx = 0; idx < MR * NR; ++idx) {
                                        C_micro_accum[idx] += C_temp[idx];
                                    }
                                }
                                
                                // Project micro-accumulator to C_accum (AccumType2 precision)
                                AccumType2 C_accum_temp[MR * NR];
                                lo_float::Project(C_micro_accum, C_accum_temp, MR * NR, rm, stoch_len);
                                
                                // Store back to C_accum matrix
                                for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                    for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                        C_accum(ic + ir + i_micro, jc + jr + j_micro) = 
                                            C_accum_temp[i_micro * NR + j_micro];
                                    }
                                }
                                
                                // On last k-iteration, project to final output C
                                if (pc + kc_eff >= k) {
                                    output_type C_output_temp[MR * NR];
                                    lo_float::Project(C_accum_temp, C_output_temp, MR * NR, rm, stoch_len);
                                    
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C(ic + ir + i_micro, jc + jr + j_micro) = 
                                                C_output_temp[i_micro * NR + j_micro];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif // _OPENMP
            
        } else if constexpr (layout_a == Layout::ColMajor && layout_b == Layout::ColMajor) {
            
            // Similar structure for column-major layout
            std::size_t NC = std::min((std::size_t)1024, n);
            std::size_t KC = std::min((std::size_t)512, k);
            std::size_t MC = std::min((std::size_t)256, m);

#ifdef _OPENMP
            // Get number of threads for allocating thread-local buffers
            int num_threads = omp_get_max_threads();
            
            // Allocate thread-local packing buffers
            std::vector<std::vector<input_type_1>> A_pack_per_thread(num_threads);
            std::vector<std::vector<input_type_2>> B_pack_per_thread(num_threads);
            
            for (int t = 0; t < num_threads; ++t) {
                A_pack_per_thread[t].resize(MC * KC, 0);
                B_pack_per_thread[t].resize(KC * NC, 0);
            }
            
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                
                // Each thread gets its own packing buffers
                input_type_1* A_pack = A_pack_per_thread[thread_id].data();
                input_type_2* B_pack = B_pack_per_thread[thread_id].data();
                MatrixA A_pack_matrix(A_pack, MC, KC, KC);
                MatrixB B_pack_matrix(B_pack, KC, NC, NC);
                
                #pragma omp for schedule(dynamic)
                for (std::size_t jc = 0; jc < n; jc += NC) {
                    const std::size_t nc_eff = std::min(NC, n - jc);
                    
                    for (std::size_t pc = 0; pc < k; pc += KC) {
                        const std::size_t kc_eff = std::min(KC, k - pc);
                        
                        auto B_slice = slice(B, range{pc, pc + kc_eff}, range{(int)jc, (int)jc + (int)nc_eff});
                        pack(B_slice, B_pack_matrix);

                        for (std::size_t ic = 0; ic < m; ic += MC) {
                            const std::size_t mc_eff = std::min(MC, m - ic);
                            
                            auto A_slice = slice(A, range{ic, ic + mc_eff}, range{(int)pc, (int)pc + (int)kc_eff});
                            pack(A_slice, A_pack_matrix);

                            for (std::size_t jr = 0; jr < nc_eff; jr += NR) {
                                const std::size_t nr_eff = std::min(NR, nc_eff - jr);
                                
                                for (std::size_t ir = 0; ir < mc_eff; ir += MR) {
                                    const std::size_t mr_eff = std::min(MR, mc_eff - ir);
                                    
                                    AccumType1 C_micro_accum[MR * NR];
                                    
                                    if (pc == 0) {
                                        for (int idx = 0; idx < MR * NR; ++idx) {
                                            C_micro_accum[idx] = AccumType1{};
                                        }
                                    } else {
                                        for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                            for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                                C_micro_accum[i_micro * NR + j_micro] = 
                                                    static_cast<AccumType1>(C_accum(ic + ir + i_micro, jc + jr + j_micro));
                                            }
                                        }
                                    }
                                    
                                    for (std::size_t kr = 0; kr < kc_eff; kr += KR) {
                                        const std::size_t kr_eff = std::min(KR, kc_eff - kr);
                                        
                                        AccumType1 C_temp[MR * NR];
                                        
                                        generic_microkernel_fma<input_type_1, input_type_2, AccumType1, MR, NR, KR>(
                                            A_pack + ir * A_pack_matrix.ld + kr,
                                            B_pack + kr * B_pack_matrix.ld + jr,
                                            C_temp,
                                            A_pack_matrix.ld,
                                            B_pack_matrix.ld,
                                            rm,
                                            stoch_len
                                        );
                                        
                                        for (int idx = 0; idx < MR * NR; ++idx) {
                                            C_micro_accum[idx] += C_temp[idx];
                                        }
                                    }
                                    
                                    AccumType2 C_accum_temp[MR * NR];
                                    lo_float::Project(C_micro_accum, C_accum_temp, MR * NR, rm, stoch_len);
                                    
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_accum(ic + ir + i_micro, jc + jr + j_micro) = 
                                                C_accum_temp[i_micro * NR + j_micro];
                                        }
                                    }
                                    
                                    if (pc + kc_eff >= k) {
                                        output_type C_output_temp[MR * NR];
                                        lo_float::Project(C_accum_temp, C_output_temp, MR * NR, rm, stoch_len);
                                        
                                        for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                            for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                                C(ic + ir + i_micro, jc + jr + j_micro) = 
                                                    C_output_temp[i_micro * NR + j_micro];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } // end parallel region
#else
            // Single-threaded fallback
            std::vector<input_type_1> A_pack(MC * KC, 0);
            std::vector<input_type_2> B_pack(KC * NC, 0);
            MatrixA A_pack_matrix(A_pack.data(), MC, KC, KC);
            MatrixB B_pack_matrix(B_pack.data(), KC, NC, NC);

            for (std::size_t jc = 0; jc < n; jc += NC) {
                const std::size_t nc_eff = std::min(NC, n - jc);
                
                for (std::size_t pc = 0; pc < k; pc += KC) {
                    const std::size_t kc_eff = std::min(KC, k - pc);
                    
                    auto B_slice = slice(B, range{pc, pc + kc_eff}, range{(int)jc, (int)jc + (int)nc_eff});
                    pack(B_slice, B_pack_matrix);

                    for (std::size_t ic = 0; ic < m; ic += MC) {
                        const std::size_t mc_eff = std::min(MC, m - ic);
                        
                        auto A_slice = slice(A, range{ic, ic + mc_eff}, range{(int)pc, (int)pc + (int)kc_eff});
                        pack(A_slice, A_pack_matrix);

                        for (std::size_t jr = 0; jr < nc_eff; jr += NR) {
                            const std::size_t nr_eff = std::min(NR, nc_eff - jr);
                            
                            for (std::size_t ir = 0; ir < mc_eff; ir += MR) {
                                const std::size_t mr_eff = std::min(MR, mc_eff - ir);
                                
                                AccumType1 C_micro_accum[MR * NR];
                                
                                if (pc == 0) {
                                    for (int idx = 0; idx < MR * NR; ++idx) {
                                        C_micro_accum[idx] = AccumType1{};
                                    }
                                } else {
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_micro_accum[i_micro * NR + j_micro] = 
                                                static_cast<AccumType1>(C_accum(ic + ir + i_micro, jc + jr + j_micro));
                                        }
                                    }
                                }
                                
                                for (std::size_t kr = 0; kr < kc_eff; kr += KR) {
                                    const std::size_t kr_eff = std::min(KR, kc_eff - kr);
                                    
                                    AccumType1 C_temp[MR * NR];
                                    
                                    generic_microkernel_fma<input_type_1, input_type_2, AccumType1, MR, NR, KR>(
                                        A_pack.data() + ir * A_pack_matrix.ld + kr,
                                        B_pack.data() + kr * B_pack_matrix.ld + jr,
                                        C_temp,
                                        A_pack_matrix.ld,
                                        B_pack_matrix.ld,
                                        rm,
                                        stoch_len
                                    );
                                    
                                    for (int idx = 0; idx < MR * NR; ++idx) {
                                        C_micro_accum[idx] += C_temp[idx];
                                    }
                                }
                                
                                AccumType2 C_accum_temp[MR * NR];
                                lo_float::Project(C_micro_accum, C_accum_temp, MR * NR, rm, stoch_len);
                                
                                for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                    for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                        C_accum(ic + ir + i_micro, jc + jr + j_micro) = 
                                            C_accum_temp[i_micro * NR + j_micro];
                                    }
                                }
                                
                                if (pc + kc_eff >= k) {
                                    output_type C_output_temp[MR * NR];
                                    lo_float::Project(C_accum_temp, C_output_temp, MR * NR, rm, stoch_len);
                                    
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C(ic + ir + i_micro, jc + jr + j_micro) = 
                                                C_output_temp[i_micro * NR + j_micro];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif // _OPENMP
        }
    }
};

} // namespace Lo_Gemm