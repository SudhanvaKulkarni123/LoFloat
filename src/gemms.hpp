#pragma  once
#include "Matrix.h"
#include "Vector.h"
#include "cache_info.h"

using namespace lo_float;
namespace Lo_Gemm {



template<MatrixType MatrixA, typename T>
void pack(const MatrixA& A, T* A_pack)
{
    const int m = A.rows();
    const int n = A.cols();
    const int ld = A.ld;
    const Layout layout = MatrixA::layout;

    for(int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if constexpr (layout == Layout::ColMajor) {
                // Column-major: A(i,j) is at A_pack[j * ld + i]
                A_pack[j * ld + i] = static_cast<T>(A(i, j));
            } else {
                // Row-major: A(i,j) is at A_pack[i * ld + j
            A_pack[i * ld + j] = static_cast<T>(A(i, j));
            }
        }
    }
}
// -------------------------------------------------------------
//  A simple signature for a micro-kernel that multiplies a      //
//  (MR×KC) packed block of A with a (KC×NR) packed block of B   //
//  and accumulates into a (MR×NR) block of C.                  //
// -------------------------------------------------------------
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

//  Default reference micro-kernel (naive), only adds loop reordering depending on formats of matrices
template<typename T_in, typename T_out, int MR, int NR, Layout layout>
static void ref_kernel(const T_in* A, const T_in* B, T_out* C,
                       std::size_t rs_c, std::size_t cs_c,
                       std::size_t K) noexcept
                       {
                        if constexpr (layout == Layout::ColMajor) {
                          
                            for (std::size_t k = 0; k < K; ++k)           // kc loop
                                for (int j = 0; j < NR; ++j)              // loop over columns
                                    for (int i = 0; i < MR; ++i)          // loop over rows
                                        C[i * rs_c + j * cs_c] +=
                                            static_cast<T_out>(A[i + k * MR]) *
                                            static_cast<T_out>(B[k * NR + j]);
                        } else if constexpr (layout == Layout::RowMajor) {
                            // RowMajor: Iterate C by rows (i), then columns (j)
                            for (std::size_t k = 0; k < K; ++k)           // kc loop
                                for (int i = 0; i < MR; ++i)              // loop over rows
                                    for (int j = 0; j < NR; ++j)          // loop over columns
                                        C[i * rs_c + j * cs_c] +=
                                            static_cast<T_out>(A[i + k * MR]) *
                                            static_cast<T_out>(B[k * NR + j]);
                        }
                    }

//default MX GEMM micreopkernel
template<typename T_in, typename T_out, int MR, int NR>
static void ref_kernel_overwrite(
    const T_in* A,      // MR x K packed A (contiguous)
    const T_in* B,      // K x NR sub-panel of B (strided)
    std::size_t ld_b,   // Leading dimension (column stride) of the B panel
    T_out* C,           // MR x NR output C (contiguous)
    std::size_t K       // Inner dimension
) noexcept {
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR; ++j) {
            T_out accum = 0; // Local accumulator for this single C(i,j) element
            for (std::size_t k = 0; k < K; ++k) {
                // A is MRxK, packed row-major: A[i*K + k]
                // B is a sub-panel of a KxNC matrix, access is B[k][j] -> B[k * ld_b + j]
                std::cout << "i = " << i << ", j = " << j << ", k = " << k << "\n";
                std::cout << "A[" << i << "][" << k << "] = " << A[i * K + k] << ", B[" << k << "][" << j << "] = " << B[k * ld_b + j] << "\n";
                accum += static_cast<T_out>(A[i * K + k]) * static_cast<T_out>(B[k * ld_b + j]);
            }
            // Overwrite the destination C, which is a temporary micro-accumulator
            C[i * NR + j] = accum;
        }
    }
}


template<MatrixType MatrixA, MatrixType MatrixB, MatrixType MatrixC, typename MatrixAccum1, typename MatrixAccum2, std::size_t MR = 4, std::size_t NR = 4, std::size_t KR = 2>
class Gemm {
    using input_type_1 = typename MatrixA::scalar_type;
    using input_type_2 = typename MatrixB::scalar_type;
    using output_type = typename MatrixC::scalar_type;
    using AccumType1 = MatrixAccum1;
    using AccumType2 = MatrixAccum2;

    static_assert(MatrixA::layout == MatrixB::layout, "Matrix A and B must have the same layout");
    static_assert(MatrixA::layout == MatrixC::layout, "Matrix A and C must have the same layout");
    static_assert(std::is_same_v<output_type, AccumType2>, "MatrixC scalar type must match AccumType2");

public:
    Gemm() = default;

    void run(const MatrixA& A, const MatrixB& B, MatrixC& C) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        std::cout << "GEMM: m = " << m << ", n = " << n << ", k = " << k << "\n";
        std::size_t NC = 1024;
        std::size_t KC = 256;
        std::size_t MC = 128;

        std::vector<input_type_1> A_pack(MC * KC, 0);
        std::vector<input_type_2> B_pack(KC * NC, 0);

        for (std::size_t jc = 0; jc < n; jc += NC) {
            const std::size_t nc_eff = std::min(NC, n - jc);
            for (std::size_t pc = 0; pc < k; pc += KC) {
                const std::size_t kc_eff = std::min(KC, k - pc);
                std::cout << "in L2 loop, packing B\n";
                std::cout << "pc = " << pc << ", kc_eff = " << kc_eff << ", nc_eff = " << nc_eff << ", jc = " << jc << "\n";
                std::cout << "B.rows() = " << B.rows() << ", B.cols() = " << B.cols() << "\n";
                auto B_slice = slice(B, range{pc, kc_eff}, range{(int)jc, (int)nc_eff});
                pack(B_slice, B_pack.data());

                for (std::size_t ic = 0; ic < m; ic += MC) {
                    const std::size_t mc_eff = std::min(MC, m - ic);
                    std::cout << "in L1 loop, packing A\n";
                    auto A_slice = slice(A, range{ic, mc_eff}, range{(int)pc, (int)kc_eff});
                    pack(A_slice, A_pack.data());

                    for (std::size_t jr = 0; jr < nc_eff; jr += NR) {
                        std::cout << "in L0 loop, computing microkernel\n";
                        const std::size_t nr_eff = std::min(NR, nc_eff - jr);
                        for (std::size_t ir = 0; ir < mc_eff; ir += MR) {
                            const std::size_t mr_eff = std::min(MR, mc_eff - ir);

                            if (mr_eff <= MR && nr_eff <= NR) {
                                // --- Stage 1: Compute into high-precision micro-accumulator ---
                                AccumType1 C_micro_accum[MR * NR];
                                for(int kr = KR; kr <= kc_eff; kr += KR) {
                                        std::cout << "Calling microkernel with MR = " << MR << ", NR = " << NR << ", KR = " << KR << " , kc_eff = " << kc_eff << "\n";
                                        ref_kernel_overwrite<input_type_1, AccumType1, MR, NR>(
                                            &A_pack[ir * kc_eff],      // Pointer to current MRxKC block in A_pack
                                            &B_pack[jr],              // Pointer to start of KCxNR block in B_pack
                                            nc_eff,                    // Stride of the B_pack panel
                                            C_micro_accum,             // Destination is the temporary micro-accumulator
                                            kr                     // K dimension
                                        );
                                    // --- Stage 2: Accumulate into final C matrix ---
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C(ic + ir + i_micro, jc + jr + j_micro) +=
                                                static_cast<AccumType2>(C_micro_accum[i_micro * NR + j_micro]);
                                        }
                                    }
                                }
                            }
                            // Note: A production-ready implementation would need to handle
                            // edge cases where mr_eff != MR or nr_eff != NR.
                        }
                    }
                }
            }
        }
    }
};


// /// @brief GEMM for MX * MX -> regular
// /// @tparam MatrixAccum1 
// /// @tparam MatrixAccum2 
// /// @tparam MatrixA 
// /// @tparam MatrixB 
// /// @tparam MatrixC 
// /// @tparam use_custom_tiling 
// template<MX_MatrixType MatrixA, MX_MatrixType MatrixB, MatrixType MatrixC, typename MatrixAccum1 = void, typename MatrixAccum2 = void, bool use_custom_tiling = false>
// class MXGemm {
//     using input_value_type = typename MatrixA::value_type;
//     using input_exp_type = typename MatrixA::shared_exp_type;

//     static_assert(MatrixA::layout == MatrixB::layout,
//                   "Matrix A and B must have the same layout");
//     static_assert(MatrixA::layout == MatrixC::layout,
//                   "Matrix A and C must have the same layout");
//     static_assert(MatrixA::layout == MatrixAccum1::layout,   
//                   "Matrix A and Accum1 must have the same layout");
//     static_assert(MatrixA::layout == MatrixAccum2::layout,
//                   "Matrix A and Accum2 must have the same layout");
    
    


//     MicroKernel<value_type> ukr_ = &ref_kernel<value_type,MR,NR>;

// public:
//     Gemm() = default; 
//     explicit Gemm(MicroKernel<value_type> user_kernel) : ukr_(user_kernel) {}

//     void set_micro_kernel(MicroKernel<value_type> k) noexcept { ukr_ = k; }


//     // void get_cache_sizes() {


//     // }


//     /*
//     @brief function to run GEMM, requires input and output matrices as arguments. Buffers for packing and accumulation are created inside the function.
//     */
//     void run(MatrixC& C, const MatrixA& A, const MatrixB& B)
//     {
//         const std::size_t m = A.rows();
//         const std::size_t n = B.cols();
//         const std::size_t k = A.cols();
        
//         Layout layout = A::layout;


//         //change loops based on layout

//         //choose m_c so that A' fills half of L2 cache
//         // Tile sizes – query cache or use sensible defaults
//         std::size_t NC = 512;
//         std::size_t KC = 128;
//         std::size_t MC = 256;
//         std::size_t MR =  4;
//         std::size_t NR =  4;

//         //keep Nc default for now
//         //TODO : wrap arrays with Matrix data structures
//         input_value_type* A_pack_ = (input_value_type*) malloc(MC * KC);
//         input_exp_type* A_pack_exp_ = (input_exp_type*) malloc(MC * KC);
        
        
//         input_value_type* B_pack_ = (input_value_type*) malloc(NC* KC);
//         input_exp_type* B_pack_exp_ = (input_exp_type*) malloc(NC* KC);

//         MatrixAccum1* C_accum_first_ = (MatrixAccum1*) malloc();


//         //for column major, we need to use GEBP, row major uses GEPB
//         if constexpr (layout == Layout::ColMajor) {
//             for(int ic = 0; ic < n; ic += NC) {
//                 for(int pc = 0; pc < k; pc += KC) {

//                     //pack block of B
//                     auto B_slice = slice(B, range{pc, pc + KC}, range{ic, ic + NC});
//                     lacpy(B_slice, B_pack_, Uplo::General);


//                     for(int jc = 0; jc < m; jc += MC) {

//                         //pack block of A
//                         auto A_slice = slice(A, range{jc, jc + MC}, range{pc, pc + KC});
//                         lacpy(A_slice, A_pack_, Uplo::General);

//                         //adjust block values to account for imperfectly tiled matrix
//                         auto nc = std::min(NC, n - ic);
//                         auto mc = std::min(MC, m - jc);
//                         auto kc = std::min(KC, k - pc);
//                         for(int jr = 0; jr < nc; j+= NR) {
//                             for(int ir = 0; ir < mc; i+= MR) {
                                
//                                 ukr_(A_pack_ + ir*KC, B_pack_ + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
//                             }
//                         } 

                       
//                     }
//                 }
//             }
            
//         } else {

//             for(int ic = 0; ic < m; ic += MC) {
//                 for(int pc = 0; pc < k; pc += KC) {

//                     //pack block of A
//                     auto A_slice = slice(A, range{ic, ic + MC}, range{pc, pc + KC});
//                     lacpy(A_slice, A_pack_, Uplo::General);

//                     for(int jc = 0; jc < n; jc += NC) {

//                         //pack block of B
//                         auto B_slice = slice(B, range{jc, jc + NC}, range{pc, pc + KC});
//                         lacpy(B_slice, B_pack_, Uplo::General);

//                         //adjust block values to account for imperfectly tiled matrix
//                         auto nc = std::min(NC, n - jc);
//                         auto mc = std::min(MC, m - ic);
//                         auto kc = std::min(KC, k - pc);
//                         for(int jr = 0; jr < nc; j+= NR) {
//                             for(int ir = 0; ir < mc; i+= MR) {
                                
//                                 ukr_(A_pack_ + ir*KC, B_pack_ + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
//                             }
//                         } 

                       
//                     }
//                 }
//             }
            
//         }
        






//     }
// };
} // namespace lo_float
