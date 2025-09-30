#pragma  once
#include "Matrix.h"
#include "Vector.h"
#include "cache_info.h"

using namespace lo_float;
namespace Lo_Gemm {


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
                       std::size_t K1,
                       std::size_t K2) noexcept
                       {
                        if constexpr (layout == Layout::ColMajor) {
                          
                            for (std::size_t k = K1; k < K2; ++k)           // kc loop
                                for (int j = 0; j < NR; ++j)              // loop over columns
                                    for (int i = 0; i < MR; ++i)          // loop over rows
                                        C[i * rs_c + j * cs_c] +=
                                            static_cast<T_out>(A[i + k * MR]) *
                                            static_cast<T_out>(B[k * NR + j]);
                        } else if constexpr (layout == Layout::RowMajor) {
                            // RowMajor: Iterate C by rows (i), then columns (j)
                            for (std::size_t k = K1; k < K2; ++k)           // kc loop
                                for (int i = 0; i < MR; ++i)              // loop over rows
                                    for (int j = 0; j < NR; ++j)          // loop over columns
                                        C[i * rs_c + j * cs_c] +=
                                            static_cast<T_out>(A[i + k * MR]) *
                                            static_cast<T_out>(B[k * NR + j]);
                        }
                    }

//default MX GEMM micreopkernel
template<typename T_in, typename T_out, int MR, int NR, int KR>
static void ref_kernel_overwrite(
    const T_in* A,      // MR x K packed A 
    const T_in* B,      // K x NR sub-panel of B (strided)
    std::size_t ld_a,   // Leading dimension (row stride) of the A panel
    std::size_t ld_b,   // Leading dimension (column stride) of the B panel
    T_out* C,           // MR x NR output C (contiguous)
    std::size_t K_max       // Inner dimension
) noexcept {


    #pragma unroll
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR; ++j) {
            T_out accum = T_out{}; // Local accumulator for this single C(i,j) element
            for (std::size_t k = 0; k < KR; ++k) {
                // A is MRxK, packed row-major: A[i*K + k]
                // B is a sub-panel of a KxNC matrix, access is B[k][j] -> B[k * ld_b + j]
                accum += static_cast<T_out>(A[i * ld_a + k] * B[k * ld_b + j]);
                // std::cout << "A[" << i << "][" << k << "] = " << A[i * ld_a + k] 
                //           << ", B[" << k << "][" << j << "] = " << B[k * ld_b + j] 
                //           << ", accum = " << accum << "\n";
             
            }
            // Overwrite the destination C, which is a temporary micro-accumulator

            C[i * NR + j] = accum;
        }
    }
}

// void printVector(const std::vector<float>& vec) {
//     for (const auto& v : vec) {
//         std::cout << v << " ";
//     }
//     std::cout << "\n";
// }

template<MatrixType MatrixA, MatrixType MatrixB, MatrixType MatrixC, typename TypeAccum1, MatrixType MatrixAccum2, std::size_t MR = 4, std::size_t NR = 4, std::size_t KR = 2>
class Gemm {
    using input_type_1 = typename MatrixA::scalar_type;
    using input_type_2 = typename MatrixB::scalar_type;
    using output_type = typename MatrixC::scalar_type;
    using AccumType1 = TypeAccum1;
    using AccumType2 = typename MatrixAccum2::scalar_type;
    using MatrixAccum1 = Matrix<AccumType1,typename MatrixA::idx_type, MatrixA::layout>;
    using FloatMatrix = Matrix<float, typename MatrixA::idx_type, MatrixA::layout>;

    static_assert(MatrixA::layout == MatrixB::layout, "Matrix A and B must have the same layout");
    static_assert(MatrixA::layout == MatrixC::layout, "Matrix A and C must have the same layout");

    constexpr static Layout layout_a = MatrixA::layout;
    constexpr static Layout layout_b = MatrixB::layout;
    constexpr static Layout layout_c = MatrixC::layout;
    constexpr static Layout layout_accum = MatrixAccum2::layout;
    

public:
    Gemm() = default;



    void run(const MatrixA& A, const MatrixB& B, MatrixC& C, MatrixAccum2& C_accum) {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        //checkl that matrix shapes add up
        assert(A.cols() == B.rows() && "Matrix A columns must match Matrix B rows");
        assert(C.rows() == m && C.cols() == n && "Matrix C must have the same dimensions as the result of A * B");


        if constexpr (layout_b == Layout::RowMajor && layout_a == Layout::RowMajor)
        {

        std::size_t NC = (std::size_t)256;
        std::size_t KC = (std::size_t)256;
        std::size_t MC = (std::size_t)128;

        std::vector<float> A_pack(MC * KC, (0));
        std::vector<float> B_pack(KC * NC, (0));
        FloatMatrix A_pack_matrix(A_pack.data(), MC, KC, KC);
        FloatMatrix B_pack_matrix(B_pack.data(), KC, NC, NC);

        #pragma omp for 
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
                             
                            
                                // --- Stage 1: Compute into high-precision micro-accumulator ---
                                AccumType1 C_micro_accum[MR * NR];
                                
                                for(int kr = 0; kr < kc_eff; kr += KR) {
                                       // std::cout << "ir = " << ir << ", jr = " << jr << ", kr = " << kr << "\n";
                                        ref_kernel_overwrite<float, AccumType1, MR, NR, KR>(
                                            A_pack.data() + ir * A_pack_matrix.ld + kr, 
                                            B_pack.data() + kr * B_pack_matrix.ld + jr,
                                            A_pack_matrix.ld, 
                                            B_pack_matrix.ld,
                                            C_micro_accum,
                                            kc_eff
                                        );
                                        //std::cout << "C_micro_accum[0] = " << C_micro_accum[0] << "\n";
                                    // --- Stage 2: Accumulate into final C matrix ---
                                    //#pragma omp parallel for collapse(2) 
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_accum(ic + ir + i_micro, jc + jr + j_micro) +=
                                                static_cast<AccumType2>(C_micro_accum[i_micro * NR + j_micro]);
                                        }
                                    }
                                    
                                }
                                if(pc + kc_eff >= k) {
                                for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C(ic + ir + i_micro, jc + jr + j_micro) +=
                                                static_cast<output_type>(C_accum(ic + ir + i_micro, jc + jr + j_micro));
                                        }
                                    }
                                }
                            
                          
                        }
                    }
                }
            }
        }


    } else if constexpr(layout_b == Layout::ColMajor && layout_b == Layout::ColMajor) {

        std::size_t NC = std::min((std::size_t)256, n);
        std::size_t KC = std::min((std::size_t)256, k);
        std::size_t MC = std::min((std::size_t)256, m);

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
                             
                            
                                // --- Stage 1: Compute into high-precision micro-accumulator ---
                                AccumType1 C_micro_accum[MR * NR];
                                for(int kr = 0; kr < kc_eff; kr += KR) {
                                       // std::cout << "ir = " << ir << ", jr = " << jr << ", kr = " << kr << "\n";
                                        ref_kernel_overwrite<AccumType1, MR, NR, KR>(
                                            A_pack.data() + ir * A_pack_matrix.ld + kr, 
                                            B_pack.data() + kr * B_pack_matrix.ld + jr,
                                            A_pack_matrix.ld, 
                                            B_pack_matrix.ld,
                                            C_micro_accum,
                                            kc_eff
                                        );
                                        //std::cout << "C_micro_accum[0] = " << C_micro_accum[0] << "\n";
                                    // --- Stage 2: Accumulate into final C matrix ---
                                    for (int i_micro = 0; i_micro < MR; ++i_micro) {
                                        for (int j_micro = 0; j_micro < NR; ++j_micro) {
                                            C_accum(ic + ir + i_micro, jc + jr + j_micro) +=
                                                static_cast<AccumType2>(C_micro_accum[i_micro * NR + j_micro]);
                                        }
                                    }
                                }
                                
                            }
                                               
                            
                          
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

// template<MatrixType MatrixA, VectorType Vectorb, VectorType Vectorc, typename TypeAccum1, VectorType VectorAccum2, std::size_t MR = 4, std::size_t NR = 4>
// class Gemv {

 

// }
} // namespace lo_float
