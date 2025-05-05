#pragma  once
#include "Matrix.h"
#include "Vector.h"
#include "cache_info.h"

namespace lo_float {

// -------------------------------------------------------------
//  A simple signature for a micro-kernel that multiplies a      //
//  (MR×KC) packed block of A with a (KC×NR) packed block of B   //
//  and accumulates into a (MR×NR) block of C.                  //
// -------------------------------------------------------------
template<typename T_in, typename T_out>
using GemmMicroKernel =
    void (*)(const T_in* A, const T_in* B, T_out* C,
             std::size_t rs_c, std::size_t cs_c,
             std::size_t k_stride /* = KC */);


template<typename T_in, typename T_out, typename T_scal>
using MX_GEMMMicroKernel =
    void (*)(const T_in* A,const T_scal* shared_exps_A, const T_in* B, const T_scal* shared_exp_B, T_out* C,
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
template<typename T_in, typename T_out, typename T_scal, int MR, int NR, Layout layout>
static void ref_kernel(const T_in* A, const T_scal* shared_exps_A, const T_in* B, const T_scal* shared_exps_B, T_out* C,
                       std::size_t rs_c, std::size_t cs_c,
                       std::size_t K) noexcept
{
    //assumption in this microkernel is that shared exp is such that there is one shared exp for each row (if row major) or column (if column major)
    if constexpr (layout == Layout::ColMajor) {
        for(int i = 0; i < MR; i++) {
            for(int j = 0; j < NR; j++) {
                C[i*rs_c + j] += static_cast<T_out>(A[i + j*MR]) * static_cast<T_out>(B[j + i*NR]) * shared_exps_A[i] * shared_exps_B[j];
            }
        }
    } else if constexpr (layout == Layout::RowMajor) {
        for(int i = 0; i < MR; i++) {
            for(int j = 0; j < MR; j++) {
                C[i*rs_c + j] += static_cast<T_out>(A[i + j*MR]) * static_cast<T_out>(B[j + i*NR]) * shared_exps_A[j] * shared_exps_B[i];
            }
        }
    
    }
}


template<MatrixType MatrixA, MatrixType MatrixB, MatrixType MatrixC, typename MatrixAccum1 = void, typename MatrixAccum2 = void, bool use_custom_tiling = false>
class Gemm {
    using input_value_type = typename MatrixA::value_type;

    static_assert(MatrixA::layout == MatrixB::layout,
                  "Matrix A and B must have the same layout");
    static_assert(MatrixA::layout == MatrixC::layout,
                  "Matrix A and C must have the same layout");
    static_assert(MatrixA::layout == MatrixAccum1::layout,   
                  "Matrix A and Accum1 must have the same layout");
    static_assert(MatrixA::layout == MatrixAccum2::layout,
                  "Matrix A and Accum2 must have the same layout");
    
    


    MicroKernel<value_type> ukr_ = &ref_kernel<value_type,MR,NR>;

public:
    Gemm() = default; 
    explicit Gemm(MicroKernel<value_type> user_kernel) : ukr_(user_kernel) {}

    void set_micro_kernel(MicroKernel<value_type> k) noexcept { ukr_ = k; }


    void autotune(int m, int n, int k) {
        


    }


    /*
    @brief function to run GEMM, requires input and output matrices as arguments. Buffers for packing and accumulation are created inside the function.
    */
    void run(MatrixC& C, const MatrixA& A, const MatrixB& B)
    {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        Layout layout = A::layout;


        //change loops based on layout

        //choose m_c so that A' fills half of L2 cache
        // Tile sizes – query cache or use sensible defaults
        std::size_t NC = 512;
        std::size_t KC = 128;
        std::size_t MC = 256;
        std::size_t MR =  4;
        std::size_t NR =  4;

        //keep Nc default for now
        //TODO : wrap arrays with Matrix data structures
        input_value_type* A_pack_ = (input_value_type*) malloc(MC * KC);
        
        input_value_type* B_pack_ = (input_value_type*) malloc(NC* KC);

        MatrixAccum1* C_accum_first_ = (MatrixAccum1*) malloc();


        //for column major, we need to use GEBP, row major uses GEPB
        if constexpr (layout == Layout::ColMajor) {
            for(int ic = 0; ic < n; ic += NC) {
                for(int pc = 0; pc < k; pc += KC) {

                    //pack block of B
                    auto B_slice = slice(B, range{pc, pc + KC}, range{ic, ic + NC});
                    lacpy(B_slice, B_pack, Uplo::General);

                    for(int jc = 0; jc < m; jc += MC) {

                        //pack block of A
                        auto A_slice = slice(A, range{jc, jc + MC}, range{pc, pc + KC});
                        lacpy(A_slice, A_pack, Uplo::General);

                        //adjust block values to account for imperfectly tiled matrix
                        auto nc = std::min(NC, n - ic);
                        auto mc = std::min(MC, m - jc);
                        auto kc = std::min(KC, k - pc);
                        for(int jr = 0; jr < nc; j+= NR) {
                            for(int ir = 0; ir < mc; i+= MR) {
                                
                                ukr_(A_pack + ir*KC, B_pack + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
                            }
                        } 

                       
                    }
                }
            }
            
        } else {

            for(int ic = 0; ic < m; ic += MC) {
                for(int pc = 0; pc < k; pc += KC) {

                    //pack block of A
                    auto A_slice = slice(A, range{ic, ic + MC}, range{pc, pc + KC});
                    lacpy(A_slice, A_pack, Uplo::General);

                    for(int jc = 0; jc < n; jc += NC) {

                        //pack block of B
                        auto B_slice = slice(B, range{jc, jc + NC}, range{pc, pc + KC});
                        lacpy(B_slice, B_pack, Uplo::General);

                        //adjust block values to account for imperfectly tiled matrix
                        auto nc = std::min(NC, n - jc);
                        auto mc = std::min(MC, m - ic);
                        auto kc = std::min(KC, k - pc);
                        for(int jr = 0; jr < nc; j+= NR) {
                            for(int ir = 0; ir < mc; i+= MR) {
                                
                                ukr_(A_pack + ir*KC, B_pack + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
                            }
                        } 

                       
                    }
                }
            }
            
        }
        






    }
};


/// @brief GEMM for MX * MX -> regular
/// @tparam MatrixAccum1 
/// @tparam MatrixAccum2 
/// @tparam MatrixA 
/// @tparam MatrixB 
/// @tparam MatrixC 
/// @tparam use_custom_tiling 
template<MX_MatrixType MatrixA, MX_MatrixType MatrixB, MatrixType MatrixC, typename MatrixAccum1 = void, typename MatrixAccum2 = void, bool use_custom_tiling = false>
class Gemm {
    using input_value_type = typename MatrixA::value_type;
    using input_exp_type = typename MatrixA::shared_exp_type;

    static_assert(MatrixA::layout == MatrixB::layout,
                  "Matrix A and B must have the same layout");
    static_assert(MatrixA::layout == MatrixC::layout,
                  "Matrix A and C must have the same layout");
    static_assert(MatrixA::layout == MatrixAccum1::layout,   
                  "Matrix A and Accum1 must have the same layout");
    static_assert(MatrixA::layout == MatrixAccum2::layout,
                  "Matrix A and Accum2 must have the same layout");
    
    


    MicroKernel<value_type> ukr_ = &ref_kernel<value_type,MR,NR>;

public:
    Gemm() = default; 
    explicit Gemm(MicroKernel<value_type> user_kernel) : ukr_(user_kernel) {}

    void set_micro_kernel(MicroKernel<value_type> k) noexcept { ukr_ = k; }


    // void get_cache_sizes() {


    // }


    /*
    @brief function to run GEMM, requires input and output matrices as arguments. Buffers for packing and accumulation are created inside the function.
    */
    void run(MatrixC& C, const MatrixA& A, const MatrixB& B)
    {
        const std::size_t m = A.rows();
        const std::size_t n = B.cols();
        const std::size_t k = A.cols();
        
        Layout layout = A::layout;


        //change loops based on layout

        //choose m_c so that A' fills half of L2 cache
        // Tile sizes – query cache or use sensible defaults
        std::size_t NC = 512;
        std::size_t KC = 128;
        std::size_t MC = 256;
        std::size_t MR =  4;
        std::size_t NR =  4;

        //keep Nc default for now
        //TODO : wrap arrays with Matrix data structures
        input_value_type* A_pack_ = (input_value_type*) malloc(MC * KC);
        input_exp_type* A_pack_exp_ = (input_exp_type*) malloc(MC * KC);
        
        
        input_value_type* B_pack_ = (input_value_type*) malloc(NC* KC);
        input_exp_type* B_pack_exp_ = (input_exp_type*) malloc(NC* KC);

        MatrixAccum1* C_accum_first_ = (MatrixAccum1*) malloc();


        //for column major, we need to use GEBP, row major uses GEPB
        if constexpr (layout == Layout::ColMajor) {
            for(int ic = 0; ic < n; ic += NC) {
                for(int pc = 0; pc < k; pc += KC) {

                    //pack block of B
                    auto B_slice = slice(B, range{pc, pc + KC}, range{ic, ic + NC});
                    lacpy(B_slice, B_pack, Uplo::General);


                    for(int jc = 0; jc < m; jc += MC) {

                        //pack block of A
                        auto A_slice = slice(A, range{jc, jc + MC}, range{pc, pc + KC});
                        lacpy(A_slice, A_pack, Uplo::General);

                        //adjust block values to account for imperfectly tiled matrix
                        auto nc = std::min(NC, n - ic);
                        auto mc = std::min(MC, m - jc);
                        auto kc = std::min(KC, k - pc);
                        for(int jr = 0; jr < nc; j+= NR) {
                            for(int ir = 0; ir < mc; i+= MR) {
                                
                                ukr_(A_pack + ir*KC, B_pack + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
                            }
                        } 

                       
                    }
                }
            }
            
        } else {

            for(int ic = 0; ic < m; ic += MC) {
                for(int pc = 0; pc < k; pc += KC) {

                    //pack block of A
                    auto A_slice = slice(A, range{ic, ic + MC}, range{pc, pc + KC});
                    lacpy(A_slice, A_pack, Uplo::General);

                    for(int jc = 0; jc < n; jc += NC) {

                        //pack block of B
                        auto B_slice = slice(B, range{jc, jc + NC}, range{pc, pc + KC});
                        lacpy(B_slice, B_pack, Uplo::General);

                        //adjust block values to account for imperfectly tiled matrix
                        auto nc = std::min(NC, n - jc);
                        auto mc = std::min(MC, m - ic);
                        auto kc = std::min(KC, k - pc);
                        for(int jr = 0; jr < nc; j+= NR) {
                            for(int ir = 0; ir < mc; i+= MR) {
                                
                                ukr_(A_pack + ir*KC, B_pack + jr*KC, C_accum_first_ + ir*rs_c + jr*cs_c, rs_c, cs_c, kc);
                            }
                        } 

                       
                    }
                }
            }
            
        }
        






    }
};
} // namespace lo_float
