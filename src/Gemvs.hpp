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

//  Default reference micro-kernel (naive), only adds loop reordering depending on formats of matrices
template<typename T_in, typename T_out, int MR, int NR, Layout layout>
static void ref_kernel(const T_in* A, const T_in* B, T_out* C,
                       std::size_t rs_c, std::size_t cs_c,
                       std::size_t K) noexcept
{
    for (std::size_t k = 0; k < K; ++k)           // kc loop
        for (int j = 0; j < NR; ++j)
            for (int i = 0; i < MR; ++i)
                C[i*rs_c + j*cs_c] += static_cast<T_out>(A[i + k*MR]) * static_cast<T_out>(B[k*NR + j]);
}


template<typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixAccum1 = void, typename MatrixAccum2 = void, bool use_custom_tiling = false>
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


    void get_cache_sizes() {


    }


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

        input_value_type* A_pack = (input_value_type*) malloc(MC * KC);

        input_value_type* B_pack = (input_value_type*) malloc(NC* KC);

        MatrixAccum1* C_accum_first = (MatrixAccum1*) malloc();






    }
};

} // namespace lo_float
