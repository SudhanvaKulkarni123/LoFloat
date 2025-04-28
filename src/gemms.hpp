#pragma  once
#include "Matrix.h"
#include "Vector.h"
#include "cache_info.h"

namespace LoGemm {

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

    //---------------------------------------------------------------------
    //  C ← C + A·B            ( no transposition for brevity )            //
    //---------------------------------------------------------------------
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

        if constexpr(use_custom_tiling) {
            //set NC = KC = sqrt(L2_cache /2) as a starting point
            double L2_size = get_cache_size(2);
            double L1_size = get_cache_size(1);

            //want KC * NR < L1_size / 2
            KC = L1_size / 8;

            MC = L2_size / (2 * KC);
        }

        //keep Nc default for now

        input_value_type* A_pack = (input_value_type*) malloc(MC * KC);

        input_value_type* B_pack = (input_value_type*) malloc(NC* KC);

        MatrixAccum1* C_accum_first = (MatrixAccum1*) malloc();





        //  ───────── outer‐most JC loop  (N dimension, B panels) ─────────
        for (std::size_t jc = 0; jc < n; jc += NC)
        {
            const std::size_t nc = std::min<std::size_t>(NC, n - jc);

            for (std::size_t pc = 0; pc < k; pc += KC)
            {
                const std::size_t kc = std::min<std::size_t>(KC, k - pc);

                // ---- pack A (m_c by k_c) into a contiguous buffer ----
                pack_submatrix(A, lo_float::range{jc, jc + nc}, lo_float::range{pc, pc + kc}, A_pack);

                //  ───── IC loop  (M dimension, A panels) ─────
                for (std::size_t ic = 0; ic < m; ic += MC)
                {
                    const std::size_t mc = std::min<std::size_t>(MC, m - ic);

                    //pack B into some B_j
                    pack_submatrix(B, lo_float::range{pc, pc+kc}, lo_float::range{ic, ic + mc}, B_pack);
                    //  ─── jr / ir loops at micro-kernel granularity ───
                    for (std::size_t jr = 0; jr < nc; jr += NR)
                    {
                        const std::size_t nr = std::min<std::size_t>(NR, nc - jr);

                        for (std::size_t ir = 0; ir < mc; ir += MR)
                        {
                            const std::size_t mr = std::min<std::size_t>(MR, mc - ir);

                            //  _Pointers into packed A, B and into C_
                            const value_type* A_block = &Ap[ir*kc];
                            const value_type* B_block = &Bp[jr];
                            value_type*       C_block = &C(ic+ir, jc+jr);

                            //  Because edge tiles may be smaller than MR×NR,
                            //  fall back to scalar update in those cases.
                            if (mr == MR && nr == NR)
                            {
                                ukr_(A_block, B_block, C_block,
                                     C.row_stride(), C.col_stride(), kc);
                            }
                            else {
                                ref_kernel<value_type,1,1>(A_block,B_block,C_block,
                                    C.row_stride(), C.col_stride(), kc);
                            }
                        } // ir
                    }     // jr
                }         // ic
            }             // pc
        }                 // jc
    }
};

} // namespace LoGemm
