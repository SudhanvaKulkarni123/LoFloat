/// @author Sudhanva Kulkarni
/// Templates to generate mixed precision routines for dot prdt of vectors
#include "lo_float_sci.hpp"
#include <cassert>

using namespace std;
using namespace lo_float;


namespace Lo_Gemm {

    template<Float FP, int block_size>
    struct int_accum_size {
       using int_type = i_n< (get_mantissa_bits_v<FP> + 2*(std::numeric_limits<FP>::max_exponent() - std::numeric_limits<FP>::min_exponent())) << block_size, Signedness::Signed>;      
    };


    template<Float Fp_in1, Float Fp_in2, Int idx, Float Fp_out, Float Fp_accum_inner, Float Fp_accum_outer, int block_size>
    Fp_out dot(Vector<Fp_in1, idx>& x, Vector<Fp_in2, idx>& y) 
    {
        int n = x.len();
        Fp_accum_outer to_ret = Fp_accum_outer{0.0f};
        int num_blocks = n/block_size;
        Fp_accum_inner partial_sum = Fp_accum_inner{0.0f};
        for(int blk = 0; blk < num_blocks; blk++) {
            
            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                    partial_sum += static_cast<Fp_accum_inner>(double(x[i])*double(y[i]));
                    
            }

            to_ret +=  static_cast<Fp_accum_outer>(partial_sum);
            partial_sum = Fp_accum_inner{0.0f};
        }

        return static_cast<Fp_out>(to_ret);

    }


    // MX-aware dot. x,y are MicroScaled: private elements (x[i], raw) + a shared scale per
    // block of r (x.shared_exps[blk]). Inner accumulation resets at each shared-exp block; the
    // two block scales are factored out and applied once per block in high precision, then the
    // partial is rounded into the (fp32-ish) outer accumulator. Mirrors the plain dot above.
    //   true_value(i) = data[i] * shared_exps[i/r], so within a block
    //   sum_{i in blk} (xd_i*xs)*(yd_i*ys) = (xs*ys) * sum_{i in blk} (xd_i*yd_i).
    // Requires x.r == y.r, x.len() == y.len(), and n % r == 0 (the trailing partial block is
    // dropped, same limitation as the plain dot's num_blocks = n/block_size above).
    template<Float Fp_in1, Float Fp_scal1, Float Fp_in2, Float Fp_scal2, Int idx,
             Float Fp_out, Float Fp_accum_inner, Float Fp_accum_outer>
    Fp_out dot(MX_Vector<Fp_in1, Fp_scal1, idx>& x, MX_Vector<Fp_in2, Fp_scal2, idx>& y)
    {
        assert(x.len() == y.len() && "MX dot: operand lengths must match");
        assert(x.r == y.r && "MX dot: operands must share the same block size r");
        const int n = x.len();
        const int r = x.r;                          // shared-exp block == inner accumulation block
        Fp_accum_outer to_ret = Fp_accum_outer{0.0f};
        const int num_blocks = n / r;
        for (int blk = 0; blk < num_blocks; blk++) {
            Fp_accum_inner partial = Fp_accum_inner{0.0f};
            for (int i = blk * r; i < (blk + 1) * r; i++) {
                partial += static_cast<Fp_accum_inner>(double(x[i]) * double(y[i]));   // raw private elems
            }
            // shared scales are constant within a block: factor them out, apply once in double
            const double scale = double(x.shared_exps[blk]) * double(y.shared_exps[blk]);
            to_ret += static_cast<Fp_accum_outer>(double(partial) * scale);
        }

        return static_cast<Fp_out>(to_ret);
    }



}


