/// @author Sudhanva Kulkarni
/// Templates to generate mixed precision routines for dot prdt of vectors
#include "lo_float_sci.hpp"

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


    
}


