/// @author Sudhanva Kulkarni
/// Templates to generate mixed precision routines for dot prdt of vectors

#include "Vector.h"
#include "lo_float.h"
#include "lo_float_sci.hpp"
#include "lo_int.h"

using namespace std;

namespace lo_float {

    template<Float FP, int block_size>
    struct int_accum_size {
       using int_type = i_n< (get_mantissa_bits_v<FP> + 2*(std::numeric_limits<FP>::max_exponent() - std::numeric_limits<FP>::min_exponent())) << block_size, Signedness::Signed>;      
    };


    template<Float Fp_in, Int idx, Float Fp_out, Float Fp_accum1, Float Fp_accum2, int block_size>
    Fp_out dot(Vector<Fp_in, idx>& x, Vector<Fp_in, idx>& y) 
    {
        int n = x.len();
        Fp_accum2 to_ret = Fp_accum2{};
        int num_blocks = n/block_size;
        Fp_accum1 partial_sum = Fp_accum1{};
        for(int blk = 0; blk < num_blocks; blk++) {

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                if constexpr (is_float) {
                    partial_sum += static_cast<Fp_accum1>(double(x[i])*double(y[i]));
                }
                    

            }

            to_ret +=  static_cast<Fp1_accum2>(partial_sum);
            partial_sum = Fp_accum1{};
        }

        return to_ret;

    }


    template<Float Fp_in, Int idx, Float Fp_out, Int Int_accum1, Float Fp_accum2, int block_size>
    Fp_out fast_dot(Vector<Fp_in, idx>& x, Vector<Fp_in, idx>& y)
    {
        int n = x.len();
        Fp_accum2 to_ret = Fp_accum2{};
        int num_blocks = n/block_size;
        Fp_accum1 partial_sum = Fp_accum1{};
        for(int blk = 0; blk < num_blocks; blk++) {

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                partial_sum += static_cast<Fp_accum2>(double(x[i])*double(y[i]));

            }

            to_ret +=  static_cast<Fp1_accum1>(partial_sum);
            partial_sum = Fp_accum1{};
        }

        return to_ret;


    }

    template<Float Fp_in, Int idx, Float Fp_out, Int Int_accum1, Float Fp_accum2, int block_size>
    Fp_out slow_dot(Vector<Fp_in, idx>& x, Vector<Fp_in, idx>& y)
    {
        int n = x.len();
        Fp_accum2 to_ret = Fp_accum2{};
        int num_blocks = n/block_size;
        Fp_accum1 partial_sum = Fp_accum1{};
        for(int blk = 0; blk < num_blocks; blk++) {

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                partial_sum += static_cast<Fp_accum2>(double(x[i])*double(y[i]));

            }

            to_ret +=  static_cast<Fp1_accum1>(partial_sum);
            partial_sum = Fp_accum1{};
        }

        return to_ret;


    }


    template<Float Fp_in, Int idx, Float Fp_out, Float Fp_accum2, int block_size>
    Fp_out exact_dot(Vector<Fp_in, idx>& x, Vector<Fp_in, idx>& y)
    {   
        using Int_accum1 = 
        int n = x.len();
        Fp_accum2 to_ret = Fp_accum2{};
        int num_blocks = n/block_size;
        Fp_accum1 partial_sum = Fp_accum1{};
        using FP_accum1 = typename int_accum_size<Fp_in, block_size>::int_type;
        for(int blk = 0; blk < num_blocks; blk++) {

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                partial_sum += static_cast<Fp_accum1>(double(x[i])*double(y[i]));

            }

            to_ret +=  static_cast<Fp_accum2>(partial_sum);
            partial_sum = Fp_accum1{};
        }

        return to_ret;


    }



    
}


