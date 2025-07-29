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


    template<Float Fp_in1, Float Fp_in2, Int idx, Float Fp_out, Int Int_accum_inner, Float Fp_accum_outer, int block_size>
    Fp_out fast_dot(Vector<Fp_in1, idx>& x, Vector<Fp_in2, idx>& y)
    {
        int n = x.len();
        Fp_accum_outer to_ret = Fp_accum_outer{};
        int num_blocks = n/block_size;
        Int_accum_inner partial_sum = Int_accum_inner{};
        const int max_exp = std::numeric_limits<Fp_in1>::max_exponent + get_bias<Fp_in1> + std::numeric_limits<Fp_in2>::max_exponent + get_bias<Fp_in2>;
        const int max_mantissa_size = get_mantissa_bits_v<Fp_in1> + get_mantissa_bits_v<Fp_in2>;
        const int shift_offset =  get_int_len_v<Int_accum_inner> - max_exp - max_mantissa_size; //this way, if something has exp, we will shift it shift_offset + exp amount and then. add
        for(int blk = 0; blk < num_blocks; blk++) {

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                const int exp_x = func_get_exponent_bits(x[i]) ;
                const int exp_y = func_get_exponent_bits(y[i]) ;

                const Int_accum_inner mantissa_x = (Int_accum_inner)(exp_x == 0 ? 0 : 1 << get_mantissa_bits_v<Fp_in1>) + func_get_mantissa_bits(x[i]);
                const Int_accum_inner mantissa_y = (Int_accum_inner)(exp_x == 0 ? 0 : 1 << get_mantissa_bits_v<Fp_in2>) + func_get_mantissa_bits(y[i]);
                const Int_accum_inner prod = mantissa_x*mantissa_y;
                const bool sign = func_get_sign_bit(x[i]) ^ func_get_sign_bit(y[i]);
                const int new_exp = exp_x + exp_y;
                
                const int num_shift = shift_offset + new_exp; 
                partial_sum += (sign ? -prod : prod);
            }

            to_ret +=  static_cast<Fp_accum_outer>(partial_sum);
            partial_sum = Int_accum_inner{};
        }

        return static_cast<Fp_out>(to_ret);;

    }

    template<Float Fp_in1, Float Fp_in2, Int idx, Float Fp_out, Int Int_accum_inner, Float Fp_accum_outer, int block_size>
    Fp_out slow_dot(Vector<Fp_in1, idx>& x, Vector<Fp_in2, idx>& y)
    {
        int n = x.len();
        Fp_accum_outer to_ret = Fp_accum_outer{};
        int num_blocks = n/block_size;
        Int_accum_inner partial_sum = Int_accum_inner{};
        const int max_exp = std::numeric_limits<Fp_in1>::max_exponent + get_bias<Fp_in1> + std::numeric_limits<Fp_in2>::max_exponent + get_bias<Fp_in2>;
        const int max_mantissa_size = get_mantissa_bits_v<Fp_in1> + get_mantissa_bits_v<Fp_in2>;
        double rel_max = 0.0;
        for(int blk = 0; blk < num_blocks; blk++) {

            //first loop finds max entry, second loop performs relative normalization and accumulates
            for(int j = blk*block_size; j < std::min(n, (blk+1)*block_size); j++) {
                rel_max = max(double(x[j]*double(y[j])), rel_max);
            }

            //now must calculate exponent shift, since we are double, exp is frexp + 1024
            int shift_offset =  0; //this way, if something has exp, we will shift it shift_offset + exp amount and then. add
            std::frexp(rel_max, &shift_offset);
            shift_offset = -shift_offset - 1024;
            shift_offset += get_int_len_v<Int_accum_inner> - max_mantissa_size;

            for(int i = blk*block_size; i < std::min(n, (blk+1)*block_size); i++) {

                const int exp_x = func_get_exponent_bits(x[i]) ;
                const int exp_y = func_get_exponent_bits(y[i]) ;

                const Int_accum_inner mantissa_x = (Int_accum_inner)(exp_x == 0 ? 0 : 1 << get_mantissa_bits_v<Fp_in1>) + func_get_mantissa_bits(x[i]);
                const Int_accum_inner mantissa_y = (Int_accum_inner)(exp_x == 0 ? 0 : 1 << get_mantissa_bits_v<Fp_in2>) + func_get_mantissa_bits(y[i]);
                const Int_accum_inner prod = mantissa_x*mantissa_y;
                const bool sign = func_get_sign_bit(x[i]) ^ func_get_sign_bit(y[i]);
                const int new_exp = exp_x + exp_y;
                
                const int num_shift = shift_offset + new_exp; 
                partial_sum += (sign ? -prod : prod);
            }
            rel_max = 0.0;
            to_ret +=  static_cast<Fp_accum_outer>(partial_sum);
            partial_sum = Int_accum_inner{};
        }

        

        return static_cast<Fp_out>(to_ret);;


    }

    
}


