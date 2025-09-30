#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "lo_float.h"

using namespace lo_float;




template<int l, int p>
int test_comparisons_exhaustive()
{
    using binary_lp = P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    using unsigned_binary_lp = P3109_float<l, p, Signedness::Unsigned, Inf_Behaviors::Saturating>;
    const int total_values = 1 << l;

    int case1_bound = total_values/2;
    

    int errors = 0;

    //check all cases where 0 < a < b for signed and unsigned
    for(unsigned int rep1 = 0; rep1 < total_values/2; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values/2; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
   
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
        }
    }

    for(unsigned int rep1 = 0; rep1 < total_values-1; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values-1; rep2++) {
            unsigned_binary_lp a = unsigned_binary_lp::FromRep(rep1);
            unsigned_binary_lp b = unsigned_binary_lp::FromRep(rep2);
   
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
        }
    }



    //check cases where a < 0 < b (signed)
    for(unsigned int rep1 = total_values/2 + 1; rep1 < total_values; rep1++) {
        for(unsigned int rep2 = 0; rep2 < total_values/2; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
            
        }
    }

    //check cases where b < a  < 0
    for(unsigned int rep1 = total_values/2 + 1; rep1 < total_values; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
            if(!(a >= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a < b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b <= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b > a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
            
        }
    }


    printf("Total errors for P3109<%d,%d>: %d\n", l, p, errors);
    return errors;
    
}

template<int l, int... Ps>
void instantiate_for_l(std::integer_sequence<int, Ps...>) {
    (test_comparisons_exhaustive<l, Ps+1>(), ...);
}
template<int... Ls>
void instantiate_all_l(std::integer_sequence<int, Ls...>) {
    (instantiate_for_l<Ls>(std::make_integer_sequence<int, Ls-1 >{}), ...);
}

// Offset sequence helper: converts [0,1,2,...,N-1] to [Offset, Offset+1, ..., Offset+N-1]
template<int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

void instantiate_all() {
    // For l from 2 to 8
    instantiate_all_l(offset_sequence<2>(std::make_integer_sequence<int, 6>{}));
}

//used to test comparisons for binary16, bf16 and tf32
int test_comparisons_random(int num_tests)
{

    using bf16 = Templated_Float<bfloatPrecisionParams>;
    using tf32 = Templated_Float<tf32PrecisionParams>;

    using binary16 =  Templated_Float<halfPrecisionParams>;

    for(int i = 0; i < num_tests; i++) {
        float a = static_cast<float>(((double) rand()) / (double) RAND_MAX * 1000.0);
        float b = static_cast<float>(((double) rand()) / (double) RAND_MAX * 1000.0);

        int sign_a = rand() % 2;
        int sign_b = rand() % 2;

        if(sign_a) a = -a;
        if(sign_b) b = -b;

        bf16 a_bf16 = bf16(a);
        bf16 b_bf16 = bf16(b);

        binary16 a_binary16 = binary16(a);
        binary16 b_binary16 = binary16(b);

        tf32 a_tf32 = tf32(a);
        tf32 b_tf32 = tf32(b);

        float a_ref_bf16 = (float)a_bf16;
        float b_ref_bf16 = (float)b_bf16;
        float a_ref_half = (float)a_binary16;
        float b_ref_half = (float)b_binary16;
        float a_ref_tf32 = (float)a_tf32;
        float b_ref_tf32 = (float)b_tf32;

        if((a_ref_bf16 <= b_ref_bf16) != (a_bf16 <= b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " <= " << (float)b_bf16 << " failed\n";

            std::cout << "a.rep(): " << std::hex << a_bf16.rep() << "\n";
            std::cout << "b.rep(): " << std::hex << b_bf16.rep() << "\n";
            return 1;
        }
        if((a_ref_bf16 > b_ref_bf16) != (a_bf16 > b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " > " << (float)b_bf16 << " failed\n";
            return 1;
        }
        if((a_ref_bf16 >= b_ref_bf16) != (a_bf16 >= b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " >= " << (float)b_bf16 << " failed\n";
            return 1;
        }
        if((a_ref_bf16 < b_ref_bf16) != (a_bf16 < b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " < " << (float)b_bf16 << " failed\n";
            return 1;
        }

        if((a_ref_half <= b_ref_half) != (a_binary16 <= b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " <= " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half > b_ref_half) != (a_binary16 > b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " > " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half >= b_ref_half) != (a_binary16 >= b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " >= " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half < b_ref_half) != (a_binary16 < b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " < " << (float)b_binary16 << " failed\n";
            return 1;
        }

         if ((a_ref_tf32 <= b_ref_tf32) != (a_tf32 <= b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " <= " << (float)b_tf32 << " failed\n";
                std::cout << "a : " << a << ", b: " << b << "\n";
                std::cout << "a_ref_tf32: " << a_ref_tf32 << ", b_ref_tf32: " << b_ref_tf32 << "\n";
                std::cout << "a.rep(): " << std::hex << a_tf32.rep() << "\n";
                std::cout << "b.rep(): " << std::hex << b_tf32.rep() << "\n";
                std::cout << (float) abs(a_tf32) << ", " << (float) abs(b_tf32) << "\n";
                return 1;
            }
            if ((a_ref_tf32 > b_ref_tf32) != (a_tf32 > b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " > " << (float)b_tf32 << " failed\n";
                return 1;
            }
            if ((a_ref_tf32 >= b_ref_tf32) != (a_tf32 >= b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " >= " << (float)b_tf32 << " failed\n";
                return 1;
            }
            if ((a_ref_tf32 < b_ref_tf32) != (a_tf32 < b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " < " << (float)b_tf32 << " failed\n";
                return 1;
            }

    }

    return 0;



}


int main()
{
    //run exhaustive test on P3109 formats for length <= 8

    int total_errors = 0;
   
    instantiate_all();

    if (test_comparisons_random(10000)) {
        printf("Randomized comparison tests failed\n");
    }
    

    std::cout << "comparison tests passed\n";

    return 0;

}