#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "lo_float.h"

using namespace lo_float;

int main() {
    //test defining custom float struct and try using TemplatedFloat

    //define Isinf and isNaN functors 

    class IsInf_f8_e4m3 {
        
        public:

        bool operator()(uint32_t bits) const {
            return bits == 0x0000007F || bits == 0x000000FF;    
        }

        uint32_t infBitPattern() const {
            return 0x0000000F;
        }

        uint32_t minNegInf() const {
            return 0x0000001F;      //1E
        }

        uint32_t minPosInf() const {
            return 0x0000000F;      //E
        }

    };

    class IsNaN_f8_e4m3 {
        
        public:

        bool operator()(uint32_t bits) const {
            return 0x0;
        }

        uint32_t qNanBitPattern() const {
            return 0x00000;
        }

        uint32_t sNanBitPattern() const {
            return 0x0000;
        }
    };

    constexpr FloatingPointParams f8_e4m3_params(5, 2, 4, 
        Rounding_Mode::RoundToNearestEven, Inf_Behaviors::HasNonTrappingInf, 
        NaN_Behaviors::NoNaN, Signedness::Signed,
        IsInf_f8_e4m3(), IsNaN_f8_e4m3()
    );

    //create templated float using these params

    using float8_e4m3 = Templated_Float<f8_e4m3_params>;
    float8_e4m3 a = std::numeric_limits<float8_e4m3>::max();
    float8_e4m3 b = std::numeric_limits<float8_e4m3>::denorm_min();
    float8_e4m3 c = a - b;
    //also print out the reps
    std::cout << "a: " << a << std::endl;
    std::cout << "a rep: " << (int)a.rep() << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "b rep: " << (int)b.rep() << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "c rep: " << (int)c.rep() << std::endl;


    return 0;
}