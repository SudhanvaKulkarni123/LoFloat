#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <cstdint>
#include "lo_float.h"
#include "lo_float_sci.hpp"

int main() {
    using namespace lo_float;

    class IsInf_f8_e4m3 {    
        public:

        bool operator()(uint32_t bits) const {
            return bits == 0x0000007F || bits == 0x000000FF;    
        }

        uint32_t infBitPattern() const {
            return 0x00000007F;
        }

        uint32_t minNegInf() const {
            return 0x000000FF;      //1E
        }

        uint32_t minPosInf() const {
            return 0x0000007F;      //E
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


    constexpr FloatingPointParams param_ufp8(
        8, 4, 3,
        Inf_Behaviors::Saturating,
        NaN_Behaviors::_3109,
        Signedness::Unsigned,
        IsInf_f8_e4m3(), IsNaN_f8_e4m3()
    );

    using float8 = Templated_Float<param_ufp8>;

    // Use deduced Project — no explicit <float8> needed
    float8 zero = Project<float8>(0);
    float8 neg_one = Project<float8>(-1);  // NegtoZero should clamp this

    std::cout << "largest number in the format: " << std::numeric_limits<float8>::max() << std::endl;
    std::cout << "largest number in the format rep: " << (int)std::numeric_limits<float8>::max().rep() << std::endl;

    // Also test the free-function API with deduced output
    float8 test_a = float8::FromRep(0x66);
    float8 test_b = float8::FromRep(0xB3);

    // Deduced output — result type comes from the assignment target
    float8 sum  = Add(test_a, test_b);
    float8 diff = Sub(test_a, test_b);
    float8 prod = Mul(test_a, test_b);
    float8 quot = Div(test_a, test_b);


    float sum_f = Add(test_a, test_b);

    std::cout << "Free-function API (deduced):  Add=" << sum << " Sub=" << diff
              << " Mul=" << prod << " Div=" << quot << "\n";
    std::cout << "Free-function API (explicit float): Add=" << sum_f << "\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (int i = 0; i < 10; ++i) {
        uint8_t rand_a = dist(rng);
        uint8_t rand_b = dist(rng);
        std::cout << "a hex: " << std::hex << (int)rand_a
                  << "  b hex: " << std::hex << (int)rand_b << "\n";

        float8 a = float8::FromRep(rand_a);
        float8 b = float8::FromRep(rand_b);

        std::cout << std::dec << "Test " << i + 1 << ":\n";
        std::cout << "  a=" << (float)a << "  b=" << (float)b << "\n";

        // Operators (use internal dispatch)
        std::cout << "  Op:  Add=" << float(a + b)
                  << " Sub=" << float(a - b)
                  << " Mul=" << float(a * b)
                  << " Div=" << (b != zero ? float(a / b) : (float)neg_one)
                  << " Eq=" << (a == b)
                  << " Lt=" << (a < b)
                  << " Gt=" << (a > b) << "\n";

        // Free functions with deduced output
        float8 fa = Add(a, b);
        float8 fs = Sub(a, b);
        float8 fm = Mul(a, b);
        std::cout << "  Free: Add=" << float(fa) << " Sub=" << float(fs) << " Mul=" << float(fm) << "\n";
        std::cout << "--------------------------------\n";
    }
    return 0;
}