//test for checking correctness of implementation of unsigned floats
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

#include <cstdint>
#include "lo_float.h"


//test comparisons, rounding negatives to zero and basic arithmetic ops

int main() {
    //define 8-bit unisgned floating point params
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
        Rounding_Mode::RoundToNearestEven,
        Inf_Behaviors::Saturating,
        NaN_Behaviors::QuietNaN,
        Signedness::Unsigned,
        IsInf_f8_e4m3(), IsNaN_f8_e4m3(), 0, Unsigned_behavior::NegtoZero
    );


    using float8 = Templated_Float<param_ufp8>;

    //0x66 = 0b0110_0110 = 2^(6 - 3)*(1 + 0.25 + 0.125) = 2^(3)*(1.375) = 11.0
    //0xb3 = 0b1011_0011 = 2^(b - 3)*(1 + 0.125 + 0.0625) = 2^(8)*(1.375) = 352.0
    //0xFF = 0b1111_1111 = 2^(f - 3)*(1 + 0.5 + 0.25 + 0.125 + 0.0625) = 7036.0


    //largest number in the format-
    std::cout << "largest number in the format: " << std::numeric_limits<float8>::max() << std::endl;
    std::cout << "largest number in the format rep: " << (int)std::numeric_limits<float8>::max().rep() << std::endl;

    //generate random numbers
    std::mt19937 rng(42); // Seed for reproducibility
    
    std::uniform_int_distribution<uint8_t> dist(0, 255); // 8-bit unsigned range
    for (int i = 0; i < 10; ++i) {
        uint8_t rand_a = dist(rng);
        std::cout << "a uint8_t repr in hex: " << std::hex << static_cast<int>(rand_a) << "\n";
        uint8_t rand_b = dist(rng);
        std::cout << "b uint8_t repr in hex: " << std::hex << static_cast<int>(rand_b) << "\n";

        float8 a = float8::FromRep(rand_a);
        float8 b = float8::FromRep(rand_b);

        // Print values
        std::cout << "Test " << i + 1 << ":\n";
        std::cout << "  a and b " << (a) << " and " << (b) << "\n";

        // Perform arithmetic operations
        std::cout << "  uint4  -> Add: " << (a + b)
                  << ", Sub: " << (a - b)
                  << ", Mul: " << (a * b)
                  << ", Comparisons: " << (a == b) << ", " << (a < b) << ", " << (a > b) 
                  << ", Div: " << (b != static_cast<float8>(0) ? (a / b) : static_cast<float8>(-1)) << "\n";

        std::cout << "--------------------------------\n";
    }

    return 0;
}