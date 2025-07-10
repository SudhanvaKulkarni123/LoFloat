#include "lo_float.h"

int main() {
    //Fp2 nan and inf checkers-

    using namespace lo_float;
    struct Fp2_InfChecker {
        bool operator()(uint8_t bits) const {
            return false;
        }
        uint8_t infBitPattern() const { return 0x0; }
        uint8_t minNegInf()     const { return 0x0; }
        uint8_t minPosInf()     const { return 0x0; }
    };

    struct Fp2_NaNChecker {
        bool operator()(uint8_t bits) const {
            return bits == 0xF;
        }
        uint8_t qNanBitPattern() const { return 0xF; } // typical QNaN
        uint8_t sNanBitPattern() const { return 0x0; } // some SNaN pattern
    };
    //fp params for signed 2-bit float->
    constexpr FloatingPointParams param_fp2(
        4, 4, -3,
        Rounding_Mode::RoundToNearestEven,
        Inf_Behaviors::Saturating,
        NaN_Behaviors::QuietNaN,
        Signedness::Unsigned,
        Fp2_InfChecker(), Fp2_NaNChecker()
    );

    using float2 = Templated_Float<param_fp2>;  

    for(int i = 0; i < 16; i++){
        float2 f = float2::FromRep(i);
        std::cout << "float2: " << f << std::endl;
    }

    return 0;
}