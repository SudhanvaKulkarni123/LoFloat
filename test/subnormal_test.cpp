//test to check rounding of subnormal numbers. Test by intializing some verey small double and rounding to fp8

#include "lo_float.h"
#include <iostream>

using namespace lo_float;
int main() {
    // Create a very small double
    double small_double = 1e-5;

    // Convert to lo_float with stochastic rounding
    static constexpr FloatingPointParams param_fp8(
        8, /*mant*/6, /*bias*/1,
        Rounding_Mode::StochasticRoundingA,
        Inf_Behaviors::Saturating, NaN_Behaviors::QuietNaN,
        Signedness::Signed,
        lo_float_internal::IEEE_F8_InfChecker(),
        lo_float_internal::IEEE_F8_NaNChecker(),
        0
    );
    using fp8 = Templated_Float<param_fp8>;   // round-to-nearest-even
    fp8 result = fp8(small_double);

    // Print the result
    std::cout << "Original double: " << small_double << std::endl;
    std::cout << "Converted to lo_float: " << static_cast<float>(result) << std::endl;

    return 0;

}
