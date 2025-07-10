//test to compare generated values against table of expected values from P3109 interim report
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>

#include "lo_float.h"


using namespace lo_float;

int main() {
    static constexpr FloatingPointParams param_fp8(
        8, /*mant*/6, /*bias*/1,
        Inf_Behaviors::Saturating, NaN_Behaviors::QuietNaN,
        Signedness::Signed,
        lo_float_internal::IEEE_F8_InfChecker(),
        lo_float_internal::IEEE_F8_NaNChecker()
    );

    using rne8 = Templated_Float<param_fp8>;   // round-to-nearest-even
    //increase number of digits displayed on terminal
    std::cout << std::setprecision(20);
    
    for(int i = 0; i < 256; i++) {
        std::cout << (double)rne8::FromRep(i) << ", " << std::endl;
    }
    std::cout << "Done" << std::endl;
    return 0;
}