#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

#include <cstdint>
#include "lo_float.h"

using namespace lo_float;
int main() {

    using fp8 = float8_ieee_p<4, Rounding_Mode::StochasticRoundingA, 4>;

    auto A = fp8(1.0);
    auto B = fp8();
}