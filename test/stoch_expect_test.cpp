#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>

#include "lo_float.h"  // Your custom library header

using namespace lo_float;

// ---------------------------------------------------------------------------
// Example "dummy" isInf/isNaN classes for 32-bit floats
// (You might have your own real versions in lo_float.)
// ---------------------------------------------------------------------------
struct IsInf_f32 {
    bool operator()(uint32_t bits) const {
        // Typical IEEE754 check: exponent=255 && fraction=0 => I nf
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t fraction = bits & 0x7FFFFF;
        return (exponent == 0xFF && fraction == 0);
    }
    uint32_t infBitPattern()    const { return 0x7F800000; }  // +Inf
    uint32_t minNegInf()        const { return 0xFF800000; }  // -Inf
    uint32_t minPosInf()        const { return 0x7F800000; }  // same as +Inf
};

struct IsNaN_f32 {
    bool operator()(uint32_t bits) const {
        // exponent=255 && fraction != 0 => NaN
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t fraction = bits & 0x7FFFFF;
        return (exponent == 0xFF && fraction != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; }  // typical QNaN
    uint32_t sNanBitPattern() const { return 0x7FA00000; }  // some SNaN pattern
};

// ---------------------------------------------------------------------------
// Build 3 param sets for a “32-bit float” style format, each with a different
// rounding mode or stochastic_rounding_length (1, 5, and 10).
// (Notice for sr10 we used RoundToNearestEven in this example.)
// ---------------------------------------------------------------------------
constexpr FloatingPointParams param_fp32_sr1(
    32, 23, 127,
    Rounding_Mode::StochasticRoundingC,
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(),
    IsNaN_f32(),
    /* stochastic_rounding_length = */ 1
);

constexpr FloatingPointParams param_fp32_rn(
    32, 23, 127,
    Rounding_Mode::RoundToNearestEven,  // a different rounding mode
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(),
    IsNaN_f32()
);

constexpr FloatingPointParams param_fp32_sr5(
    32, 23, 127,
    Rounding_Mode::StochasticRoundingC,
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(),
    IsNaN_f32(),
    /* stochastic_rounding_length = */ 5
);

constexpr FloatingPointParams param_fp32_sr10(
    32, 23, 127,
    Rounding_Mode::StochasticRoundingC,  // a different rounding mode
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(),
    IsNaN_f32(),
    /* stochastic_rounding_length = */ 10
);

// Create the custom “fp32”-ish types
using float32_sr1  = Templated_Float<param_fp32_sr1>;
using float32_sr5  = Templated_Float<param_fp32_sr5>;
using float32_sr10 = Templated_Float<param_fp32_sr10>;
using float32_rn   = Templated_Float<param_fp32_rn>;  // RoundToNearestEven




// ---------------------------------------------------------------------------
// Helper function: compute the relative error = |(val - ref)/ref|, if ref != 0
// If |ref| < a tiny epsilon, return 0 or skip the sample if you prefer.
// ---------------------------------------------------------------------------
double relative_error(double val, double ref, double eps = 1e-15)
{
    if(std::fabs(ref) < eps) {
        // Could return 0, or treat as 'no valid measurement'
        return 0.0; 
    }
    return std::fabs((val - ref) / ref);
}

int main()
{
    // Seed the library's *internal* RNG for stochastic rounding
    lo_float::set_seed(248);

    // Also seed our local random generator for test input data
    std::mt19937 rng(248);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    int N = 5; // number of random pairs

    // We’ll accumulate sum of REL errors for each op in each type (including native float)
    double sumErrAdd_sr1  = 0.0, sumErrSub_sr1  = 0.0, sumErrMul_sr1  = 0.0, sumErrDiv_sr1  = 0.0;
    double sumErrAdd_sr5  = 0.0, sumErrSub_sr5  = 0.0, sumErrMul_sr5  = 0.0, sumErrDiv_sr5  = 0.0;
    double sumErrAdd_sr10 = 0.0, sumErrSub_sr10 = 0.0, sumErrMul_sr10 = 0.0, sumErrDiv_sr10 = 0.0;
    double sumErrAdd_rn   = 0.0, sumErrSub_rn   = 0.0, sumErrMul_rn   = 0.0, sumErrDiv_rn   = 0.0;

    double sumErrAdd_float = 0.0, sumErrSub_float = 0.0, sumErrMul_float = 0.0, sumErrDiv_float = 0.0;

    // We'll track how many valid samples for each operation
    int countAdd = 0, countSub = 0, countMul = 0, countDiv = 0;

    //print min/max values with numeric_limits
std::cout << "float32_sr1 min rep: " << (double)std::numeric_limits<float32_sr1>::min().rep() << "\n";
std::cout << "float32_sr1 max: " << (double)std::numeric_limits<float32_sr1>::max() << "\n";
std::cout << "float32_sr5 min: " << (double)std::numeric_limits<float32_sr5>::min() << "\n";
std::cout << "float32_sr5 max: " << (double)std::numeric_limits<float32_sr5>::max() << "\n";
std::cout << "float32_sr10 min: " << (double)std::numeric_limits<float32_sr10>::min() << "\n";
std::cout << "float32_sr10 max: " << (double)std::numeric_limits<float32_sr10>::max() << "\n";

std::cout << "float32_sr1 denorm_min: " << (double)std::numeric_limits<float32_sr1>::denorm_min() << "\n";

std::cout << "float32_sr1 mantissa_bits : " << float32_sr1::mantissa_bits << "\n";

    for(int i = 0; i < N; i++) {
        double x_d = dist(rng);
        double y_d = dist(rng);

        // Avoid division by zero (or near zero) for the test
        if(std::fabs(y_d) < 1e-15) {
            y_d = 1.0;
        }

        // Convert input doubles into each custom float type
        float32_sr1  x_sr1  = static_cast<float32_sr1>(x_d);
        float32_sr1  y_sr1  = static_cast<float32_sr1>(y_d);

        std::cout << "x_sr1: " << double(x_sr1) << ", y_sr1: " << double(y_sr1) << "\n";

        float32_sr5  x_sr5  = static_cast<float32_sr5>(x_d);
        float32_sr5  y_sr5  = static_cast<float32_sr5>(y_d);

        float32_sr10 x_sr10 = static_cast<float32_sr10>(x_d);
        float32_sr10 y_sr10 = static_cast<float32_sr10>(y_d);

        float32_rn   x_rn   = static_cast<float32_rn>(x_d);
        float32_rn   y_rn   = static_cast<float32_rn>(y_d);



        // Convert to native float
        float x_f = static_cast<float>(x_d);
        float y_f = static_cast<float>(y_d);

        std::cout << "x_f: " << x_f << ", y_f: " << y_f << "\n";

    //     // Double-precision "references"
        double add_ref = x_d + y_d;
        double sub_ref = x_d - y_d;
        double mul_ref = x_d * y_d;
        double div_ref = x_d / y_d; // safe now that y_d != 0

        // Perform arithmetic in each custom type
        double add_sr1   = static_cast<double>(x_sr1 + y_sr1);
        double sub_sr1   = static_cast<double>(x_sr1 - y_sr1);
        double mul_sr1   = static_cast<double>(x_sr1 * y_sr1);
        double div_sr1   = static_cast<double>(x_sr1 / y_sr1);

        double add_sr5   = static_cast<double>(x_sr5 + y_sr5);
        double sub_sr5   = static_cast<double>(x_sr5 - y_sr5);
        double mul_sr5   = static_cast<double>(x_sr5 * y_sr5);
        double div_sr5   = static_cast<double>(x_sr5 / y_sr5);

        double add_sr10  = static_cast<double>(x_sr10 + y_sr10);
        double sub_sr10  = static_cast<double>(x_sr10 - y_sr10);
        double mul_sr10  = static_cast<double>(x_sr10 * y_sr10);
        double div_sr10  = static_cast<double>(x_sr10 / y_sr10);

        double add_srrn   = static_cast<double>(x_rn + y_rn);
        double sub_srrn   = static_cast<double>(x_rn - y_rn);
        double mul_srrn   = static_cast<double>(x_rn * y_rn);
        double div_srrn   = static_cast<double>(x_rn / y_rn);

        // Perform arithmetic in native float, then cast back to double
        double add_f   = static_cast<double>(x_f + y_f);
        double sub_f   = static_cast<double>(x_f - y_f);
        double mul_f   = static_cast<double>(x_f * y_f);
        double div_f   = static_cast<double>(x_f / y_f);

        // For each operation, measure relative error if reference is not too small

        // Add
        {
            double re_sr1   = relative_error(add_sr1,  add_ref);
            double re_sr5   = relative_error(add_sr5,  add_ref);
            double re_sr10  = relative_error(add_sr10, add_ref);
            double re_srrn  = relative_error(add_srrn, add_ref);
            double re_f     = relative_error(add_f,    add_ref);

            if(re_sr1 >= 0.0) {
                sumErrAdd_sr1  += re_sr1;
                sumErrAdd_sr5  += re_sr5;
                sumErrAdd_sr10 += re_sr10;
                sumErrAdd_rn   += re_srrn;
                sumErrAdd_float += re_f;
                countAdd++;
            }
        }

        // Sub
        {
            double re_sr1   = relative_error(sub_sr1,  sub_ref);
            double re_sr5   = relative_error(sub_sr5,  sub_ref);
            double re_sr10  = relative_error(sub_sr10, sub_ref);
            double re_srrn  = relative_error(sub_srrn, sub_ref);
            double re_f     = relative_error(sub_f,    sub_ref);

            if(re_sr1 >= 0.0) {
                sumErrSub_sr1  += re_sr1;
                sumErrSub_sr5  += re_sr5;
                sumErrSub_sr10 += re_sr10;
                sumErrSub_float += re_f;
                sumErrSub_rn   += re_srrn;
                countSub++;
            }
        }

        // Mul
        {
            double re_sr1   = relative_error(mul_sr1,  mul_ref);
            double re_sr5   = relative_error(mul_sr5,  mul_ref);
            double re_sr10  = relative_error(mul_sr10, mul_ref);
            double re_f     = relative_error(mul_f,    mul_ref);
            double re_srrn  = relative_error(mul_srrn, mul_ref);

            if(re_sr1 >= 0.0) {
                sumErrMul_sr1  += re_sr1;
                sumErrMul_sr5  += re_sr5;
                sumErrMul_sr10 += re_sr10;
                sumErrMul_float += re_f;
                sumErrMul_rn   += re_srrn;
                countMul++;
            }
        }

        // Div
        {
            double re_sr1   = relative_error(div_sr1,  div_ref);
            double re_sr5   = relative_error(div_sr5,  div_ref);
            double re_sr10  = relative_error(div_sr10, div_ref);
            double re_f     = relative_error(div_f,    div_ref);
            double re_srrn  = relative_error(div_srrn, div_ref);

            if(re_sr1 >= 0.0) {
                sumErrDiv_sr1  += re_sr1;
                sumErrDiv_sr5  += re_sr5;
                sumErrDiv_sr10 += re_sr10;
                sumErrDiv_float += re_f;
                sumErrDiv_rn   += re_srrn;
                countDiv++;
            }
        }
    }

    // Compute mean relative errors (avoid dividing by zero if count=0)
    auto safeDivide = [](double sum, int count) {
        return (count > 0) ? (sum / count) : 0.0;
    };

    double meanErrAdd_sr1   = safeDivide(sumErrAdd_sr1,   countAdd);
    double meanErrSub_sr1   = safeDivide(sumErrSub_sr1,   countSub);
    double meanErrMul_sr1   = safeDivide(sumErrMul_sr1,   countMul);
    double meanErrDiv_sr1   = safeDivide(sumErrDiv_sr1,   countDiv);

    double meanErrAdd_sr5   = safeDivide(sumErrAdd_sr5,   countAdd);
    double meanErrSub_sr5   = safeDivide(sumErrSub_sr5,   countSub);
    double meanErrMul_sr5   = safeDivide(sumErrMul_sr5,   countMul);
    double meanErrDiv_sr5   = safeDivide(sumErrDiv_sr5,   countDiv);

    double meanErrAdd_sr10  = safeDivide(sumErrAdd_sr10,  countAdd);
    double meanErrSub_sr10  = safeDivide(sumErrSub_sr10,  countSub);
    double meanErrMul_sr10  = safeDivide(sumErrMul_sr10,  countMul);
    double meanErrDiv_sr10  = safeDivide(sumErrDiv_sr10,  countDiv);

    double meanErrAdd_float = safeDivide(sumErrAdd_float, countAdd);
    double meanErrSub_float = safeDivide(sumErrSub_float, countSub);
    double meanErrMul_float = safeDivide(sumErrMul_float, countMul);
    double meanErrDiv_float = safeDivide(sumErrDiv_float, countDiv);
    
    double meanErrAdd_rn    = safeDivide(sumErrAdd_rn,    countAdd);
    double meanErrSub_rn    = safeDivide(sumErrSub_rn,    countSub);
    double meanErrMul_rn    = safeDivide(sumErrMul_rn,    countMul);
    double meanErrDiv_rn    = safeDivide(sumErrDiv_rn,    countDiv);
    // Print results
    std::cout << std::fixed << std::setprecision(8);

    std::cout << "\n=== Stochastic Rounding: length=1 ===\n";
    std::cout << "Add mean REL error: " << meanErrAdd_sr1  <<  " log2 of error : " << log2f(meanErrAdd_sr1) << "\n";
    std::cout << "Sub mean REL error: " << meanErrSub_sr1  << "\n";
    std::cout << "Mul mean REL error: " << meanErrMul_sr1  << "\n";
    std::cout << "Div mean REL error: " << meanErrDiv_sr1  << "\n";

    std::cout << "\n=== Stochastic Rounding: length=5 ===\n";
    std::cout << "Add mean REL error: " << meanErrAdd_sr5  <<  " log2 of error : " << log2f(meanErrAdd_sr5) << "\n";
    std::cout << "Sub mean REL error: " << meanErrSub_sr5  << "\n";
    std::cout << "Mul mean REL error: " << meanErrMul_sr5  << "\n";
    std::cout << "Div mean REL error: " << meanErrDiv_sr5  << "\n";

    std::cout << "\n=== Stochastic Rounding: length=10 ===\n";
    std::cout << "Add mean REL error: " << meanErrAdd_sr10 <<  " log2 of error : " << log2f(meanErrAdd_sr10) << "\n";
    std::cout << "Sub mean REL error: " << meanErrSub_sr10 << "\n";
    std::cout << "Mul mean REL error: " << meanErrMul_sr10 << "\n";
    std::cout << "Div mean REL error: " << meanErrDiv_sr10 << "\n";

    std::cout << "\n=== RoundToNearestEven ===\n";
    std::cout << "Add mean REL error: " << meanErrAdd_rn   <<  " log2 of error : " << log2f(meanErrAdd_rn) << "\n";
    std::cout << "Sub mean REL error: " << meanErrSub_rn   << "\n";
    std::cout << "Mul mean REL error: " << meanErrMul_rn   << "\n";
    std::cout << "Div mean REL error: " << meanErrDiv_rn   << "\n";

    std::cout << "\n=== Native float (C++ built-in) ===\n";
    std::cout << "Add mean REL error: " << meanErrAdd_float <<  " log2 of error : " << log2f(meanErrAdd_float) << "\n";
    std::cout << "Sub mean REL error: " << meanErrSub_float << "\n";
    std::cout << "Mul mean REL error: " << meanErrMul_float << "\n";
    std::cout << "Div mean REL error: " << meanErrDiv_float << "\n";

    std::cout << std::endl;
    
    return 0;
}
