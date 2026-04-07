#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>

#include "lo_float.h"

using namespace lo_float;

// ---------------------------------------------------------------------------
// IsInf/IsNaN classes for 32-bit floats
// ---------------------------------------------------------------------------
struct IsInf_f32 {
    bool operator()(uint32_t bits) const {
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t fraction = bits & 0x7FFFFF;
        return (exponent == 0xFF && fraction == 0);
    }
    uint32_t infBitPattern()    const { return 0x7F800000; }
    uint32_t minNegInf()        const { return 0xFF800000; }
    uint32_t minPosInf()        const { return 0x7F800000; }
};

struct IsNaN_f32 {
    bool operator()(uint32_t bits) const {
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t fraction = bits & 0x7FFFFF;
        return (exponent == 0xFF && fraction != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; }
    uint32_t sNanBitPattern() const { return 0x7FA00000; }
};

// ---------------------------------------------------------------------------
// One single FloatingPointParams for fp32 (no rounding mode or stochastic_len)
// Per the paper: FloatingPointParams(bitwidth, mantissa_bits, bias,
//                  Inf_Behaviors, NaN_Behaviors, Signedness, IsInf, IsNaN)
// ---------------------------------------------------------------------------
constexpr FloatingPointParams param_fp32(
    32, 23, 127,
    Inf_Behaviors::Extended,
    NaN_Behaviors::_754,
    Signedness::Signed,
    IsInf_f32(),
    IsNaN_f32()
);

using float32 = Templated_Float<param_fp32>;

// ---------------------------------------------------------------------------
// Helper: relative error
// ---------------------------------------------------------------------------
double relative_error(double val, double ref, double eps = 1e-15)
{
    if (std::fabs(ref) < eps) return 0.0;
    return std::fabs((val - ref) / ref);
}

int main()
{
    lo_float::set_seed(248);

    std::mt19937 rng(248);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    int N = 5;

    // Accumulators for mean relative error per (rounding_mode, operation)
    // Modes: SR1, SR5, SR10, RN, native float
    double sumErrAdd_sr1  = 0, sumErrSub_sr1  = 0, sumErrMul_sr1  = 0, sumErrDiv_sr1  = 0;
    double sumErrAdd_sr5  = 0, sumErrSub_sr5  = 0, sumErrMul_sr5  = 0, sumErrDiv_sr5  = 0;
    double sumErrAdd_sr10 = 0, sumErrSub_sr10 = 0, sumErrMul_sr10 = 0, sumErrDiv_sr10 = 0;
    double sumErrAdd_rn   = 0, sumErrSub_rn   = 0, sumErrMul_rn   = 0, sumErrDiv_rn   = 0;
    double sumErrAdd_float= 0, sumErrSub_float= 0, sumErrMul_float= 0, sumErrDiv_float= 0;

    int countAdd = 0, countSub = 0, countMul = 0, countDiv = 0;

    // Print limits
    std::cout << "float32 min: " << (double)std::numeric_limits<float32>::min() << "\n";
    std::cout << "float32 max: " << (double)std::numeric_limits<float32>::max() << "\n";
    std::cout << "float32 denorm_min: " << (double)std::numeric_limits<float32>::denorm_min() << "\n";
    std::cout << "float32 mantissa_bits: " << float32::mantissa_bits << "\n";

    for (int i = 0; i < N; i++) {
        double x_d = dist(rng);
        double y_d = dist(rng);
        if (std::fabs(y_d) < 1e-15) y_d = 1.0;

        // Convert inputs to float32 (rounding applied at conversion via Round)
        float32 x_f32 = static_cast<float32>(x_d);
        float32 y_f32 = static_cast<float32>(y_d);

        std::cout << "x_f32: " << double(x_f32) << ", y_f32: " << double(y_f32) << "\n";

        // Native float for comparison
        float x_f = static_cast<float>(x_d);
        float y_f = static_cast<float>(y_d);
        std::cout << "x_f: " << x_f << ", y_f: " << y_f << "\n";

        // Double-precision references
        double add_ref = x_d + y_d;
        double sub_ref = x_d - y_d;
        double mul_ref = x_d * y_d;
        double div_ref = x_d / y_d;

        // --- StochasticRoundingC, stochastic_len=1 ---
        // Per paper: Add(a, b, Rounding_Mode, stochastic_len) or
        //            StochasticAdd(a, b, stochastic_len) with mode set elsewhere.
        //            Using the explicit rounding mode + stochastic_len API.
        {
            float32 add_sr1 = Add(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 1);
            float32 sub_sr1 = Sub(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 1);
            float32 mul_sr1 = Mul(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 1);
            float32 div_sr1 = Div(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 1);

            sumErrAdd_sr1 += relative_error(double(add_sr1), add_ref);
            sumErrSub_sr1 += relative_error(double(sub_sr1), sub_ref);
            sumErrMul_sr1 += relative_error(double(mul_sr1), mul_ref);
            sumErrDiv_sr1 += relative_error(double(div_sr1), div_ref);
        }

        // --- StochasticRoundingC, stochastic_len=5 ---
        {
            float32 add_sr5 = Add(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 5);
            float32 sub_sr5 = Sub(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 5);
            float32 mul_sr5 = Mul(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 5);
            float32 div_sr5 = Div(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 5);

            sumErrAdd_sr5 += relative_error(double(add_sr5), add_ref);
            sumErrSub_sr5 += relative_error(double(sub_sr5), sub_ref);
            sumErrMul_sr5 += relative_error(double(mul_sr5), mul_ref);
            sumErrDiv_sr5 += relative_error(double(div_sr5), div_ref);
        }

        // --- StochasticRoundingC, stochastic_len=10 ---
        {
            float32 add_sr10 = Add(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 10);
            float32 sub_sr10 = Sub(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 10);
            float32 mul_sr10 = Mul(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 10);
            float32 div_sr10 = Div(x_f32, y_f32, Rounding_Mode::StochasticRoundingC, 10);

            sumErrAdd_sr10 += relative_error(double(add_sr10), add_ref);
            sumErrSub_sr10 += relative_error(double(sub_sr10), sub_ref);
            sumErrMul_sr10 += relative_error(double(mul_sr10), mul_ref);
            sumErrDiv_sr10 += relative_error(double(div_sr10), div_ref);
        }

        // --- RoundTiesToEven (deterministic, no stochastic_len) ---
        {
            float32 add_rn = Add(x_f32, y_f32, Rounding_Mode::RoundToNearestEven);
            float32 sub_rn = Sub(x_f32, y_f32, Rounding_Mode::RoundToNearestEven);
            float32 mul_rn = Mul(x_f32, y_f32, Rounding_Mode::RoundToNearestEven);
            float32 div_rn = Div(x_f32, y_f32, Rounding_Mode::RoundToNearestEven);

            sumErrAdd_rn += relative_error(double(add_rn), add_ref);
            sumErrSub_rn += relative_error(double(sub_rn), sub_ref);
            sumErrMul_rn += relative_error(double(mul_rn), mul_ref);
            sumErrDiv_rn += relative_error(double(div_rn), div_ref);
        }

        // --- Native float ---
        {
            double add_f = static_cast<double>(x_f + y_f);
            double sub_f = static_cast<double>(x_f - y_f);
            double mul_f = static_cast<double>(x_f * y_f);
            double div_f = static_cast<double>(x_f / y_f);

            sumErrAdd_float += relative_error(add_f, add_ref);
            sumErrSub_float += relative_error(sub_f, sub_ref);
            sumErrMul_float += relative_error(mul_f, mul_ref);
            sumErrDiv_float += relative_error(div_f, div_ref);
        }

        countAdd++; countSub++; countMul++; countDiv++;
    }

    auto safeDivide = [](double sum, int count) {
        return (count > 0) ? (sum / count) : 0.0;
    };

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "\n=== StochasticRoundingC: stochastic_len=1 ===\n";
    std::cout << "Add mean REL error: " << safeDivide(sumErrAdd_sr1, countAdd)
              << " log2: " << log2(safeDivide(sumErrAdd_sr1, countAdd)) << "\n";
    std::cout << "Sub mean REL error: " << safeDivide(sumErrSub_sr1, countSub) << "\n";
    std::cout << "Mul mean REL error: " << safeDivide(sumErrMul_sr1, countMul) << "\n";
    std::cout << "Div mean REL error: " << safeDivide(sumErrDiv_sr1, countDiv) << "\n";

    std::cout << "\n=== StochasticRoundingC: stochastic_len=5 ===\n";
    std::cout << "Add mean REL error: " << safeDivide(sumErrAdd_sr5, countAdd)
              << " log2: " << log2(safeDivide(sumErrAdd_sr5, countAdd)) << "\n";
    std::cout << "Sub mean REL error: " << safeDivide(sumErrSub_sr5, countSub) << "\n";
    std::cout << "Mul mean REL error: " << safeDivide(sumErrMul_sr5, countMul) << "\n";
    std::cout << "Div mean REL error: " << safeDivide(sumErrDiv_sr5, countDiv) << "\n";

    std::cout << "\n=== StochasticRoundingC: stochastic_len=10 ===\n";
    std::cout << "Add mean REL error: " << safeDivide(sumErrAdd_sr10, countAdd)
              << " log2: " << log2(safeDivide(sumErrAdd_sr10, countAdd)) << "\n";
    std::cout << "Sub mean REL error: " << safeDivide(sumErrSub_sr10, countSub) << "\n";
    std::cout << "Mul mean REL error: " << safeDivide(sumErrMul_sr10, countMul) << "\n";
    std::cout << "Div mean REL error: " << safeDivide(sumErrDiv_sr10, countDiv) << "\n";

    std::cout << "\n=== RoundTiesToEven ===\n";
    std::cout << "Add mean REL error: " << safeDivide(sumErrAdd_rn, countAdd)
              << " log2: " << log2(safeDivide(sumErrAdd_rn, countAdd)) << "\n";
    std::cout << "Sub mean REL error: " << safeDivide(sumErrSub_rn, countSub) << "\n";
    std::cout << "Mul mean REL error: " << safeDivide(sumErrMul_rn, countMul) << "\n";
    std::cout << "Div mean REL error: " << safeDivide(sumErrDiv_rn, countDiv) << "\n";

    std::cout << "\n=== Native float (C++ built-in) ===\n";
    std::cout << "Add mean REL error: " << safeDivide(sumErrAdd_float, countAdd)
              << " log2: " << log2(safeDivide(sumErrAdd_float, countAdd)) << "\n";
    std::cout << "Sub mean REL error: " << safeDivide(sumErrSub_float, countSub) << "\n";
    std::cout << "Mul mean REL error: " << safeDivide(sumErrMul_float, countMul) << "\n";
    std::cout << "Div mean REL error: " << safeDivide(sumErrDiv_float, countDiv) << "\n";

    std::cout << std::endl;
    return 0;
}