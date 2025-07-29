// -------------------------------------------------------------
// stochastic-rounding-distribution.cpp
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <random>
#include <limits>
#include "lo_float.h"          // your existing library

using namespace lo_float;

// ---------- helper traits for IEEE-754 single precision ----------
struct IsInf_f32 {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) == 0);
    }
    uint32_t infBitPattern() const { return 0x7F800000; }
    uint32_t minNegInf()     const { return 0xFF800000; }
    uint32_t minPosInf()     const { return 0x7F800000; }
};

struct IsNaN_f32 {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; }
    uint32_t sNanBitPattern() const { return 0x7FA00000; }
};


constexpr FloatingPointParams param_fp32(
    /*total bits*/ 32, /*mantissa*/ 23, /*bias*/ 127,
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(), IsNaN_f32());

using float32 = Templated_Float<param_fp32>;   // always the same storage layout

// ---------- generic experiment for any stochastic-length ----------
void run_one_len(const char* label,
                 double      testVal,
                 int         stoch_len,
                 int         trials,
                const float32 d_down, const float32 d_up)
{
    int hitLower = 0, hitUpper = 0;

    for (int i = 0; i < trials; ++i) {
        // --- perform conversion with stochastic rounding of 'stoch_len' bits
        float32 fp = Round<double, float32>(
            testVal, Rounding_Mode::StochasticRoundingC, stoch_len);

        double d = static_cast<double>(fp);

        // --- reference bounds using deterministic rounding

        if (d == (double)d_down) ++hitLower;
        else if (d == (double)d_up) ++hitUpper;
        
    }

    std::cout << "  " << std::left << std::setw(4) << label
              << ":  P(down) = " << std::setw(6)
              << std::fixed << std::setprecision(2)
              << (100.0 * hitLower / trials) << "%   "
              << "P(up) = " << std::setw(6)
              << (100.0 * hitUpper / trials) << "%\n";
}

// -----------------------------------------------------------------
int main()
{
    // 0. configuration -------------------------------------------------
    constexpr int trials = 1'000'000;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float>  pick_float(-100.0f, 100.0f);
    std::uniform_real_distribution<double> pick_frac (0.0, 1.0);

    // 1. pick a random double strictly between two neighbouring floats --
    float f_low, f_up;
    while (true) {
        f_low = pick_float(rng);
        f_up  = std::nextafterf(f_low, std::numeric_limits<float>::infinity());
        if (std::isfinite(f_up) && f_up != f_low) break;
    }
    double t;
    do { t = pick_frac(rng); } while (t == 0.0 || t == 1.0);

    const double x   = static_cast<double>(f_low) +
                       t * (static_cast<double>(f_up) - f_low);
    const double ulp = static_cast<double>(f_up) - f_low;
    const double p_up = (x - static_cast<double>(f_low)) / ulp;
    const double p_dn = 1.0 - p_up;

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "\nRandom double x        = " << x   << '\n'
              << "Lower single f_low      = " << f_low << '\n'
              << "Upper single f_up       = " << f_up  << '\n'
              << "Ideal P(down)           = " << p_dn * 100.0 << "%\n"
              << "Ideal P(up)             = " << p_up * 100.0 << "%\n\n";

    // 2. Monte-Carlo experiment ----------------------------------------
    lo_float::set_seed(static_cast<std::uint64_t>(std::random_device{}()));
    std::cout << "Observed (" << trials << " trials)\n\n";

    for (int len = 1; len <= 10; ++len) {
        char name[8]; std::snprintf(name, sizeof(name), "sr%d", len);
        run_one_len(name, x, len, trials, float32(f_low), float32(f_up));
    }
    return 0;
}
