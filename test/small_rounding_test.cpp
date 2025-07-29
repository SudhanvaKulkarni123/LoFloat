// -----------------------------------------------------------------------------
//  fp64  ➜  fp8  stochastic-rounding experiment  (very small inputs)
//  Uses the same `lo_float` infrastructure you already have
// -----------------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdint>
#include <cmath>

#include "lo_float.h"

using namespace lo_float;

// -------- 8-bit “E4M3” helpers ------------------------------------------------
struct IsInf_f8 {
    bool operator()(uint8_t bits) const      // exp == 0xF, frac == 0
    { return false; }
    uint8_t infBitPattern() const { return 0x0; }        // +inf
    uint8_t minNegInf()     const { return 0x0; }        // -inf
    uint8_t minPosInf()     const { return 0x0; }
};

struct IsNaN_f8 {
    bool operator()(uint8_t bits) const      // exp == 0xF, frac != 0
    { return false; }
    uint8_t qNanBitPattern() const { return 0x0; }
    uint8_t sNanBitPattern() const { return 0x0; }
};

// -------- fp8 parameter packs -------------------------------------------------
constexpr FloatingPointParams param_fp8_sr1(
     8, /*mant*/3, /*bias*/7,
     Rounding_Mode::StochasticRounding,
     Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
     Signedness::Signed,
     IsInf_f8(), IsNaN_f8(), 1    // 1 random bit
);
constexpr FloatingPointParams param_fp8_sr2(
    8, /*mant*/3, /*bias*/7,
    Rounding_Mode::StochasticRounding,
    Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f8(), IsNaN_f8(), 2    // 1 random bit
);
constexpr FloatingPointParams param_fp8_sr3(
    8, /*mant*/3, /*bias*/7,
    Rounding_Mode::StochasticRounding,
    Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f8(), IsNaN_f8(), 3   // 1 random bit
);

constexpr FloatingPointParams param_fp8_sr4(
    8, /*mant*/3, /*bias*/7,
    Rounding_Mode::StochasticRounding,
    Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f8(), IsNaN_f8(), 4    // 1 random bit
);

constexpr FloatingPointParams param_fp8_sr5(
    8, /*mant*/3, /*bias*/7,
    Rounding_Mode::StochasticRounding,
    Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f8(), IsNaN_f8(), 5    // 1 random bit
);






constexpr FloatingPointParams param_fp8_rd(   // deterministic “toward -∞”
     8, 3, 7, Rounding_Mode::RoundDown,
     Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
     Signedness::Signed, IsInf_f8(), IsNaN_f8()
);
constexpr FloatingPointParams param_fp8_ru(   // deterministic “toward +∞”
     8, 3, 7, Rounding_Mode::RoundUp,
     Inf_Behaviors::Extended, NaN_Behaviors::QuietNaN,
     Signedness::Signed, IsInf_f8(), IsNaN_f8()
);

// -------- type aliases --------------------------------------------------------
using float8_sr1 = Templated_Float<param_fp8_sr1>;
using float8_sr2 = Templated_Float<param_fp8_sr2>;
using float8_sr3 = Templated_Float<param_fp8_sr3>;
using float8_sr4 = Templated_Float<param_fp8_sr4>;
using float8_sr5 = Templated_Float<param_fp8_sr5>;
using float8_rd  = Templated_Float<param_fp8_rd>;
using float8_ru  = Templated_Float<param_fp8_ru>;

// -------- generic runner ------------------------------------------------------
template<class Fp8>
void run_one_type(const char* tag, double x, int trials)
{
    int hitLow = 0, hitUp = 0;

    for (int i = 0; i < trials; ++i) {
        double d = static_cast<double>( static_cast<Fp8>(x) );
        if (d == static_cast<double>( float8_rd(x) )) ++hitLow;
        else                                          ++hitUp;   // must be Up
    }

    std::cout << "  " << std::left << std::setw(5) << tag
              << ":  P(0) = "
              << std::setw(6) << std::fixed << std::setprecision(2)
              << 100.0*hitLow/trials << "%   "
              << "P(minSub) = "
              << std::setw(6)
              << 100.0*hitUp/trials  << "%\n";
}

// -------- main experiment -----------------------------------------------------
int main()
{
    constexpr int trials = 1'000'0;
    std::mt19937 rng{ std::random_device{}() };

    // --- 1. choose a random *strictly* sub-normal interval  ------------------
    std::uniform_real_distribution<float> pick_small(0.0, 1.0e-5);

    float x, f_low, f_up;
    x      = pick_small(rng);

        f_low  = static_cast<float>( float8_rd(x) );
        f_up   = static_cast<float>( float8_ru(x) );
    
        std::cout << "rd = " << f_low << "\n";
        std::cout << "ru = " << f_up << "\n";
    std::cout << "x = " << x << "\n";
    const double ulp   = f_up - f_low;
    const double p_up  = (x - f_low) / ulp;
    const double p_dn  = 1.0 - p_up;

    // // --- 2. echo the test point ---------------------------------------------
    // std::cout << std::fixed << std::setprecision(10);
    // std::cout << "\nRandom x              = " << x      << '\n';
    // std::cout << "RoundDown (0)          = " << f_low  << '\n';
    // std::cout << "RoundUp   (minSub)     = " << f_up   << '\n';
    // std::cout << "Ideal P(0)             = " << p_dn*100.0 << "%\n";
    // std::cout << "Ideal P(minSub)        = " << p_up*100.0 << "%\n\n";

    // // --- 3. stochastic Monte-Carlo ------------------------------------------
    // lo_float::set_seed( static_cast<std::uint64_t>(std::random_device{}()) );

    // std::cout << "Observed (" << trials << " trials)\n\n";
    // run_one_type<float8_sr1>("sr1", x, trials);
    // run_one_type<float8_sr2>("sr2", x, trials);
    // run_one_type<float8_sr3>("sr3", x, trials);
    // run_one_type<float8_sr4>("sr4", x, trials);
    // run_one_type<float8_sr5>("sr5", x, trials);

    return 0;
}
