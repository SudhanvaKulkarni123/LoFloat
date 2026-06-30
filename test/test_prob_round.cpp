// -------------------------------------------------------------
// test_prob_round.cpp
// Empirical round-up-probability checks (paper Table II) for the
// non-deterministic rounding modes that are NOT swept elsewhere:
//   - ProbabilisticRounding  : P(up) = 0.5 for any nonzero tail
//   - StochasticRoundingD     : P(up) = 0.5*( floor(q*xf)/q + ceil(q*xf)/q )
// Model: pick a double strictly between two fp32 neighbours, round it many
// times, and compare the observed up-fraction against the ideal probability
// within a Bernoulli confidence band.
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <limits>
#include "lo_float.h"

using namespace lo_float;

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
    32, 23, 127,
    Inf_Behaviors::Extended, NaN_Behaviors::_3109, Signedness::Signed,
    IsInf_f32(), IsNaN_f32());
using fp32 = Templated_Float<param_fp32>;

// Round x `trials` times with `ps`; return observed fraction that landed on
// the upper neighbour f_up (vs the lower neighbour f_low).
//
// The library's stochastic modes re-seed a local std::mt19937 from the global
// thread_local `lof_seed` on every call, so a fixed seed makes the result
// deterministic. To Monte-Carlo the round-up probability we therefore reseed
// from a master generator before each trial (well-spread seeds avoid the
// correlated-initial-output issue of sequentially-seeded mt19937).
static double observed_p_up(double x, fp32 f_low, fp32 f_up,
                            ProjSpec ps, int trials, std::mt19937& master)
{
    int hitUpper = 0, hitLower = 0;
    for (int i = 0; i < trials; ++i) {
        lo_float::set_seed(static_cast<unsigned int>(master()));
        fp32 r = Round<fp32, double>(x, ps);
        double d = static_cast<double>(r);
        if      (d == (double)f_up)  ++hitUpper;
        else if (d == (double)f_low) ++hitLower;
    }
    return (double)hitUpper / trials;
}

// Ideal P(up) for StochasticRoundingD (Table II): 0.5*(floor(q*xf)/q + ceil(q*xf)/q).
static double ideal_p_up_D(double xf, int n) {
    const double q = std::ldexp(1.0, n);   // 2^n
    return 0.5 * (std::floor(q * xf) / q + std::ceil(q * xf) / q);
}

int main()
{
    constexpr int trials = 200000;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<float> pick_float(-100.0f, 100.0f);
    std::uniform_real_distribution<double> pick_frac(0.0, 1.0);

    // Confidence band: for N Bernoulli trials at p, sigma = sqrt(p(1-p)/N).
    // 5*sigma at p=0.5, N=200k is ~0.0056; use a comfortable 0.02 tolerance.
    const double tol = 0.02;

    lo_float::set_seed(0xC0FFEEu);
    int num_errors = 0;

    // --- pick a random double strictly between two neighbouring fp32 values ---
    auto pick_between = [&](double& x, float& f_low, float& f_up, double& xf) {
        while (true) {
            f_low = pick_float(rng);
            f_up  = std::nextafterf(f_low, std::numeric_limits<float>::infinity());
            if (std::isfinite(f_up) && f_up != f_low) break;
        }
        double t; do { t = pick_frac(rng); } while (t == 0.0 || t == 1.0);
        x  = (double)f_low + t * ((double)f_up - (double)f_low);
        xf = (x - (double)f_low) / ((double)f_up - (double)f_low);
    };

    std::cout << std::fixed << std::setprecision(4);

    // ===== ProbabilisticRounding: P(up) = 0.5 regardless of xf =====
    {
        double x, xf; float f_low, f_up; pick_between(x, f_low, f_up, xf);
        double p = observed_p_up(x, fp32(f_low), fp32(f_up),
                                 ProjSpec{Rounding_Mode::ProbabilisticRounding}, trials, rng);
        std::cout << "ProbabilisticRounding: xf=" << xf
                  << "  observed P(up)=" << p << "  ideal=0.5000\n";
        if (std::fabs(p - 0.5) > tol) {
            std::cout << "  FAIL: |P(up)-0.5| = " << std::fabs(p - 0.5) << " > " << tol << "\n";
            num_errors++;
        }
    }

    // ===== StochasticRoundingD: coin-flip variant, ideal per Table II =====
    for (int n = 1; n <= 4; ++n) {
        double x, xf; float f_low, f_up; pick_between(x, f_low, f_up, xf);
        double ideal = ideal_p_up_D(xf, n);
        double p = observed_p_up(x, fp32(f_low), fp32(f_up),
                                 ProjSpec{Rounding_Mode::StochasticRoundingD,
                                          Saturation_Mode::OvfInf, n}, trials, rng);
        std::cout << "StochasticRoundingD n=" << n << ": xf=" << xf
                  << "  observed P(up)=" << p << "  ideal=" << ideal << "\n";
        if (std::fabs(p - ideal) > tol) {
            std::cout << "  FAIL: |P(up)-ideal| = " << std::fabs(p - ideal) << " > " << tol << "\n";
            num_errors++;
        }
    }

    std::cout << (num_errors == 0 ? "\nprob/stochastic-D probability test: pass\n"
                                  : "\nprob/stochastic-D probability test: FAIL\n");
    return num_errors == 0 ? 0 : 1;
}
