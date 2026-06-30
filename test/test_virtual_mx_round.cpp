// -------------------------------------------------------------
// test_virtual_mx_round.cpp  --  tests for lo_float::virtual_mx_round
// (CPU / xsimd microscaling block quantization).
//
// Restructured to follow test_rounding_modes.cpp: the same per-mode rounding
// bounds (get_denom / mach_eps / UNT / directional checks / FromRep tie cases)
// and the same P_3109<l,p> template sweep (l=2..8, p=1..l-1). No hardcoded magic
// constants -- every value comes from the format under test (FromRep, its
// max-normal / denorm_min) or from a random draw inside that format's range.
//
// MX quantization rounds TWO numbers, so this file runs the test_rounding_modes
// test on each of them:
//
//   (1) the SCALE element.  per block:  amax = max|x_i| (in fp32, as the kernel
//       computes it), scale = RNE_round(amax/priv_max_normal) into the public
//       E8M0 (power-of-two) format. We assert `scale` is the RoundToNearestEven
//       projection of amax/priv_max_normal: one of the two E8M0 neighbours, and
//       the nearer one.
//
//   (2) the PRIVATE elements.  the DUT stores q_i = round(x_i/scale) into the
//       private P_3109<l,p> format and writes back out_i = q_i*scale. With S =
//       scale a power of two, x_i/scale and q_i*scale are EXACT in fp32, so
//       z_i = out_i/S = round(x_i/S) is the true private rounding of x_i/S, with
//       no rescale slop. We feed the DUT each deterministic mode and apply the
//       verbatim test_rounding_modes per-mode bound to (x_i/S, z_i). Inputs span
//       the private range down into the subnormals and below, so e.g.
//       RoundAwayFromZero of a tiny value must reach denorm_min (never 0) -- a
//       result of 0 there is a real bug, not tolerated by these bounds.
//
//   The oracle is double / the tight test_rounding_modes bounds (rule 1).
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <vector>
#include <cstring>

#include "lo_float.h"
#include "lo_float_sci.hpp"
#include "fp_tools.hpp"

using namespace lo_float;

// ---- shared helpers (mirror test_rounding_modes.cpp) ------------------------
static double get_denom(double d) {
    if (d == 0.0 || !std::isfinite(d)) return 1.0;
    int exp = 0;
    std::frexp(d, &exp);          // d == mantissa * 2^exp,  mantissa in [0.5,1)
    return std::ldexp(1.0, exp);  // 2^exp
}

// Public (scale) format: E8M0 -- power-of-two only. MUST be Unsigned so the
// scale gets all 8 exponent bits (see loop/NOTES.md MX-scale caveat).
static constexpr FloatingPointParams params_e8m0(
    8, 0, 127, Inf_Behaviors::Extended, NaN_Behaviors::_754, Signedness::Unsigned,
    SingleInfChecker(), SingleNaNChecker());                 // 2^-126 .. 2^128

template <typename P>
static double priv_max_normal(const P& p) {
    const int me = p.is_signed == Signedness::Signed
        ? (1 << (p.bitwidth - p.mantissa_bits - 1)) - 1 - p.bias
        : (1 << (p.bitwidth - p.mantissa_bits))     - 1 - p.bias;
    return std::ldexp(1.0, me) * (2.0 - std::ldexp(1.0, -p.mantissa_bits));
}

static bool bit_eq(float a, float b) {
    uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4);
    return x == y;
}

static const char* mode_name(Rounding_Mode rm) {
    switch (rm) {
        case Rounding_Mode::RoundDown:           return "RoundDown";
        case Rounding_Mode::RoundUp:             return "RoundUp";
        case Rounding_Mode::RoundTowardsZero:    return "RoundTowardsZero";
        case Rounding_Mode::RoundAwayFromZero:   return "RoundAwayFromZero";
        case Rounding_Mode::RoundToNearestEven:  return "RoundToNearestEven";
        case Rounding_Mode::RoundToNearestOdd:   return "RoundToNearestOdd";
        case Rounding_Mode::RoundTiesToAway:     return "RoundTiesToAway";
        default:                                 return "Round?";
    }
}

// The exact test_rounding_modes per-mode bound, factored so it can run on either
// rounded number (scale or private element). `r` is the rounded result, `x` the
// exact value; `norm` says whether r landed in the format's NORMAL range (uses
// the fixed subnormal tolerance UNT below it). Returns 0/1 errors; caps prints.
//
// NOTE vs test_rounding_modes::is_normal(): that helper is `min() < f`, false for
// every NEGATIVE value -- harmless there (its inputs are positive) but wrong here
// (MX inputs are signed), so we key off |r| >= min_normal.
static int check_round_bound(const char* tag, Rounding_Mode rm,
                             double x, double r, bool norm,
                             double mach_eps, double UNT, int& printed) {
    const double denom  = get_denom(x);
    const double rel    = std::fabs(r - x) / denom;
    const double abserr = std::fabs(r - x);
    const double mag_ok = norm ? (rel <= mach_eps) : (abserr <= UNT);
    bool fail = false;
    switch (rm) {
        case Rounding_Mode::RoundDown:         fail = (r > x) || !mag_ok; break;
        case Rounding_Mode::RoundUp:           fail = (r < x) || !mag_ok; break;
        case Rounding_Mode::RoundTowardsZero:  fail = (std::fabs(r) > std::fabs(x)) || !mag_ok; break;
        case Rounding_Mode::RoundAwayFromZero: fail = (std::fabs(r) < std::fabs(x)) || !mag_ok; break;
        case Rounding_Mode::RoundToNearestEven:
        case Rounding_Mode::RoundToNearestOdd:
        case Rounding_Mode::RoundTiesToAway:   fail = !mag_ok; break;
        default: break;
    }
    if (fail && ++printed <= 8)
        std::cout << tag << " " << mode_name(rm) << " bound fail (x=" << x
                  << " r=" << r << " rel=" << rel << " abs=" << abserr << ")\n";
    return fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
template<int l, int p>
int test_mx_3109() {
    using P_3109_type = P_3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    constexpr auto priv_params =
        lo_float_internal::param_float_p_3109<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;

    const double mach_eps   = std::pow(2.0, -p + 1);                 // 2^{-(p-1)}
    const double UNT        = (double)std::numeric_limits<P_3109_type>::denorm_min();
    const double min_normal = std::ldexp(1.0, 1 - priv_params.bias); // 2^{1-bias}
    const double priv_maxn  = priv_max_normal(priv_params);

    const Rounding_Mode det_modes[] = {
        Rounding_Mode::RoundDown,          Rounding_Mode::RoundUp,
        Rounding_Mode::RoundTowardsZero,   Rounding_Mode::RoundAwayFromZero,
        Rounding_Mode::RoundToNearestEven, Rounding_Mode::RoundToNearestOdd,
        Rounding_Mode::RoundTiesToAway,
    };
    int num_errors = 0, printed = 0;
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    // log-uniform magnitude in [denorm_min/4, priv_max_normal], random sign.
    // Covers normals, subnormals, and sub-denorm values (which directed modes
    // must still round to denorm_min, not 0). All representable in fp32.
    auto rnd_in_range = [&]() -> double {
        const double u  = (double)std::rand() / (double)RAND_MAX;        // [0,1]
        const double lo = std::log2(UNT / 4.0), hi = std::log2(priv_maxn);
        const double mag = std::exp2(lo + u * (hi - lo));
        return ((std::rand() & 1) ? 1.0 : -1.0) * mag;
    };

    // ---- PART 1 : SCALE element ---------------------------------------------
    // wide-dynamic-range blocks exercise the E8M0 scale rounding; assert the
    // scale is the RNE projection of amax/priv_max_normal (nearest neighbour).
    {
        const int block_sizes[] = {1, 7, 16, 32, 33, 64};
        for (int bs : block_sizes) {
            const int n = 257;
            std::vector<float> base(n);
            for (auto& x : base) {
                const int   e = (std::rand() % 41) - 20;            // 2^[-20,20]
                const float u = ((float)std::rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
                base[(&x - base.data())] = ((std::rand() & 1) ? 1.f : -1.f) * std::ldexp(u, e);
            }
            for (int b0 = 0; b0 < n; b0 += bs) {
                const int len = std::min(bs, n - b0);
                float amax = 0.0f;
                for (int i = 0; i < len; ++i) amax = std::max(amax, std::fabs(base[b0 + i]));
                const float t_scale = (float)amax / (float)priv_maxn;
                if (!(t_scale > 0.0f)) continue;
                const float scale = lo_float::virtual_round(t_scale, params_e8m0,
                                       ProjSpec{Rounding_Mode::RoundToNearestEven});
                const float sdn = lo_float::virtual_round(t_scale, params_e8m0,
                                       ProjSpec{Rounding_Mode::RoundDown});
                const float sup = lo_float::virtual_round(t_scale, params_e8m0,
                                       ProjSpec{Rounding_Mode::RoundUp});
                if (!(scale == sdn || scale == sup)) {
                    if (++printed <= 8)
                        std::cout << "scale not a neighbour (t=" << (double)t_scale
                                  << " scale=" << scale << " dn=" << sdn << " up=" << sup << ")\n";
                    ++num_errors;
                }
                const double dsc = std::fabs((double)scale - (double)t_scale);
                const double dlo = std::fabs((double)sdn  - (double)t_scale);
                const double dhi = std::fabs((double)sup  - (double)t_scale);
                if (dsc > std::min(dlo, dhi) + 1e-9 * (double)t_scale) {
                    if (++printed <= 8)
                        std::cout << "scale not nearest (t=" << (double)t_scale
                                  << " scale=" << scale << ")\n";
                    ++num_errors;
                }
            }
        }
    }

    // ---- PART 2 : PRIVATE elements ------------------------------------------
    // For a few power-of-two block scales S = 2^E: build a block of values in
    // [-priv_maxn, priv_maxn]*S with one element pinned to priv_maxn*S so the MX
    // scale is exactly S. Then out_i/S = round(x_i/S) and we apply the verbatim
    // test_rounding_modes bound to (x_i/S, out_i/S).
    {
        const int scale_exps[] = {0, 3, -5, 9, -12};
        const int n = 256;
        for (int E : scale_exps) {
            const double S = std::ldexp(1.0, E);
            std::vector<float> base(n);
            base[0] = (float)(priv_maxn * S);                       // pin amax -> scale = S
            for (int i = 1; i < n; ++i) base[i] = (float)(rnd_in_range() * S);

            for (Rounding_Mode rm : det_modes) {
                std::vector<float> dut = base;
                lo_float::virtual_mx_round(dut.data(), n, n, params_e8m0, priv_params,
                                           ProjSpec{Rounding_Mode::RoundToNearestEven},
                                           ProjSpec{rm});
                const float scale = lo_float::virtual_round((float)base[0] / (float)priv_maxn,
                                       params_e8m0, ProjSpec{Rounding_Mode::RoundToNearestEven});
                if (scale != (float)S) {                            // scale must be exactly S
                    if (++printed <= 8)
                        std::cout << "PART2 scale!=S (E=" << E << " scale=" << scale << ")\n";
                    ++num_errors; continue;
                }
                for (int i = 0; i < n; ++i) {
                    const double z = (double)base[i] / (double)scale; // value rounded (exact)
                    const double q = (double)dut[i]  / (double)scale; // DUT result   (exact)
                    const bool   norm = std::fabs(q) > min_normal;   // strict, like test_rounding_modes is_normal
                    num_errors += check_round_bound("priv", rm, z, q, norm, mach_eps, UNT, printed);
                }
            }
        }
    }

    // ---- PART 3a : exact case -- every representable value round-trips -------
    // FromRep enumerates exactly-representable private values; at scale==1 they
    // must pass through the DUT bit-exact under EVERY mode (rule 2). No hardcode.
    {
        std::vector<float> reps;
        for (uint32_t rep = 0; rep < (1u << l); ++rep) {
            const float v = (float)(double)P_3109_type::FromRep(rep);
            if (std::isfinite(v) && std::fabs(v) <= priv_maxn) reps.push_back(v);
        }
        reps.push_back((float)priv_maxn);                           // pin amax -> scale = 1
        for (Rounding_Mode rm : det_modes) {
            std::vector<float> dut = reps;
            lo_float::virtual_mx_round(dut.data(), (int)dut.size(), (int)dut.size(),
                                       params_e8m0, priv_params,
                                       ProjSpec{Rounding_Mode::RoundToNearestEven}, ProjSpec{rm});
            for (size_t i = 0; i < reps.size(); ++i)
                if (!bit_eq(dut[i], reps[i])) {
                    if (++printed <= 8)
                        std::cout << "exact-case changed @" << i << " in=" << reps[i]
                                  << " out=" << dut[i] << " (" << mode_name(rm) << ")\n";
                    ++num_errors;
                }
        }
    }

    // ---- PART 3b : tie cases routed THROUGH the DUT (scale forced to 1) ------
    // The test_rounding_modes tie loop, but the midpoint is fed to the MX kernel.
    // Prepending priv_max_normal pins amax => scale == 1, so out == round(tie);
    // check the parity / away rule on the DUT output's rep.
    if constexpr (p > 1) {
        auto is_even = [](uint32_t b){ return (b & 1u) == 0; };
        auto is_odd  = [](uint32_t b){ return (b & 1u) == 1; };
        for (uint32_t rep = 1; rep < (1u << l) - 2; ++rep) {
            const double a_d = (double)P_3109_type::FromRep(rep);
            const double b_d = (double)P_3109_type::FromRep(rep + 1);
            const double tie = (a_d + b_d) / 2.0;
            if (std::isnan(tie) || std::fabs(tie) >= priv_maxn) continue;
            for (Rounding_Mode rm : { Rounding_Mode::RoundToNearestEven,
                                      Rounding_Mode::RoundToNearestOdd,
                                      Rounding_Mode::RoundTiesToAway }) {
                float blk[2] = { (float)priv_maxn, (float)tie };
                lo_float::virtual_mx_round(blk, 2, 2, params_e8m0, priv_params,
                                           ProjSpec{Rounding_Mode::RoundToNearestEven}, ProjSpec{rm});
                const P_3109_type out = P_3109_type((double)blk[1]);   // scale==1
                if (rm == Rounding_Mode::RoundToNearestEven && !is_even(out.rep())) {
                    if (++printed <= 8) std::cout << "RNE tie not even (tie=" << tie << ")\n"; ++num_errors;
                }
                if (rm == Rounding_Mode::RoundToNearestOdd && !is_odd(out.rep())) {
                    if (++printed <= 8) std::cout << "RNO tie not odd (tie=" << tie << ")\n"; ++num_errors;
                }
                if (rm == Rounding_Mode::RoundTiesToAway) {
                    const double ref_next = std::nextafter(tie, tie > 0 ? 1e300 : -1e300);
                    if (std::fabs((double)out) < std::fabs(ref_next) - 1e-20) {
                        if (++printed <= 8) std::cout << "RTA tie not away (tie=" << tie << ")\n"; ++num_errors;
                    }
                }
            }
        }
    }

    // ---- PART 4 : stochastic two-neighbour support (rule 5) -----------------
    {
        std::srand(99 + l * 16 + p);
        const int n = 128;
        std::vector<float> base(n);
        base[0] = (float)priv_maxn;                                 // scale = 1
        for (int i = 1; i < n; ++i) base[i] = (float)rnd_in_range();
        std::vector<float> dut = base;
        lo_float::virtual_mx_round(dut.data(), n, n, params_e8m0, priv_params,
                                   ProjSpec{Rounding_Mode::RoundToNearestEven},
                                   ProjSpec{Rounding_Mode::StochasticRoundingA, Saturation_Mode::OvfInf, 8});
        for (int i = 0; i < n; ++i) {
            const float dn = lo_float::virtual_round(base[i], priv_params,
                                                     ProjSpec{Rounding_Mode::RoundDown});
            const float up = lo_float::virtual_round(base[i], priv_params,
                                                     ProjSpec{Rounding_Mode::RoundUp});
            if (!bit_eq(dut[i], dn) && !bit_eq(dut[i], up)) {
                if (++printed <= 8)
                    std::cout << "stoch non-neighbour @" << i << " in=" << base[i]
                              << " out=" << dut[i] << " dn=" << dn << " up=" << up << "\n";
                ++num_errors;
            }
        }
    }

    std::cout << "virtual_mx_round P_3109<" << l << "," << p << "> : "
              << (num_errors == 0 ? "pass" : "FAIL") << "\n";
    return num_errors;
}

// ---- template sweep machinery (mirrors test_rounding_modes.cpp) ------------
template<int l, int... Ps>
int test_mx_for_l(std::integer_sequence<int, Ps...>) {
    int e = 0; ((e += test_mx_3109<l, Ps + 1>()), ...); return e;
}
template<int... Ls>
int test_mx_all_l(std::integer_sequence<int, Ls...>) {
    int e = 0; ((e += test_mx_for_l<Ls>(std::make_integer_sequence<int, Ls - 1>{})), ...); return e;
}
template<int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

int main() {
    std::srand(7);
    // l from 2 to 8, p from 1 to l-1 -- identical sweep to test_rounding_modes.
    int total = test_mx_all_l(offset_sequence<2>(std::make_integer_sequence<int, 7>{}));
    std::cout << "\n=== TOTAL ERRORS: " << total
              << (total == 0 ? "  (ALL PASS)" : "  (FAIL)") << " ===\n";
    return total == 0 ? 0 : 1;
}
