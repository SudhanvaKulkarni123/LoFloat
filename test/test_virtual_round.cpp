// -------------------------------------------------------------
// test_virtual_round.cpp  --  tests for the two scalar
// virtual_round overloads in lo_float.h
//
//   1)  virtual_round(value, ToMantissaBits, Rounding_Mode)
//        - mantissa-only rounding, no exponent constraint
//   2)  virtual_round(value, FloatingPointParams, Rounding_Mode)
//        - full target-format rounding (range + precision)
//
// Modeled after test_rounding_modes.cpp.
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <vector>

#include "lo_float.h"
#include "lo_float_sci.hpp"
#include "fp_tools.hpp"

using namespace lo_float;

// -------------------------------------------------------------
// helpers
// -------------------------------------------------------------
static double get_denom(double d) {
    if (d == 0.0 || !std::isfinite(d)) return 1.0;
    int exp = 0;
    std::frexp(d, &exp);
    return std::ldexp(1.0, exp);
}

static float rnd32_signed() {
    // random in roughly [-1, 1]
    float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    return (u * 2.0f - 1.0f) * v;
}

static float rnd32_wide() {
    // random across many magnitudes:  sign * 2^e * uniform[0,1)
    float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    int e = (std::rand() % 41) - 20;        // exponent in [-20, 20]
    float sign = (std::rand() & 1) ? 1.0f : -1.0f;
    return sign * std::ldexp(u, e);
}

// "is_normal" for native float (used as the subnormal-vs-normal switch in
// the reference test). For float, min() == 2^-126.
static bool float_is_normal(float f) {
    return std::fabs(f) >= std::numeric_limits<float>::min();
}

// -------------------------------------------------------------
// 1)  virtual_round(value, ToMantissaBits, mode)
// -------------------------------------------------------------
// Only the mantissa is truncated/rounded; the exponent is unchanged, so
// the relative error must respect 2^-ToMantissaBits.
template <int ToMantissaBits>
int test_mantissa_round(int n_iters = 2000) {
    int num_errors = 0;

    const double mach_eps      = std::pow(2.0, -ToMantissaBits);
    const double mach_eps_half = std::pow(2.0, -ToMantissaBits - 1);

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (int i = 0; i < n_iters; ++i) {
        float d = rnd32_wide();
        if (!std::isfinite(d)) continue;

        float fd   = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundDown);
        float fu   = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundUp);
        float frne = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundToNearestEven);
        float frno = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundToNearestOdd);
        float frta = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundTiesToAway);
        float frtz = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundTowardsZero);
        float fraw = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundAwayFromZero);

        double denom = get_denom(d);

        double rel_down = std::fabs((double)fd   - d) / denom;
        double rel_up   = std::fabs((double)fu   - d) / denom;
        double rel_rne  = std::fabs((double)frne - d) / denom;
        double rel_rno  = std::fabs((double)frno - d) / denom;
        double rel_rta  = std::fabs((double)frta - d) / denom;
        double rel_rtz  = std::fabs((double)frtz - d) / denom;
        double rel_raw  = std::fabs((double)fraw - d) / denom;

        // direction
        if ((double)fd > d || rel_down > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundDown failed (x=" << d
                      << " fd=" << (double)fd << " rel=" << rel_down << ")\n";
            ++num_errors;
        }
        if ((double)fu < d || rel_up > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundUp failed (x=" << d
                      << " fu=" << (double)fu << " rel=" << rel_up << ")\n";
            ++num_errors;
        }
        if (std::fabs((double)frtz) > std::fabs(d) || rel_rtz > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundTowardsZero failed (x=" << d
                      << " frtz=" << (double)frtz << " rel=" << rel_rtz << ")\n";
            ++num_errors;
        }
        if (std::fabs((double)fraw) < std::fabs(d) || rel_raw > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundAwayFromZero failed (x=" << d
                      << " fraw=" << (double)fraw << " rel=" << rel_raw << ")\n";
            ++num_errors;
        }

        // nearest modes: tighter bound (eps/2)
        if (rel_rne > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundToNearestEven failed (x=" << d
                      << " frne=" << (double)frne << " rel=" << rel_rne << ")\n";
            ++num_errors;
        }
        if (rel_rno > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundToNearestOdd failed (x=" << d
                      << " frno=" << (double)frno << " rel=" << rel_rno << ")\n";
            ++num_errors;
        }
        if (rel_rta > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundTiesToAway failed (x=" << d
                      << " frta=" << (double)frta << " rel=" << rel_rta << ")\n";
            ++num_errors;
        }
    }

    // ------------------------------------------------------------------
    // tie-breaking: construct exact ties between two adjacent representable
    // values at the target precision and check RNE -> even, RNO -> odd,
    // RTA -> away.
    // ------------------------------------------------------------------
    if constexpr (ToMantissaBits >= 1 && ToMantissaBits < 23) {
        const int kFromMantissa = 23;                  // float
        const int shift         = kFromMantissa - ToMantissaBits;     // dropped bits
        const uint32_t step     = uint32_t{1} << shift;               // ulp at target
        const uint32_t half     = uint32_t{1} << (shift - 1);         // tie offset

        auto kept_lsb = [&](float v) {
            uint32_t bits = std::bit_cast<uint32_t>(v);
            return (bits >> shift) & 1u;
        };

        for (int rep = 0; rep < 200; ++rep) {
            // pick a random representable value at target precision
            float base = rnd32_wide();
            if (!std::isfinite(base) || base == 0.0f) continue;
            uint32_t b = std::bit_cast<uint32_t>(base);
            b &= ~(step - 1);                       // align to target ulp
            float a = std::bit_cast<float>(b);
            float a_next = std::bit_cast<float>(b + step);
            // exact midpoint:  a + ulp/2
            float tie = std::bit_cast<float>(b + half);
            if (!std::isfinite(tie) || !std::isfinite(a_next)) continue;

            float rne = lo_float::virtual_round(tie, ToMantissaBits, Rounding_Mode::RoundToNearestEven);
            float rno = lo_float::virtual_round(tie, ToMantissaBits, Rounding_Mode::RoundToNearestOdd);
            float rta = lo_float::virtual_round(tie, ToMantissaBits, Rounding_Mode::RoundTiesToAway);

            if (kept_lsb(rne) != 0u) {
                std::cout << "[mant=" << ToMantissaBits << "] RNE tie not even (tie=" << (double)tie
                          << " rne=" << (double)rne << ")\n";
                ++num_errors;
            }
            if (kept_lsb(rno) != 1u) {
                std::cout << "[mant=" << ToMantissaBits << "] RNO tie not odd (tie=" << (double)tie
                          << " rno=" << (double)rno << ")\n";
                ++num_errors;
            }
            // RTA: result magnitude should be the further of {a, a_next}
            double further = std::max(std::fabs((double)a), std::fabs((double)a_next));
            if (std::fabs((double)rta) + 1e-30 < further) {
                std::cout << "[mant=" << ToMantissaBits << "] RTA tie not away (tie=" << (double)tie
                          << " rta=" << (double)rta << ")\n";
                ++num_errors;
            }
        }
    }

    // ------------------------------------------------------------------
    // stochastic modes: result must be one of {round-down, round-up}.
    // ------------------------------------------------------------------
    {
        const Rounding_Mode stoch_modes[] = {
            Rounding_Mode::StochasticRoundingA,
            Rounding_Mode::StochasticRoundingB,
            Rounding_Mode::StochasticRoundingC,
            Rounding_Mode::True_StochasticRounding,
        };
        const char* stoch_names[] = {"StochA", "StochB", "StochC", "TrueStoch"};

        for (int m = 0; m < 4; ++m) {
            for (int i = 0; i < 500; ++i) {
                float d = rnd32_wide();
                if (!std::isfinite(d)) continue;
                float fd = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundDown);
                float fu = lo_float::virtual_round(d, ToMantissaBits, Rounding_Mode::RoundUp);
                float rs = lo_float::virtual_round(d, ToMantissaBits, ProjSpec{stoch_modes[m], Saturation_Mode::OvfInf, 8});
                if (rs != fd && rs != fu) {
                    std::cout << "[mant=" << ToMantissaBits << "] " << stoch_names[m]
                              << " produced non-adjacent result (x=" << d
                              << " rs=" << (double)rs
                              << " fd=" << (double)fd
                              << " fu=" << (double)fu << ")\n";
                    ++num_errors;
                }
            }
        }
    }

    std::cout << "virtual_round<float, mant=" << ToMantissaBits << "> : "
              << (num_errors == 0 ? "pass" : "FAIL") << "\n";
    return num_errors;
}

// -------------------------------------------------------------
// 2)  virtual_round(value, FloatingPointParams, mode)
// -------------------------------------------------------------
// Tests the full overload that respects target bias/range/inf behavior.
template <typename ToInf, typename ToNaN>
int test_fp_params_round(const char* name,
                         FloatingPointParams<ToInf, ToNaN> ToFp,
                         int n_iters = 5000) {
    int num_errors = 0;

    const double mach_eps      = std::pow(2.0, -ToFp.mantissa_bits);
    const double mach_eps_half = std::pow(2.0, -ToFp.mantissa_bits - 1);

    // smallest representable positive (subnormal) magnitude in target
    const double UNT = std::pow(2.0, 1 - ToFp.bias) *
                       std::pow(2.0, -ToFp.mantissa_bits);

    // Largest finite magnitude in target's exponent range. We allow
    // results to grow up to the next power of two beyond the value
    // (round-up of a near-max number). For normal values strictly inside
    // the target range, the bounds below are correct.
    const int ToMax_exp = (ToFp.is_signed == Signedness::Signed
                              ? (1 << (ToFp.bitwidth - ToFp.mantissa_bits - 1)) - 1
                              : (1 << (ToFp.bitwidth - ToFp.mantissa_bits))) - 1
                         - ToFp.bias;

    const float max_abs = std::ldexp(1.0f, ToMax_exp);  // upper bound for in-range tests

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    auto check_modes = [&](float d) {
        if (!std::isfinite(d)) return;
        if (std::fabs(d) >= (float)max_abs) return;     // skip overflow region
        float local = d;

        float fd   = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundDown);
        float fu   = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundUp);
        float frne = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundToNearestEven);
        float frno = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundToNearestOdd);
        float frta = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundTiesToAway);
        float frtz = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundTowardsZero);
        float fraw = lo_float::virtual_round(local, ToFp, Rounding_Mode::RoundAwayFromZero);

        double denom    = get_denom(d);
        double abs_down = std::fabs((double)fd   - d);
        double abs_up   = std::fabs((double)fu   - d);
        double abs_rne  = std::fabs((double)frne - d);
        double abs_rno  = std::fabs((double)frno - d);
        double abs_rta  = std::fabs((double)frta - d);
        double abs_rtz  = std::fabs((double)frtz - d);
        double abs_raw  = std::fabs((double)fraw - d);

        double rel_down = abs_down / denom;
        double rel_up   = abs_up   / denom;
        double rel_rne  = abs_rne  / denom;
        double rel_rno  = abs_rno  / denom;
        double rel_rta  = abs_rta  / denom;
        double rel_rtz  = abs_rtz  / denom;
        double rel_raw  = abs_raw  / denom;

        // Choose the relative-vs-absolute branch the same way the reference
        // test does: based on whether the result is in the *target* normal
        // range. For target subnormals we fall back to UNT.
        const double abs_d = std::fabs((double)d);
        const bool in_target_normal = abs_d >= std::ldexp(1.0, 1 - ToFp.bias);

        if (in_target_normal) {
            if ((double)fd > d || rel_down > mach_eps) {
                std::cout << name << " RoundDown failed (x=" << d << " fd=" << (double)fd
                          << " rel=" << rel_down << ")\n"; ++num_errors;
            }
            if ((double)fu < d || rel_up > mach_eps) {
                std::cout << name << " RoundUp failed (x=" << d << " fu=" << (double)fu
                          << " rel=" << rel_up << ")\n"; ++num_errors;
            }
            if (std::fabs((double)frtz) > std::fabs(d) || rel_rtz > mach_eps) {
                std::cout << name << " RoundTowardsZero failed (x=" << d << " frtz=" << (double)frtz
                          << " rel=" << rel_rtz << ")\n"; ++num_errors;
            }
            if (std::fabs((double)fraw) < std::fabs(d) || rel_raw > mach_eps) {
                std::cout << name << " RoundAwayFromZero failed (x=" << d << " fraw=" << (double)fraw
                          << " rel=" << rel_raw << ")\n"; ++num_errors;
            }
            if (rel_rne > mach_eps_half) {
                std::cout << name << " RoundToNearestEven failed (x=" << d << " frne=" << (double)frne
                          << " rel=" << rel_rne << ")\n"; ++num_errors;
            }
            if (rel_rno > mach_eps_half) {
                std::cout << name << " RoundToNearestOdd failed (x=" << d << " frno=" << (double)frno
                          << " rel=" << rel_rno << ")\n"; ++num_errors;
            }
            if (rel_rta > mach_eps_half) {
                std::cout << name << " RoundTiesToAway failed (x=" << d << " frta=" << (double)frta
                          << " rel=" << rel_rta << ")\n"; ++num_errors;
            }
        } else {
            // target-subnormal regime: error bounded by unit-in-the-last-place
            if ((double)fd > d || abs_down > UNT) {
                std::cout << name << " (sub) RoundDown failed (x=" << d << " fd=" << (double)fd
                          << " abs=" << abs_down << ")\n"; ++num_errors;
            }
            if ((double)fu < d || abs_up > UNT) {
                std::cout << name << " (sub) RoundUp failed (x=" << d << " fu=" << (double)fu
                          << " abs=" << abs_up << ")\n"; ++num_errors;
            }
            if (std::fabs((double)frtz) > std::fabs(d) || abs_rtz > UNT) {
                std::cout << name << " (sub) RoundTowardsZero failed (x=" << d << " frtz=" << (double)frtz
                          << " abs=" << abs_rtz << ")\n"; ++num_errors;
            }
            if (std::fabs((double)fraw) < std::fabs(d) || abs_raw > UNT) {
                std::cout << name << " (sub) RoundAwayFromZero failed (x=" << d << " fraw=" << (double)fraw
                          << " abs=" << abs_raw << ")\n"; ++num_errors;
            }
            if (abs_rne > UNT) {
                std::cout << name << " (sub) RoundToNearestEven failed (x=" << d << " frne=" << (double)frne
                          << " abs=" << abs_rne << ")\n"; ++num_errors;
            }
            if (abs_rno > UNT) {
                std::cout << name << " (sub) RoundToNearestOdd failed (x=" << d << " frno=" << (double)frno
                          << " abs=" << abs_rno << ")\n"; ++num_errors;
            }
            if (abs_rta > UNT) {
                std::cout << name << " (sub) RoundTiesToAway failed (x=" << d << " frta=" << (double)frta
                          << " abs=" << abs_rta << ")\n"; ++num_errors;
            }
        }
    };

    // sweep across magnitudes
    for (int i = 0; i < n_iters; ++i) {
        check_modes(rnd32_signed());
        check_modes(rnd32_wide());
    }

    // explicit zero / NaN
    {
        float zero = 0.0f;
        float r = lo_float::virtual_round(zero, ToFp, Rounding_Mode::RoundToNearestEven);
        if (r != 0.0f) {
            std::cout << name << " zero not preserved (got " << (double)r << ")\n";
            ++num_errors;
        }
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        float rn = lo_float::virtual_round(nan_val, ToFp, Rounding_Mode::RoundToNearestEven);
        if (!std::isnan(rn)) {
            std::cout << name << " NaN not preserved (got " << (double)rn << ")\n";
            ++num_errors;
        }
    }

    // tie-breaking: construct ties at the target precision (in the normal
    // exponent range) and check RNE/RNO/RTA semantics.
    {
        const int target_mant = ToFp.mantissa_bits;
        const int shift       = 23 - target_mant;
        if (shift > 0) {
            const uint32_t step = uint32_t{1} << shift;
            const uint32_t half = uint32_t{1} << (shift - 1);

            auto kept_lsb = [&](float v) {
                uint32_t bits = std::bit_cast<uint32_t>(v);
                return (bits >> shift) & 1u;
            };

            for (int i = 0; i < 200; ++i) {
                // pick a random in-range positive value at target ulp
                int e = (std::rand() % 6) - 2;          // small exponent
                float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                float base = std::ldexp(u + 1.0f, e);   // in [2^e, 2^(e+1))
                if (!std::isfinite(base)) continue;
                uint32_t b = std::bit_cast<uint32_t>(base);
                b &= ~(step - 1);
                float a      = std::bit_cast<float>(b);
                float a_next = std::bit_cast<float>(b + step);
                float tie    = std::bit_cast<float>(b + half);
                if (!std::isfinite(a_next) || !std::isfinite(tie)) continue;

                float rne = lo_float::virtual_round(tie, ToFp, Rounding_Mode::RoundToNearestEven);
                float rno = lo_float::virtual_round(tie, ToFp, Rounding_Mode::RoundToNearestOdd);
                float rta = lo_float::virtual_round(tie, ToFp, Rounding_Mode::RoundTiesToAway);

                if (kept_lsb(rne) != 0u) {
                    std::cout << name << " RNE tie not even (tie=" << (double)tie
                              << " rne=" << (double)rne << ")\n"; ++num_errors;
                }
                if (kept_lsb(rno) != 1u) {
                    std::cout << name << " RNO tie not odd (tie=" << (double)tie
                              << " rno=" << (double)rno << ")\n"; ++num_errors;
                }
                double further = std::max(std::fabs((double)a), std::fabs((double)a_next));
                if (std::fabs((double)rta) + 1e-30 < further) {
                    std::cout << name << " RTA tie not away (tie=" << (double)tie
                              << " rta=" << (double)rta << ")\n"; ++num_errors;
                }
            }
        }
    }

    // stochastic sanity for the FP-params overload
    {
        const Rounding_Mode stoch_modes[] = {
            Rounding_Mode::StochasticRoundingA,
            Rounding_Mode::StochasticRoundingB,
            Rounding_Mode::StochasticRoundingC,
            Rounding_Mode::True_StochasticRounding,
        };
        const char* stoch_names[] = {"StochA", "StochB", "StochC", "TrueStoch"};

        for (int m = 0; m < 4; ++m) {
            for (int i = 0; i < 500; ++i) {
                float d = rnd32_signed();
                if (!std::isfinite(d)) continue;
                if (std::fabs(d) >= (float)max_abs) continue;
                float local_d = d, local_u = d, local_s = d;
                float fd = lo_float::virtual_round(local_d, ToFp, Rounding_Mode::RoundDown);
                float fu = lo_float::virtual_round(local_u, ToFp, Rounding_Mode::RoundUp);
                float rs = lo_float::virtual_round(local_s, ToFp, ProjSpec{stoch_modes[m], Saturation_Mode::OvfInf, 8});
                if (rs != fd && rs != fu) {
                    std::cout << name << " " << stoch_names[m]
                              << " produced non-adjacent result (x=" << d
                              << " rs=" << (double)rs
                              << " fd=" << (double)fd
                              << " fu=" << (double)fu << ")\n";
                    ++num_errors;
                }
            }
        }
    }

    std::cout << "virtual_round<" << name << "> : "
              << (num_errors == 0 ? "pass" : "FAIL") << "\n";
    return num_errors;
}

// -------------------------------------------------------------
int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int total = 0;

    // 1) mantissa-only overload, several target precisions
    total += test_mantissa_round<7>();    // bf16-like
    total += test_mantissa_round<10>();   // half/tf32-like
    total += test_mantissa_round<15>();
  //  total += test_mantissa_round<23>();   // no-op; should be exact

    // 2) full FloatingPointParams overload
    //total += test_fp_params_round("halfPrecision",  halfPrecisionParams);
    // total += test_fp_params_round("bfloatPrecision", bfloatPrecisionParams);
    // total += test_fp_params_round("tf32Precision",  tf32PrecisionParams);

    std::cout << "\n=== TOTAL ERRORS: " << total
              << (total == 0 ? "  (ALL PASS)" : "  (FAIL)") << " ===\n";
    return total == 0 ? 0 : 1;
}
