// -------------------------------------------------------------
// test_sat_mode.cpp  —  Saturation_Mode (SatMode) behaviour test.
//
// Mirrors test_rounding_modes.cpp in design: it sweeps the same P_3109
// formats (l = 2..8, p = 1..l-1) and rounds values through the call-site
// Round<T>() entry point.  Where test_rounding_modes varies the rounding
// mode, this file fixes the rounding mode (RNE) and varies the
// Saturation_Mode carried by the ProjSpec, then checks the over-range /
// infinite-input results against the spec:
//
//   OvfInf       (default): out-of-range -> +/-inf when the target format has
//                           infinity, else the extremal finite value.
//   SatFinite             : every returned value is clamped to the finite
//                           range (a finite or infinite input that overflows
//                           -> +/-max-finite; an infinite input -> +/-max).
//   SatPropagate          : finite values clamp to +/-max-finite; a true
//                           infinite input is preserved as +/-inf (when the
//                           format has infinity).
//
// To actually distinguish the three modes we test BOTH the Saturating
// (no-infinity) and Extended (has-infinity) variants of each P_3109 format:
// on a no-inf format all three modes clamp to max; only on a has-inf format
// do OvfInf/SatFinite/SatPropagate diverge.  The reference oracle is double
// (the format's max-finite and infinity), per loop/NUMERICAL_TESTING.md.
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "lo_float.h"
#include "lo_float_sci.hpp"

using namespace lo_float;

// One format under test.  Returns the number of failures.
template <typename T>
int test_sat_one(const char* label) {
    int num_errors = 0;

    const double maxf     = static_cast<double>(std::numeric_limits<T>::max());
    // Degenerate format with an empty finite-normal range (e.g. P_3109<2,1>
    // Extended: 1 sign + 1 exponent bit reserved for inf, 0 mantissa bits ->
    // max() == 0, every nonzero value overflows). Saturation behaviour is
    // vacuous there, so skip it.
    if (!(maxf > 0.0)) return 0;
    const bool   has_inf  = std::numeric_limits<T>::has_infinity;
    // A format may declare has_infinity but (for degenerate tiny widths) not
    // actually produce a true inf bit-pattern; only formats that really yield
    // inf can distinguish OvfInf/SatPropagate from SatFinite.
    const double fmt_inf  = static_cast<double>(std::numeric_limits<T>::infinity());
    const bool   real_inf = has_inf && std::isinf(fmt_inf);

    const Rounding_Mode rm = Rounding_Mode::RoundToNearestEven;
    const ProjSpec ps_ovf {rm, Saturation_Mode::OvfInf};
    const ProjSpec ps_fin {rm, Saturation_Mode::SatFinite};
    const ProjSpec ps_prop{rm, Saturation_Mode::SatPropagate};

    auto is_pos_inf = [](double v){ return std::isinf(v) && v > 0; };
    auto is_neg_inf = [](double v){ return std::isinf(v) && v < 0; };

    auto check = [&](const char* what, bool ok, double x, double got, double want) {
        if (!ok) {
            std::cout << label << " " << what << " FAILED (x=" << x
                      << " got=" << got << " want=" << want
                      << " maxf=" << maxf << " real_inf=" << real_inf << ")\n";
            ++num_errors;
        }
    };

    // ---- (a) finite, over-range inputs (both signs) ----------------------
    // maxf * k overflows the finite range for any k > 1, for every rounding
    // mode, so we don't depend on directed-rounding-at-overflow subtleties.
    const double mults[] = {2.0, 8.0, 137.0, 1e6};
    for (double m : mults) {
        for (double sign : {+1.0, -1.0}) {
            const double x = sign * maxf * m;

            const double r_ovf  = static_cast<double>(Round<T>(static_cast<float>(x), ps_ovf));
            const double r_fin  = static_cast<double>(Round<T>(static_cast<float>(x), ps_fin));
            const double r_prop = static_cast<double>(Round<T>(static_cast<float>(x), ps_prop));

            // SatFinite and (finite) SatPropagate always clamp to +/-max-finite.
            check("SatFinite(finite-ovf)",  r_fin  == sign * maxf, x, r_fin,  sign * maxf);
            check("SatPropagate(finite-ovf)", r_prop == sign * maxf, x, r_prop, sign * maxf);

            if (real_inf) {
                // OvfInf -> +/-inf for a has-inf format.
                check("OvfInf(finite-ovf)->inf",
                      (sign > 0 ? is_pos_inf(r_ovf) : is_neg_inf(r_ovf)),
                      x, r_ovf, sign * fmt_inf);
            } else {
                // No real infinity: OvfInf clamps to +/-max-finite too.
                check("OvfInf(finite-ovf)->max", r_ovf == sign * maxf, x, r_ovf, sign * maxf);
            }
        }
    }

    // ---- (b) infinite inputs (both signs) --------------------------------
    for (double sign : {+1.0, -1.0}) {
        const float xinf = sign * std::numeric_limits<float>::infinity();

        const double r_ovf  = static_cast<double>(Round<T>(xinf, ps_ovf));
        const double r_fin  = static_cast<double>(Round<T>(xinf, ps_fin));
        const double r_prop = static_cast<double>(Round<T>(xinf, ps_prop));

        // SatFinite: an infinite input is always clamped to +/-max-finite.
        check("SatFinite(inf-in)->max", r_fin == sign * maxf,
              static_cast<double>(xinf), r_fin, sign * maxf);

        if (real_inf) {
            // OvfInf and SatPropagate preserve a true infinity.
            check("OvfInf(inf-in)->inf",
                  (sign > 0 ? is_pos_inf(r_ovf) : is_neg_inf(r_ovf)),
                  static_cast<double>(xinf), r_ovf, sign * fmt_inf);
            check("SatPropagate(inf-in)->inf",
                  (sign > 0 ? is_pos_inf(r_prop) : is_neg_inf(r_prop)),
                  static_cast<double>(xinf), r_prop, sign * fmt_inf);
        } else {
            check("OvfInf(inf-in)->max",      r_ovf  == sign * maxf,
                  static_cast<double>(xinf), r_ovf, sign * maxf);
            check("SatPropagate(inf-in)->max", r_prop == sign * maxf,
                  static_cast<double>(xinf), r_prop, sign * maxf);
        }
    }

    // ---- (c) in-range values are unaffected by Saturation_Mode -----------
    // 1.0 is representable in every P_3109 format here (exponent 0); all three
    // saturation modes must return the identical finite value.
    {
        const float in_range = 1.0f;
        const double r_ovf  = static_cast<double>(Round<T>(in_range, ps_ovf));
        const double r_fin  = static_cast<double>(Round<T>(in_range, ps_fin));
        const double r_prop = static_cast<double>(Round<T>(in_range, ps_prop));
        check("in-range invariance (ovf==fin)",  r_ovf == r_fin,  1.0, r_ovf, r_fin);
        check("in-range invariance (ovf==prop)", r_ovf == r_prop, 1.0, r_ovf, r_prop);
        check("in-range finite", std::isfinite(r_ovf), 1.0, r_ovf, 1.0);
    }

    return num_errors;
}

template <int l, int p>
int test_sat_3109() {
    int e = 0;
    // Saturating (no infinity) variant — all three modes clamp to max-finite.
    e += test_sat_one<P_3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>>(
        ("P_3109<l,p>(Sat) "));
    // Extended (has infinity) variant — the three modes diverge.
    e += test_sat_one<P_3109_float<l, p, Signedness::Signed, Inf_Behaviors::Extended>>(
        ("P_3109<l,p>(Ext) "));
    if (e == 0)
        std::cout << "P_3109<" << l << "," << p << "> : pass\n";
    else
        std::cout << "P_3109<" << l << "," << p << "> : FAIL (" << e << ")\n";
    return e;
}

// ---- instantiation machinery (mirrors test_rounding_modes.cpp) ----------
static int g_total_errors = 0;

template <int l, int... Ps>
void sat_for_l(std::integer_sequence<int, Ps...>) {
    ((g_total_errors += test_sat_3109<l, Ps + 1>()), ...);
}
template <int... Ls>
void sat_all_l(std::integer_sequence<int, Ls...>) {
    (sat_for_l<Ls>(std::make_integer_sequence<int, Ls - 1>{}), ...);
}
template <int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

int main() {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // l from 2 to 8, p from 1 to l-1 — same sweep as test_rounding_modes.
    sat_all_l(offset_sequence<2>(std::make_integer_sequence<int, 7>{}));

    if (g_total_errors == 0)
        std::cout << "test_sat_mode: ALL PASS\n";
    else
        std::cout << "test_sat_mode: FAIL with " << g_total_errors << " errors\n";
    return g_total_errors == 0 ? 0 : 1;
}
