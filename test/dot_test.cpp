// Mixed-precision DOT validation tests.
//
//   T6  (test_mixed_dot) — accumulator-roundoff test, randomized "Strategy 1":
//       construct inputs whose true dot product is ~0 so the measured error is
//       dominated by accumulator roundoff, then confirm the DOT accumulated at the
//       claimed precision Fp_accum by checking the error against the bound of [16].
//   T10 (test_mx_dot)    — MXDot: same fill-and-cancel backbone on MicroScaling
//       vectors (private low-precision elements + a shared power-of-two scale per block).
//
// Both use a single accumulation precision (inner == outer). See
// loop/LOFLOAT_ARITH_paper.md sections T6 / T10.

#include "Vector.h"
#include "Dot.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <limits>

using namespace lo_float;
using namespace Lo_Gemm;

static bool g_verbose = false;

template<typename T>
void print_vec(T* x, int n) {
    for (int i = 0; i < n; i++) std::cout << (double)x[i] << " ";
    std::cout << "\n";
}

// ---- double-precision references over typed buffers ----
template<Float Fp1, Float Fp2>
double ref_dot(const Fp1* x, const Fp2* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * (double)y[i];
    return s;
}
template<Float Fp>
double one_norm(const Fp* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += std::abs((double)x[i]);
    return s;
}
template<Float Fp1, Float Fp2>
double ref_abs_dot(const Fp1* x, const Fp2* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += std::abs((double)x[i] * (double)y[i]);
    return s;
}

// ---- double-precision references over plain double buffers ----
static double ddot(const double* x, const double* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}
static double dabs_dot(const double* x, const double* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += std::abs(x[i] * y[i]);
    return s;
}
static double d1norm(const double* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += std::abs(x[i]);
    return s;
}

// Unit roundoff (machine epsilon) of a float format, computed from its digit count.
// numeric_limits<Templated_Float>::epsilon() is buggy for the custom types, so we
// derive eps = 2^(1-digits) directly (matches std::numeric_limits<float>::epsilon()).
template<Float Fp>
double format_eps() {
    return std::ldexp(1.0, 1 - std::numeric_limits<Fp>::digits);
}

// Error bound of [16] for the accumulator-roundoff test (Strategy 1):
//   bound = (n+2)*eps_accum*S + U + (eps_out/2)*|s|,
// where S = sum|x_i*y_i|, s = accurate dot, and U is the underflow term.
template<Float Fp_accum, Float Fp_out>
double accum_bound(int n, double S, double abs_s, double x1norm, double y1norm) {
    const double dn = (double)std::numeric_limits<Fp_accum>::denorm_min();
    double U = std::max((double)(2 * n + 3),
                        std::max(y1norm + 2 * n + 1, x1norm + 2 * n + 1));
    U = U * dn + dn;
    return (n + 2) * format_eps<Fp_accum>() * S + U + 0.5 * format_eps<Fp_out>() * abs_s;
}

// =====================================================================================
// T6 — mixed-precision DOT, accumulator-roundoff (randomized, Strategy 1)
// =====================================================================================
template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, int block_size>
bool test_mixed_dot(const char* name, int N) {
    const double fp2_max = (double)std::numeric_limits<Fp2>::max();

    // Working buffers with headroom so overflow-splitting never reallocates.
    const int cap = 4 * N + block_size + 16;
    Fp1* xb = (Fp1*)malloc(cap * sizeof(Fp1));
    Fp2* yb = (Fp2*)malloc(cap * sizeof(Fp2));
    int cnt = 0;

    // x drawn in [0.5, 1.0) (quantized to Fp1) — keeps divisors away from 0 so
    // s/x stays bounded and overflow-splitting terminates quickly.
    auto draw_x = []() -> double {
        double v = 0.5 + 0.5 * ((double)rand() / (double)RAND_MAX);
        return (double)(Fp1)v;
    };

    // Append one logical term x*y == target, splitting y across several entries
    // (with duplicated x) when target is not representable in Fp2 (T6 overflow case).
    // When target is representable the while-loop body never runs, so this reduces
    // to a single (x, Fp2(target)) entry in the common case.
    auto emit = [&](double xi, double target) {
        double remaining = target;
        while (std::abs(remaining) > fp2_max) {
            double chunk = (remaining > 0 ? fp2_max : -fp2_max);
            xb[cnt] = (Fp1)xi;
            yb[cnt] = (Fp2)chunk;             // exactly representable (== fp2_max)
            remaining -= (double)yb[cnt];
            cnt++; assert(cnt < cap);
        }
        xb[cnt] = (Fp1)xi;
        yb[cnt] = (Fp2)remaining;
        cnt++; assert(cnt < cap);
    };

    // Number of cancellation steps (leading-bit cancellation, ~digits/3 of double).
    const int k = std::max(2, (int)std::ceil(std::numeric_limits<double>::digits / 3.0));

    // Small-type fill-priming: if Fp2's dynamic range can't fill the accumulator
    // mantissa starting from exponent 0, prime the accumulator near the top of Fp2's
    // range so the geometric descent walks down through every accumulator bit.
    const int fp2_range = std::numeric_limits<Fp2>::max_exponent
                        - std::numeric_limits<Fp2>::min_exponent;
    const bool small_type = fp2_range < std::numeric_limits<Fp_accum>::digits;
    double term = small_type ? 0.5 * fp2_max : ((double)rand() / (double)RAND_MAX);
    const double descent = 0.25;   // 2^-2 per the paper

    // term-0
    {
        double x0 = draw_x();
        emit(x0, term / x0);
    }
    // fill loop: each new term ~ term, geometrically decreasing
    const int n_fill = std::max(1, N - k);
    for (int j = 1; j < n_fill; j++) {
        term *= descent;
        double xj = draw_x();
        emit(xj, term / xj);
    }
    // cancel loop: drive the running sum toward 0 by cancelling it each step
    for (int j = 0; j < k; j++) {
        double s_run = ref_dot(xb, yb, cnt);
        double xj = draw_x();
        emit(xj, -s_run / xj);
    }
    // pad to a multiple of block_size with zero-product terms — the plain dot drops
    // the trailing partial block (num_blocks = n/block_size), and our cancel terms
    // are at the end, so they must not be dropped.
    while (cnt % block_size != 0) { xb[cnt] = (Fp1)1.0; yb[cnt] = (Fp2)0.0; cnt++; }
    const int n = cnt;

    Vector<Fp1, int> X(xb, n);
    Vector<Fp2, int> Y(yb, n);

    const double s = ref_dot(xb, yb, n);
    auto result = dot<Fp1, Fp2, int, Fp_out, Fp_accum, Fp_accum, block_size>(X, Y);

    const double abs_s = std::abs(s);
    const double S = ref_abs_dot(xb, yb, n);
    const double bound = accum_bound<Fp_accum, Fp_out>(
        n, S, abs_s, one_norm(xb, n), one_norm(yb, n));
    const double err = std::abs((double)result - s);
    const bool pass = err <= bound;

    std::cout << "  T6  " << std::left << std::setw(22) << name
              << " n=" << std::setw(6) << n
              << " err=" << std::setw(12) << err
              << " bound=" << std::setw(12) << bound
              << (pass ? "  pass" : "  FAIL") << "\n";
    if (g_verbose)
        std::cout << "        s=" << s << " result=" << (double)result
                  << " S=" << S << " primed=" << small_type << "\n";

    free(xb); free(yb);
    return pass;
}

// =====================================================================================
// T10 — MXDot: same backbone on MicroScaling vectors.
//   Build near-zero, accumulator-filling true values (doubles), then microscale each
//   block of r elements (shared power-of-two E8M0 scale + low-precision private elems)
//   and run the MX dot. The shared exponents descend naturally across blocks because
//   the true values descend geometrically (the MX analogue of T6's 2^-2 descent).
// =====================================================================================
template<Float Fp1, Float Fp2, Float Fp_scal, Float Fp_accum, Float Fp_out, int r>
bool test_mx_dot(const char* name, int N) {
    const int n = (N / r) * r;                 // exact multiple of r (MX dot needs n%r==0)
    const int nb = n / r;

    double* xt = (double*)malloc(n * sizeof(double));   // true values (doubles)
    double* yt = (double*)malloc(n * sizeof(double));
    Fp1* xd = (Fp1*)malloc(n * sizeof(Fp1));            // private elements
    Fp2* yd = (Fp2*)malloc(n * sizeof(Fp2));
    Fp_scal* xs = (Fp_scal*)malloc(nb * sizeof(Fp_scal)); // shared scales
    Fp_scal* ys = (Fp_scal*)malloc(nb * sizeof(Fp_scal));

    auto draw_x = []() -> double {
        return 0.5 + 0.5 * ((double)rand() / (double)RAND_MAX);
    };

    // --- build near-zero true values for x and y (in double) ---
    // The block magnitude descends by `sexp` binades per block (the T10 shared-exponent
    // "rate of descent"): block b has products ~ 2^(-sexp*b). Spanning more binades than
    // the accumulator mantissa fills it and stresses accumulator roundoff. The descent is
    // gentle (per block, not per element) so values stay in the E8M0 / private ranges.
    const int sexp = 2;
    const int cancel_blocks = std::max(1, std::min(nb / 4, 4));
    const int fill_blocks = std::max(1, nb - cancel_blocks);
    for (int b = 0; b < fill_blocks; b++) {
        const double Mb = std::ldexp(1.0, -sexp * b);     // block product magnitude
        for (int i = b * r; i < (b + 1) * r; i++) {
            xt[i] = draw_x();
            yt[i] = Mb / xt[i];                            // x*y ~ Mb
        }
    }
    // cancel: each cancel block cancels the running sum with its first entry (the rest
    // of the block is zero) — T10's "compute accurate_dot over j..j+r and cast the whole
    // block simultaneously". Per-block (not per-element) cancellation keeps the residual
    // from free-falling into the denormal/underflow range across the cancel region.
    for (int b = fill_blocks; b < nb; b++) {
        const int i0 = b * r;
        const double s_run = ddot(xt, yt, i0);
        xt[i0] = draw_x();
        yt[i0] = -s_run / xt[i0];                  // cancel the running sum
        for (int i = i0 + 1; i < (b + 1) * r; i++) { xt[i] = draw_x(); yt[i] = 0.0; }
    }

    // --- microscale each block of r into MX form (per-block power-of-two scale) ---
    const double xpriv_max = (double)std::numeric_limits<Fp1>::max();
    const double ypriv_max = (double)std::numeric_limits<Fp2>::max();
    // E8M0 represents 2^(X-127), X in [0,254] -> exponents [-127, 127]; clamp so the scale
    // never underflows to 0 (which would make true/scale infinite).
    auto choose_scale = [](double amax, double priv_max) -> double {
        if (!(amax > 0.0)) return 1.0;
        int e = (int)std::ceil(std::log2(amax / priv_max));   // 2^e * priv_max >= amax
        if (e < -100) e = -100;   // stay clear of the E8M0 exponent edges (avoid scale->0)
        if (e > 100)  e = 100;
        return std::ldexp(1.0, e);
    };
    for (int b = 0; b < nb; b++) {
        double xamax = 0.0, yamax = 0.0;
        for (int i = b * r; i < (b + 1) * r; i++) {
            xamax = std::max(xamax, std::abs(xt[i]));
            yamax = std::max(yamax, std::abs(yt[i]));
        }
        Fp_scal sx = (Fp_scal)choose_scale(xamax, xpriv_max);   // E8M0: exact power of two
        Fp_scal sy = (Fp_scal)choose_scale(yamax, ypriv_max);
        xs[b] = sx; ys[b] = sy;
        const double sxd = (double)sx, syd = (double)sy;
        for (int i = b * r; i < (b + 1) * r; i++) {
            xd[i] = (Fp1)(xt[i] / sxd);          // private element (low precision)
            yd[i] = (Fp2)(yt[i] / syd);
        }
    }

    MX_Vector<Fp1, Fp_scal, int> X(xd, xs, n, nb, 1, r);
    MX_Vector<Fp2, Fp_scal, int> Y(yd, ys, n, nb, 1, r);

    // --- reference over the stored (descaled) values, in double ---
    double s = 0.0, S = 0.0, x1 = 0.0, y1 = 0.0;
    for (int b = 0; b < nb; b++) {
        const double sxd = (double)xs[b], syd = (double)ys[b];
        for (int i = b * r; i < (b + 1) * r; i++) {
            const double xv = (double)xd[i] * sxd;
            const double yv = (double)yd[i] * syd;
            s  += xv * yv;
            S  += std::abs(xv * yv);
            x1 += std::abs(xv);
            y1 += std::abs(yv);
        }
    }

    auto result = dot<Fp1, Fp_scal, Fp2, Fp_scal, int, Fp_out, Fp_accum, Fp_accum>(X, Y);

    const double abs_s = std::abs(s);
    const double bound = accum_bound<Fp_accum, Fp_out>(n, S, abs_s, x1, y1);
    const double err = std::abs((double)result - s);
    const bool pass = err <= bound;

    std::cout << "  T10 " << std::left << std::setw(22) << name
              << " n=" << std::setw(6) << n
              << " err=" << std::setw(12) << err
              << " bound=" << std::setw(12) << bound
              << (pass ? "  pass" : "  FAIL") << "\n";
    if (g_verbose)
        std::cout << "        s=" << s << " result=" << (double)result
                  << " S=" << S << " r=" << r << "\n";

    free(xt); free(yt); free(xd); free(yd); free(xs); free(ys);
    return pass;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose")
            g_verbose = true;

    std::srand(12345);
    std::cout << std::setprecision(6);

    // FP8 P3109 input/element formats
    using binary8p3   = P_3109_float<8, 3, Signedness::Signed, Inf_Behaviors::Extended>;
    using binary8p4   = P_3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>;
    using binary8p4sf = P_3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Saturating>;
    using scale_t     = lo_float::ocp_e8m0;   // E8M0 shared scale (power-of-two only)

    int failures = 0;
    const int N = 1024;

    std::cout << "=== T6: mixed-precision DOT accumulator-roundoff (Strategy 1) ===\n";
    // Output type == accumulator type so the measured error is pure accumulator roundoff
    // (a coarse output would mask the accumulator precision the test is backing out).
    failures += !test_mixed_dot<binary8p3, binary8p4, half,  half,  4>("p3*p4 / half-acc",  N);
    failures += !test_mixed_dot<binary8p3, binary8p4, float, float, 4>("p3*p4 / float-acc", N);
    // small saturating type pair — exercises fill-priming with a float accumulator
    failures += !test_mixed_dot<binary8p4sf, binary8p4sf, float, float, 4>("p4sf*p4sf / float-acc", N);
    failures += !test_mixed_dot<binary8p4,   binary8p4,   half,  half,  8>("p4*p4 / half-acc blk8", N);

    // MX private elements use the full element-format range, so r products of full-range
    // elements need an fp32+ inner accumulator (a half accumulator would overflow) — use
    // float and double accumulation, the realistic MX configurations.
    std::cout << "=== T10: MXDot ===\n";
    failures += !test_mx_dot<binary8p3, binary8p4, scale_t, float,  float,  32>("p3*p4 / float-acc r32",  N);
    failures += !test_mx_dot<binary8p3, binary8p4, scale_t, double, double, 32>("p3*p4 / double-acc r32", N);
    failures += !test_mx_dot<binary8p4, binary8p4, scale_t, float,  float,  16>("p4*p4 / float-acc r16",  N);

    std::cout << "\n=== TOTAL FAILURES: " << failures
              << (failures == 0 ? "  (ALL PASS) ===\n" : "  (FAIL) ===\n");
    return failures ? 1 : 0;
}
