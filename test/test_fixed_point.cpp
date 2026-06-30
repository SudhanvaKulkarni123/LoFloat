///@author loft loop — fixed_point<IntBits,FracBits,Sign> tests
//
// Verifies the Q(IntBits.FracBits) fixed-point type added to lo_int.h:
//   value = int_part + 2^{-FracBits} * frac_part = raw * 2^{-FracBits}.
//
// Following loop/NUMERICAL_TESTING.md:
//  - oracle is computed independently in double / int64_t, never the DUT itself;
//  - exactly-representable grid values round-trip bit-exact;
//  - small formats are enumerated exhaustively (every raw bit pattern);
//  - from_double rounding is checked against std::llround (round-half-away),
//    a *different* code path from the type's internal +/-0.5 trick;
//  - tie/midpoint inputs are constructed explicitly;
//  - saturating overflow is exercised at both ends.
//
// A test accumulates an error count; 0 = pass (quiet), nonzero prints the
// failing case and the program exits nonzero.

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>

#include "lo_int.h"

using namespace lo_float;

static int g_errors = 0;

template <typename... Args>
static void fail(const char* what, Args... a) {
    ++g_errors;
    std::cerr << "FAIL [" << what << "]";
    ((std::cerr << ' ' << a), ...);
    std::cerr << '\n';
}

// ---- per-format constants computed independently of the DUT ----
template <int I, int F, Signedness S>
struct Oracle {
    static constexpr int  W   = I + F;
    static constexpr bool sgn = (S == Signedness::Signed);
    static constexpr long long ONE = 1LL << F;
    static constexpr long long HI  = sgn ? ((1LL << (W - 1)) - 1) : ((1LL << W) - 1);
    static constexpr long long LO  = sgn ? -(1LL << (W - 1)) : 0LL;

    static long long clamp(long long r) {
        if (r > HI) return HI;
        if (r < LO) return LO;
        return r;
    }
    // round-half-away-from-zero of x*2^F, then saturate (independent libc path)
    static long long from_double(double x) {
        if (std::isnan(x)) return 0;
        double maxv = double(HI) / double(ONE);
        double minv = double(LO) / double(ONE);
        if (x >= maxv) return HI;
        if (x <= minv) return LO;
        return clamp((long long)std::llround(x * (double)ONE));
    }
};

// ---------------------------------------------------------------------------
//  Test 1: exhaustive enumeration of one small format — round-trip, field
//          decomposition, value, and unary negate against the oracle.
// ---------------------------------------------------------------------------
template <int I, int F, Signedness S>
void enum_unary() {
    using FP  = fixed_point<I, F, S>;
    using Ora = Oracle<I, F, S>;
    for (long long r = Ora::LO; r <= Ora::HI; ++r) {
        FP x = FP::FromRaw(static_cast<decltype(x.raw())>(r));

        // raw stored exactly
        if ((long long)x.raw() != r) fail("raw-store", I, F, (int)S, r, (long long)x.raw());

        // value = raw / 2^F  (exact for these small W)
        double v   = x.to_double();
        double ref = double(r) / double(Ora::ONE);
        if (v != ref) fail("to_double", I, F, (int)S, r, v, ref);

        // field decomposition: int_part + frac_part/2^F == value (exact)
        long long ip = (long long)x.int_part();
        long long fp = (long long)x.frac_part();
        if (fp < 0 || fp >= Ora::ONE) fail("frac-range", I, F, (int)S, r, fp);
        double recon = double(ip) + double(fp) / double(Ora::ONE);
        if (recon != v) fail("decompose", I, F, (int)S, r, recon, v);

        // grid value must round-trip bit-exact through from_double
        FP rt(v);
        if ((long long)rt.raw() != r) fail("roundtrip", I, F, (int)S, r, (long long)rt.raw());

        // unary minus, saturating
        FP neg = -x;
        long long expect = Ora::clamp(-r);
        if ((long long)neg.raw() != expect) fail("neg", I, F, (int)S, r, (long long)neg.raw(), expect);
    }
}

// ---------------------------------------------------------------------------
//  Test 2: exhaustive pairwise arithmetic against int64 oracle.
// ---------------------------------------------------------------------------
template <int I, int F, Signedness S>
void enum_binary() {
    using FP  = fixed_point<I, F, S>;
    using Ora = Oracle<I, F, S>;
    for (long long a = Ora::LO; a <= Ora::HI; ++a) {
        FP A = FP::FromRaw(static_cast<decltype(A.raw())>(a));
        for (long long b = Ora::LO; b <= Ora::HI; ++b) {
            FP B = FP::FromRaw(static_cast<decltype(B.raw())>(b));

            long long add = Ora::clamp(a + b);
            long long sub = Ora::clamp(a - b);
            // mul: product at scale 2F, shift back by F, trunc toward zero
            long long mul = Ora::clamp((a * b) / Ora::ONE);

            if ((long long)(A + B).raw() != add) fail("add", I, F, (int)S, a, b, (long long)(A + B).raw(), add);
            if ((long long)(A - B).raw() != sub) fail("sub", I, F, (int)S, a, b, (long long)(A - B).raw(), sub);
            if ((long long)(A * B).raw() != mul) fail("mul", I, F, (int)S, a, b, (long long)(A * B).raw(), mul);

            if (b != 0) {
                long long div = Ora::clamp((a << F) / b);  // trunc toward zero
                if ((long long)(A / B).raw() != div) fail("div", I, F, (int)S, a, b, (long long)(A / B).raw(), div);
            }

            // comparisons mirror the integer ordering of raw
            if ((A < B)  != (a < b))  fail("lt",  I, F, (int)S, a, b);
            if ((A == B) != (a == b)) fail("eq",  I, F, (int)S, a, b);
        }
    }
}

// ---------------------------------------------------------------------------
//  Test 3: from_double over a sweep incl. off-grid, tie midpoints, overflow.
// ---------------------------------------------------------------------------
template <int I, int F, Signedness S>
void from_double_sweep() {
    using FP  = fixed_point<I, F, S>;
    using Ora = Oracle<I, F, S>;
    const double step = 1.0 / double(Ora::ONE);

    // dense sweep across (and beyond) the representable range, including the
    // exact midpoints (k+0.5)*ULP that distinguish the rounding rule
    double lo = double(Ora::LO) / double(Ora::ONE) - 3.0;
    double hi = double(Ora::HI) / double(Ora::ONE) + 3.0;
    for (double x = lo; x <= hi; x += step * 0.25) {
        long long expect = Ora::from_double(x);
        FP f(x);
        if ((long long)f.raw() != expect) fail("from_double", I, F, (int)S, x, (long long)f.raw(), expect);
    }
    // explicit half-ULP ties at a few grid points
    for (long long k = Ora::LO; k < Ora::HI; ++k) {
        double mid = (double(k) + 0.5) / double(Ora::ONE);
        long long expect = Ora::from_double(mid);
        FP f(mid);
        if ((long long)f.raw() != expect) fail("tie", I, F, (int)S, mid, (long long)f.raw(), expect);
    }
    // saturation at both ends
    FP up(1e30), dn(-1e30);
    if ((long long)up.raw() != Ora::HI) fail("sat-hi", I, F, (int)S, (long long)up.raw(), Ora::HI);
    if ((long long)dn.raw() != Ora::LO) fail("sat-lo", I, F, (int)S, (long long)dn.raw(), Ora::LO);

    // numeric_limits agreement
    if ((long long)std::numeric_limits<FP>::max().raw()    != Ora::HI) fail("nl-max", I, F, (int)S);
    if ((long long)std::numeric_limits<FP>::lowest().raw() != Ora::LO) fail("nl-low", I, F, (int)S);
    if ((long long)std::numeric_limits<FP>::epsilon().raw()!= 1)        fail("nl-eps", I, F, (int)S);
}

int main() {
    // exhaustive unary + decomposition on a spread of small formats
    enum_unary<4, 4, Signedness::Signed>();
    enum_unary<4, 4, Signedness::Unsigned>();
    enum_unary<2, 2, Signedness::Signed>();
    enum_unary<1, 7, Signedness::Signed>();
    enum_unary<6, 2, Signedness::Unsigned>();
    enum_unary<8, 8, Signedness::Signed>();   // W=16 (65536 patterns)

    // exhaustive pairwise arithmetic on W<=8 formats (cheap)
    enum_binary<4, 4, Signedness::Signed>();
    enum_binary<4, 4, Signedness::Unsigned>();
    enum_binary<2, 2, Signedness::Signed>();
    enum_binary<1, 7, Signedness::Signed>();
    enum_binary<5, 3, Signedness::Unsigned>();

    // from_double rounding / saturation
    from_double_sweep<4, 4, Signedness::Signed>();
    from_double_sweep<4, 4, Signedness::Unsigned>();
    from_double_sweep<8, 8, Signedness::Signed>();
    from_double_sweep<2, 6, Signedness::Signed>();

    if (g_errors == 0) {
        std::cout << "test_fixed_point: ALL PASSED\n";
        return 0;
    }
    std::cerr << "test_fixed_point: " << g_errors << " FAILURES\n";
    return 1;
}
