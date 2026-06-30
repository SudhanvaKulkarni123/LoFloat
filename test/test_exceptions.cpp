// Tests IEEE-754 exception-flag signaling (no trapping).
//
// Build with -DENABLE_EXCEPT (the Makefile target does this). The flags are only
// raised for formats whose NA_behavior == _754; a parallel _3109 format is used
// to confirm the gate stays silent.
#include <cmath>
#include <cstdint>
#include <iostream>
#include "lo_float.h"
#include "lo_float_sci.hpp"

using namespace lo_float;
namespace lfi = lo_float::lo_float_internal;

// ---- e5m2 inf/NaN checkers (sign | 5 exp | 2 mant) ------------------------
struct E5M2Inf {
    bool operator()(uint32_t b) const { return ((b >> 2) & 0x1F) == 0x1F && (b & 0x3) == 0; }
    uint32_t infBitPattern() const { return 0x7C; }
    uint32_t minPosInf()    const { return 0x7C; }
    uint32_t minNegInf()    const { return 0xFC; }
};
struct E5M2NaN {
    bool operator()(uint32_t b) const { return ((b >> 2) & 0x1F) == 0x1F && (b & 0x3) != 0; }
    uint32_t qNanBitPattern() const { return 0x7E; }
    uint32_t sNanBitPattern() const { return 0x7D; }
};

constexpr FloatingPointParams param_754(
    8, 2, 15, Inf_Behaviors::Extended, NaN_Behaviors::_754,
    Signedness::Signed, E5M2Inf(), E5M2NaN());
constexpr FloatingPointParams param_3109(
    8, 2, 15, Inf_Behaviors::Extended, NaN_Behaviors::_3109,
    Signedness::Signed, E5M2Inf(), E5M2NaN());

using f754  = Templated_Float<param_754>;
using f3109 = Templated_Float<param_3109>;

static int failures = 0;

static bool has(uint8_t f, lfi::LF_exception_flags bit) {
    return (f & static_cast<uint8_t>(bit)) != 0;
}

// Run `body`, then assert the named flag is (raised==true) raised afterwards.
template <typename F>
static void expect(const char* name, lfi::LF_exception_flags bit, bool raised, F&& body) {
    lfi::f_env.reset_exception_flags();
    body();
    const uint8_t f = lfi::f_env.get_exception_flags();
    const bool ok = has(f, bit) == raised;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name
              << " (flags=0x" << std::hex << (int)f << std::dec << ")\n";
    if (!ok) ++failures;
}

// Assert NO flags at all were raised.
template <typename F>
static void expect_none(const char* name, F&& body) {
    lfi::f_env.reset_exception_flags();
    body();
    const uint8_t f = lfi::f_env.get_exception_flags();
    const bool ok = (f == 0);
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name
              << " (flags=0x" << std::hex << (int)f << std::dec << ")\n";
    if (!ok) ++failures;
}

int main() {
    using IF = lfi::LF_exception_flags;
    volatile double sink = 0.0;  // keep results live

    const f754 one   = f754(1.0);
    const f754 zero  = f754(0.0);
    const f754 inf_  = std::numeric_limits<f754>::infinity();

    // ---- division ---------------------------------------------------------
    expect("finite/0 -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ f754 r = one / zero; sink = (double)r; });
    expect("finite/0 NOT InvalidOperation", IF::InvalidOperation, false,
           [&]{ f754 r = one / zero; sink = (double)r; });
    expect("0/0 -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = zero / zero; sink = (double)r; });
    expect("0/0 NOT DivisionByZero", IF::DivisionByZero, false,
           [&]{ f754 r = zero / zero; sink = (double)r; });

    // ---- invalid arithmetic ----------------------------------------------
    expect("inf - inf -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = inf_ - inf_; sink = (double)r; });
    expect("0 * inf -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = zero * inf_; sink = (double)r; });

    // ---- overflow / underflow on conversion ------------------------------
    expect("overflow on Project(1e30) -> Overflow", IF::Overflow, true,
           [&]{ f754 r = Project<f754>(1.0e30); sink = (double)r; });
    expect("tiny value -> Underflow", IF::Underflow, true,
           [&]{ f754 r = Project<f754>(std::ldexp(1.0, -15)); sink = (double)r; });

    // ---- float -> int ----------------------------------------------------
    expect("inf -> int -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ int v = (int)inf_; sink = v; });

    // ---- transcendentals (lo_float_sci.hpp) ------------------------------
    expect("sqrt(-1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = sqrt(f754(-1.0)); sink = (double)r; });
    expect("acosh(0) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = acosh(f754(0.0)); sink = (double)r; });
    expect("log(-1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = log(f754(-1.0)); sink = (double)r; });
    expect("log2(0) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = log2(zero); sink = (double)r; });
    expect("atanh(1) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = atanh(one); sink = (double)r; });
    expect("rSqrt(0) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = rSqrt(zero); sink = (double)r; });

    // ---- fma -------------------------------------------------------------
    expect("fma(0, inf, 1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = fma<param_754, param_754, param_754, param_754>(zero, inf_, one);
                sink = (double)r; });
    expect_none("fma(2, 3, 1) raises nothing",
           [&]{ auto r = fma<param_754, param_754, param_754, param_754>(f754(2.0), f754(3.0), one);
                sink = (double)r; });

    // ---- value sanity for a few new transcendentals ----------------------
    {
        const double got = (double)log10(f754(100.0));
        const bool ok = std::abs(got - 2.0) < 0.2;   // coarse: e5m2 is low precision
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << "log10(100) ~= 2 (got " << got << ")\n";
        if (!ok) ++failures;
    }

    // ---- gate: a _3109 format must raise NOTHING -------------------------
    const f3109 q_one  = f3109(1.0);
    const f3109 q_zero = f3109(0.0);
    expect_none("_3109 finite/0 raises nothing",
           [&]{ f3109 r = q_one / q_zero; sink = (double)r; });
    expect_none("_3109 sqrt(-1) raises nothing",
           [&]{ auto r = sqrt(f3109(-1.0)); sink = (double)r; });
    expect_none("_3109 overflow raises nothing",
           [&]{ f3109 r = Project<f3109>(1.0e30); sink = (double)r; });

    (void)sink;
    if (failures == 0) {
        std::cout << "All tests passed\n";
        return 0;
    }
    std::cout << failures << " test(s) FAILED\n";
    return 1;
}
