///@author Sudhanva Kulkarni, UC Berkeley
// -----------------------------------------------------------------------------
// T5 — integer-type tests for i_n<len, Signedness>.
//
// Exhaustively checks arithmetic (+ - * / %), bitwise shifts, and the full set
// of comparison operators for INT2..INT10, both signed and unsigned, against a
// wide-integer reference.
//
// SATURATION semantics: i_n clamps out-of-range results to [min, max] (see
// clamp_to_storage in src/lo_int.h). This is an INTENTIONAL deviation from the
// paper, which specifies wrap-around overflow for integer types; the reference
// here therefore saturates, matching the implementation rather than the spec.
// -----------------------------------------------------------------------------
#include <iostream>
#include <cstdint>
#include <utility>
#include "lo_int.h"

using namespace lo_float;

// Wide-integer saturating reference for a len-bit value of signedness S.
template <int len, Signedness S>
struct ref_range {
    static constexpr long long LO = (S == Signedness::Signed) ? -(1LL << (len - 1)) : 0LL;
    static constexpr long long HI = (S == Signedness::Signed) ? ((1LL << (len - 1)) - 1)
                                                              : ((1LL << len) - 1);
    static constexpr long long sat(long long x) {
        return x < LO ? LO : (x > HI ? HI : x);
    }
};

template <int len, Signedness S>
int test_intn() {
    using T = i_n<len, S>;
    using R = ref_range<len, S>;
    int errors = 0;

    auto as_ll = [](T v) { return static_cast<long long>(static_cast<int>(v)); };

    for (long long a = R::LO; a <= R::HI; ++a) {
        T ta(a);
        // Construction round-trips exactly (a is already in range).
        if (as_ll(ta) != a) {
            std::cout << "  ctor mismatch a=" << a << " got " << as_ll(ta) << "\n";
            ++errors;
        }

        for (long long b = R::LO; b <= R::HI; ++b) {
            T tb(b);

            if (as_ll(ta + tb) != R::sat(a + b)) { std::cout << "  ADD " << a << "+" << b << "\n"; ++errors; }
            if (as_ll(ta - tb) != R::sat(a - b)) { std::cout << "  SUB " << a << "-" << b << "\n"; ++errors; }
            if (as_ll(ta * tb) != R::sat(a * b)) { std::cout << "  MUL " << a << "*" << b << "\n"; ++errors; }
            if (b != 0) {
                if (as_ll(ta / tb) != R::sat(a / b)) { std::cout << "  DIV " << a << "/" << b << "\n"; ++errors; }
                if (as_ll(ta % tb) != R::sat(a % b)) { std::cout << "  MOD " << a << "%" << b << "\n"; ++errors; }
            }

            // Comparisons must match plain integer comparison.
            if ((ta == tb) != (a == b)) { std::cout << "  EQ " << a << "," << b << "\n"; ++errors; }
            if ((ta != tb) != (a != b)) { std::cout << "  NE " << a << "," << b << "\n"; ++errors; }
            if ((ta <  tb) != (a <  b)) { std::cout << "  LT " << a << "," << b << "\n"; ++errors; }
            if ((ta >  tb) != (a >  b)) { std::cout << "  GT " << a << "," << b << "\n"; ++errors; }
            if ((ta <= tb) != (a <= b)) { std::cout << "  LE " << a << "," << b << "\n"; ++errors; }
            if ((ta >= tb) != (a >= b)) { std::cout << "  GE " << a << "," << b << "\n"; ++errors; }
        }

        // Shifts (k in [0, len)). Left shift saturates; right shift stays in range
        // (arithmetic for signed, logical for unsigned — both equal a >> k here).
        for (int k = 0; k < len; ++k) {
            if (as_ll(ta << k) != R::sat(a << k)) { std::cout << "  SHL " << a << "<<" << k << "\n"; ++errors; }
            if (as_ll(ta >> k) != (a >> k))       { std::cout << "  SHR " << a << ">>" << k << "\n"; ++errors; }
        }
    }

    std::cout << "INT" << len << (S == Signedness::Signed ? " signed  " : " unsigned")
              << " : " << (errors == 0 ? "pass" : "FAIL") << "\n";
    return errors;
}

// Explicit boundary spot-checks (0, +max, min) documenting saturation behaviour.
template <int len, Signedness S>
int test_boundaries() {
    using T = i_n<len, S>;
    using R = ref_range<len, S>;
    int errors = 0;
    auto as_ll = [](T v) { return static_cast<long long>(static_cast<int>(v)); };

    T mx(R::HI), mn(R::LO), one(1), zero(0);
    if (as_ll(mx + one) != R::HI) { std::cout << "  max+1 not saturated (INT" << len << ")\n"; ++errors; }   // overflow -> max
    if (as_ll(mn - one) != R::LO) { std::cout << "  min-1 not saturated (INT" << len << ")\n"; ++errors; }   // underflow -> min
    if (as_ll(zero * mx) != 0)    { std::cout << "  0*max != 0 (INT" << len << ")\n"; ++errors; }
    return errors;
}

template <int len, Signedness S, int... Bonus>
int run_one() {
    return test_intn<len, S>() + test_boundaries<len, S>();
}

template <int... Ls>
int run_all_lengths(std::integer_sequence<int, Ls...>) {
    int e = 0;
    ((e += run_one<Ls + 2, Signedness::Signed>()), ...);
    ((e += run_one<Ls + 2, Signedness::Unsigned>()), ...);
    return e;
}

int main() {
    // INT2..INT10 -> lengths {2,3,...,10} == {0..8} + 2
    int errors = run_all_lengths(std::make_integer_sequence<int, 9>{});
    std::cout << "\nlo_int (T5) exhaustive INT2-INT10 saturating test: "
              << (errors == 0 ? "PASS" : "FAIL") << "\n";
    return errors == 0 ? 0 : 1;
}
