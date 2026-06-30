// Tests the arithmetic operators of lo_float types that were previously
// exercised only by the print-only demo in lo_float_test.cpp.
//
// Convention (same spirit as test_rounding_modes.cpp / recip_test.cpp):
// enumerate each format and compare every operator against a *double*
// reference computed from the same operands:
//
//   format_op(a, b)  ==  round_to_format( double(a) OP double(b) )
//
// The operands are low-precision values, so their exact sum/difference/product
// /quotient is representable in double; rounding that double result back to the
// format must reproduce the operator output exactly. NaN and Inf results are
// compared NaN/Inf-aware (see same()).
//
// Formats mirror test_rounding_modes.cpp:
//   * P_3109_float<l, p, Signed, Saturating> for l = 2..8, p = 1..l-1
//     -> exhaustive over all (a, b) rep pairs.
//   * bf16 / tf32 / binary16 -> 1000 x 1000 random pairs.

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>

#include "lo_float.h"
#include "lo_float_sci.hpp"

using namespace lo_float;

namespace {

// NaN/Inf-aware equality for the float-projected results.
bool same(float x, float y)
{
    if (std::isnan(x) && std::isnan(y)) return true;
    return x == y;   // Inf == Inf and -0 == 0 behave as desired here
}

// Exhaustive +,-,*,/ , unary - and compound assignment for one format.
template <class Fp>
int test_ops_exhaustive(const char* tag)
{
    constexpr unsigned total = 1u << Fp::bitwidth;
    int errors = 0;

    for (unsigned i = 0; i < total; ++i)
    {
        Fp a = Fp::FromRep(i);
        if (isnan(a)) continue;
        const double da = static_cast<double>(a);

        // unary minus
        if (!same(static_cast<float>(-a), static_cast<float>(Fp(-da))))
        {
            if (errors < 10) std::cout << "  [" << tag << "] unary- mismatch @ " << da << "\n";
            ++errors;
        }

        for (unsigned j = 0; j < total; ++j)
        {
            Fp b = Fp::FromRep(j);
            if (isnan(b)) continue;
            const double db = static_cast<double>(b);

            const float add_got = static_cast<float>(a + b);
            const float sub_got = static_cast<float>(a - b);
            const float mul_got = static_cast<float>(a * b);
            const float div_got = static_cast<float>(a / b);

            const float add_ref = static_cast<float>(Fp(da + db));
            const float sub_ref = static_cast<float>(Fp(da - db));
            const float mul_ref = static_cast<float>(Fp(da * db));
            const float div_ref = static_cast<float>(Fp(da / db));

            if (!same(add_got, add_ref)) { if (errors < 10) std::cout << "  [" << tag << "] + mismatch " << da << "+" << db << ": got " << add_got << " ref " << add_ref << "\n"; ++errors; }
            if (!same(sub_got, sub_ref)) { if (errors < 10) std::cout << "  [" << tag << "] - mismatch " << da << "-" << db << ": got " << sub_got << " ref " << sub_ref << "\n"; ++errors; }
            if (!same(mul_got, mul_ref)) { if (errors < 10) std::cout << "  [" << tag << "] * mismatch " << da << "*" << db << ": got " << mul_got << " ref " << mul_ref << "\n"; ++errors; }
            if (!same(div_got, div_ref)) { if (errors < 10) std::cout << "  [" << tag << "] / mismatch " << da << "/" << db << ": got " << div_got << " ref " << div_ref << "\n"; ++errors; }

            // compound assignment must match the corresponding binary operator
            Fp c = a; c += b;
            if (!same(static_cast<float>(c), add_got)) { if (errors < 10) std::cout << "  [" << tag << "] += inconsistent with +\n"; ++errors; }
            c = a; c -= b;
            if (!same(static_cast<float>(c), sub_got)) { if (errors < 10) std::cout << "  [" << tag << "] -= inconsistent with -\n"; ++errors; }
            c = a; c *= b;
            if (!same(static_cast<float>(c), mul_got)) { if (errors < 10) std::cout << "  [" << tag << "] *= inconsistent with *\n"; ++errors; }
            c = a; c /= b;
            if (!same(static_cast<float>(c), div_got)) { if (errors < 10) std::cout << "  [" << tag << "] /= inconsistent with /\n"; ++errors; }
        }
    }

    std::cout << "P_3109 arith [" << tag << "]: " << (errors == 0 ? "pass" : "FAIL")
              << " (" << errors << " errors)\n";
    return errors;
}

// 1000 x 1000 random pairs for the 16-bit formats.
template <class Fp>
int test_ops_random(const char* tag, int outer = 1000, int inner = 1000)
{
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    int errors = 0;

    for (int oi = 0; oi < outer; ++oi)
    {
        Fp a = Fp(dist(gen));
        if (isnan(a)) continue;
        const double da = static_cast<double>(a);

        if (!same(static_cast<float>(-a), static_cast<float>(Fp(-da)))) ++errors;

        for (int ii = 0; ii < inner; ++ii)
        {
            Fp b = Fp(dist(gen));
            if (isnan(b)) continue;
            const double db = static_cast<double>(b);

            if (!same(static_cast<float>(a + b), static_cast<float>(Fp(da + db)))) ++errors;
            if (!same(static_cast<float>(a - b), static_cast<float>(Fp(da - db)))) ++errors;
            if (!same(static_cast<float>(a * b), static_cast<float>(Fp(da * db)))) ++errors;
            if (db != 0.0 && !same(static_cast<float>(a / b), static_cast<float>(Fp(da / db)))) ++errors;
        }
    }

    std::cout << tag << " arith: " << (errors == 0 ? "pass" : "FAIL")
              << " (" << errors << " errors)\n";
    return errors;
}

// --- instantiate P_3109<l,p> for l = 2..8, p = 1..l-1 (mirror rounding test) -

int g_errors = 0;

template <int l, int p>
void run_one()
{
    char tag[32];
    std::snprintf(tag, sizeof(tag), "<%d,%d>", l, p);
    g_errors += test_ops_exhaustive<P_3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>>(tag);
}

template <int l, int... Ps>
void run_for_l(std::integer_sequence<int, Ps...>) { (run_one<l, Ps + 1>(), ...); }

template <int... Ls>
void run_all_l(std::integer_sequence<int, Ls...>) { (run_for_l<Ls>(std::make_integer_sequence<int, Ls - 1>{}), ...); }

template <int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

} // namespace

int main()
{
    // Exhaustive over the low-precision P_3109 formats (l = 2..8).
    run_all_l(offset_sequence<2>(std::make_integer_sequence<int, 7>{}));

    // 1000 x 1000 random over the 16-bit formats.
    g_errors += test_ops_random<Templated_Float<halfPrecisionParams>>("binary16");
    g_errors += test_ops_random<Templated_Float<bfloatPrecisionParams>>("bf16");
    g_errors += test_ops_random<Templated_Float<tf32PrecisionParams>>("tf32");

    std::cout << "\n";
    if (g_errors == 0)
    {
        std::cout << "ALL lo_float arithmetic tests PASSED\n";
        return 0;
    }
    std::cout << "lo_float arithmetic tests FAILED (" << g_errors << " errors)\n";
    return 1;
}
