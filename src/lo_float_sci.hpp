#pragma once
//#ifndef LO_FLOAT_ALL_HPP
//#define LO_FLOAT_ALL_HPP

#include "platform_macros.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#if !LOFLOAT_NVCC
#include <cfenv>
#endif
#include "lo_float.h"                // hypothetical header where your types are declared



namespace lo_float {


    namespace lo_float_internal {

        template <typename RangeReducer, typename Approx_func>
        class FuncApprox {
            RangeReducer reducer;
            Approx_func approx_func;

        public:
            
            FuncApprox(RangeReducer r, Approx_func f) : reducer(r), approx_func(f) {}

            template <FloatingPointParams Fp>
            Templated_Float<Fp> operator()(Templated_Float<Fp> x) const {
                auto [reduced, ctx] = reducer(x);
                return approx_func(reduced, ctx);
                
            }
        };
    }


//template to get required datatype for exact multiplication based on number of mantissa bits (just pick the type with at least 2n mantissa bits)
template<Float T1, Float T2>
struct exact_mult_type {
    using value = std::conditional< std::max((get_mantissa_bits_v<T1>, get_mantissa_bits_v<T2>)) < 8, float, double>;
};

template<Float T1, Float T2>
struct exact_add_type {
    using value = std::conditional< std::max((get_mantissa_bits_v<T1>, get_mantissa_bits_v<T2>)) < 7, float, double>;
};

template <Float T1, Float T2>
using exact_mult_type_v = typename exact_mult_type<T1, T2>::value;


#if LOFLOAT_ENABLE_EXCEPT
// Gate helper for the math wrappers below: raise exception `f` on the global
// environment, but only for formats that opted into IEEE-754 semantics
// (NA_behavior == _754). Mirrors lo_float.h's call sites.
template <FloatingPointParams Fp>
inline void sci_raise(lo_float_internal::LF_exception_flags f) {
    lo_float_internal::signal_if_754<
        lo_float_internal::get_NaN_Behavior_v<Templated_Float<Fp>>>(f);
}
#endif




// 1) Input operator>>
template <FloatingPointParams Fp>
inline std::istream& operator>>(std::istream& is, Templated_Float<Fp>& x)
{
    float f;
    is >> f;
    x = Templated_Float<Fp>(f);
    return is;
}

template <FloatingPointParams Fp>
inline std::ostream& operator<<(std::ostream& os, const Templated_Float<Fp>& x)
{
    os << static_cast<double>(x);
    return os;
}

// 2) ceil
template <FloatingPointParams Fp>
inline Templated_Float<Fp> ceil(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(ConstexprCeil(static_cast<double>(x)), ps);
}

// 3) floor
template <FloatingPointParams Fp>
inline Templated_Float<Fp> floor(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    // same trick: -ceil(-x)
    return Project<Templated_Float<Fp>>(
        -ConstexprCeil(-static_cast<double>(x)), ps
    );
}

// ============================================================================
//  Math wrappers (Table 9.1).  Uniform pattern: round the input up to double,
//  compute with the matching std:: routine, round the result back into the
//  format.  Overflow/underflow fall out of the round-back conversion; here we
//  add the per-function domain (invalid operation) and pole (divide-by-zero)
//  signals from §9.2, each gated on _754 via sci_raise<Fp>.
// ============================================================================
#if LOFLOAT_ENABLE_EXCEPT
  #define LOF_INVALID(Fp) sci_raise<Fp>(lo_float_internal::LF_exception_flags::InvalidOperation)
  #define LOF_DIVZERO(Fp) sci_raise<Fp>(lo_float_internal::LF_exception_flags::DivisionByZero)
#else
  #define LOF_INVALID(Fp) ((void)0)
  #define LOF_DIVZERO(Fp) ((void)0)
#endif

// π — defined locally because lof_pi is not guaranteed under strict -std=c++20.
inline constexpr double lof_pi = 3.14159265358979323846264338327950288;

// ---- logarithms: x<0 -> invalid, x==0 -> divideByZero --------------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> log(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log(xd), ps);
}

// log2 (base-2)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> log2(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log2(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log10(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log10(xd), ps);
}

// ---- log(1+x) family: x<-1 -> invalid, x==-1 -> divideByZero -------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> logp1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log1p(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log2p1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log2(1.0 + xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log10p1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::log10(1.0 + xd), ps);
}

// 5) max
template <FloatingPointParams Fp>
inline Templated_Float<Fp> max(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    return (x > y) ? x : y;
}

// 6) min
template <FloatingPointParams Fp>
inline Templated_Float<Fp> min(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    return (x > y) ? y : x;
}

// ---- powers / roots ------------------------------------------------------
// 7) sqrt: x<0 -> invalid
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sqrt(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::sqrt(xd), ps);
}

// rSqrt = 1/sqrt(x): x<0 -> invalid, x==±0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> rSqrt(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(1.0 / std::sqrt(xd), ps);
}

// hypot: no domain/pole exceptions (overflow/underflow only)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> hypot(Templated_Float<Fp> x, Templated_Float<Fp> y, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::hypot(static_cast<double>(x), static_cast<double>(y)), ps);
}

// 8) pow (integer base, float exponent) — original convenience overload
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pow(int base, Templated_Float<Fp> expVal, ProjSpec ps = ProjSpec{})
{
    return Project<Templated_Float<Fp>>(
        std::pow(static_cast<double>(base), static_cast<double>(expVal)), ps
    );
}

// pow(x, y): x<0 with non-integer y -> invalid; x==0 with y<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pow(Templated_Float<Fp> x, Templated_Float<Fp> y, ProjSpec ps = ProjSpec{})
{
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);
    if (xd < 0.0 && yd != std::floor(yd)) LOF_INVALID(Fp);
    else if (xd == 0.0 && yd < 0.0)       LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::pow(xd, yd), ps);
}

// powr(x, y) = exp(y*log(x)); domain excludes negative x.
// x<0 -> invalid; (±0,±0)/(+∞,±0)/(+1,±∞) -> invalid; ±0 with y<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> powr(Templated_Float<Fp> x, Templated_Float<Fp> y, ProjSpec ps = ProjSpec{})
{
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);
    if (xd < 0.0 ||
        (xd == 0.0 && yd == 0.0) ||
        (std::isinf(xd) && xd > 0.0 && yd == 0.0) ||
        (xd == 1.0 && std::isinf(yd)))
        LOF_INVALID(Fp);
    else if (xd == 0.0 && yd < 0.0)
        LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::exp(yd * std::log(xd)), ps);
}

// compound(x, n) = (1+x)^n: x<-1 -> invalid; x==-1 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> compound(Templated_Float<Fp> x, int n, ProjSpec ps = ProjSpec{})
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)                 LOF_INVALID(Fp);
    else if (xd == -1.0 && n < 0)  LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::pow(1.0 + xd, static_cast<double>(n)), ps);
}

// rootn(x, n) = x^(1/n): n==0 -> invalid; x<0 with even n -> invalid;
//                        x==0 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> rootn(Templated_Float<Fp> x, int n, ProjSpec ps = ProjSpec{})
{
    const double xd = static_cast<double>(x);
    if (n == 0 || (xd < 0.0 && (n % 2 == 0))) LOF_INVALID(Fp);
    else if (xd == 0.0 && n < 0)              LOF_DIVZERO(Fp);
    // sign-preserving root for odd n and negative x.
    const double mag = std::pow(std::fabs(xd), 1.0 / static_cast<double>(n));
    return Project<Templated_Float<Fp>>((xd < 0.0 && (n % 2 != 0)) ? -mag : mag, ps);
}

// pown(x, n) = x^n (integer n): x==±0 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pown(Templated_Float<Fp> x, int n, ProjSpec ps = ProjSpec{})
{
    const double xd = static_cast<double>(x);
    if (xd == 0.0 && n < 0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::pow(xd, static_cast<double>(n)), ps);
}

// ---- exponentials: no domain/pole exceptions (overflow/underflow only) ---
template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::exp(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> expm1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::expm1(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp2(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::exp2(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp2m1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::exp2(static_cast<double>(x)) - 1.0, ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp10(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::pow(10.0, static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp10m1(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::pow(10.0, static_cast<double>(x)) - 1.0, ps);
}

// ---- trigonometric: |x|==∞ -> invalid ------------------------------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sin(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::sin(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cos(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::cos(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> tan(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::tan(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> sinPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::sin(lof_pi * xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cosPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::cos(lof_pi * xd), ps);
}

// tanPi: |x|==∞ -> invalid; x a half-integer (2x is an odd integer) -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> tanPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) {
        LOF_INVALID(Fp);
    } else {
        const double two_x = 2.0 * xd;
        if (two_x == std::trunc(two_x) && std::fmod(two_x, 2.0) != 0.0)
            LOF_DIVZERO(Fp);
    }
    return Project<Templated_Float<Fp>>(std::tan(lof_pi * xd), ps);
}

// ---- inverse trigonometric: |x|>1 -> invalid for asin/acos ---------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> asin(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::asin(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> acos(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::acos(xd), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::atan(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan2(Templated_Float<Fp> y, Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::atan2(static_cast<double>(y), static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> asinPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::asin(xd) / lof_pi, ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> acosPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::acos(xd) / lof_pi, ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atanPi(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::atan(static_cast<double>(x)) / lof_pi, ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan2Pi(Templated_Float<Fp> y, Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::atan2(static_cast<double>(y), static_cast<double>(x)) / lof_pi, ps);
}

// ---- hyperbolic ----------------------------------------------------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sinh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::sinh(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cosh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::cosh(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> tanh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::tanh(static_cast<double>(x)), ps);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> asinh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    return Project<Templated_Float<Fp>>(std::asinh(static_cast<double>(x)), ps);
}

// acosh: x<1 -> invalid
template <FloatingPointParams Fp>
inline Templated_Float<Fp> acosh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 1.0) LOF_INVALID(Fp);
    return Project<Templated_Float<Fp>>(std::acosh(xd), ps);
}

// atanh: |x|==1 -> divideByZero; |x|>1 -> invalid
template <FloatingPointParams Fp>
inline Templated_Float<Fp> atanh(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    const double a  = std::fabs(xd);
    if (a > 1.0)       LOF_INVALID(Fp);
    else if (a == 1.0) LOF_DIVZERO(Fp);
    return Project<Templated_Float<Fp>>(std::atanh(xd), ps);
}

// Fast2Sum (Dekker): returns {c, e} such that c = fl(a + b) and a + b = c + e
// exactly, where e captures the rounding error of the floating-point sum.
// Requires a >= b (more precisely exp(a) >= exp(b)), which holds here since the
// operands are sorted before being passed in. Costs only 3 floating-point ops.
template <Float T>
inline std::pair<T, T> two_sum(T a, T b) noexcept
{
    T c = a + b;        // c = fl(a + b)
    T z = c - a;        // recovered value of b in the sum
    T e = b - z;        // exact rounding error
    return {c, e};
}

// ExactAdd (Knuth's 2Sum): same postcondition as two_sum but with no ordering
// requirement on the operands, at the cost of 6 flops instead of 3.
template <Float T>
inline std::pair<T, T> two_sum_full(T a, T b) noexcept
{
    T c  = a + b;
    T bv = c - a;               // b as it was seen by the sum
    T av = c - bv;              // a as it was seen by the sum
    T e  = (a - av) + (b - bv); // exact rounding error
    return {c, e};
}

// □_odd for the working precision: the correctly-rounded-to-odd sum a + b.
// Round to nearest first, then, whenever the sum was inexact, force the last
// significand bit to 1 by nudging one ulp in the direction of the discarded
// part.  That single sticky bit is what makes a later rounding into the narrow
// output format land on the same value as one rounding of the exact sum
// (Boldo & Melquiond, §3) -- i.e. it defeats double rounding.
template <Float T>
inline T add_odd(T a, T b) noexcept
{
    auto [s, e] = two_sum_full(a, b);
    if (e == T(0)) return s;   // sum was exact: nothing was discarded
    using bits_t = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    if (lo_float_internal::bit_cast<bits_t>(s) & bits_t(1))
        return s;              // significand already odd
    return std::nextafter(s, e > T(0) ? std::numeric_limits<T>::infinity()
                                      : -std::numeric_limits<T>::infinity());
}

//9) FMA — computes (x*y)+z with a single rounding into the output format.
// §7.2(b,c,d): 0×∞ or ∞−∞ have no usefully definable result -> invalid operation.
// Uses the RTO trick from https://ens-lyon.hal.science/inria-00080427v2/document
// (Boldo & Melquiond).  With a, b, c the operands widened to the working type:
//
//     (u_h, u_l) = ExactMult(a, b)
//     (t_h, t_l) = ExactAdd(c, u_h)
//     v          = □_odd(t_l + u_l)      // round to odd
//     z          = ∘(t_h + v)            // the one user-visible rounding
//
// u_l and t_l are the exact residuals of the two operations, so t_h + (t_l + u_l)
// is the exact value of x*y + z.  Splitting it as t_h + □_odd(tail) lets the final
// ∘ round once into the output format even when the exact value needs more than
// the working type's precision: the odd-rounded tail carries the sticky bit that a
// plain round-to-nearest tail would have thrown away.
template <FloatingPointParams Fp1, FloatingPointParams Fp2, FloatingPointParams Fp3, FloatingPointParams Fp_out>
LOFLOAT_HOST LOFLOAT_INLINE Templated_Float<Fp_out> fma(
    Templated_Float<Fp1> x, Templated_Float<Fp2> y, Templated_Float<Fp3> z, ProjSpec ps = ProjSpec{}) noexcept
{
    using out_t = Templated_Float<Fp_out>;
    using acc   = double;   // working type: wide enough for an exact ExactMult

    // ExactMult is only exact if the full product of the two significands fits.
    static_assert((Fp1.mantissa_bits + 1) + (Fp2.mantissa_bits + 1)
                      <= std::numeric_limits<acc>::digits,
                  "fma: operand formats too wide for an exact ExactMult in double");

    const acc a = static_cast<acc>(x);
    const acc b = static_cast<acc>(y);
    const acc c = static_cast<acc>(z);

    // (u_h, u_l) = ExactMult(a, b):  u_h = fl(a·b), u_l = a·b − u_h, exactly
    // (a·b is exact in `acc` by the static_assert, so u_h is it and u_l is 0;
    //  the fma keeps the split correct for wider working types as well).
    const acc u_h = a * b;
    const acc u_l = std::fma(a, b, -u_h);
    // §7.2(b,c,d): 0×∞ and ∞−∞ are invalid; a NaN operand merely propagates.
    // Non-finite values also have no meaningful error terms, so they skip the
    // algorithm and go straight through the output conversion.
    if (!std::isfinite(u_h) || !std::isfinite(c)) {
        if ((std::isnan(u_h) && !std::isnan(a) && !std::isnan(b)) ||
            (std::isinf(u_h) && std::isinf(c) && (std::signbit(u_h) != std::signbit(c))))
            LOF_INVALID(Fp_out);
        return Project<out_t>(u_h + c, ps);
    }
    // (t_h, t_l) = ExactAdd(c, u_h)
    const auto [t_h, t_l] = two_sum_full<acc>(c, u_h);

    // v = □_odd(t_l + u_l)
    const acc v = add_odd<acc>(t_l, u_l);

    // z = ∘(t_h + v).  v == 0 means t_h is already the exact result; returning it
    // untouched also keeps the sign of a zero result (t_h + (+0) would flip −0).
    return Project<out_t>(v == acc(0) ? t_h : add_odd<acc>(t_h, v), ps);
}

#undef LOF_INVALID
#undef LOF_DIVZERO


//10) FAA - uses the algorithm from https://hal.science/hal-04575249/document
// NOTE (loft loop): faa() is unfinished WIP — its body references an `accum_type`
// that is no longer declared (the template's accumulator param was renamed), so it
// does not compile. Commented out for now per the author's instruction so the rest
// of the build is unblocked; finish/restore separately.
// template <FloatingPointParams Fp1, FloatingPointParams Fp2, FloatingPointParams Fp3, FloatingPointParams Fp_Out>
// inline Templated_Float<Fp3> faa(
//     Templated_Float<Fp1> x, Templated_Float<Fp2> y, Templated_Float<Fp3> z) noexcept
// {
//     using result_type = Templated_Float<Fp_Out>;
//     auto x1 = max(x, max(y, z));
//     auto x2 = x1 == x ? max(y, z) : max(x, z);
//     auto x3 = min(x, min(y, z));
//
//     // c = fl(x1 + x2), e = error term so that x1 + x2 = c + e exactly
//     auto [x_h,x_l] = two_sum<accum_type>(static_cast<accum_type>(x1),
//                                       static_cast<accum_type>(x2));
//     auto [s_h, s_l] = two_sum<accum_type>(static_cast<accum_type>(x_h),static_cast<accum_type>(x3));
//     auto [v_h, v_l] = two_sum<accum_type>(static_cast<accum_type>(x_l),static_cast<accum_type>(s_l));
//
//     return static_cast<result_type>(
//         static_cast<accum_type>(x) + static_cast<accum_type>(y) + static_cast<accum_type>(z)
//     );
// }



// ============================================================================
//  IEEE P3109 operations.  Scalar ops only; block operations (§5) and FAA
//  (§4.10.7) are intentionally not implemented here.  Exception flags are
//  ignored.  The P3109 number model has a single NaN and no negative zero.
// ============================================================================

// ---- §4.13 classification predicates + §4.13.1 Class ----

// §4.13 IsZero: NaN/±inf are not zero; otherwise the value equals zero.
template <FloatingPointParams Fp>
inline bool IsZero(Templated_Float<Fp> x) noexcept {
    if (isnan(x) || isinf(x)) return false;
    return static_cast<double>(x) == 0.0;
}

// §4.13 IsOne: NaN/±inf are not one; otherwise the value equals one.
template <FloatingPointParams Fp>
inline bool IsOne(Templated_Float<Fp> x) noexcept {
    if (isnan(x) || isinf(x)) return false;
    return static_cast<double>(x) == 1.0;
}

// §4.13 IsNaN.
template <FloatingPointParams Fp>
inline bool IsNaN(Templated_Float<Fp> x) noexcept {
    return isnan(x);
}

// §4.13 IsInfinite.
template <FloatingPointParams Fp>
inline bool IsInfinite(Templated_Float<Fp> x) noexcept {
    return isinf(x);
}

// §4.13 IsFinite: neither infinite nor NaN.
template <FloatingPointParams Fp>
inline bool IsFinite(Templated_Float<Fp> x) noexcept {
    return !IsInfinite<Fp>(x) && !IsNaN<Fp>(x);
}

// §4.13 IsSignMinus: NaN and +inf false; -inf true; no negative zero, so zero is false.
template <FloatingPointParams Fp>
inline bool IsSignMinus(Templated_Float<Fp> x) noexcept {
    if (isnan(x)) return false;
    return x < Templated_Float<Fp>{};   // only -inf and negatives are < 0
}

// §4.13 IsNormal: finite nonzero with magnitude at least the smallest normal.
template <FloatingPointParams Fp>
inline bool IsNormal(Templated_Float<Fp> x) noexcept {
    if (IsZero<Fp>(x) || IsInfinite<Fp>(x) || IsNaN<Fp>(x)) return false;
    return abs(x) >= std::numeric_limits<Templated_Float<Fp>>::min();
}

// §4.13 IsSubnormal: finite nonzero with magnitude below the smallest normal.
template <FloatingPointParams Fp>
inline bool IsSubnormal(Templated_Float<Fp> x) noexcept {
    if (IsZero<Fp>(x) || IsInfinite<Fp>(x) || IsNaN<Fp>(x)) return false;
    return !IsNormal<Fp>(x);
}

// §4.13.1 the eight disjoint classes.
enum Class_Enum : uint8_t {
    ClsNaN,
    ClsNegativeInfinity,
    ClsNegativeNormal,
    ClsNegativeSubnormal,
    ClsZero,
    ClsPositiveSubnormal,
    ClsPositiveNormal,
    ClsPositiveInfinity
};

// §4.13.1 Class: map x to exactly one of the eight disjoint classes.
template <FloatingPointParams Fp>
inline Class_Enum Class(Templated_Float<Fp> x) noexcept {
    if (IsNaN<Fp>(x)) return ClsNaN;
    if (IsInfinite<Fp>(x)) return IsSignMinus<Fp>(x) ? ClsNegativeInfinity : ClsPositiveInfinity;
    if (IsZero<Fp>(x)) return ClsZero;
    if (IsSignMinus<Fp>(x))
        return IsSubnormal<Fp>(x) ? ClsNegativeSubnormal : ClsNegativeNormal;
    return IsSubnormal<Fp>(x) ? ClsPositiveSubnormal : ClsPositiveNormal;
}

// ---- §4.11 extrema family ----

// §4.11 Minimum — NaN-propagating; tie returns second operand y.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> Minimum(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x) || isnan(y)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    return (x < y) ? x : y;   // tie (x==y) -> y
}

// §4.11 Maximum — NaN-propagating; tie returns first operand x.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> Maximum(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x) || isnan(y)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    return (x < y) ? y : x;   // tie (x==y) -> x
}

// §4.11 MinimumNumber — a lone NaN is ignored; (NaN,NaN)->NaN.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinimumNumber(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x)) return isnan(y) ? std::numeric_limits<Templated_Float<Fp>>::quiet_NaN() : y;
    if (isnan(y)) return x;
    return Minimum<Fp>(x, y);
}

// §4.11 MaximumNumber — a lone NaN is ignored; (NaN,NaN)->NaN.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaximumNumber(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x)) return isnan(y) ? std::numeric_limits<Templated_Float<Fp>>::quiet_NaN() : y;
    if (isnan(y)) return x;
    return Maximum<Fp>(x, y);
}

// §4.11 MinimumMagnitude — NaN-propagating; compare by |x|, ties broken by signed value.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinimumMagnitude(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x) || isnan(y)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    const Templated_Float<Fp> ax = abs(x), ay = abs(y);
    if (ax < ay) return x;
    if (ay < ax) return y;
    return (x < y) ? x : y;   // magnitude tie -> lesser value
}

// §4.11 MaximumMagnitude — NaN-propagating; compare by |x|, ties broken by signed value.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaximumMagnitude(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x) || isnan(y)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    const Templated_Float<Fp> ax = abs(x), ay = abs(y);
    if (ax > ay) return x;
    if (ay > ax) return y;
    return (x < y) ? y : x;   // magnitude tie -> greater value
}

// §4.11 MinimumMagnitudeNumber — a lone NaN is ignored; (NaN,NaN)->NaN.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinimumMagnitudeNumber(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x)) return isnan(y) ? std::numeric_limits<Templated_Float<Fp>>::quiet_NaN() : y;
    if (isnan(y)) return x;
    return MinimumMagnitude<Fp>(x, y);
}

// §4.11 MaximumMagnitudeNumber — a lone NaN is ignored; (NaN,NaN)->NaN.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaximumMagnitudeNumber(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x)) return isnan(y) ? std::numeric_limits<Templated_Float<Fp>>::quiet_NaN() : y;
    if (isnan(y)) return x;
    return MaximumMagnitude<Fp>(x, y);
}

// §4.11.3 MinimumFinite — NaN-ignoring, identical to MinimumNumber (±inf passed through).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinimumFinite(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    return MinimumNumber<Fp>(x, y);
}

// §4.11.3 MaximumFinite — NaN-ignoring, identical to MaximumNumber (±inf passed through).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaximumFinite(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    return MaximumNumber<Fp>(x, y);
}

// §4.11.4 Clamp — constrain x to [lo,hi]; any-NaN or inverted bounds (lo>hi) -> NaN.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> Clamp(Templated_Float<Fp> x, Templated_Float<Fp> lo, Templated_Float<Fp> hi) noexcept
{
    if (isnan(x) || isnan(lo) || isnan(hi)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    if (lo > hi) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    return x < lo ? lo : (x > hi ? hi : x);
}

// ---- §4.10 misc scalar arithmetic ----

// §4.10.8 Recip = 1/x: NaN->NaN; 0->NaN (P3109 divide-by-zero); ±inf->0; else 1/x.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> Recip(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    if (isnan(x))  return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    if (xd == 0.0) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    return Project<Templated_Float<Fp>>(1.0 / xd, ps);   // 1.0/±inf == 0.0 naturally
}

// §4.10.2 CopySign — magnitude of x with sign of y; NaN in either operand -> NaN; y==0 => positive.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> CopySign(Templated_Float<Fp> x, Templated_Float<Fp> y, ProjSpec ps = ProjSpec{}) noexcept
{
    if (isnan(x) || isnan(y)) return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    const double mag = static_cast<double>(abs(x));   // abs(±inf) = +inf
    const bool   neg = static_cast<double>(y) < 0.0;  // y==0 -> positive
    return Project<Templated_Float<Fp>>(neg ? -mag : mag, ps);
}

// §4.10.13 Softplus = log(1+e^x): NaN->NaN; +inf->+inf; -inf->0 (stable form handles the limits).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> Softplus(Templated_Float<Fp> x, ProjSpec ps = ProjSpec{}) noexcept
{
    const double xd = static_cast<double>(x);
    const double r  = xd > 0.0 ? xd + std::log1p(std::exp(-xd)) : std::log1p(std::exp(xd));
    return Project<Templated_Float<Fp>>(r, ps);
}

// ---- §4.12 ordering ----

// §4.12.1 TotalOrder — single NaN sorts below everything; otherwise x <= y.
template <FloatingPointParams Fp>
inline bool TotalOrder(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    if (isnan(x)) return true;
    if (isnan(y)) return false;
    return x <= y;
}

// ---- §4.16 next-value operations ----

// SmallestNegative = -MinPositive = value -denorm_min (signed formats only).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> SmallestNegative() noexcept
{
    using L = std::numeric_limits<Templated_Float<Fp>>;
    using R = decltype(L::denorm_min().rep());
    return Templated_Float<Fp>::FromRep(L::denorm_min().rep() | static_cast<R>(R{1} << (Fp.bitwidth - 1)));
}

// §4.16 NextGreaterThan — least value comparing greater than x, or NaN if none.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> NextGreaterThan(Templated_Float<Fp> x) noexcept
{
    const bool signed_fmt = (Fp.is_signed == Signedness::Signed);
    const bool extended   = (Fp.OV_behavior != Inf_Behaviors::Saturating);
    const bool neg        = (static_cast<double>(x) < 0.0);   // false for +0/+inf; true for negatives/-inf
    using L = std::numeric_limits<Templated_Float<Fp>>;
    auto r  = x.rep();
    using R = decltype(r);

    if (isnan(x)) return L::quiet_NaN();                                    // NaN
    if (extended && isinf(x) && !neg) return L::quiet_NaN();                // +inf -> none
    if (x == L::max()) return extended ? L::infinity() : L::quiet_NaN();    // MaxFinite -> +inf / NaN
    if (signed_fmt && extended && isinf(x)) return L::lowest();             // -inf -> MinFinite
    if (signed_fmt) {                                                       // SmallestNegative -> 0
        if (x == SmallestNegative<Fp>()) return Templated_Float<Fp>::FromRep(static_cast<R>(0));
    }
    return Templated_Float<Fp>::FromRep(static_cast<R>((signed_fmt && neg) ? (r - 1) : (r + 1)));
}

// §4.16 NextLessThan — greatest value comparing less than x, or NaN if none.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> NextLessThan(Templated_Float<Fp> x) noexcept
{
    const bool signed_fmt = (Fp.is_signed == Signedness::Signed);
    const bool extended   = (Fp.OV_behavior != Inf_Behaviors::Saturating);
    const bool neg        = (static_cast<double>(x) < 0.0);   // false for +0/+inf; true for negatives/-inf
    using L = std::numeric_limits<Templated_Float<Fp>>;
    auto r  = x.rep();
    using R = decltype(r);

    if (isnan(x)) return L::quiet_NaN();                                        // NaN
    if (signed_fmt && extended && isinf(x) && neg) return L::quiet_NaN();       // -inf -> none
    if (extended && isinf(x) && !neg) return L::max();                          // +inf -> MaxFinite
    if (x == L::lowest())                                                       // MinFinite (or unsigned 0)
        return (signed_fmt && extended) ? (-L::infinity()) : L::quiet_NaN();    //   signed Extended -> -inf else NaN
    if (signed_fmt && x == Templated_Float<Fp>::FromRep(static_cast<R>(0)))     // 0 -> SmallestNegative
        return SmallestNegative<Fp>();
    return Templated_Float<Fp>::FromRep(static_cast<R>((signed_fmt && neg) ? (r + 1) : (r - 1)));
}

// ---- §4.14 format-level queries (callable as NameOf(x) or NameOf<Fp>()) ----

// §4.14 storage width in bits (K).
template <FloatingPointParams Fp>
inline int BitwidthOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.bitwidth;
}

// §4.14 precision P = trailing significand bits + 1.
template <FloatingPointParams Fp>
inline int PrecisionOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.mantissa_bits + 1;
}

// §4.14 trailing significand field width (P-1).
template <FloatingPointParams Fp>
inline int TrailingSignificandBitwidthOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.mantissa_bits;
}

// §4.14 exponent field width = K - (P-1) - signbit.
template <FloatingPointParams Fp>
inline int ExponentBitwidthOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.bitwidth - Fp.mantissa_bits - (Fp.is_signed == Signedness::Signed ? 1 : 0);
}

// §4.14 exponent bias.
template <FloatingPointParams Fp>
inline int ExponentBiasOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.bias;
}

// §4.14 signedness of the format.
template <FloatingPointParams Fp>
inline Signedness SignednessOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.is_signed;
}

// §4.14 domain: Inf_Behaviors::Extended = ±inf in the datum set; Saturating = Finite domain.
template <FloatingPointParams Fp>
inline Inf_Behaviors DomainOf(Templated_Float<Fp> = {}) noexcept {
    return Fp.OV_behavior;
}

// §4.14 largest finite value.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaxFiniteOf(Templated_Float<Fp> = {}) noexcept {
    return std::numeric_limits<Templated_Float<Fp>>::max();
}

// §4.14 most-negative finite value (0 for unsigned formats).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinFiniteOf(Templated_Float<Fp> = {}) noexcept {
    return std::numeric_limits<Templated_Float<Fp>>::lowest();
}

// §4.14 smallest positive value (smallest subnormal).
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinPositiveOf(Templated_Float<Fp> = {}) noexcept {
    return std::numeric_limits<Templated_Float<Fp>>::denorm_min();
}

// §4.14 smallest positive normal value.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MinNormalOf(Templated_Float<Fp> = {}) noexcept {
    return std::numeric_limits<Templated_Float<Fp>>::min();
}

// §4.14 largest subnormal (exp field 0, all mantissa bits set), or NaN if no subnormals.
template <FloatingPointParams Fp>
inline Templated_Float<Fp> MaxSubnormalOf(Templated_Float<Fp> = {}) noexcept {
    if constexpr (Fp.mantissa_bits == 0) {
        return std::numeric_limits<Templated_Float<Fp>>::quiet_NaN();
    } else {
        using RepT = decltype(std::numeric_limits<Templated_Float<Fp>>::denorm_min().rep());
        return Templated_Float<Fp>::FromRep(static_cast<RepT>((RepT{1} << Fp.mantissa_bits) - RepT{1}));
    }
}


    template<FloatingPointParams Fp>
    inline constexpr auto func_get_mantissa_bits(Templated_Float<Fp>& x) {
        return x.rep() & ((1 << get_mantissa_bits_v<Templated_Float<Fp>>) - 1);
    }

    template<FloatingPointParams Fp>
    inline constexpr auto func_get_exponent_bits(Templated_Float<Fp>& x) {
        return abs(x).rep() >> get_mantissa_bits_v<Templated_Float<Fp>>;
    }

    template<FloatingPointParams Fp>
    inline constexpr bool func_get_sign_bit(Templated_Float<Fp>& x) {
        return x < Templated_Float<Fp>(0.0f);
    }




} // namespace tlapack
// namespace lo_float






//#endif // LO_FLOAT_ALL_HPP
