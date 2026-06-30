#pragma once
//#ifndef LO_FLOAT_ALL_HPP
//#define LO_FLOAT_ALL_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#ifndef USE_CUDA
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


#ifdef ENABLE_EXCEPT
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
inline Templated_Float<Fp> ceil(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(ConstexprCeil(static_cast<double>(x)));
}

// 3) floor
template <FloatingPointParams Fp>
inline Templated_Float<Fp> floor(Templated_Float<Fp> x) noexcept
{
    // same trick: -ceil(-x)
    return Templated_Float<Fp>(
        -ConstexprCeil(-static_cast<double>(x))
    );
}

// ============================================================================
//  Math wrappers (Table 9.1).  Uniform pattern: round the input up to double,
//  compute with the matching std:: routine, round the result back into the
//  format.  Overflow/underflow fall out of the round-back conversion; here we
//  add the per-function domain (invalid operation) and pole (divide-by-zero)
//  signals from §9.2, each gated on _754 via sci_raise<Fp>.
// ============================================================================
#ifdef ENABLE_EXCEPT
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
inline Templated_Float<Fp> log(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log(xd));
}

// log2 (base-2)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> log2(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log2(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log10(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log10(xd));
}

// ---- log(1+x) family: x<-1 -> invalid, x==-1 -> divideByZero -------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> logp1(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log1p(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log2p1(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log2(1.0 + xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> log10p1(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)       LOF_INVALID(Fp);
    else if (xd == -1.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::log10(1.0 + xd));
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
inline Templated_Float<Fp> sqrt(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::sqrt(xd));
}

// rSqrt = 1/sqrt(x): x<0 -> invalid, x==±0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> rSqrt(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 0.0)       LOF_INVALID(Fp);
    else if (xd == 0.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(1.0 / std::sqrt(xd));
}

// hypot: no domain/pole exceptions (overflow/underflow only)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> hypot(Templated_Float<Fp> x, Templated_Float<Fp> y) noexcept
{
    return Templated_Float<Fp>(std::hypot(static_cast<double>(x), static_cast<double>(y)));
}

// 8) pow (integer base, float exponent) — original convenience overload
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pow(int base, Templated_Float<Fp> expVal)
{
    return Templated_Float<Fp>(
        std::pow(static_cast<double>(base), static_cast<double>(expVal))
    );
}

// pow(x, y): x<0 with non-integer y -> invalid; x==0 with y<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pow(Templated_Float<Fp> x, Templated_Float<Fp> y)
{
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);
    if (xd < 0.0 && yd != std::floor(yd)) LOF_INVALID(Fp);
    else if (xd == 0.0 && yd < 0.0)       LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::pow(xd, yd));
}

// powr(x, y) = exp(y*log(x)); domain excludes negative x.
// x<0 -> invalid; (±0,±0)/(+∞,±0)/(+1,±∞) -> invalid; ±0 with y<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> powr(Templated_Float<Fp> x, Templated_Float<Fp> y)
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
    return Templated_Float<Fp>(std::exp(yd * std::log(xd)));
}

// compound(x, n) = (1+x)^n: x<-1 -> invalid; x==-1 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> compound(Templated_Float<Fp> x, int n)
{
    const double xd = static_cast<double>(x);
    if (xd < -1.0)                 LOF_INVALID(Fp);
    else if (xd == -1.0 && n < 0)  LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::pow(1.0 + xd, static_cast<double>(n)));
}

// rootn(x, n) = x^(1/n): n==0 -> invalid; x<0 with even n -> invalid;
//                        x==0 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> rootn(Templated_Float<Fp> x, int n)
{
    const double xd = static_cast<double>(x);
    if (n == 0 || (xd < 0.0 && (n % 2 == 0))) LOF_INVALID(Fp);
    else if (xd == 0.0 && n < 0)              LOF_DIVZERO(Fp);
    // sign-preserving root for odd n and negative x.
    const double mag = std::pow(std::fabs(xd), 1.0 / static_cast<double>(n));
    return Templated_Float<Fp>((xd < 0.0 && (n % 2 != 0)) ? -mag : mag);
}

// pown(x, n) = x^n (integer n): x==±0 with n<0 -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pown(Templated_Float<Fp> x, int n)
{
    const double xd = static_cast<double>(x);
    if (xd == 0.0 && n < 0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::pow(xd, static_cast<double>(n)));
}

// ---- exponentials: no domain/pole exceptions (overflow/underflow only) ---
template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::exp(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> expm1(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::expm1(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp2(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::exp2(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp2m1(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::exp2(static_cast<double>(x)) - 1.0);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp10(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::pow(10.0, static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> exp10m1(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::pow(10.0, static_cast<double>(x)) - 1.0);
}

// ---- trigonometric: |x|==∞ -> invalid ------------------------------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sin(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::sin(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cos(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::cos(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> tan(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::tan(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> sinPi(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::sin(lof_pi * xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cosPi(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::cos(lof_pi * xd));
}

// tanPi: |x|==∞ -> invalid; x a half-integer (2x is an odd integer) -> divideByZero
template <FloatingPointParams Fp>
inline Templated_Float<Fp> tanPi(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::isinf(xd)) {
        LOF_INVALID(Fp);
    } else {
        const double two_x = 2.0 * xd;
        if (two_x == std::trunc(two_x) && std::fmod(two_x, 2.0) != 0.0)
            LOF_DIVZERO(Fp);
    }
    return Templated_Float<Fp>(std::tan(lof_pi * xd));
}

// ---- inverse trigonometric: |x|>1 -> invalid for asin/acos ---------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> asin(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::asin(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> acos(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::acos(xd));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::atan(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan2(Templated_Float<Fp> y, Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> asinPi(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::asin(xd) / lof_pi);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> acosPi(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (std::fabs(xd) > 1.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::acos(xd) / lof_pi);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atanPi(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::atan(static_cast<double>(x)) / lof_pi);
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> atan2Pi(Templated_Float<Fp> y, Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::atan2(static_cast<double>(y), static_cast<double>(x)) / lof_pi);
}

// ---- hyperbolic ----------------------------------------------------------
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sinh(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::sinh(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> cosh(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::cosh(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> tanh(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::tanh(static_cast<double>(x)));
}

template <FloatingPointParams Fp>
inline Templated_Float<Fp> asinh(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::asinh(static_cast<double>(x)));
}

// acosh: x<1 -> invalid
template <FloatingPointParams Fp>
inline Templated_Float<Fp> acosh(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    if (xd < 1.0) LOF_INVALID(Fp);
    return Templated_Float<Fp>(std::acosh(xd));
}

// atanh: |x|==1 -> divideByZero; |x|>1 -> invalid
template <FloatingPointParams Fp>
inline Templated_Float<Fp> atanh(Templated_Float<Fp> x) noexcept
{
    const double xd = static_cast<double>(x);
    const double a  = std::fabs(xd);
    if (a > 1.0)       LOF_INVALID(Fp);
    else if (a == 1.0) LOF_DIVZERO(Fp);
    return Templated_Float<Fp>(std::atanh(xd));
}

#undef LOF_INVALID
#undef LOF_DIVZERO



//9) FMA — computes (x*y)+z by rounding the operands up to double, evaluating
// std::fma (one rounding), and rounding the result back into the output format.
// §7.2(b,c,d): 0×∞ or ∞−∞ have no usefully definable result -> invalid operation.
template <FloatingPointParams Fp1, FloatingPointParams Fp2, FloatingPointParams Fp3, FloatingPointParams Fp_out>
inline Templated_Float<Fp_out> fma(
    Templated_Float<Fp1> x, Templated_Float<Fp2> y, Templated_Float<Fp3> z) noexcept
{
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);
    const double zd = static_cast<double>(z);
    const double res = std::fma(xd, yd, zd);
#ifdef ENABLE_EXCEPT
    // A NaN result from non-NaN operands means an invalid (0×∞ or ∞−∞) arose.
    if (std::isnan(res) && !std::isnan(xd) && !std::isnan(yd) && !std::isnan(zd))
        sci_raise<Fp_out>(lo_float_internal::LF_exception_flags::InvalidOperation);
#endif
    return Templated_Float<Fp_out>(res);
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

//11) FMA into fixed point


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
