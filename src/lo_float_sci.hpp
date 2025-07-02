#pragma once
//#ifndef LO_FLOAT_ALL_HPP
//#define LO_FLOAT_ALL_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

#include "lo_float.h"                // hypothetical header where your types are declared



namespace lo_float {


//template to get required datatype for exact multiplication based on number of mantissa bits (just pick the type with at least 2n mantissa bits)
template<Float T1, Float T2>
struct exact_mult_type {
    using value = std::conditional< std::max((get_mantissa_bits_v<T1>, get_mantissa_bits_v<T2>)) < 8, float, double>;
}

template<Float T1, Float T2>
struct exact_add_type {
    using value = std::conditional< std::max((get_mantissa_bits_v<T1>, get_mantissa_bits_v<T2>)) < 7, float, double>;
}

template <Float T1, Float T2>
using exact_mult_type_v = typename exact_mult_type<T1, T2>::value;




// 1) Input operator>>
template <FloatingPointParams Fp>
inline std::istream& operator>>(std::istream& is, Templated_Float<Fp>& x)
{
    float f;
    is >> f;
    x = Templated_Float<Fp>(f);
    return is;
}

// 2) ceil
template <FloatingPointParams Fp>
inline Templated_Float<Fp> ceil(Templated_Float<Fp> x) noexcept
{
    // If you prefer a compile-time version for small integer values,
    // you can still call lo_float_internal::ConstexprCeil
    // or just use std::ceil:
    return Templated_Float<Fp>(lo_float_internal::ConstexprCeil(static_cast<double>(x)));
}

// 3) floor
template <FloatingPointParams Fp>
inline Templated_Float<Fp> floor(Templated_Float<Fp> x) noexcept
{
    // same trick: -ceil(-x)
    return Templated_Float<Fp>(
        -lo_float_internal::ConstexprCeil(-static_cast<double>(x))
    );
}

// 4) log2 (though the original uses std::log, which is base-e, not base-2)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> log2(Templated_Float<Fp> x) noexcept
{
    // The original code calls std::log() (natural log). If you actually
    // want base-2, consider std::log2(). We'll replicate the original:
    return Templated_Float<Fp>(std::log(static_cast<double>(x)));
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

// 7) sqrt
template <FloatingPointParams Fp>
inline Templated_Float<Fp> sqrt(Templated_Float<Fp> x) noexcept
{
    return Templated_Float<Fp>(std::sqrt(static_cast<double>(x)));
}

// 8) pow (integer base, float exponent)
template <FloatingPointParams Fp>
inline Templated_Float<Fp> pow(int base, Templated_Float<Fp> expVal)
{
    return Templated_Float<Fp>(
        std::pow(static_cast<double>(base), static_cast<double>(expVal))
    );
}

//9) FMA
template <FloatingPointParams Fp_x, FloatingPointParams Fp_y, FloatingPointParams Fp_Acc, FloatingPointParams Fp_out>
inline Templated_Float<Fp_out> fma(
    Templated_Float<Fp_x> x, Templated_Float<Fp_y> y, Templated_Float<Fp_out> z) noexcept
{
    using accum_type = Templated_Float<Fp_Acc>;
    using result_type = Templated_Float<Fp_out>;   
    using x_type = Templated_Float<Fp_x>;
    using y_type = Templated_Float<Fp_y>;
    using mult_type = exact_mult_type_v<x_type, y_type>;
    return static_cast<result_type>(
        static_cast<accum_type>(static_cast<exact_mult_type>(x) * static_cast<exact_mult_type>(y)) +
        static_cast<accum_type>(z)
    );


} 

//10) FAA
template <FloatingPointParams Fp_A, FloatingPointParams Fp_B, FloatingPointParams Fp_C, FloatingPointParams Fp_Acc>
inline Templated_Float<Fp1> faa(
    Templated_Float<Fp1> x, Templated_Float<Fp2> y, Templated_Float<Fp3> z) noexcept
{
    using accum_type = Templated_Float<Fp_Acc>;
    using result_type = Templated_Float<Fp3>;   
    return static_cast<result_type>(
        static_cast<accum_type>(x) + static_cast<accum_type>(y) + static_cast<accum_type>(z)
    );
} 

} // namespace tlapack
// namespace lo_float






//#endif // LO_FLOAT_ALL_HPP
