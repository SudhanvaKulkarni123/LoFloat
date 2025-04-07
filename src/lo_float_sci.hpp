#pragma once
//#ifndef LO_FLOAT_ALL_HPP
//#define LO_FLOAT_ALL_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

#include "lo_float.h"                // hypothetical header where your types are declared


using namespace lo_float;

namespace tlapack {


//
// 1) 6-bit floats
//

//---------------------------------
// float6_e3m2
//---------------------------------

inline std::istream& operator>>(std::istream& is, float6_e3m2& x)
{
    float f;
    is >> f;
    x = float6_e3m2(f);  
    return is;
}

inline float6_e3m2 ceil(float6_e3m2 x) noexcept
{
    return float6_e3m2(lo_float_internal::ConstexprCeil(double(x)));
}
inline float6_e3m2 floor(float6_e3m2 x) noexcept
{
    return float6_e3m2(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float6_e3m2 log2(float6_e3m2 x) noexcept
{
    return float6_e3m2(std::log(double(x)));  // or std::log2 if desired
}
inline float6_e3m2 max(float6_e3m2 x, float6_e3m2 y) noexcept
{
    return (x > y) ? x : y;
}
inline float6_e3m2 min(float6_e3m2 x, float6_e3m2 y) noexcept
{
    return (x > y) ? y : x;
}
inline float6_e3m2 sqrt(float6_e3m2 x) noexcept
{
    return float6_e3m2(std::sqrt(double(x)));
}
inline float6_e3m2 pow(int base, float6_e3m2 expVal)
{
    return float6_e3m2(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float6_e2m3
//---------------------------------

inline std::istream& operator>>(std::istream& is, float6_e2m3& x)
{
    float f;
    is >> f;
    x = float6_e2m3(f);
    return is;
}
inline float6_e2m3 ceil(float6_e2m3 x) noexcept
{
    return float6_e2m3(lo_float_internal::ConstexprCeil(double(x)));
}
inline float6_e2m3 floor(float6_e2m3 x) noexcept
{
    return float6_e2m3(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float6_e2m3 log2(float6_e2m3 x) noexcept
{
    return float6_e2m3(std::log(double(x)));
}
inline float6_e2m3 max(float6_e2m3 x, float6_e2m3 y) noexcept
{
    return (x > y) ? x : y;
}
inline float6_e2m3 min(float6_e2m3 x, float6_e2m3 y) noexcept
{
    return (x > y) ? y : x;
}
inline float6_e2m3 sqrt(float6_e2m3 x) noexcept
{
    return float6_e2m3(std::sqrt(double(x)));
}
inline float6_e2m3 pow(int base, float6_e2m3 expVal)
{
    return float6_e2m3(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float6_p<p>
//---------------------------------

template<int p>
inline std::istream& operator>>(std::istream& is, float6_p<p>& x)
{
    float f;
    is >> f;
    x = float6_p<p>(f);  // constructor float6_p<p>(float)
    return is;
}
template<int p>
inline float6_p<p> ceil(float6_p<p> x) noexcept
{
    return float6_p<p>(lo_float_internal::ConstexprCeil(double(x)));
}
template<int p>
inline float6_p<p> floor(float6_p<p> x) noexcept
{
    return float6_p<p>(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
template<int p>
inline float6_p<p> log2(float6_p<p> x) noexcept
{
    return float6_p<p>(std::log(double(x)));
}
template<int p>
inline float6_p<p> max(float6_p<p> x, float6_p<p> y) noexcept
{
    return (x > y) ? x : y;
}
template<int p>
inline float6_p<p> min(float6_p<p> x, float6_p<p> y) noexcept
{
    return (x > y) ? y : x;
}
template<int p>
inline float6_p<p> sqrt(float6_p<p> x) noexcept
{
    return float6_p<p>(std::sqrt(double(x)));
}
template<int p>
inline float6_p<p> pow(int base, float6_p<p> expVal)
{
    return float6_p<p>(std::pow(double(base), double(expVal)));
}

//
// 2) 8-bit floats 
//

//---------------------------------
// float8_e4m3fn
//---------------------------------

inline std::istream& operator>>(std::istream& is, float8_e4m3fn& x)
{
    float f;
    is >> f;
    x = float8_e4m3fn(f);
    return is;
}
inline float8_e4m3fn ceil(float8_e4m3fn x) noexcept
{
    return float8_e4m3fn(lo_float_internal::ConstexprCeil(double(x)));
}
inline float8_e4m3fn floor(float8_e4m3fn x) noexcept
{
    return float8_e4m3fn(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float8_e4m3fn log2(float8_e4m3fn x) noexcept
{
    return float8_e4m3fn(std::log(double(x)));
}
inline float8_e4m3fn max(float8_e4m3fn x, float8_e4m3fn y) noexcept
{
    return (x > y) ? x : y;
}
inline float8_e4m3fn min(float8_e4m3fn x, float8_e4m3fn y) noexcept
{
    return (x > y) ? y : x;
}
inline float8_e4m3fn sqrt(float8_e4m3fn x) noexcept
{
    return float8_e4m3fn(std::sqrt(double(x)));
}
inline float8_e4m3fn pow(int base, float8_e4m3fn expVal)
{
    return float8_e4m3fn(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float8_e4m3fnuz
//---------------------------------

inline std::istream& operator>>(std::istream& is, float8_e4m3fnuz& x)
{
    float f;
    is >> f;
    x = float8_e4m3fnuz(f);
    return is;
}
inline float8_e4m3fnuz ceil(float8_e4m3fnuz x) noexcept
{
    return float8_e4m3fnuz(lo_float_internal::ConstexprCeil(double(x)));
}
inline float8_e4m3fnuz floor(float8_e4m3fnuz x) noexcept
{
    return float8_e4m3fnuz(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float8_e4m3fnuz log2(float8_e4m3fnuz x) noexcept
{
    return float8_e4m3fnuz(std::log(double(x)));
}
inline float8_e4m3fnuz max(float8_e4m3fnuz x, float8_e4m3fnuz y) noexcept
{
    return (x > y) ? x : y;
}
inline float8_e4m3fnuz min(float8_e4m3fnuz x, float8_e4m3fnuz y) noexcept
{
    return (x > y) ? y : x;
}
inline float8_e4m3fnuz sqrt(float8_e4m3fnuz x) noexcept
{
    return float8_e4m3fnuz(std::sqrt(double(x)));
}
inline float8_e4m3fnuz pow(int base, float8_e4m3fnuz expVal)
{
    return float8_e4m3fnuz(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float8_e4m3b11fnuz
//---------------------------------

inline std::istream& operator>>(std::istream& is, float8_e4m3b11fnuz& x)
{
    float f;
    is >> f;
    x = float8_e4m3b11fnuz(f);
    return is;
}
inline float8_e4m3b11fnuz ceil(float8_e4m3b11fnuz x) noexcept
{
    return float8_e4m3b11fnuz(lo_float_internal::ConstexprCeil(double(x)));
}
inline float8_e4m3b11fnuz floor(float8_e4m3b11fnuz x) noexcept
{
    return float8_e4m3b11fnuz(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float8_e4m3b11fnuz log2(float8_e4m3b11fnuz x) noexcept
{
    return float8_e4m3b11fnuz(std::log(double(x)));
}
inline float8_e4m3b11fnuz max(float8_e4m3b11fnuz x, float8_e4m3b11fnuz y) noexcept
{
    return (x > y) ? x : y;
}
inline float8_e4m3b11fnuz min(float8_e4m3b11fnuz x, float8_e4m3b11fnuz y) noexcept
{
    return (x > y) ? y : x;
}
inline float8_e4m3b11fnuz sqrt(float8_e4m3b11fnuz x) noexcept
{
    return float8_e4m3b11fnuz(std::sqrt(double(x)));
}
inline float8_e4m3b11fnuz pow(int base, float8_e4m3b11fnuz expVal)
{
    return float8_e4m3b11fnuz(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float8_e5m2
//---------------------------------

inline std::istream& operator>>(std::istream& is, float8_e5m2& x)
{
    float f;
    is >> f;
    x = float8_e5m2(f);
    return is;
}
inline float8_e5m2 ceil(float8_e5m2 x) noexcept
{
    return float8_e5m2(lo_float_internal::ConstexprCeil(double(x)));
}
inline float8_e5m2 floor(float8_e5m2 x) noexcept
{
    return float8_e5m2(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float8_e5m2 log2(float8_e5m2 x) noexcept
{
    return float8_e5m2(std::log(double(x)));
}
inline float8_e5m2 max(float8_e5m2 x, float8_e5m2 y) noexcept
{
    return (x > y) ? x : y;
}
inline float8_e5m2 min(float8_e5m2 x, float8_e5m2 y) noexcept
{
    return (x > y) ? y : x;
}
inline float8_e5m2 sqrt(float8_e5m2 x) noexcept
{
    return float8_e5m2(std::sqrt(double(x)));
}
inline float8_e5m2 pow(int base, float8_e5m2 expVal)
{
    return float8_e5m2(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float8_e5m2fnuz
//---------------------------------

inline std::istream& operator>>(std::istream& is, float8_e5m2fnuz& x)
{
    float f;
    is >> f;
    x = float8_e5m2fnuz(f);
    return is;
}
inline float8_e5m2fnuz ceil(float8_e5m2fnuz x) noexcept
{
    return float8_e5m2fnuz(lo_float_internal::ConstexprCeil(double(x)));
}
inline float8_e5m2fnuz floor(float8_e5m2fnuz x) noexcept
{
    return float8_e5m2fnuz(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float8_e5m2fnuz log2(float8_e5m2fnuz x) noexcept
{
    return float8_e5m2fnuz(std::log(double(x)));
}
inline float8_e5m2fnuz max(float8_e5m2fnuz x, float8_e5m2fnuz y) noexcept
{
    return (x > y) ? x : y;
}
inline float8_e5m2fnuz min(float8_e5m2fnuz x, float8_e5m2fnuz y) noexcept
{
    return (x > y) ? y : x;
}
inline float8_e5m2fnuz sqrt(float8_e5m2fnuz x) noexcept
{
    return float8_e5m2fnuz(std::sqrt(double(x)));
}
inline float8_e5m2fnuz pow(int base, float8_e5m2fnuz expVal)
{
    return float8_e5m2fnuz(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float8_ieee_p<p>
//---------------------------------

template<int p>
inline std::istream& operator>>(std::istream& is, float8_ieee_p<p>& x)
{
    float f;
    is >> f;
    x = float8_ieee_p<p>(f);
    return is;
}
template<int p>
inline float8_ieee_p<p> ceil(float8_ieee_p<p> x) noexcept
{
    return float8_ieee_p<p>(lo_float_internal::ConstexprCeil(double(x)));
}
template<int p>
inline float8_ieee_p<p> floor(float8_ieee_p<p> x) noexcept
{
    return float8_ieee_p<p>(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
template<int p>
inline float8_ieee_p<p> log2(float8_ieee_p<p> x) noexcept
{
    return float8_ieee_p<p>(std::log(double(x)));
}
template<int p>
inline float8_ieee_p<p> max(float8_ieee_p<p> x, float8_ieee_p<p> y) noexcept
{
    return (x > y) ? x : y;
}
template<int p>
inline float8_ieee_p<p> min(float8_ieee_p<p> x, float8_ieee_p<p> y) noexcept
{
    return (x > y) ? y : x;
}
template<int p>
inline float8_ieee_p<p> sqrt(float8_ieee_p<p> x) noexcept
{
    return float8_ieee_p<p>(std::sqrt(double(x)));
}
template<int p>
inline float8_ieee_p<p> pow(int base, float8_ieee_p<p> expVal)
{
    return float8_ieee_p<p>(std::pow(double(base), double(expVal)));
}

//
// 3) 4-bit floats
//

//---------------------------------
// float4_e2m1
//---------------------------------

inline std::istream& operator>>(std::istream& is, float4_e2m1& x)
{
    float f;
    is >> f;
    x = float4_e2m1(f);
    return is;
}
inline float4_e2m1 ceil(float4_e2m1 x) noexcept
{
    return float4_e2m1(lo_float_internal::ConstexprCeil(double(x)));
}
inline float4_e2m1 floor(float4_e2m1 x) noexcept
{
    return float4_e2m1(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
inline float4_e2m1 log2(float4_e2m1 x) noexcept
{
    return float4_e2m1(std::log(double(x)));
}
inline float4_e2m1 max(float4_e2m1 x, float4_e2m1 y) noexcept
{
    return (x > y) ? x : y;
}
inline float4_e2m1 min(float4_e2m1 x, float4_e2m1 y) noexcept
{
    return (x > y) ? y : x;
}
inline float4_e2m1 sqrt(float4_e2m1 x) noexcept
{
    return float4_e2m1(std::sqrt(double(x)));
}
inline float4_e2m1 pow(int base, float4_e2m1 expVal)
{
    return float4_e2m1(std::pow(double(base), double(expVal)));
}

//---------------------------------
// float4_p<p>
//---------------------------------

template<int p>
inline std::istream& operator>>(std::istream& is, float4_p<p>& x)
{
    float f;
    is >> f;
    x = float4_p<p>(f); 
    return is;
}
template<int p>
inline float4_p<p> ceil(float4_p<p> x) noexcept
{
    return float4_p<p>(lo_float_internal::ConstexprCeil(double(x)));
}
template<int p>
inline float4_p<p> floor(float4_p<p> x) noexcept
{
    return float4_p<p>(-lo_float_internal::ConstexprCeil(-1.0 * double(x)));
}
template<int p>
inline float4_p<p> log2(float4_p<p> x) noexcept
{
    return float4_p<p>(std::log(double(x)));
}
template<int p>
inline float4_p<p> max(float4_p<p> x, float4_p<p> y) noexcept
{
    return (x > y) ? x : y;
}
template<int p>
inline float4_p<p> min(float4_p<p> x, float4_p<p> y) noexcept
{
    return (x > y) ? y : x;
}
template<int p>
inline float4_p<p> sqrt(float4_p<p> x) noexcept
{
    return float4_p<p>(std::sqrt(double(x)));
}
template<int p>
inline float4_p<p> pow(int base, float4_p<p> expVal)
{
    return float4_p<p>(std::pow(double(base), double(expVal)));
}

//---------------------------------


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
template <FloatingPointParams Fp_A, FloatingPointParams Fp_B, FloatingPointParams Fp_C, FloatingPointParams Fp_Acc>
inline Templated_Float<Fp1> fma(
    Templated_Float<Fp1> x, Templated_Float<Fp2> y, Templated_Float<Fp3> z) noexcept
{
    using accum_type = Templated_Float<Fp_Acc>;
    using result_type = Templated_Float<Fp3>;   
    return static_cast<result_type>(
        static_cast<accum_type>(x) * static_cast<accum_type>(y) +
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






// //-------------------------------------
// // Optionally: TLAPACK traits for each
// //-------------------------------------
// namespace tlapack {
// namespace traits {

// // float6_e3m2
// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float6_e3m2, int> {
//     using type = ml_dtypes::float8_internal::float6_e3m2;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float6_e3m2, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float6_e3m2>;
//     static constexpr bool is_complex = false;
// };

// // float6_e2m3
// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float6_e2m3, int> {
//     using type = ml_dtypes::float8_internal::float6_e2m3;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float6_e2m3, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float6_e2m3>;
//     static constexpr bool is_complex = false;
// };

// // float6_p<p>
// template <int p>
// struct real_type_traits<ml_dtypes::float8_internal::float6_p<p>, int> {
//     using type = ml_dtypes::float8_internal::float6_p<p>;
//     static constexpr bool is_real = true;
// };
// template <int p>
// struct complex_type_traits<ml_dtypes::float8_internal::float6_p<p>, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float6_p<p>>;
//     static constexpr bool is_complex = false;
// };

// // Similarly for all the 8-bit floats:
// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
//     using type = ml_dtypes::float8_internal::float8_e4m3fn;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_e4m3fn>;
//     static constexpr bool is_complex = false;
// };

// // ... and so on for float8_e4m3fnuz, float8_e4m3b11fnuz, float8_e5m2, float8_e5m2fnuz ...
// // Just replicate the same pattern:

// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3fnuz, int> {
//     using type = ml_dtypes::float8_internal::float8_e4m3fnuz;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3fnuz, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_e4m3fnuz>;
//     static constexpr bool is_complex = false;
// };

// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3b11fnuz, int> {
//     using type = ml_dtypes::float8_internal::float8_e4m3b11fnuz;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3b11fnuz, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_e4m3b11fnuz>;
//     static constexpr bool is_complex = false;
// };

// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
//     using type = ml_dtypes::float8_internal::float8_e5m2;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_e5m2>;
//     static constexpr bool is_complex = false;
// };

// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float8_e5m2fnuz, int> {
//     using type = ml_dtypes::float8_internal::float8_e5m2fnuz;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_e5m2fnuz, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_e5m2fnuz>;
//     static constexpr bool is_complex = false;
// };

// // float8_ieee_p<p>
// template <int p>
// struct real_type_traits<ml_dtypes::float8_internal::float8_ieee_p<p>, int> {
//     using type = ml_dtypes::float8_internal::float8_ieee_p<p>;
//     static constexpr bool is_real = true;
// };
// template <int p>
// struct complex_type_traits<ml_dtypes::float8_internal::float8_ieee_p<p>, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float8_ieee_p<p>>;
//     static constexpr bool is_complex = false;
// };

// // 4-bit floats:
// template <>
// struct real_type_traits<ml_dtypes::float8_internal::float4_e2m1, int> {
//     using type = ml_dtypes::float8_internal::float4_e2m1;
//     static constexpr bool is_real = true;
// };
// template <>
// struct complex_type_traits<ml_dtypes::float8_internal::float4_e2m1, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float4_e2m1>;
//     static constexpr bool is_complex = false;
// };

// template <int p>
// struct real_type_traits<ml_dtypes::float8_internal::float4_p<p>, int> {
//     using type = ml_dtypes::float8_internal::float4_p<p>;
//     static constexpr bool is_real = true;
// };
// template <int p>
// struct complex_type_traits<ml_dtypes::float8_internal::float4_p<p>, int> {
//     using type = std::complex<ml_dtypes::float8_internal::float4_p<p>>;
//     static constexpr bool is_complex = false;
// };

// } // namespace traits
// } // namespace tlapack


namespace tlapack {
namespace traits {

    //-------------------------------------------------------------------------
    // 6-bit types
    //-------------------------------------------------------------------------

    // float6_e3m2
    template <>
    struct real_type_traits<lo_float::float6_e3m2, int> {
        using type = lo_float::float6_e3m2;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float6_e3m2, int> {
        using type = std::complex<lo_float::float6_e3m2>;
        constexpr static bool is_complex = false;
    };

    // float6_e2m3
    template <>
    struct real_type_traits<lo_float::float6_e2m3, int> {
        using type = lo_float::float6_e2m3;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float6_e2m3, int> {
        using type = std::complex<lo_float::float6_e2m3>;
        constexpr static bool is_complex = false;
    };

    // float6_p<p>
    template <int p>
    struct real_type_traits<lo_float::float6_p<p>, int> {
        using type = lo_float::float6_p<p>;
        constexpr static bool is_real = true;
    };

    template <int p>
    struct complex_type_traits<lo_float::float6_p<p>, int> {
        using type = std::complex<lo_float::float6_p<p>>;
        constexpr static bool is_complex = false;
    };

    //-------------------------------------------------------------------------
    // 8-bit types
    //-------------------------------------------------------------------------

    // float8_e4m3fn
    template <>
    struct real_type_traits<lo_float::float8_e4m3fn, int> {
        using type = lo_float::float8_e4m3fn;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float8_e4m3fn, int> {
        using type = std::complex<lo_float::float8_e4m3fn>;
        constexpr static bool is_complex = false;
    };

    // float8_e4m3fnuz
    template <>
    struct real_type_traits<lo_float::float8_e4m3fnuz, int> {
        using type = lo_float::float8_e4m3fnuz;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float8_e4m3fnuz, int> {
        using type = std::complex<lo_float::float8_e4m3fnuz>;
        constexpr static bool is_complex = false;
    };

    // float8_e4m3b11fnuz
    template <>
    struct real_type_traits<lo_float::float8_e4m3b11fnuz, int> {
        using type = lo_float::float8_e4m3b11fnuz;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float8_e4m3b11fnuz, int> {
        using type = std::complex<lo_float::float8_e4m3b11fnuz>;
        constexpr static bool is_complex = false;
    };

    // float8_e5m2
    template <>
    struct real_type_traits<lo_float::float8_e5m2, int> {
        using type = lo_float::float8_e5m2;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float8_e5m2, int> {
        using type = std::complex<lo_float::float8_e5m2>;
        constexpr static bool is_complex = false;
    };

    // float8_e5m2fnuz
    template <>
    struct real_type_traits<lo_float::float8_e5m2fnuz, int> {
        using type = lo_float::float8_e5m2fnuz;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float8_e5m2fnuz, int> {
        using type = std::complex<lo_float::float8_e5m2fnuz>;
        constexpr static bool is_complex = false;
    };

    // float8_ieee_p<p>
    template <int p>
    struct real_type_traits<lo_float::float8_ieee_p<p>, int> {
        using type = lo_float::float8_ieee_p<p>;
        constexpr static bool is_real = true;
    };

    template <int p>
    struct complex_type_traits<lo_float::float8_ieee_p<p>, int> {
        using type = std::complex<lo_float::float8_ieee_p<p>>;
        constexpr static bool is_complex = false;
    };

    //-------------------------------------------------------------------------
    // 4-bit types
    //-------------------------------------------------------------------------

    // float4_e2m1
    template <>
    struct real_type_traits<lo_float::float4_e2m1, int> {
        using type = lo_float::float4_e2m1;
        constexpr static bool is_real = true;
    };

    template <>
    struct complex_type_traits<lo_float::float4_e2m1, int> {
        using type = std::complex<lo_float::float4_e2m1>;
        constexpr static bool is_complex = false;
    };

    // float4_p<p>
    template <int p>
    struct real_type_traits<lo_float::float4_p<p>, int> {
        using type = lo_float::float4_p<p>;
        constexpr static bool is_real = true;
    };

    template <int p>
    struct complex_type_traits<lo_float::float4_p<p>, int> {
        using type = std::complex<lo_float::float4_p<p>>;
        constexpr static bool is_complex = false;
    };

} // namespace traits
} // namespace tlapack

//#endif // LO_FLOAT_ALL_HPP
