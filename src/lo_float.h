/// @author Sudhanva Kulkarni
/* This file contains code for software defined 6 bit and 4 bit floats. It follows the same software design as Andrew Fitzgibbon's float8.h and uses std::bitset instead of uint8
*/
#ifndef ML_DTYPES_FLOAT6_4_H_
#define ML_DTYPES_FLOAT6_4_H_
//#define STOCHASTIC_ROUND
//#define STOCHASTIC_ARITH
#define LEN 13  
#include <random> 
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>
#include <math.h>
#include <bitset>
#include <complex>
#include <limits>
#include <ostream>
#include <type_traits>
// #include "tlapack/base/types.hpp"
// #include "tlapack/base/scalar_type_traits.hpp"
// #include "tlapack/base/types.hpp"
// #include "tlapack/base/scalar_type_traits.hpp"
#include "fp_tools.hpp"
#include "f_exceptions.hpp"
#include "template_helpers.h"

#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif








namespace lo_float {
namespace lo_float_internal {


    //forward decl of classes
    template<typename Derived, typename UnderlyingType>
    class lo_float_base;

    template<typename Derived, FloatingPointParams Fp>
    class Var_lo_float;

    template<FloatingPointParams Fp>
    class Templated_Float;





  static std::mt19937 mt(static_cast<int>(time(nullptr)));

  //global function to set seed
  void set_seed(int a) {
    mt.seed(a);
  }

    // @brief helper struct that picks underlying float that should be used for the simulation. We require 2*mantissa_bits + 1 mantissa bits in the simulation type 
    template <int N>
    struct AOpTypeSelector
    {
        using type = std::conditional_t<
            (N < 12),
            float,
            double
        >;
    };

    //alias for the helper
    template <int mantissa_bits>
    using AOpType = typename AOpTypeSelector<mantissa_bits>::type;

    // Helper sruct to decide underlying type for sqrt. Here we need 2*mantissa_bits + 2 mantissa bits in the simulation type

    template <int mantissa_bits>
    struct SqrtTypeSelector
    {
        using type = std::conditional_t<
        (mantissa_bits < 11), 
        float,
        double
        >;
    };

    //alias
    template <int mantissa_bits>
    using SqrtType = typename SqrtTypeSelector<mantissa_bits>::type;





    template<typename Derived, typename UnderlyingType = uint8_t>
    class lo_float_base {
    protected:
        struct ConstructFromRepTag {};

        // "Core" constructor storing rep_ in the base
        constexpr lo_float_base(UnderlyingType rep, ConstructFromRepTag)
            : rep_(rep)
        {}

        // CRTP friend declaration
        template<typename T, FloatingPointParams Fp> friend class Var_lo_float;
        template<FloatingPointParams Fp> friend class Templated_Float;
        


    public:
        constexpr lo_float_base() : rep_(0) {}

        constexpr UnderlyingType rep() const {
            return rep_;
        }

        // Templated constructor
        template <typename T,
                typename EnableIf = std::enable_if_t<std::is_arithmetic_v<T>>>
        explicit lo_float_base(T f)
            : lo_float_base(ConvertFrom(static_cast<float>(f)).rep(),
                            ConstructFromRepTag{}) {}

        explicit lo_float_base(double f64)
            : lo_float_base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}

        explicit lo_float_base(float f32)
            : lo_float_base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}

        // CRTP helpers
        constexpr const Derived& derived() const {
            return *static_cast<const Derived*>(this);
        }
        constexpr Derived& derived() {
            return *static_cast<Derived*>(this);
        }

        static constexpr Derived FromRep(UnderlyingType rep) {
            return Derived(rep, ConstructFromRepTag{});
        }

        // -------------------------------------------
        // Declarations for ConvertFrom / ConvertTo
        // -------------------------------------------
        template <typename From>
        static inline Derived ConvertFrom(const From& from);

        template <typename To>
        static inline To ConvertTo(const Derived& from);


        template <typename T,
                typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
            explicit operator T() const {
                return static_cast<T>(static_cast<float>(derived()));
            }
            explicit operator double() const {
                return ConvertTo<double>(derived());
            }
            explicit operator float() const {
                return ConvertTo<float>(derived());
            }

            explicit operator bool() const {
                return if constexpr (get_signedness_v<Derived> == Signedness::Signed) {
                    return (rep() & 0x7F) != 0;
                } else {
                    return rep() != 0;
                }
            }

        // define underlying float before defining arithemtic types
        using UnderlyingFloat = AOpType<get_mantissa_bits_v<Derived>>;
        __attribute__((always_inline)) inline  Derived
        operator+(const Derived& other) const {
            return Derived{UnderlyingFloat{derived()} + UnderlyingFloat{other}};
        }
        __attribute__((always_inline)) inline  Derived
        operator=(const Derived& other) const {
            return Derived{UnderlyingFloat{other}};
        }
        __attribute__((always_inline)) inline  Derived
        operator-(const Derived& other) const {
            return Derived{UnderlyingFloat{derived()} - UnderlyingFloat{other}};
        }
        __attribute__((always_inline)) inline  Derived
        operator*(const Derived& other) const {
            return Derived{UnderlyingFloat{derived()} * UnderlyingFloat{other}};
        }
        __attribute__((always_inline)) inline  Derived
        operator/(const Derived& other) const {
            return Derived{UnderlyingFloat{derived()} / UnderlyingFloat{other}};
        }

        // Example comparison
        enum Ordering : int8_t {
            kLess = -1,
            kEquivalent = 0,
            kGreater = 1,
            kUnordered = 2,
        };

        template<typename T>
        constexpr bool operator==(const T& other) const {
            return Compare(derived(), other) == Ordering::kEquivalent;
        }

        template<typename T>
        constexpr bool operator!=(const T& other) const {
            return Compare(derived(), other) != Ordering::kEquivalent;
        }

        template<typename T>
        __attribute__((always_inline)) inline  bool operator<(
            const T& other) const {
            return Compare(derived(), other) == Ordering::kLess;
        }

        template<typename T>
        __attribute__((always_inline)) inline  bool operator<=(
            const T& other) const {
            return Compare(derived(), other) <= Ordering::kEquivalent;
        }

        template<typename T>
        __attribute__((always_inline)) inline  bool operator>(
            const T& other) const {
            return Compare(derived(), other) == Ordering::kGreater;
        }

        template<typename T>
        __attribute__((always_inline)) inline  bool operator>=(
            const T& other) const {
            auto ordering = Compare(derived(), other);
            return ordering == kGreater || ordering == kEquivalent;
        }
        
        __attribute__((always_inline)) inline  Derived& operator+=(
            const Derived& other) {
            derived() = derived() + other;
            return derived();
        }
        __attribute__((always_inline)) inline  Derived& operator-=(
            const Derived& other) {
            derived() = derived() - other;
            return derived();
        }
        __attribute__((always_inline)) inline  Derived& operator*=(
            const Derived& other) {
            derived() = derived() * other;
            return derived();
        }
        __attribute__((always_inline)) inline  Derived& operator/=(
            const Derived& other) {
            derived() = derived() / other;
            return derived();
        }

    private:
        //-----------------------------------------
        // Single shared 'rep_' in the base
        //-----------------------------------------
        UnderlyingType rep_;
        using Signed_type = typename std::make_signed<UnderlyingType>::type;

        template <typename T>
        concept SignedFloat = (get_signedness_v<T> == lo_float::Signedness::Signed);

        template <typename T>
        concept UnsignedFloat = (get_signedness_v<T> == lo_float::Signedness::Unsigned);

        // Helper for compare:
        static __attribute__((always_inline)) inline  std::pair<UnderlyingType, UnderlyingType>
        SignAndMagnitude(Derived x) {
            const UnderlyingType x_abs_bits =
                std::bit_cast<UnderlyingType>(std::abs(x));
            const UnderlyingType x_bits = std::bit_cast<UnderlyingType>(x);
            const UnderlyingType x_sign = x_bits ^ x_abs_bits;
            return {x_sign, x_abs_bits};
        }

        static __attribute__((always_inline)) inline  Signed_type
        SignAndMagnitudeToTwosComplement(UnderlyingType sign, UnderlyingType magnitude) {
            return magnitude ^ (static_cast<Signed_type>(sign) < 0 ? -1 : 0);
        }

        // Compare function. For signed floats tak eTwos complement path. For unsigned just compare represntations
        template<typename T>
        __attribute__((always_inline)) inline  friend constexpr Ordering Compare(
            const Derived& lhs, const T& rhs) {
            if (std::isnan(lhs) || std::isnan(rhs)) {
                return kUnordered;
            }
            auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
            auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
            if (lhs_mag == 0 && rhs_mag == 0) {
                return kEquivalent;
            }
            Signed_type lhs_tc = SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
            Signed_type rhs_tc = SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
            if (lhs_tc < rhs_tc) return kLess;
            if (lhs_tc > rhs_tc) return kGreater;
            return kEquivalent;
        }

        template<UnsignedFloat T>
        __attribute__((always_inline)) inline  friend constexpr Ordering Compare(
            const Derived& lhs, const T& rhs)
            requires UnsignedFloat<Derived> {
            if (std::isnan(lhs) || std::isnan(rhs)) {
                return kUnordered;
            }
            auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
            auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
            if (lhs_mag == 0 && rhs_mag == 0) {
                return kEquivalent;
            }
            if (lhs_mag < rhs_mag) return kLess;
            if (lhs_mag > rhs_mag) return kGreater;
            return kEquivalent;
        }
    }; //lo_float_base


    //helper template to pick storage format
    template<int Len>
    using Base_repr_select = std::conditional_t<(Len <= 8), uint8_t, std::conditional_t<(Len <= 16), uint16_t, uint32_t>>;


    /// varfloat base
    template<typename Derived, FloatingPointParams Fp>
    class Var_lo_float : public lo_float_base<Derived, Base_repr_select<Fp.bitwidth>> {
        private :
        using UType = Base_repr_select<Fp.bitwidth>;
        using Base  = lo_float_base<Derived, UType>;

        friend class lo_float_base<Derived, UType>;

        friend class Templated_Float<Fp>;

        // Inherit constructors from lo_float_base
        using Base::Base;

        using SType = typename std::make_signed<UType>::type;


        static __attribute__((always_inline)) inline  SType
        SignAndMagnitudeToTwosComplement(UType sign, UType magnitude) {
                return magnitude ^ (static_cast<SType>(sign << Fp.Len) < 0 ? -1 : 0);
        }

        protected:
            using typename Base::ConstructFromRepTag;

            constexpr Var_lo_float(UType rep, ConstructFromRepTag tag)
                : Base(rep, tag)
            {}

        public:

            explicit  operator bool() const {
                return (this->rep() & ((1 << Fp.bitwidth) - 1)) != 0;
            }
            constexpr Derived operator-() const {
                return Base::FromRep(static_cast<UType>(this->rep() ^ (1 << (Fp.bitwidth - 1))));
            }

            Derived operator-(const Derived& other) const {
                return Base::operator-(other);
            }
            //declare structs/enums from template arg as static fields so that they can be accessed later
            static constexpr NaNChecker auto IsNaNFunctor = Fp.IsNaN;

            static constexpr InfChecker auto IsInfFunctor = Fp.IsInf;

            static constexpr Rounding_Mode rounding_mode = Fp.rounding_mode;

            static constexpr  Inf_Behaviors Overflow_behavior = Fp.OV_behavior;
            static constexpr  NaN_Behaviors NaN_behavior = Fp.NA_behavior;

            static constexpr int bitwidth = Fp.bitwidth;

            static constexpr Signedness is_signed = Fp.is_signed;

            static constexpr Unsigned_behavior unsigned_behavior = Fp.unsigned_behavior;

            static constexpr  int bias = Fp.bias;

            static constexpr int mantissa_bits = Fp.mantissa_bits;

            static constexpr int stochastic_rounding_length = Fp.StochasticRoundingBits;

    };

    template<FloatingPointParams Fp>
    class Templated_Float : public Var_lo_float<Templated_Float<Fp>, Fp> {
    private:
    using Base = Var_lo_float<Templated_Float<Fp>, Fp>;


    public:
    
    using Base::Base;

    static constexpr NaNChecker auto IsNaNFunctor = Fp.IsNaN;

    static constexpr InfChecker auto IsInfFunctor = Fp.IsInf;

    static constexpr Rounding_Mode rounding_mode = Fp.rounding_mode;

    static constexpr  Inf_Behaviors Overflow_behavior = Fp.OV_behavior;
    static constexpr  NaN_Behaviors NaN_behavior = Fp.NA_behavior;

    static constexpr int bitwidth = Fp.bitwidth;

    static constexpr Signedness is_signed = Fp.is_signed;

    static constexpr Unsigned_behavior unsigned_behavior = Fp.unsigned_behavior;

    static constexpr  int bias = Fp.bias;

    static constexpr int mantissa_bits = Fp.mantissa_bits;

    static constexpr int stochastic_rounding_length = Fp.StochasticRoundingBits;


    };


    //define FloatingPointParams for float8e4m3_fn
    struct OCP_F8E4M3_NaNChecker {
        bool operator()(uint32_t bits) {
            return bits == 0x000000FF;
        }

        uint32_t qNanBitPattern() const {
            return 0x000000FF;
        }  // typical QNaN

        uint32_t sNanBitPattern() const {
            return 0x000000FF;
        }  // some SNaN pattern
    };

    struct OCP_F8E4M3_InfChecker {
        bool operator()(uint32_t bits) {
            return false;
        }

        uint32_t minNegInf() const {
            return 0x0;
        }  // -∞ => 0xFF800000

        uint32_t minPosInf() const {
            return 0x0;
        }  // +∞ => 0x7F800000
    };

    // FloatingPointParams for float8e4m3_fn -> f is finite, n is NaN
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_e4m3fn(
        8, //totoal bitwidth
        3, // mantissa bits
        7,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );
        

      
    // FloatingPointParams for float8e4m3b11_fnuz -> f is finite, n is NaN, u unsigned, z zero
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_e4m3b11fnuz(
        8, //totoal bitwidth
        3, // mantissa bits
        11,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Unsigned,       //It is unsigned
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );

        // FloatingPointParams for float8e4m3b11_fnuz -> f is finite, n is NaN, u unsigned, z zero
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_e4m3fnuz(
        8, //totoal bitwidth
        3, // mantissa bits
        7,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Unsigned,       //It is unsigned
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );

    //NaNChecker for float8e5m2
    struct OCP_F8E5M2_NaNChecker {
        bool operator()(uint32_t bits) {
            return bits == 0x000000FF || bits == 0x000000FE || bits == 0x000000FD;
        }

        uint32_t qNanBitPattern() const {
            return 0x000000FF;
        }  

        uint32_t sNanBitPattern() const {
            return 0x000000FF;
        } 
    };


    // FloatingPoint params for float8e5m2
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_e5m2(
        8, //totoal bitwidth
        2, // mantissa bits
        15,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );


    // FloatingPointParams for float8e5m2fnuz -> f is finite, n is NaN, u unsigned, z zero
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_e5m2fnuz(
        8, //totoal bitwidth
        2, // mantissa bits
        15,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Unsigned,       //It is unsigned
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );


    // FloatingPointParams for float8ieee_p
    template<int p, Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float8_ieee_p(
        8, //totoal bitwidth
        p - 1, // mantissa bits
        (1 << (8 - p)) - 1,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );


    
    // FloatingPointParams for float6e3m2
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float6_e3m2(
        6, //totoal bitwidth
        2, // mantissa bits
        3,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );

    // FloatingPointParams for float6e2m3
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float6_e2m3(
        6, //totoal bitwidth
        3, // mantissa bits
        1,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );

    // FloatingPointParams for float6_p
    template<int p, Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float6_p(
        6, //totoal bitwidth
        p - 1, // mantissa bits
        (1 << (6 - p)) - 1,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );

    // FloatingPointParams for float4_e2m1
    template<Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float4_e2m1(
        4, //totoal bitwidth
        1, // mantissa bits
        1,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );


    // FloatingPointParams for float4_p
    template<int p, Rounding_Mode round_mode>
    constexpr FloatingPointParams param_float4_p(
        4, //totoal bitwidth
        p - 1, // mantissa bits
        (1 << (4 - p)) - 1,  //bias
        round_mode,  // rounding mode
        Inf_Behaviors::Saturating,  //No infinity
        NaN_Behaviors::QuietNaN,    //NaN behavior
        Signedness::Signed,         //It is signed
        OCP_F8E4M3_InfChecker(),    //Inf Functor
        OCP_F8E4M3_NaNChecker()     //NaN Functor
    );
   

    
} //namepsace lo_float_internal


    //now define the types using the previously defined FloatingPointParams

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_e4m3_fn = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3fn<round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_e4m3b11_fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3b11fnuz<round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_e4m3_fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3fnuz<round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_e5m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e5m2<round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_e5m2fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e5m2fnuz<round_mode>>;

    template<int p, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float8_ieee_p = lo_float_internal::Templated_Float<lo_float_internal::param_float8_ieee_p<p, round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float6_e3m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float6_e3m2<round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float6_e2m3 = lo_float_internal::Templated_Float<lo_float_internal::param_float6_e2m3<round_mode>>;

    template<int p, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float6_p = lo_float_internal::Templated_Float<lo_float_internal::param_float6_p<p, round_mode>>;

    template<Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float4_e2m1 = lo_float_internal::Templated_Float<lo_float_internal::param_float4_e2m1<round_mode>>;

    template<int p, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven>
    using float4_p = lo_float_internal::Templated_Float<lo_float_internal::param_float4_p<p, round_mode>>;


    template<typename T>
constexpr T ConstexprAbs(T x) { return x < T{0.0} ? -x : x; }

template<typename T>
constexpr T ConstexprCeil(T x) {
  constexpr T kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<T>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;\
  }
  const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
  return x_trunc < x ? x_trunc + 1.0 : x_trunc;
}

constexpr double ConstexprFloor(double x) { return -ConstexprCeil(-x); }

constexpr double kLog10Of2 = 0.3010299956639812;
// C17 5.2.4.2.2p11:
// "number of decimal digits, q, such that any floating-point number with q
// decimal digits can be rounded into a floating-point number with p radix b
// digits and back again without change to the q decimal digits"
// floor((p - 1) * log10(2));
constexpr int Digits10FromDigits(int digits) {
  return static_cast<int>(ConstexprFloor((digits - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "number of decimal digits, n, such that any floating-point number with p
// radix b digits can be rounded to a floating-point number with n decimal
// digits and back again without change to the value" 
// ceil(1 + p * log10(2));
constexpr int MaxDigits10FromDigits(int digits) {
  return static_cast<int>(ConstexprCeil(1.0 + (digits * kLog10Of2)));
}

// C17 5.2.4.2.2p11:
// "minimum negative integer such that 10 raised to that power is in the range
// of normalized floating-point numbers"
// TODO: https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent10 says "representable"
// ceil(log10(2**(emin - 1))) == ceil((emin - 1) * log10(2));
constexpr int MinExponent10FromMinExponent(int min_exponent) {
  return static_cast<int>(ConstexprCeil((min_exponent - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "maximum integer such that 10 raised to that power is in the range of
// representable finite floating-point numbers"
// floor(log10((1 - 2**-p) * 2**emax)) == floor(log10(1 - 2**-p) +
// emax * log10(2))
constexpr int MaxExponent10FromMaxExponentAndDigits(int max_exponent,
                                                    int digits) {
  // We only support digits in {3,4}. This table would grow if we wanted to
  // handle more values.
  constexpr double kLog10OfOnePredecessor[] = {
    //log10(1 - 2**-1)
    -0.3010299956639812,
    //log10(1 - 2**-2)
      -0.12493873660829993,
      // log10(1 - 2**-3)
      -0.057991946977686754,
      // log10(1 - 2**-4)
      -0.028028723600243537,
      // log10(1 - 2**-5)
      -0.013788284485633295,
    // log10(1 - 2**-6)
        -0.006931471805599453,
        // log10(1 - 2**-7)
    -0.003415655202992898,
    // log10(1 - 2**-8)
        -0.0017077547810594657
  };
  // 
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 1] +
                                         max_exponent * kLog10Of2));
}


namespace lo_float_internal {

    



template<FloatingPointParams Fp>
struct numeric_limits_flexible {

    static inline constexpr const bool is_specialized = true;
    static inline constexpr const bool is_signed = Fp.is_signed == lo_float::Signedness::Signed;
    static inline constexpr const bool is_integer = false;
    static inline constexpr const bool is_exact = false;
    static inline constexpr const bool has_quiet_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::QuietNaN;
    static inline constexpr const bool has_signaling_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::SignalingNaN;
    static inline constexpr const bool has_denorm = true;
    static inline constexpr const bool has_denorm_loss = false;
    static inline constexpr const bool round_style = std::round_indeterminate;
    static inline constexpr const bool is_iec559 = false;
    static inline constexpr const int radix = std::numeric_limits<float>::radix;
    static inline constexpr const bool traps = false;
    static inline constexpr const bool tinyness_before = true;
    static inline constexpr const int kExponentBias = Fp.bias;
    static inline constexpr const int kMantissaBits = Fp.mantissa_bits;
    static inline constexpr const int digits = Fp.mantissa_bits + 1;
    static inline constexpr const int digits10 = Digits10FromDigits(digits);
    static inline constexpr const int max_digits10 =
        MaxDigits10FromDigits(digits);
    static inline constexpr const int min_exponent = (1 - kExponentBias);
    static inline constexpr const int min_exponent10 =
        MinExponent10FromMinExponent(min_exponent);
    static inline constexpr const int max_exponent = is_signed ? (1 << (Fp.bitwidth - Fp.mantissa_bits - 1)) - 1 - Fp.bias : (1 << (Fp.bitwidth - Fp.mantissa_bits)) - 1 - Fp.bias;
    static inline constexpr const int max_exponent10 =
        MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
    static inline constexpr const bool has_infinity = Fp.OV_behavior != lo_float::Inf_Behaviors::Saturating;

    using Base_Type = Base_repr_select<Fp.bitwidth>;
            static constexpr Templated_Float<Fp> min() {
                return Templated_Float<Fp>::FromRep((1 << (Fp.mantissa_bits)));
              }
              static constexpr Templated_Float<Fp> lowest() {
                return Templated_Float<Fp>::FromRep(Fp.IsInf.minNegInf() - 1);
              }
              static constexpr Templated_Float<Fp> max() {
                return Templated_Float<Fp>::FromRep(Fp.IsInf.minPosInf() - 1);
              }
              static constexpr Templated_Float<Fp> epsilon() {
                return Templated_Float<Fp>::FromRep(
                  static_cast<Base_Type>(((-1 + kExponentBias) << kMantissaBits))
                );
              }
              static constexpr Templated_Float<Fp> round_error() {
                return Templated_Float<Fp>::FromRep(
                    ((-1 + kExponentBias) << kMantissaBits)
                );
              }
              static constexpr Templated_Float<Fp> infinity() {
                return Templated_Float<Fp>::FromRep(Fp.IsInf.minPosInf());
              }
              static constexpr Templated_Float<Fp> quiet_NaN() {
                return Templated_Float<Fp>::FromRep(Fp.IsNaN.qNanBitPattern());
              }
              static constexpr Templated_Float<Fp> signaling_NaN() {
                return Templated_Float<Fp>::FromRep(Fp.IsNaN.sNanBitPattern());
              }
              static constexpr Templated_Float<Fp> denorm_min() {
                return Templated_Float<Fp>::FromRep(0x1);
              }

};

} // namespace lo_float_internal 

template <FloatingPointParams Fp>
using Templated_Float = lo_float_internal::Templated_Float<Fp>;

void set_seed(unsigned int seedVal)
{
    // Just forward to the internal function or directly:
    lo_float_internal::mt.seed(seedVal);

    // or lo_float_internal::set_seed_internal(seedVal);
}
}


namespace lo_float {
    namespace lo_float_internal {

template<FloatingPointParams Fp>
constexpr inline Templated_Float<Fp> abs(const Templated_Float<Fp>& a) {
    return Templated_Float<Fp>::FromRep(a.rep() & ((1 << (Fp.bitwidth - 1)) - 1));
}

template<FloatingPointParams Fp>
constexpr inline bool isnan(const Templated_Float<Fp>& a) {
    return Fp.IsNaN(a.rep());
}

template<FloatingPointParams Fp>
constexpr inline bool isinf(const Templated_Float<Fp>& a) {
    return Fp.IsInf(a.rep());
}

template <typename LoFloat>
constexpr inline bool(isinf)(const lo_float_base<LoFloat>& lf) {
  return std::numeric_limits<LoFloat>::has_infinity
             ? abs(lf.derived()).rep() ==
                   std::numeric_limits<LoFloat>::infinity().rep()
             : false;  // No inf representation.
}


template <typename LoFloat>
constexpr inline bool(isfinite)(const lo_float_base<LoFloat>& a) {
  return !isnan(a.derived()) && !isinf(a.derived());
}

template <typename LoFloat>
std::ostream& operator<<(std::ostream& os, const lo_float_base<LoFloat>& lf) {
  os << static_cast<float>(lf.derived());
  return os;
}



}
}




namespace std {


    template <lo_float::FloatingPointParams Fp>
    struct numeric_limits<lo_float::Templated_Float<Fp>>
      : lo_float::lo_float_internal::numeric_limits_flexible<Fp> {};
    
    


    //abs override
    template <lo_float::FloatingPointParams Fp>
    lo_float::Templated_Float<Fp> abs(
        const lo_float::Templated_Float<Fp>& a) {
        return lo_float::lo_float_internal::abs(a);
    }

    //isnan override
    template <lo_float::FloatingPointParams Fp>
    bool isnan(const lo_float::Templated_Float<Fp>& a) {
        return lo_float::lo_float_internal::isnan(a);
    }

    //isinf override
    template <lo_float::FloatingPointParams Fp>
    bool isinf(const lo_float::Templated_Float<Fp>& a) {
        return lo_float::lo_float_internal::isinf(a);
    }


    

} //namespace std








namespace lo_float {
    namespace lo_float_internal {

    template<int le_size, typename uint>
    static constexpr inline int countl_zero(uint x) {
        int zeroes = le_size - 4;
        if constexpr (le_size > 64) {
            if (x >> 64) {
                zeroes -= 64;
                x >>= 64;
            }
        }
        if constexpr (le_size > 32) {
            if (x >> 32) {
                zeroes -= 32;
                x >>= 32;
            }
        }
        if constexpr (le_size > 16) {
            if (x >> 16) {
                zeroes -= 16;
                x >>= 16;
            }
        }
        if constexpr (le_size > 8) {
            if(x >> 8) {
                zeroes -= 8;
                x >>= 8;
            }
        }
        if constexpr (le_size > 4) {
            if (x >> 4) {
                zeroes -= 4;
                x >>= 4;
            }
        }

        return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
    }

   



// template <int kNumBytes>
// using GetUnsignedInteger =
//     typename std::get_integer_by_size<kNumBytes>::unsigned_type;



template <int a, typename Enable = void>
struct IntegerBySize;

template <int a>
struct IntegerBySize<a, std::enable_if_t<a == 1>> {
    using unsigned_type = uint8_t;
    using signed_type = int8_t;
};

template <int a>
struct IntegerBySize<a, std::enable_if_t<a == 2>> {
    using unsigned_type = uint16_t;
    using signed_type = int16_t;
};


template <int a>
struct IntegerBySize<a, std::enable_if_t<(a > 2 && a <= 4)>> {
    using unsigned_type = uint32_t;
    using signed_type = int32_t;
};

template <int a>
struct IntegerBySize<a, std::enable_if_t<(a > 4)>> {
    using unsigned_type = uint64_t;
    using signed_type = int64_t;
};

// Alias to get the unsigned type directly
template <int kNumBytes>
using GetUnsignedInteger = typename IntegerBySize<kNumBytes>::unsigned_type;

// template <>
// struct IntegerBySize<2> {
//     using unsigned_type = uint16_t;
//     using signed_type = int16_t;
// };

// template <>
// struct IntegerBySize<4> {
//     using unsigned_type = uint32_t;
//     using signed_type = int32_t;
// };

// template <>
// struct IntegerBySize<8> {
//     using unsigned_type = uint64_t;
//     using signed_type = int64_t;
// };




// Converts between two floating-point types.
template <typename From, typename To,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static  inline Scalar run(const Scalar& from) {
    return from;
  }
};

template <typename Scalar>
struct ConvertImpl<Scalar, Scalar> : public IdentityConversion<Scalar> {};



template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = std::numeric_limits<Float>::digits - 1;
  static constexpr int kExponentBits = get_signedness_v<Float> == Signedness::Signed
                                ? Fp.bitwidth - Fp.mantissa_bits - 1
                                : Fp.bitwidth - Fp.mantissa_bits;
  static constexpr BitsType kExponentMask = get_signedness_v<Templated_Float<Fp>> == Signedness::Signed
                                ? (BitsType{1} << (kExponentBits - 1)) - 1
                                : (BitsType{1} << kExponentBits) - 1;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = std::numeric_limits<Float>::kExponentBias;
};


template <typename Float>
struct Traits : public TraitsBase<Float> {};

template <FloatingPointParams Fp>
struct Traits<Templated_Float<Fp>> : public TraitsBase<Templated_Float<Fp>> {
    using Base = TraitsBase<Templated_Float<Fp>>;
    static constexpr int kBits = Fp.bitwidth;
    static constexpr int kMantissaBits = Fp.mantissa_bits;
    static constexpr int kExponentBits = get_signedness_v<Templated_Float<Fp>> == Signedness::Signed
                                             ? Fp.bitwidth - Fp.mantissa_bits - 1
                                             : Fp.bitwidth - Fp.mantissa_bits;
    static constexpr int kExponentBias = Fp.bias;
};
 
template <>
struct Traits<float> : public TraitsBase<float> {
  static constexpr int kBits = sizeof(float) * CHAR_BIT;
  static constexpr int kExponentBits = 8;
  static constexpr int kMantissaBits = 23;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};

template <>
struct Traits<double> : public TraitsBase<double> {
  static constexpr int kBits = sizeof(double) * CHAR_BIT;
  static constexpr int kExponentBits = 11;
  static constexpr int kMantissaBits = 52;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};




template <typename Bits>
constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
   // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = roundoff == 0
                  ? 0
                  : ((bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
  return bits + bias;
}




template <typename Bits>
inline Bits Stochastic_Round(Bits bits, int roundoff, int len = 0) {
  //given pattern FFF...FLRTT...T,rounds stochastically by generating random bits
  // corresponding to  RTT...T and adding the genned number.
  //Then we truncate the mantissa
  //auto len = 2;
  std::uniform_int_distribution<unsigned int> distribution(0, (1<< len) - 1);
  unsigned int samp = distribution(mt); // Generate a random integer of length len, next get top "roundoff" bits
  unsigned int to_add;
  if (roundoff > len) {
    to_add = samp << (roundoff - len);
    } else {
        to_add = samp >> (len - roundoff);
    }

  Bits to_ret = bits + static_cast<Bits>(to_add);
  return to_ret;
}

template <typename Bits>
inline Bits RoundBitsTowardsZero(Bits bits, int roundoff) {
    // Round towards zero by just truncating the bits
    //in bits FFF...FLRTT....T RTT....T needs to be rounded off, so just set  RTT..T to be 0
    auto mask = ~((Bits{1} << roundoff) - 1);
    return bits & mask;
}


template<typename Bits>
inline Bits RoundBitsAwayFromZero(Bits bits, int roundoff) {
    //Round away from Zero by truncating bits and adding one to the remaining bit pattern
    // in bits FFF...FRTT...T, set RTT...T to be zero and add 1 to FFF...F
    auto mask = ~((Bits{1} << roundoff) - 1);
    Bits truncated = bits & mask;
    return truncated + (bits > 0 ? Bits{1} << roundoff : 0);
}

template <typename Bits>
constexpr inline Bits RoundBitsToNearestOdd(Bits bits, int roundoff) {
    // Round to nearest odd by adding a bias term.
    // Consider a bit pattern:
    //   FFF...FLRTT...T,
    // where bits RTT...T need to be rounded-off. We add a bias term to the
    // bit pattern such that a carry is introduced to round up only if
    // - L is 0, R is 1, OR
    // - L is 1, R is 1, any T is one.
    // This ensures the final result is odd.
    
    Bits bias = roundoff == 0
                    ? 0
                    : ((~bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
    return bits + bias;
}

template<typename Bits>
inline Bits RoundUp(Bits bits, int roundoff) {
  //round bit pattern up by adding 1 to the bit pattern
  //in bits FFF...FLRTT...T, set RTT...T to be 0 and add 1 to FFF...F
  auto mask = ~((Bits{1} << roundoff) - 1);
  Bits truncated = bits & mask;
  return truncated + (bits > 0 ? Bits{1} << roundoff : 0);

}
template <typename Bits>
inline Bits RoundDown(Bits bits, int roundoff) {
  //just truncate the bits
  //in bits FFF...FLRTT...T, set RTT...T to be 0
  auto mask = ~((Bits{1} << roundoff) - 1);
  return bits & mask;
 
}

template <typename From, typename To>
struct ConvertImpl<From, To,
                   std::enable_if_t<!std::is_same_v<From, To>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;
  //none are void

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToBits = ToTraits::kBits;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;
    //none are void
  // `WideBits` is wide enough to accommodate the largest exponent and mantissa
  // in either `From` or `To`.
  static constexpr int kWideBits =
      (std::max(kToMantissaBits, kFromMantissaBits)) +  // Max significand.
      (std::max(kToExponentBits, kFromExponentBits));   // Max exponent.
  static constexpr int kWideBytes = (kWideBits + (CHAR_BIT - 1)) / CHAR_BIT;   //kWideBits = 0????
  using WideBits = GetUnsignedInteger<kWideBytes>;
  //static_assert(std::is_same_v<WideBits, uint32_t>, "WideBits<8> must be uint64_t");
  //using WideBits = u_int32_t;
  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;


//need to change the bool to an enum to support other rounding modes
  static  inline To run(const From& from, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven) {
    // Shift bits to destination type, without sign bit.
    const bool from_sign_bit = get_signedness_v<From> == Signedness::Unsigned ? false :
        std::bit_cast<FromBits>(from) >> (kFromBits - 1);
    
    if(get_signedness_v<To> == Signedness::Unsigned && from_sign_bit) {
        if( get_unsigned_behavior_v<To> == Unsigned_behavior::NegtoZero) {
            return To{};
        } else {
            if( get_unsigned_behavior_v<To> == Unsigned_behavior::NegtoNaN) {
                
            }
            return std::numeric_limits<To>::
        }
    }
    const FromBits from_bits =
        std::bit_cast<FromBits>(std::abs(from));

    // Special values, preserving sign.
    if (std::isinf(from)) {
      return from_sign_bit ? -std::numeric_limits<To>::infinity()
                           : std::numeric_limits<To>::infinity();
    }
    if (std::isnan(from)) {
      return std::numeric_limits<To>::quiet_NaN();
    }
    if (from_bits == 0) {
      return from_sign_bit ? -To{} : To{};
    }

    const int biased_from_exponent = from_bits >> kFromMantissaBits;  //check if number is subnormal

    // `To` supports more exponents near zero which means that some subnormal
    // values in `From` may become normal. 
    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent) {
      if (biased_from_exponent == 0) {
        // Subnormals.
        WideBits bits = from_bits;

        // Determine exponent in target type.
        const int normalization_factor =
            countl_zero<kFromBits>(from_bits) - (kFromBits - kFromMantissaBits) + 1;
        const int biased_exponent = kExponentOffset - normalization_factor + 1;
        if (biased_exponent <= 0) {
          // Result is subnormal.  Adjust the subnormal bits to account for
          // the difference in exponent bias.
          if constexpr (kExponentOffset < (kWideBits) ) {        //is this ok for fp6/4?
            bits <<= kExponentOffset;
          }
        } else {
          // Result is normal. Shift the mantissa to account for the number of
          // leading zero digits, and clear the hidden bit.
          bits <<= normalization_factor;
          bits &= ~(WideBits{1} << kFromMantissaBits);
          // Insert the exponent bits.
          bits |= static_cast<WideBits>(biased_exponent) << kFromMantissaBits;
        }

        // Truncate/round mantissa if necessary.
        if constexpr (kDigitShift > 0) {
          bits <<= kDigitShift;
        } else {
            
            switch(round_mode){
                case Rounding_Mode::RoundToNearestOdd :
                    bits = RoundBitsToNearestOdd(bits, -kDigitShift);
                    break;
                case Rounding_Mode::RoundTowardsZero :
                    bits = RoundBitsTowardsZero(bits, -kDigitShift);
                    break;
                case Rounding_Mode::RoundAwayFromZero :
                    bits = RoundBitsAwayFromZero(bits, -kDigitShift);
                    break;
                case Rounding_Mode::StochasticRounding :
                    bits = Stochastic_Round(bits, -kDigitShift, get_stochastic_length_v<To>);
                    break;
                default :
                    bits = RoundBitsToNearestEven(bits, -kDigitShift);
            } 
            
          
          bits >>= -kDigitShift;
        }
        To to = std::bit_cast<To>(static_cast<ToBits>(bits));
        return from_sign_bit ? -to : to;
      }
    }
    // `To` supports fewer exponents near zero which means that some values in
    // `From` may become subnormal.
    if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent) {
      const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
      const int biased_to_exponent = unbiased_exponent + kToExponentBias;
      // Subnormals and zero.
      if (biased_to_exponent <= 0) {
        // Round and shift mantissa down.
        FromBits from_has_leading_one = (biased_from_exponent > 0 ? 1 : 0);
        int exponent_shift =
            -kDigitShift - biased_to_exponent + from_has_leading_one;
        // Insert the implicit leading 1 bit on the mantissa for normalized
        // inputs.
        FromBits rounded_from_bits =
            (from_bits & FromTraits::kMantissaMask) |
            (from_has_leading_one << kFromMantissaBits);
        ToBits bits = 0;
        // To avoid UB, limit rounding and shifting to the full mantissa plus
        // leading 1.
        if (exponent_shift <= kFromMantissaBits + 1) {
            // NOTE: we need to round again from the original from_bits,
            // otherwise the lower precision bits may already be lost.  There is
            // an edge-case where rounding to a normalized value would normally
            // round down, but for a subnormal, we need to round up.
            switch (round_mode) {
                case Rounding_Mode::RoundToNearestEven:
                  rounded_from_bits = RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
                  break;
                case Rounding_Mode::RoundToNearestOdd:
                  rounded_from_bits = RoundBitsToNearestOdd(rounded_from_bits, exponent_shift);
                  break;
                case Rounding_Mode::RoundTowardsZero:
                  rounded_from_bits = RoundBitsTowardsZero(rounded_from_bits, exponent_shift);
                  break;
                case Rounding_Mode::RoundAwayFromZero:
                  rounded_from_bits = RoundBitsAwayFromZero(rounded_from_bits, exponent_shift);
                  break;
                case Rounding_Mode::StochasticRounding:
                  rounded_from_bits = Stochastic_Round(rounded_from_bits, exponent_shift, get_stochastic_length_v<To>);
                  break;
              }
              
          bits = (rounded_from_bits >> exponent_shift);
        }
        // Insert sign and return.
        To to = std::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }

    // Round the mantissa if it is shrinking.
    WideBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
        switch (round_mode) {
            case Rounding_Mode::RoundToNearestEven:
              rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
              break;
            case Rounding_Mode::RoundToNearestOdd:
              rounded_from_bits = RoundBitsToNearestOdd(from_bits, -kDigitShift);
              break;
            case Rounding_Mode::RoundTowardsZero:
              rounded_from_bits = RoundBitsTowardsZero(from_bits, -kDigitShift);
              break;
            case Rounding_Mode::RoundAwayFromZero:
              rounded_from_bits = RoundBitsAwayFromZero(from_bits, -kDigitShift);
              break;
            case Rounding_Mode::StochasticRounding:
              rounded_from_bits = Stochastic_Round(from_bits, -kDigitShift, get_stochastic_length_v<To>);
              break;
          }
          
      
      // Zero-out tail bits.
      rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
    }

    // Re-bias the exponent.
    rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                         << kFromMantissaBits;

    ToBits bits;
    // Check for overflows by aligning the significands. We always align the
    // narrower significand to the wider significand.
    const WideBits kToHighestRep =
        std::bit_cast<ToBits>(std::numeric_limits<To>::max());
    WideBits aligned_highest{kToHighestRep};
    if constexpr (kDigitShift < 0) {
      aligned_highest <<= -kDigitShift;
      // Shift down, all dropped bits should already be zero.
      bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
    } else if constexpr (kDigitShift >= 0) {
      // Shift up, inserting zeros in the newly created digits.
      rounded_from_bits <<= kDigitShift;
      bits = ToBits{static_cast<ToBits>(rounded_from_bits)};
    }

    To to = std::bit_cast<To>(bits);
    // `From` supports larger values than `To`, we may overflow.
    if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                 std::numeric_limits<To>::digits) <
                  std::make_pair(std::numeric_limits<From>::max_exponent,
                                 std::numeric_limits<From>::digits)) {
        
      if (rounded_from_bits > aligned_highest) {
        // Overflowed values map to highest or infinity depending on kSaturate.
        if(std::numeric_limits<To>::has_infinity) {
          to = from_sign_bit ? -std::numeric_limits<To>::infinity()
                             : std::numeric_limits<To>::infinity();
        } else {
          to = from_sign_bit ? -std::numeric_limits<To>::max()
                             : std::numeric_limits<To>::max();
        }
      }
    }
    // Insert sign bit.
    return from_sign_bit ? -to : to;
  }

  
};



template <typename Derived, typename UnderlyingType>
template <typename From>
 Derived lo_float_base<Derived, UnderlyingType>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived>::run(from, Derived::rounding_mode);
}

template <typename Derived, typename UnderlyingType>
template <typename To>
 To lo_float_base<Derived, UnderlyingType>::ConvertTo(const Derived& from) {

  return ConvertImpl<Derived, To>::run(from, Derived::rounding_mode);

}






    } //namespace lo_float_internal

    template<FloatingPointParams Fp>
    using Templated_Float = lo_float_internal::Templated_Float<Fp>;


}   //namespace lo_float




        

#endif //FLOAT_6_4
  