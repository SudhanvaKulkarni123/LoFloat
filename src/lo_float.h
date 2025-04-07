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
#include  "eigen/Eigen/Core"  

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


  static std::mt19937 mt(static_cast<int>(time(nullptr)));

  //global function to set seed
  void set_seed(int a) {
    mt.seed(a);
  }
  
  
 // 6-bit floats
// First two classes had no template params, now they each get the Rounding_Mode template:
template <Rounding_Mode rm>
class float6_e3m2;

template <Rounding_Mode rm>
class float6_e2m3;


template<int p, Rounding_Mode rm>
class float6_p;

// 8-bit data types
template<Rounding_Mode rm>
class float8_e4m3fn;

template<Rounding_Mode rm>
class float8_e4m3fnuz;

template<Rounding_Mode rm>
class float8_e4m3b11fnuz;

template<Rounding_Mode rm>
class float8_e5m2;

template<Rounding_Mode rm>
class float8_e5m2fnuz;

// Previously: template<int p> class float8_ieee_p;
// Now: template<int p, Rounding_Mode rm> class float8_ieee_p;
template<int p, Rounding_Mode rm>
class float8_ieee_p;

// 4-bit types
template<Rounding_Mode rm>
class float4_e2m1;

template<int p, Rounding_Mode rm>
class float4_p;



template<typename Derived, typename UnderlyingType = uint8_t>
class lo_float_base {
protected:
    struct ConstructFromRepTag {};

    // "Core" constructor storing rep_ in the base
    constexpr lo_float_base(uint8_t rep, ConstructFromRepTag)
        : rep_(rep)
    {}

    // CRTP friend declarations
    template<typename T> friend class float4_base;
    template<typename T> friend class float6_base;
    template<typename T> friend class float8_base;
    template<typename T, typename V> friend class VarFloat;

public:
    constexpr lo_float_base() : rep_(0) {}

    constexpr uint8_t rep() const {
        return rep_;
    }

    // Templated constructor
    template <typename T,
              typename EnableIf = std::enable_if_t<std::is_arithmetic_v<T>>>
    explicit EIGEN_DEVICE_FUNC lo_float_base(T f)
        : lo_float_base(ConvertFrom(static_cast<float>(f)).rep(),
                        ConstructFromRepTag{}) {}

    explicit EIGEN_DEVICE_FUNC lo_float_base(double f64)
        : lo_float_base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}

    explicit EIGEN_DEVICE_FUNC lo_float_base(float f32)
        : lo_float_base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}

    explicit EIGEN_DEVICE_FUNC lo_float_base(Eigen::bfloat16 bf16)
        : lo_float_base(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}

    explicit EIGEN_DEVICE_FUNC lo_float_base(Eigen::half f16)
        : lo_float_base(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

    // CRTP helpers
    constexpr const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
    constexpr Derived& derived() {
        return *static_cast<Derived*>(this);
    }

    static constexpr Derived FromRep(uint8_t rep) {
        return Derived(rep, ConstructFromRepTag{});
    }

    // -------------------------------------------
    // Declarations for ConvertFrom / ConvertTo
    // -------------------------------------------
    template <bool kSaturate = false, bool kTruncate = false, typename From>
    static inline EIGEN_DEVICE_FUNC Derived ConvertFrom(const From& from);

    template <typename To, bool kSaturate = false, bool kTruncate = false>
    static inline EIGEN_DEVICE_FUNC To ConvertTo(const Derived& from);

    template <bool kSaturate = false, bool kTruncate = false>
    static inline EIGEN_DEVICE_FUNC double ConvertTo(const Derived& from);

    template <bool kSaturate = false, bool kTruncate = false>
    static inline EIGEN_DEVICE_FUNC Derived ConvertFrom(const double& from);


    template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
        explicit EIGEN_DEVICE_FUNC operator T() const {
            return static_cast<T>(static_cast<float>(derived()));
        }
        explicit EIGEN_DEVICE_FUNC operator double() const {
            return ConvertTo<double>(derived());
        }
        explicit EIGEN_DEVICE_FUNC operator float() const {
            return ConvertTo<float>(derived());
        }
        explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
            return ConvertTo<Eigen::bfloat16>(derived());
        }
        explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
            return ConvertTo<Eigen::half>(derived());
        }

        explicit EIGEN_DEVICE_FUNC operator bool() const {
            return (rep() & 0x7F) != 0;
        }

    // -------------------------------------------
    // Example arithmetic operators
    // -------------------------------------------
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
    operator+(const Derived& other) const {
        return Derived{double{derived()} + double{other}};
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
    operator=(const Derived& other) const {
        return Derived{double{other}};
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
    operator-(const Derived& other) const {
        return Derived{double{derived()} - double{other}};
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
    operator*(const Derived& other) const {
        return Derived{double{derived()} * double{other}};
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
    operator/(const Derived& other) const {
        return Derived{double{derived()} / double{other}};
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
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
        const T& other) const {
        return Compare(derived(), other) == Ordering::kLess;
    }

    template<typename T>
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
        const T& other) const {
        return Compare(derived(), other) <= Ordering::kEquivalent;
    }

    template<typename T>
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
        const T& other) const {
        return Compare(derived(), other) == Ordering::kGreater;
    }

    template<typename T>
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
        const T& other) const {
        auto ordering = Compare(derived(), other);
        return ordering == kGreater || ordering == kEquivalent;
    }
    
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator+=(
        const Derived& other) {
        derived() = derived() + other;
        return derived();
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator-=(
        const Derived& other) {
        derived() = derived() - other;
        return derived();
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator*=(
        const Derived& other) {
        derived() = derived() * other;
        return derived();
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator/=(
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

    // Helper for compare:
    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::pair<UnderlyingType, UnderlyingType>
    SignAndMagnitude(Derived x) {
        const UnderlyingType x_abs_bits =
            Eigen::numext::bit_cast<UnderlyingType>(Eigen::numext::abs(x));
        const UnderlyingType x_bits = Eigen::numext::bit_cast<UnderlyingType>(x);
        const UnderlyingType x_sign = x_bits ^ x_abs_bits;
        return {x_sign, x_abs_bits};
    }

    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Signed_type
    SignAndMagnitudeToTwosComplement(UnderlyingType sign, UnderlyingType magnitude) {
        return magnitude ^ (static_cast<Signed_type>(sign) < 0 ? -1 : 0);
    }

    // Compare function
    template<typename T>
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC friend constexpr Ordering Compare(
        const Derived& lhs, const T& rhs) {
        if (Eigen::numext::isnan(lhs) || Eigen::numext::isnan(rhs)) {
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
}; //lo_float_base


//helper template to pick storage format
template<int Len>
using Base_repr_select = std::conditional_t<(Len <= 8), uint8_t, std::conditional_t<(Len <= 16), uint16_t, uint32_t>>;


///hypothetical for varfloat
template<typename Derived, FloatingPointParams Fp>
class Var_lo_float : public lo_float_base<Derived, Base_repr_select<Fp.bitwidth>> {
    private :
    using UType = Base_repr_select<Fp.bitwidth>;
    using Base  = lo_float_base<Derived, UType>;

    friend class lo_float_base<Derived, UType>;

    // Inherit constructors from lo_float_base
    using Base::Base;

    using SType = typename std::make_signed<UType>::type;


    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC SType
    SignAndMagnitudeToTwosComplement(UType sign, UType magnitude) {
            return magnitude ^ (static_cast<SType>(sign << Fp.Len) < 0 ? -1 : 0);
    }

    protected:
        using typename Base::ConstructFromRepTag;

        constexpr Var_lo_float(UType rep, ConstructFromRepTag tag)
            : Base(rep, tag)
        {}

    public:

        explicit EIGEN_DEVICE_FUNC operator bool() const {
            return (this->rep() & ((1 << Fp.bitwidth) - 1)) != 0;
        }
        constexpr Derived operator-() const {
            return Base::FromRep(static_cast<UType>(this->rep() ^ (1 << Fp.bitwidth)));
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

        static constexpr  int bias = Fp.bias;

        static constexpr int p = Fp.mantissa_bits;

};

template<FloatingPointParams Fp>
class Templated_Float : public Var_lo_float<Templated_Float<Fp>, Fp> {
 private:
 using Base = Var_lo_float<Templated_Float<Fp>, Fp>;

 friend class Var_lo_float<Templated_Float<Fp>, Fp>;

 using Base::Base;

};


//For f8_e4m3fn, I would define the struct as follows-
//FPParams(8, 3, 7, )

        
template<Rounding_Mode round_mode>
class float8_e4m3fn : public float8_base<float8_e4m3fn<round_mode>> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by 0bS111'1111.
  // The "fn" suffix is for consistency with the corresponding LLVM/MLIR type,
  // signaling this type is not consistent with IEEE-754.  The "f" indicates
  // it is finite values only. The "n" indicates it includes NaNs, but only
  // at the outer range.
 private:
 using float8_e4m3fn_type = float8_e4m3fn<round_mode>;
  using Base = float8_base<float8_e4m3fn_type>;
  friend class float8_base<float8_e4m3fn_type>;
  using Base::Base;

 public:
 template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e5m2<rm2>& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
    template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e4m3b11fnuz<rm2>& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };
  
  template<Rounding_Mode rm2>
  constexpr bool operator==(const float8_e4m3fn<rm2>& other) const {
    return Compare(this->derived(), other) == Ordering::kEquivalent;
  }
  
  static constexpr Rounding_Mode rounding_mode = round_mode;
  

  
      
};
template<Rounding_Mode round_mode>
class float8_e4m3b11fnuz : public float8_base<float8_e4m3b11fnuz<round_mode>> {
 private:
  // CRTP "self" alias:
  using float8_e4m3b11fnuz_type = float8_e4m3b11fnuz<round_mode>;
  // Base class depends on this instantiation
  using Base = float8_base<float8_e4m3b11fnuz_type>;
  friend class float8_base<float8_e4m3b11fnuz_type>;
  using Base::Base;  // Inherit constructors

 public:
  // Example "convert from" constructors (parametric in source's rounding mode):
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3b11fnuz(const float8_e5m2<rm2>& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3b11fnuz(const float8_e5m2fnuz<rm2>& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3b11fnuz(const float8_e4m3fn<rm2>& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3b11fnuz(const float8_e4m3fnuz<rm2>& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}

  // Possibly more constructors for other float8 types...
  // template <int p, Rounding_Mode rm2>
  // EIGEN_DEVICE_FUNC explicit float8_e4m3b11fnuz(const float8_ieee_p<p, rm2>& f8)
  //     : float8_e4m3b11fnuz(ConvertFrom(f8)) {}

  // Match the style of the first class: store the round_mode as a static constexpr
  static constexpr Rounding_Mode rounding_mode = round_mode;
};

// Legacy name used in XLA (TODO(jewillco): remove).
template<Rounding_Mode rm>
using float8_e4m3b11 = float8_e4m3b11fnuz<rm>;

template<Rounding_Mode round_mode>
class float8_e4m3fnuz : public float8_base<float8_e4m3fnuz<round_mode>> {
  // 8-bit floating point with 3 bit mantissa.
  //
  // An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
  // mantissa. The suffix "fnuz" is consistent with LLVM/MLIR naming and is
  // derived from the differences to IEEE floating point conventions. `F` is
  // for "finite" (no infinities), `N` for with special NaN encoding, `UZ` for
  // unsigned zero.
  //
  // This type has the following characteristics:
  // * bit encoding: S1E4M3 - `0bSEEEEMMM`
  // * exponent bias: 8
  // * infinities: Not supported
  // * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits
  // set to all 0s - `0b10000000`
  // * denormals when exponent is 0
 private:
  using float8_e4m3fnuz_type = float8_e4m3fnuz<round_mode>;
  using Base = float8_base<float8_e4m3fnuz_type>;
  friend class float8_base<float8_e4m3fnuz_type>;
  using Base::Base;

 public:
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3fnuz(const float8_e5m2<rm2>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3fnuz(const float8_e5m2fnuz<rm2>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3fnuz(const float8_e4m3b11fnuz<rm2>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3fnuz(const float8_e4m3fn<rm2>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  template <int p, Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e4m3fnuz(const float8_ieee_p<p, rm2>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;
};

template<Rounding_Mode round_mode>
class float8_e5m2 : public float8_base<float8_e5m2<round_mode>> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using float8_e5m2_type = float8_e5m2<round_mode>;
  using Base = float8_base<float8_e5m2_type>;
  friend class float8_base<float8_e5m2_type>;
  using Base::Base;

 public:
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e5m2(const float8_e4m3fn<rm2>& f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e5m2(const float8_e4m3fnuz<rm2>& f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e5m2(const float8_e4m3b11fnuz<rm2>& f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  template<Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e5m2(const float8_e5m2fnuz<rm2>& f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  template <int p, Rounding_Mode rm2>
  EIGEN_DEVICE_FUNC explicit float8_e5m2(const float8_ieee_p<p, rm2>& f8)
      : float8_e5m2(ConvertFrom(f8)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;
};



////////////////////////////////////////////////////////////////////////
// float8_e5m2fnuz
////////////////////////////////////////////////////////////////////////

template<Rounding_Mode round_mode>
class float8_e5m2fnuz : public float8_base<float8_e5m2fnuz<round_mode>> {
  // 8-bit floating point with 2 bit mantissa.
  //
  // An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
  // mantissa. The suffix "fnuz" is consistent with LLVM/MLIR naming and is
  // derived from the differences to IEEE floating point conventions. `F` is
  // for "finite" (no infinities), `N` for with special NaN encoding, `UZ` for
  // unsigned zero.
  //
  // This type has the following characteristics:
  // * bit encoding: S1E5M2 - `0bSEEEEEMM`
  // * exponent bias: 16
  // * infinities: Not supported
  // * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits
  // set to all 0s - `0b10000000`
  // * denormals when exponent is 0
 private:
  using float8_e5m2fnuz_type = float8_e5m2fnuz<round_mode>;
  using Base = float8_base<float8_e5m2fnuz_type>;
  friend class float8_base<float8_e5m2fnuz_type>;
  using Base::Base;

 public:
  // Constructors accepting various other float8 types (templated by their rounding modes)
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e5m2<rm2>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3b11fnuz<rm2>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3fn<rm2>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3fnuz<rm2>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}

  template<int p, Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_ieee_p<p, rm2>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;
};

////////////////////////////////////////////////////////////////////////
// float8_ieee_p
////////////////////////////////////////////////////////////////////////

template <int p, Rounding_Mode round_mode>
class float8_ieee_p : public float8_base<float8_ieee_p<p, round_mode>> {
  // IEEE P3109 WG 8-bit floating point with p bits of precision.
  //
  // An 8-bit floating point type with 1 sign bit,
  // 8-p bits exponent, and p-1 bits mantissa.
  //
  // This type has the following characteristics:
  // * bit encoding: S1E<8-p>M<p-1>
  // * exponent bias: 2^(7-p)
  // * infinities: +Inf at 0x7f, -Inf at 0xff
  // * NaNs: Single NaN at `0b10000000`
  // * denormals when exponent is 0

 private:
  using this_t = float8_ieee_p<p, round_mode>;
  using Base = float8_base<this_t>;
  friend class float8_base<this_t>;
  using Base::Base;

 public:
  // Constructors accepting various other float8 types:
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e5m2<rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e5m2fnuz<rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3b11fnuz<rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3fnuz<rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3fn<rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<int q, Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_ieee_p<q, rm2>& f8)
      : float8_ieee_p(ConvertFrom(f8)) {}

  template<int q, Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float6_p<q, rm2>& f6)
      : float8_ieee_p(ConvertFrom(f6)) {}

  template<int q, Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float4_p<q, rm2>& f4)
      : float8_ieee_p(ConvertFrom(f4)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;

  constexpr float8_ieee_p<p, round_mode> operator-() const {
    // TODO: use isnan()
    if ((this->rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  float8_ieee_p<p, round_mode> operator-(const float8_ieee_p<p, round_mode>& other) const {
    return Base::operator-(other);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  constexpr bool operator==(const float8_ieee_p<p, round_mode>& other) const {
    return Compare(this->derived(), other) == kEquivalent;
  }

  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return this->rep() != 0;
  }
};



////////////////////////////////////////////////////////////////////////
// float6_e3m2
////////////////////////////////////////////////////////////////////////
template<Rounding_Mode round_mode>
class float6_e3m2 : public float6_base<float6_e3m2<round_mode>> {
    // 1S3E2M, bias = 3, saturated rounding, no Inf or NaN
 private:
  using float6_e3m2_type = float6_e3m2<round_mode>;
  using Base = float6_base<float6_e3m2_type>;
  friend class float6_base<float6_e3m2_type>;
  using Base::Base;

 public:
  // Make the constructor accept float6_e2m3 with any rounding_mode:
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float6_e3m2(const float6_e2m3<rm2>& f6)
      : float6_e3m2(ConvertFrom(f6)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;
};

////////////////////////////////////////////////////////////////////////
// float6_e2m3
////////////////////////////////////////////////////////////////////////
template<Rounding_Mode round_mode>
class float6_e2m3 : public float6_base<float6_e2m3<round_mode>> {
    // 1S3E2M, bias = 1, saturated rounding, no Inf or NaN
 private:
  using float6_e2m3_type = float6_e2m3<round_mode>;
  using Base = float6_base<float6_e2m3_type>;
  friend class float6_base<float6_e2m3_type>;
  using Base::Base;

 public:
  // Make the constructor accept float6_e3m2 with any rounding_mode:
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float6_e2m3(const float6_e3m2<rm2>& f6)
      : float6_e2m3(ConvertFrom(f6)) {}

  static constexpr Rounding_Mode rounding_mode = round_mode;
};

////////////////////////////////////////////////////////////////////////
// float6_p
////////////////////////////////////////////////////////////////////////
template<int p, Rounding_Mode round_mode>
class float6_p : public float6_base<float6_p<p, round_mode>> {
    // 1S(6-p)E(p-1)M, bias = 2^(6 - p) - 1 , Inf at 0x3F and 0x1F. NaN at 0x20
 private:
  using float6_p_type = float6_p<p, round_mode>;
  using Base = float6_base<float6_p_type>;
  friend class float6_base<float6_p_type>;
  using Base::Base;

 public:
  // Templated ctors for cross-type construction:
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float6_p(const float6_e3m2<rm2>& f6)
      : float6_p(this->ConvertFrom(f6)) {}
  template<Rounding_Mode rm2>
  explicit EIGEN_DEVICE_FUNC float6_p(const float6_e2m3<rm2>& f6)
      : float6_p(this->ConvertFrom(f6)) {}

  constexpr float6_p<p, round_mode> operator-() const {
    // TODO: use isnan()
    if ((this->rep() & 0x1f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  float6_p<p, round_mode> operator-(const float6_p<p, round_mode>& other) const {
    return Base::operator-(other);
  }

  static constexpr Rounding_Mode rounding_mode = round_mode;
};

////////////////////////////////////////////////////////////////////////
// float4_e2m1
////////////////////////////////////////////////////////////////////////
template<Rounding_Mode round_mode>
class float4_e2m1 : public float4_base<float4_e2m1<round_mode>> {
 private:
  using float4_e2m1_type = float4_e2m1<round_mode>;
  using Base = float4_base<float4_e2m1_type>;
  friend class float4_base<float4_e2m1_type>;
  using Base::Base;

  // (No public constructors given here in the snippet.)

 public:
  static constexpr Rounding_Mode rounding_mode = round_mode;
};

////////////////////////////////////////////////////////////////////////
// float4_p
////////////////////////////////////////////////////////////////////////
template<int p, Rounding_Mode round_mode>
class float4_p : public float4_base<float4_p<p, round_mode>> {
    // 1S2E1M, bias = 2^(4 - p) - 1, Inf at 0xF and 0x7, NaN at 0x4
 private:
  using float4_p_type = float4_p<p, round_mode>;
  using Base = float4_base<float4_p_type>;
  friend class float4_base<float4_p_type>;
  using Base::Base;

 public:
  float4_p<p, round_mode> operator-(const float4_p<p, round_mode>& other) const {
    return Base::operator-(other);
  }

  constexpr float4_p<p, round_mode> operator-() const {
    // TODO: use isnan()
    if ((this->rep() & 0x7) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  static constexpr Rounding_Mode rounding_mode = round_mode;
};


constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }

constexpr double ConstexprCeil(double x) {
  constexpr double kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<double>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;
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
      -0.013788284485633295
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 1] +
                                         max_exponent * kLog10Of2));
}


                // Structures for use in specializing std::numeric_limits.
        struct numeric_limits_float8_base {
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const bool is_specialized = true;
        static inline constexpr const bool is_signed = true;
        static inline constexpr const bool is_integer = false;
        static inline constexpr const bool is_exact = false;
        static inline constexpr const bool has_quiet_NaN = true;
        static inline constexpr const std::float_denorm_style has_denorm =
            std::denorm_present;
        static inline constexpr const bool has_denorm_loss = false;
        static inline constexpr const std::float_round_style round_style =
            std::round_indeterminate;
        static inline constexpr const bool is_bounded = true;
        static inline constexpr const bool is_modulo = false;
        static inline constexpr const int radix = std::numeric_limits<float>::radix;
        static inline constexpr const bool traps = std::numeric_limits<float>::traps;
        static inline constexpr const bool tinyness_before =
            std::numeric_limits<float>::tinyness_before;
        // NOLINTEND
        };

        
        struct numeric_limits_float8_e4m3fn : public numeric_limits_float8_base {
        public:
        static inline constexpr const int kExponentBias = 7;
        static inline constexpr const int kMantissaBits = 3;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = kMantissaBits + 1;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent =
            (0b1111 - 7) + 1;  // Extended format.
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false;
        static inline constexpr const bool has_infinity = true;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        template<Rounding_Mode rm>
        struct numeric_limits_f8_e4m3fn_rm : numeric_limits_float8_e4m3fn {
        // 1.0 * 2^(0b0001 - 7) = 1.0 * 2^-6 = 0.015625
        static constexpr float8_e4m3fn<rm> min() {
            return float8_e4m3fn<rm>::FromRep(0b0'0001 << kMantissaBits);
        }
        // -(1 + 0b110 * 2^-3) * 2^(0b1111 - 7) = -1.75 * 2^8 = 448
        static constexpr float8_e4m3fn<rm> lowest() {
            return float8_e4m3fn<rm>::FromRep(0b1'1111'110);
        }
        // (1 + 0b110 * 2^-3) * 2**(0b1111 - 7) = 1.75 * 2^8 = 448
        static constexpr float8_e4m3fn<rm> max() {
            return float8_e4m3fn<rm>::FromRep(0b0'1111'110);
        }
        // 1.0 * 2^-3 = 0.125
        static constexpr float8_e4m3fn<rm> epsilon() {
            return float8_e4m3fn<rm>::FromRep((-kMantissaBits + kExponentBias)
                                        << kMantissaBits);
        }
        // 1.0 * 2^-1 = 0.5
        static constexpr float8_e4m3fn<rm> round_error() {
            return float8_e4m3fn<rm>::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3fn<rm> infinity() {
            return float8_e4m3fn<rm>::FromRep(0b0'1111'111);
        }
        // NaN.
        static constexpr float8_e4m3fn<rm> quiet_NaN() {
            return float8_e4m3fn<rm>::FromRep(0b1'0000'000);
        }
        static constexpr float8_e4m3fn<rm> signaling_NaN() {
            return float8_e4m3fn<rm>::FromRep(0b1'0000'000);
        }
        // 1.0 * 2^(-7 - 3 + 1) = 1.0 * 2^-9 = 0.001953125
        static constexpr float8_e4m3fn<rm> denorm_min() {
            return float8_e4m3fn<rm>::FromRep(0b0'0000'001);
        }
        };

        struct numeric_limits_float8_e4m3b11fnuz : public numeric_limits_float8_base {
  
        static inline constexpr const int kExponentBias = 11;
        static inline constexpr const int kMantissaBits = 3;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = kMantissaBits + 1;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent =
            (0b1111 - kExponentBias) + 1;  // Extended format.
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false;
        static inline constexpr const bool has_infinity = false;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        template<Rounding_Mode rm> 
        struct numeric_limits_f8_e4m3b11fnuz_rm : numeric_limits_float8_e4m3b11fnuz {

        // 1.0 * 2^(0b0001 - 11) = 1.0 * 2^-10 = 0.0009765625
        static constexpr float8_e4m3b11fnuz<rm> min() {
            return float8_e4m3b11fnuz<rm>::FromRep(1 << kMantissaBits);
        }
        // -(1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = -1.875 * 2^4 = -30
        static constexpr float8_e4m3b11fnuz<rm> lowest() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b1'1111'111);
        }
        // (1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = 1.875 * 2^4 = 30
        static constexpr float8_e4m3b11fnuz<rm> max() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b0'1111'111);
        }
        // 1.0 * 2^-3 = 0.125
        static constexpr float8_e4m3b11fnuz<rm> epsilon() {
            return float8_e4m3b11fnuz<rm>::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
        }
        // 1.0 * 2^-1 = 0.5
        static constexpr float8_e4m3b11fnuz<rm> round_error() {
            return float8_e4m3b11fnuz<rm>::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3b11fnuz<rm> infinity() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b0'1111'111);
        }
        // NaN.
        static constexpr float8_e4m3b11fnuz<rm> quiet_NaN() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b1'0000'000);
        }
        static constexpr float8_e4m3b11fnuz<rm> signaling_NaN() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b1'0000'000);
        }
        // 1.0 * 2^(-11 - 3 + 1) = 1.0 * 2^-13 = 0.0001220703125
        static constexpr float8_e4m3b11fnuz<rm> denorm_min() {
            return float8_e4m3b11fnuz<rm>::FromRep(0b0'0000'001);
        }
        };

        struct numeric_limits_float8_e4m3fnuz : public numeric_limits_float8_base {

        static inline constexpr const int kExponentBias = 8;
        static inline constexpr const int kMantissaBits = 3;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = kMantissaBits + 1;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent =
            (0b1111 - kExponentBias) + 1;  // Extended format.
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false;
        static inline constexpr const bool has_infinity = false;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        template<Rounding_Mode rm>
        struct numeric_limits_f8_e4m3_fnuz_rm : numeric_limits_float8_e4m3fnuz {

        static constexpr float8_e4m3fnuz<rm> min() {
            return float8_e4m3fnuz<rm>::FromRep(0x08);
        }
        static constexpr float8_e4m3fnuz<rm> lowest() {
            return float8_e4m3fnuz<rm>::FromRep(0xFF);
        }
        static constexpr float8_e4m3fnuz<rm> max() {
            return float8_e4m3fnuz<rm>::FromRep(0x7F);
        }
        static constexpr float8_e4m3fnuz<rm> epsilon() {
            return float8_e4m3fnuz<rm>::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
        }
        static constexpr float8_e4m3fnuz<rm> round_error() {
            return float8_e4m3fnuz<rm>::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3fnuz<rm> infinity() {
            return float8_e4m3fnuz<rm>::FromRep(0x80);
        }  // NaN.
        static constexpr float8_e4m3fnuz<rm> quiet_NaN() {
            return float8_e4m3fnuz<rm>::FromRep(0x80);
        }
        static constexpr float8_e4m3fnuz<rm> signaling_NaN() {
            return float8_e4m3fnuz<rm>::FromRep(0x80);
        }
        static constexpr float8_e4m3fnuz<rm> denorm_min() {
            return float8_e4m3fnuz<rm>::FromRep(0x01);
        }
        };


        struct numeric_limits_float8_e5m2 : public numeric_limits_float8_base {

        static inline constexpr const int kExponentBias = 15;
        static inline constexpr const int kMantissaBits = 2;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = kMantissaBits + 1;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent = 0b11111 - kExponentBias;
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false;
        static inline constexpr const bool has_infinity = true;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        template<Rounding_Mode rm>
struct numeric_limits_f8_e5m2_rm : numeric_limits_float8_e5m2 {
  static constexpr float8_e5m2<rm> min() {
    return float8_e5m2<rm>::FromRep(1 << kMantissaBits);
  }
  static constexpr float8_e5m2<rm> lowest() {
    return float8_e5m2<rm>::FromRep(0b1'11110'11);
  }
  static constexpr float8_e5m2<rm> max() {
    return float8_e5m2<rm>::FromRep(0b0'11110'11);
  }
  static constexpr float8_e5m2<rm> epsilon() {
    return float8_e5m2<rm>::FromRep(
      (-kMantissaBits + kExponentBias) << kMantissaBits
    );
  }
  static constexpr float8_e5m2<rm> round_error() {
    return float8_e5m2<rm>::FromRep(
      (-1 + kExponentBias) << kMantissaBits
    );
  }
  static constexpr float8_e5m2<rm> infinity() {
    return float8_e5m2<rm>::FromRep(0b0'11111'00);
  }
  static constexpr float8_e5m2<rm> quiet_NaN() {
    return float8_e5m2<rm>::FromRep(0b0'11111'10);
  }
  static constexpr float8_e5m2<rm> signaling_NaN() {
    return float8_e5m2<rm>::FromRep(0b0'11111'01);
  }
  static constexpr float8_e5m2<rm> denorm_min() {
    return float8_e5m2<rm>::FromRep(0b0'00000'01);
  }
};


        struct numeric_limits_float8_e5m2fnuz : public numeric_limits_float8_base {

        static inline constexpr const int kExponentBias = 16;
        static inline constexpr const int kMantissaBits = 2;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = kMantissaBits + 1;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent =
            (0b11111 - kExponentBias) + 1;
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false;
        static inline constexpr const bool has_infinity = false;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        template<Rounding_Mode rm>
        struct numeric_limits_f8_e5m2fnuz_rm : numeric_limits_float8_e5m2fnuz {
          static constexpr float8_e5m2fnuz<rm> min() {
            return float8_e5m2fnuz<rm>::FromRep(0x04);
          }
          static constexpr float8_e5m2fnuz<rm> lowest() {
            return float8_e5m2fnuz<rm>::FromRep(0xFF);
          }
          static constexpr float8_e5m2fnuz<rm> max() {
            return float8_e5m2fnuz<rm>::FromRep(0x7F);
          }
          static constexpr float8_e5m2fnuz<rm> epsilon() {
            return float8_e5m2fnuz<rm>::FromRep(
              (-kMantissaBits + kExponentBias) << kMantissaBits
            );
          }
          static constexpr float8_e5m2fnuz<rm> round_error() {
            return float8_e5m2fnuz<rm>::FromRep(
              (-1 + kExponentBias) << kMantissaBits
            );
          }
          static constexpr float8_e5m2fnuz<rm> infinity() {
            return float8_e5m2fnuz<rm>::FromRep(0x80);
          }
          static constexpr float8_e5m2fnuz<rm> quiet_NaN() {
            return float8_e5m2fnuz<rm>::FromRep(0x80);
          }
          static constexpr float8_e5m2fnuz<rm> signaling_NaN() {
            return float8_e5m2fnuz<rm>::FromRep(0x80);
          }
          static constexpr float8_e5m2fnuz<rm> denorm_min() {
            return float8_e5m2fnuz<rm>::FromRep(0x01);
          }
        };

        template <int p>
        struct numeric_limits_float8_ieee_p : public numeric_limits_float8_base {

        static inline constexpr const int kExponentBias = (1 << (7-p));
        static inline constexpr const int kMantissaBits = p - 1;

        public:
        // NOLINTBEGIN: these names must match std::numeric_limits.
        static inline constexpr const int digits = p;
        static inline constexpr const int digits10 = Digits10FromDigits(digits);
        static inline constexpr const int max_digits10 =
            MaxDigits10FromDigits(digits);
        static inline constexpr const int min_exponent = (1 - kExponentBias);
        static inline constexpr const int min_exponent10 =
            MinExponent10FromMinExponent(min_exponent);
        static inline constexpr const int max_exponent = kExponentBias - 1;
        static inline constexpr const int max_exponent10 =
            MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
        static inline constexpr const bool is_iec559 = false; // TODO
        static inline constexpr const bool has_infinity = true;
        static inline constexpr const bool has_signaling_NaN = false;
        // NOLINTEND
        };

        
template<int p, Rounding_Mode rm>
struct numeric_limits_f8_ieee_p_rm : numeric_limits_float8_ieee_p<p> {
  static constexpr float8_ieee_p<p, rm> min() {
    return float8_ieee_p<p, rm>::FromRep(1 << (p - 1));
  }
  static constexpr float8_ieee_p<p, rm> lowest() {
    return float8_ieee_p<p, rm>::FromRep(0xfe);
  }
  static constexpr float8_ieee_p<p, rm> max() {
    return float8_ieee_p<p, rm>::FromRep(0x7e);
  }
  static constexpr float8_ieee_p<p, rm> epsilon() {
    if constexpr (p < 5) {
      constexpr int expeps =
        ((- (p - 1)) + (1 << (7 - p))) << (p - 1);
      return float8_ieee_p<p, rm>::FromRep(static_cast<uint8_t>(expeps));
    }
    return float8_ieee_p<p, rm>::FromRep(
      static_cast<uint8_t>(1 << ((1 << (7 - p)) - 1))
    );
  }
  static constexpr float8_ieee_p<p, rm> round_error() {
    return float8_ieee_p<p, rm>::FromRep(
      ((-1 + (1 << (7 - p))) << (p - 1))
    );
  }
  static constexpr float8_ieee_p<p, rm> infinity() {
    return float8_ieee_p<p, rm>::FromRep(0x7f);
  }
  static constexpr float8_ieee_p<p, rm> quiet_NaN() {
    return float8_ieee_p<p, rm>::FromRep(0x80);
  }
  static constexpr float8_ieee_p<p, rm> signaling_NaN() {
    return float8_ieee_p<p, rm>::FromRep(0x80);
  }
  static constexpr float8_ieee_p<p, rm> denorm_min() {
    return float8_ieee_p<p, rm>::FromRep(0x01);
  }
};




        struct numeric_limits_float6_base {
            static inline constexpr const bool is_specialized = true;
            static inline constexpr const bool is_signed = true;
            static inline constexpr const bool is_integer = false;
            static inline constexpr const bool is_exact = false;
            static inline constexpr const bool has_quiet_NaN = true;
            static inline constexpr const bool has_signaling_NaN = false;
            static inline constexpr const bool has_denorm = true;
            static inline constexpr const bool has_denorm_loss = false;
            static inline constexpr const bool round_style = std::round_to_nearest;
            static inline constexpr const bool is_iec559 = false;
            static inline constexpr const int radix = std::numeric_limits<float>::radix;
            static inline constexpr const bool traps = std::numeric_limits<float>::traps;
            static inline constexpr const bool tinyness_before = std::numeric_limits<float>::tinyness_before;
        };

        struct numeric_limits_float6_e3m2 : public numeric_limits_float6_base {

            static inline constexpr const int kExponentBias = 3;
            static inline constexpr const int kMantissaBits = 2;
            public:
            // NOLINTBEGIN: these names must match std::numeric_limits.
            static inline constexpr const int digits = kMantissaBits + 1;
            static inline constexpr const int digits10 = Digits10FromDigits(digits);
            static inline constexpr const int max_digits10 =
                MaxDigits10FromDigits(digits);
            static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
            static inline constexpr const int min_exponent10 =
                MinExponent10FromMinExponent(min_exponent);
            static inline constexpr const int max_exponent = 0b111 - kExponentBias;
            static inline constexpr const int max_exponent10 =
                MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static inline constexpr const bool is_iec559 = false;
            static inline constexpr const bool has_infinity = false;
            static inline constexpr const bool has_signaling_NaN = false;
            // NOLINTEND
        };


template<Rounding_Mode rm>
struct numeric_limits_f6_e3m2_rm : numeric_limits_float6_e3m2 {
  static constexpr float6_e3m2<rm> min() {
    return float6_e3m2<rm>::FromRep(1 << kMantissaBits);
  }
  static constexpr float6_e3m2<rm> lowest() {
    return float6_e3m2<rm>::FromRep(0b1'111'11);
  }
  static constexpr float6_e3m2<rm> max() {
    return float6_e3m2<rm>::FromRep(0b0'111'11);
  }
  static constexpr float6_e3m2<rm> epsilon() {
    return float6_e3m2<rm>::FromRep(
      ((-kMantissaBits) + kExponentBias) << kMantissaBits
    );
  }
  static constexpr float6_e3m2<rm> round_error() {
    return float6_e3m2<rm>::FromRep(
      ((-1 + kExponentBias) << kMantissaBits)
    );
  }
  static constexpr float6_e3m2<rm> infinity() {
    return float6_e3m2<rm>::FromRep(0);
  }
  static constexpr float6_e3m2<rm> quiet_NaN() {
    return float6_e3m2<rm>::FromRep(0b0'11111'10);
  }
  static constexpr float6_e3m2<rm> signaling_NaN() {
    return float6_e3m2<rm>::FromRep(0b0'111'01);
  }
  static constexpr float6_e3m2<rm> denorm_min() {
    return float6_e3m2<rm>::FromRep(0b0'000'01);
  }
};
        struct numeric_limits_float6_e2m3 : public numeric_limits_float6_base {

            static inline constexpr const int kExponentBias = 1;
            static inline constexpr const int kMantissaBits = 3;
            public:
            // NOLINTBEGIN: these names must match std::numeric_limits.
            static inline constexpr const int digits = kMantissaBits + 1;
            static inline constexpr const int digits10 = Digits10FromDigits(digits);
            static inline constexpr const int max_digits10 =
                MaxDigits10FromDigits(digits);
            static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
            static inline constexpr const int min_exponent10 =
                MinExponent10FromMinExponent(min_exponent);
            static inline constexpr const int max_exponent = 0b11 - kExponentBias;
            static inline constexpr const int max_exponent10 =
                MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static inline constexpr const bool is_iec559 = false;
            static inline constexpr const bool has_infinity = false;
            static inline constexpr const bool has_signaling_NaN = false;
            // NOLINTEND
        };

         
template<Rounding_Mode rm>
struct numeric_limits_f6_e2m3_rm : numeric_limits_float6_e2m3 {
  static constexpr float6_e2m3<rm> min() {
    return float6_e2m3<rm>::FromRep(1 << kMantissaBits);
  }
  static constexpr float6_e2m3<rm> lowest() {
    return float6_e2m3<rm>::FromRep(0b1'11'111);
  }
  static constexpr float6_e2m3<rm> max() {
    return float6_e2m3<rm>::FromRep(0b0'11'111);
  }
  static constexpr float6_e2m3<rm> epsilon() {
    return float6_e2m3<rm>::FromRep(
      ((-kMantissaBits) + kExponentBias) << kMantissaBits
    );
  }
  static constexpr float6_e2m3<rm> round_error() {
    return float6_e2m3<rm>::FromRep(
      ((-1 + kExponentBias) << kMantissaBits)
    );
  }
  static constexpr float6_e2m3<rm> infinity() {
    return float6_e2m3<rm>::FromRep(0);
  }
  static constexpr float6_e2m3<rm> quiet_NaN() {
    return float6_e2m3<rm>::FromRep(0b0'11'110);
  }
  static constexpr float6_e2m3<rm> signaling_NaN() {
    return float6_e2m3<rm>::FromRep(0b0'11'001);
  }
  static constexpr float6_e2m3<rm> denorm_min() {
    return float6_e2m3<rm>::FromRep(0b0'00'001);
  }
};


        template<int p>
        struct numeric_limits_float6_p :  public numeric_limits_float6_base {
            static inline constexpr const int kExponentBias = (1 << (5 - p));
            static inline constexpr const int kMantissaBits = p - 1;

            public:
            // NOLINTBEGIN: these names must match std::numeric_limits.
            static inline constexpr const int digits = p;
            static inline constexpr const int digits10 = Digits10FromDigits(digits);
            static inline constexpr const int max_digits10 =
                MaxDigits10FromDigits(digits);
            static inline constexpr const int min_exponent = (1 - kExponentBias);
            static inline constexpr const int min_exponent10 =
                MinExponent10FromMinExponent(min_exponent);
            static inline constexpr const int max_exponent = kExponentBias - 1;
            static inline constexpr const int max_exponent10 =
                MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static inline constexpr const bool is_iec559 = false; // TODO
            static inline constexpr const bool has_infinity = true;
            static inline constexpr const bool has_signaling_NaN = false;
            // NOLINTEND
        };
           

      

template<int p, Rounding_Mode rm>
struct numeric_limits_f6_p_rm : numeric_limits_float6_p<p> {
  static constexpr float6_p<p, rm> min() {
    return float6_p<p, rm>::FromRep(1 << (p - 1));
  }
  static constexpr float6_p<p, rm> lowest() {
    return float6_p<p, rm>::FromRep(0x3e);
  }
  static constexpr float6_p<p, rm> max() {
    return float6_p<p, rm>::FromRep(0x1e);
  }
  static constexpr float6_p<p, rm> epsilon() {
    if constexpr (p < 4) {
      constexpr int expeps = (
        (- (p - 1) + (1 << (5 - p))) << (p - 1)
      );
      return float6_p<p, rm>::FromRep(static_cast<uint8_t>(expeps));
    }
    return float6_p<p, rm>::FromRep(
      static_cast<uint8_t>(1 << ((1 << (5 - p)) - 1))
    );
  }
  static constexpr float6_p<p, rm> round_error() {
    return float6_p<p, rm>::FromRep(
      ((-1 + (1 << (5 - p))) << (p - 1))
    );
  }
  static constexpr float6_p<p, rm> infinity() {
    return float6_p<p, rm>::FromRep(0x1f);
  }
  static constexpr float6_p<p, rm> quiet_NaN() {
    return float6_p<p, rm>::FromRep(0x20);
  }
  static constexpr float6_p<p, rm> signaling_NaN() {
    return float6_p<p, rm>::FromRep(0x20);
  }
  static constexpr float6_p<p, rm> denorm_min() {
    return float6_p<p, rm>::FromRep(0x01);
  }
};


        struct numeric_limits_float4_base {
            static inline constexpr const bool is_specialized = true;
            static inline constexpr const bool is_signed = true;
            static inline constexpr const bool is_integer = false;
            static inline constexpr const bool is_exact = false;
            static inline constexpr const bool has_quiet_NaN = true;
            static inline constexpr const bool has_signaling_NaN = false;
            static inline constexpr const bool has_denorm = true;
            static inline constexpr const bool has_denorm_loss = false;
            static inline constexpr const bool round_style = std::round_to_nearest;
            static inline constexpr const bool is_iec559 = false;
            static inline constexpr const int radix = std::numeric_limits<float>::radix;
            static inline constexpr const bool traps = std::numeric_limits<float>::traps;
            static inline constexpr const bool tinyness_before = std::numeric_limits<float>::tinyness_before;
       
        };

        struct numeric_limits_float4_e2m1 : public numeric_limits_float4_base {

   
            static inline constexpr const int kExponentBias = 1;
            static inline constexpr const int kMantissaBits = 1;

        public:
            // NOLINTBEGIN: these names must match std::numeric_limits.
            static inline constexpr const int digits = kMantissaBits + 1; // 2
            static inline constexpr const int digits10 = Digits10FromDigits(digits);
            static inline constexpr const int max_digits10 = MaxDigits10FromDigits(digits);
            static inline constexpr const int min_exponent = (1 - kExponentBias) + 1; // 1
            static inline constexpr const int min_exponent10 = MinExponent10FromMinExponent(min_exponent);
            static inline constexpr const int max_exponent = (0b11) - kExponentBias; // 2
            static inline constexpr const int max_exponent10 = MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static inline constexpr const bool is_iec559 = false;
            static inline constexpr const bool has_infinity = false;
            static inline constexpr const bool has_signaling_NaN = false;
            // NOLINTEND
        };

         
template<Rounding_Mode rm>
struct numeric_limits_f4_e2m1_rm : numeric_limits_float4_e2m1 {
  static constexpr float4_e2m1<rm> min() {
    return float4_e2m1<rm>::FromRep(0b0010);
  }
  static constexpr float4_e2m1<rm> lowest() {
    return float4_e2m1<rm>::FromRep(0b1111);
  }
  static constexpr float4_e2m1<rm> max() {
    return float4_e2m1<rm>::FromRep(0b0111);
  }
  static constexpr float4_e2m1<rm> epsilon() {
    return float4_e2m1<rm>::FromRep(0b0010);
  }
  static constexpr float4_e2m1<rm> round_error() {
    return float4_e2m1<rm>::FromRep(0b0001);
  }
  static constexpr float4_e2m1<rm> infinity() {
    return float4_e2m1<rm>::FromRep(0b0000);
  }
  static constexpr float4_e2m1<rm> quiet_NaN() {
    return float4_e2m1<rm>::FromRep(0b1000);
  }
  static constexpr float4_e2m1<rm> signaling_NaN() {
    return float4_e2m1<rm>::FromRep(0b0000);
  }
  static constexpr float4_e2m1<rm> denorm_min() {
    return float4_e2m1<rm>::FromRep(0b0001);
  }
};


        template<int p>
        struct numeric_limits_float4_p : public numeric_limits_float4_base {

            static inline constexpr const int kExponentBias = (1 << (3 - p));
            static inline constexpr const int kMantissaBits = p - 1;


            public:
            // NOLINTBEGIN: these names must match std::numeric_limits.
            static inline constexpr const int digits = p;
            static inline constexpr const int digits10 = Digits10FromDigits(digits);
            static inline constexpr const int max_digits10 =
                MaxDigits10FromDigits(digits);
            static inline constexpr const int min_exponent = (1 - kExponentBias);
            static inline constexpr const int min_exponent10 =
                MinExponent10FromMinExponent(min_exponent);
            static inline constexpr const int max_exponent = kExponentBias - 1;
            static inline constexpr const int max_exponent10 =
                MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static inline constexpr const bool is_iec559 = false; // TODO
            static inline constexpr const bool has_infinity = true;
            static inline constexpr const bool has_signaling_NaN = false;
            // NOLINTEND
        };

       
template<int p, Rounding_Mode rm>
struct numeric_limits_f4_p_rm : numeric_limits_float4_p<p> {
  static constexpr float4_p<p, rm> min() {
    return float4_p<p, rm>::FromRep(1 << (p - 1));
  }
  static constexpr float4_p<p, rm> lowest() {
    return float4_p<p, rm>::FromRep(0xE);
  }
  static constexpr float4_p<p, rm> max() {
    return float4_p<p, rm>::FromRep(0x6);
  }
  static constexpr float4_p<p, rm> epsilon() {
    if constexpr (p < 3) {
      constexpr int expeps =
        ((- (p - 1)) + (1 << (3 - p))) << (p - 1);
      return float4_p<p, rm>::FromRep(static_cast<uint8_t>(expeps));
    }
    return float4_p<p, rm>::FromRep(
      static_cast<uint8_t>(1 << ((1 << (3 - p)) - 1))
    );
  }
  static constexpr float4_p<p, rm> round_error() {
    return float4_p<p, rm>::FromRep(
      ((-1 + (1 << (3 - p))) << (p - 1))
    );
  }
  static constexpr float4_p<p, rm> infinity() {
    return float4_p<p, rm>::FromRep(0x7);
  }
  static constexpr float4_p<p, rm> quiet_NaN() {
    return float4_p<p, rm>::FromRep(0x8);
  }
  static constexpr float4_p<p, rm> signaling_NaN() {
    return float4_p<p, rm>::FromRep(0x8);
  }
  static constexpr float4_p<p, rm> denorm_min() {
    return float4_p<p, rm>::FromRep(0x1);
  }
};

template<FloatingPointParams Fp>
struct numeric_limits_flexible {

    static inline constexpr const bool is_specialized = true;
    static inline constexpr const bool is_signed = Fp.is_signed == lo_float::Signedness::Signed;
    static inline constexpr const bool is_integer = false;
    static inline constexpr const bool is_exact = false;
    static inline constexpr const bool has_quiet_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::HasQuietNaN;
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

}
}

namespace std {

    // 8-bit
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_e4m3fn<rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_e4m3fn_rm<rm> {};
    
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_e4m3b11fnuz<rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_e4m3b11fnuz_rm<rm> {};
    
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_e4m3fnuz<rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_e4m3_fnuz_rm<rm> {};
    
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_e5m2<rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_e5m2_rm<rm> {};
    
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_e5m2fnuz<rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_e5m2fnuz_rm<rm> {};
    
    template <int p, lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float8_ieee_p<p, rm>>
      : lo_float::lo_float_internal::numeric_limits_f8_ieee_p_rm<p, rm> {};
    
    // 6-bit
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float6_e2m3<rm>>
      : lo_float::lo_float_internal::numeric_limits_f6_e2m3_rm<rm> {};
    
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float6_e3m2<rm>>
      : lo_float::lo_float_internal::numeric_limits_f6_e3m2_rm<rm> {};
    
    template <int p, lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float6_p<p, rm>>
      : lo_float::lo_float_internal::numeric_limits_f6_p_rm<p, rm> {};
    
    // 4-bit
    template <lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float4_e2m1<rm>>
      : lo_float::lo_float_internal::numeric_limits_f4_e2m1_rm<rm> {};
    
    template <int p, lo_float::Rounding_Mode rm>
    struct numeric_limits<lo_float::lo_float_internal::float4_p<p, rm>>
      : lo_float::lo_float_internal::numeric_limits_f4_p_rm<p, rm> {};

    template <lo_float::FloatingPointParams Fp>
    struct numeric_limits<lo_float::lo_float_internal::Templated_Float<Fp>>
      : lo_float::lo_float_internal::numeric_limits_flexible<Fp> {};
    
    }  // namespace std
    

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


// float6_e2m3<rm>
template <Rounding_Mode rm>
constexpr inline float6_e2m3<rm> abs(const float6_e2m3<rm>& a) {
  return float6_e2m3<rm>::FromRep(a.rep() & 0b00'0'11'111);
}

template <Rounding_Mode rm>
inline bool isnan(const float6_e2m3<rm>& a) {
  return abs(a).rep() == std::numeric_limits<float6_e2m3<rm>>::quiet_NaN().rep();
}

// float6_e3m2<rm>
template <Rounding_Mode rm>
constexpr inline float6_e3m2<rm> abs(const float6_e3m2<rm>& a) {
  return float6_e3m2<rm>::FromRep(a.rep() & 0b00'0'11'111);
}

template <Rounding_Mode rm>
inline bool isnan(const float6_e3m2<rm>& a) {
  return a.rep() == std::numeric_limits<float6_e3m2<rm>>::quiet_NaN().rep();
}

// float6_p<p, rm>
template <int p, Rounding_Mode rm>
constexpr inline bool isnan(const float6_p<p, rm>& a) {
  return a.rep() == 0x20;
}

template <int p, Rounding_Mode rm>
constexpr inline float6_p<p, rm> abs(const float6_p<p, rm>& a) {
  return isnan(a) ? a
                  : float6_p<p, rm>::FromRep(a.rep() & 0x1F);
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
// float8_e4m3fn<rm>

template <Rounding_Mode rm>
constexpr inline float8_e4m3fn<rm> abs(const float8_e4m3fn<rm>& a) {
  return float8_e4m3fn<rm>::FromRep(a.rep() & 0b0'1111'111);
}

template <Rounding_Mode rm>
constexpr inline bool isnan(const float8_e4m3fn<rm>& a) {
  return abs(a).rep() ==
         std::numeric_limits<float8_e4m3fn<rm>>::quiet_NaN().rep();
}

// float8_e4m3b11fnuz<rm>

template <Rounding_Mode rm>
constexpr inline float8_e4m3b11fnuz<rm> abs(const float8_e4m3b11fnuz<rm>& a) {
  return (a.rep() & 0b0'1111'111) == 0
      ? float8_e4m3b11fnuz<rm>::FromRep(a.rep())
      : float8_e4m3b11fnuz<rm>::FromRep(a.rep() & 0b0'1111'111);
}

template <Rounding_Mode rm>
constexpr inline bool isnan(const float8_e4m3b11fnuz<rm>& a) {
  return a.rep() ==
         std::numeric_limits<float8_e4m3b11fnuz<rm>>::quiet_NaN().rep();
}

// float8_e4m3fnuz<rm>

template <Rounding_Mode rm>
constexpr inline float8_e4m3fnuz<rm> abs(const float8_e4m3fnuz<rm>& a) {
  return (a.rep() & 0x7F) == 0
      ? float8_e4m3fnuz<rm>::FromRep(a.rep())
      : float8_e4m3fnuz<rm>::FromRep(a.rep() & 0x7F);
}

template <Rounding_Mode rm>
constexpr inline bool isnan(const float8_e4m3fnuz<rm>& a) {
  return abs(a).rep() ==
         std::numeric_limits<float8_e4m3fnuz<rm>>::quiet_NaN().rep();
}

// float8_e5m2<rm>

template <Rounding_Mode rm>
constexpr inline float8_e5m2<rm> abs(const float8_e5m2<rm>& a) {
  return float8_e5m2<rm>::FromRep(a.rep() & 0b0'11111'11);
}

template <Rounding_Mode rm>
constexpr inline bool isnan(const float8_e5m2<rm>& a) {
  return abs(a).rep() >
         std::numeric_limits<float8_e5m2<rm>>::infinity().rep();
}

// float8_e5m2fnuz<rm>

template <Rounding_Mode rm>
constexpr inline float8_e5m2fnuz<rm> abs(const float8_e5m2fnuz<rm>& a) {
  return (a.rep() & 0x7F) == 0
      ? float8_e5m2fnuz<rm>::FromRep(a.rep())
      : float8_e5m2fnuz<rm>::FromRep(a.rep() & 0x7F);
}

template <Rounding_Mode rm>
constexpr inline bool isnan(const float8_e5m2fnuz<rm>& a) {
  return a.rep() == 0x80;
}

// float8_ieee_p<p, rm>

template <int p, Rounding_Mode rm>
constexpr inline bool isnan(const float8_ieee_p<p, rm>& a) {
  return a.rep() == 0x80;
}

template <int p, Rounding_Mode rm>
constexpr inline float8_ieee_p<p, rm> abs(const float8_ieee_p<p, rm>& a) {
  return isnan(a) ? a
                  : float8_ieee_p<p, rm>::FromRep(a.rep() & 0x7F);
}

// float4_e2m1<rm>

template <Rounding_Mode rm>
inline float4_e2m1<rm> abs(const float4_e2m1<rm>& a) {
  return float4_e2m1<rm>::FromRep(a.rep() & 0b0000'0'111);
}

template <Rounding_Mode rm>
inline bool isnan(const float4_e2m1<rm>& a) {
  return a.rep() ==
         std::numeric_limits<float4_e2m1<rm>>::quiet_NaN().rep();
}

// float4_p<p, rm>

template <int p, Rounding_Mode rm>
inline float4_p<p, rm> abs(const float4_p<p, rm>& a) {
  return float4_p<p, rm>::FromRep(a.rep() & 0b0000'0'111);
}

template <int p, Rounding_Mode rm>
inline bool isnan(const float4_p<p, rm>& a) {
  return a.rep() ==
         std::numeric_limits<float4_p<p, rm>>::quiet_NaN().rep();
}



}
}








namespace lo_float {
    namespace lo_float_internal {

    template<int le_size, typename uint>
    static constexpr inline int countl_zero(uint x) {
        int zeroes = le_size - 4;
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
//     typename Eigen::numext::get_integer_by_size<kNumBytes>::unsigned_type;



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
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& from) {
    return from;
  }
};

template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};


template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = Eigen::NumTraits<Float>::digits() - 1;
  static constexpr int kExponentBits = kBits - kMantissaBits - 1;
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};


template <typename Float>
struct Traits : public TraitsBase<Float> {};

template <Rounding_Mode rm>
struct Traits<float8_e4m3b11fnuz<rm>> : public TraitsBase<float8_e4m3b11fnuz<rm>> {
  static constexpr int kExponentBias = 11;
};

template <Rounding_Mode rm>
struct Traits<float8_e4m3fnuz<rm>> : public TraitsBase<float8_e4m3fnuz<rm>> {
  using Base = TraitsBase<float8_e4m3fnuz<rm>>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <Rounding_Mode rm>
struct Traits<float8_e5m2fnuz<rm>> : public TraitsBase<float8_e5m2fnuz<rm>> {
  using Base = TraitsBase<float8_e5m2fnuz<rm>>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <int p, Rounding_Mode rm>
struct Traits<float8_ieee_p<p, rm>> : public TraitsBase<float8_ieee_p<p, rm>> {
  using Base = TraitsBase<float8_ieee_p<p, rm>>;
  static constexpr int kExponentBias = 1 << (7 - p);
};

template <Rounding_Mode rm>
struct Traits<float6_e3m2<rm>> : public TraitsBase<float6_e3m2<rm>> {
    using Base =  TraitsBase<float6_e3m2<rm>>;
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 3;
};

template <Rounding_Mode rm>
struct Traits<float6_e2m3<rm>> : public TraitsBase<float6_e2m3<rm>> {
    using Base =  TraitsBase<float6_e2m3<rm>>;
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 1;
};

template <int p, Rounding_Mode rm>
struct Traits<float6_p<p, rm>> : public TraitsBase<float6_p<p, rm>> {
  using Base = TraitsBase<float6_p<p, rm>>;
  static constexpr int kBits = 6;
  static constexpr int kExponentBias = 1 << (5 - p);
};

template <Rounding_Mode rm>
struct Traits<float4_e2m1<rm>> : public TraitsBase<float4_e2m1<rm>> {
    using Base =  TraitsBase<float4_e2m1<rm>>;
    static constexpr int kBits = 4;
    static constexpr int kExponentBias = 1;
};


template <int p, Rounding_Mode rm>
struct Traits<float4_p<p, rm>> : public TraitsBase<float4_p<p, rm>> {
  using Base = TraitsBase<float4_p<p, rm>>;
  static constexpr int kBits = 4;
  static constexpr int kExponentBias = 1 << (3 - p);
};


template <FloatingPointParams Fp>
struct Traits<Templated_Float<Fp>> : public TraitsBase<Templated_Float<Fp>> {
    using Base = TraitsBase<Templated_Float<Fp>>;
    static constexpr int kBits = Fp.bitwidth;
    static constexpr int kExponentBias = Fp.bias;
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
  int samp = distribution(mt); // Generate a random integer
  Bits complement = (Bits{1} << (len)) - 1;
  Bits to_add = static_cast<Bits>(samp & complement); 
  Bits to_ret = bits + (to_add << (roundoff - len)); // Add random bits to the input bits
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

template<tyepname Bits>
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

template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
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
  static EIGEN_DEVICE_FUNC inline To run(const From& from, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven) {
    // Shift bits to destination type, without sign bit.
    const bool from_sign_bit =
        Eigen::numext::bit_cast<FromBits>(from) >> (kFromBits - 1);
    const FromBits from_bits =
        Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::infinity()
                           : Eigen::NumTraits<To>::infinity();
    }
    if (Eigen::numext::isnan(from)) {
      return Eigen::NumTraits<To>::quiet_NaN();
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
          if constexpr (!kTruncate) {
            
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
                    bits = Stochastic_Round(bits, -kDigitShift);
                    break;
                default :
                bits = RoundBitsToNearestEven(bits, -kDigitShift);

            } 
            
          }
          bits >>= -kDigitShift;
        }
        To to = Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits));
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
          if constexpr (!kTruncate) {
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
                  rounded_from_bits = Stochastic_Round(rounded_from_bits, exponent_shift);
                  break;
              }
              
          }
          bits = (rounded_from_bits >> exponent_shift);
        }
        // Insert sign and return.
        To to = Eigen::numext::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }

    // Round the mantissa if it is shrinking.
    WideBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
      if constexpr (!kTruncate) {
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
              rounded_from_bits = Stochastic_Round(from_bits, -kDigitShift);
              break;
          }
          
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
        Eigen::numext::bit_cast<ToBits>(Eigen::NumTraits<To>::highest());
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

    To to = Eigen::numext::bit_cast<To>(bits);
    // `From` supports larger values than `To`, we may overflow.
    if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                 std::numeric_limits<To>::digits) <
                  std::make_pair(std::numeric_limits<From>::max_exponent,
                                 std::numeric_limits<From>::digits)) {
        
      if (rounded_from_bits > aligned_highest) {
        // Overflowed values map to highest or infinity depending on kSaturate.
        to = kSaturate ? Eigen::NumTraits<To>::highest()
                       : Eigen::NumTraits<To>::infinity();
      }
    }
    // Insert sign bit.
    return from_sign_bit ? -to : to;
  }

  
};



template <typename Derived, typename UnderlyingType>
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC Derived lo_float_base<Derived, UnderlyingType>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from, Derived::rounding_mode);
}

template <typename Derived, typename UnderlyingType>
template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To lo_float_base<Derived, UnderlyingType>::ConvertTo(const Derived& from) {

  return ConvertImpl<Derived, To, kSaturate, kTruncate>::run(from, Derived::rounding_mode);

}

template <typename Derived, typename UnderlyingType>
template <bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC double lo_float_base<Derived, UnderlyingType>::ConvertTo(const Derived& from) {

  return ConvertImpl<Derived, double, kSaturate, kTruncate>::run(from, Derived::rounding_mode);



}

template <typename Derived, typename UnderlyingType>
template <bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC Derived lo_float_base<Derived, UnderlyingType>::ConvertFrom(const double& from) {

  return ConvertImpl<double, Derived, kSaturate, kTruncate>::run(from, Derived::rounding_mode);


}










    } //namespace lo_float_internal

    // 8-bit
template<Rounding_Mode rm>
using float8_e4m3fn = lo_float_internal::float8_e4m3fn<rm>;

template<Rounding_Mode rm>
using float8_e4m3b11 = lo_float_internal::float8_e4m3b11<rm>;

template<Rounding_Mode rm>
using float8_e4m3fnuz = lo_float_internal::float8_e4m3fnuz<rm>;

template<Rounding_Mode rm>
using float8_e4m3b11fnuz = lo_float_internal::float8_e4m3b11fnuz<rm>;

template<Rounding_Mode rm>
using float8_e5m2 = lo_float_internal::float8_e5m2<rm>;

template<Rounding_Mode rm>
using float8_e5m2fnuz = lo_float_internal::float8_e5m2fnuz<rm>;

template<int p, Rounding_Mode rm>
using float8_ieee_p = lo_float_internal::float8_ieee_p<p, rm>;

// 6-bit
template<Rounding_Mode rm>
using float6_e3m2 = lo_float_internal::float6_e3m2<rm>;

template<Rounding_Mode rm>
using float6_e2m3 = lo_float_internal::float6_e2m3<rm>;

template<int p, Rounding_Mode rm>
using float6_p = lo_float_internal::float6_p<p, rm>;

// 4-bit
template<Rounding_Mode rm>
using float4_e2m1 = lo_float_internal::float4_e2m1<rm>;

template<int p, Rounding_Mode rm>
using float4_p = lo_float_internal::float4_p<p, rm>;

template<FloatingPointParams Fp>
using Templated_Float = lo_float_internal::Templated_Float<Fp>;


}   //namespace lo_float


namespace Eigen {
    namespace numext {

// float8_e4m3fn
template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
lo_float::float8_e4m3fn<rm> bit_cast(const uint8_t& src) {
  return lo_float::float8_e4m3fn<rm>::FromRep(src);
}

template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
uint8_t bit_cast(const lo_float::float8_e4m3fn<rm>& src) {
  return src.rep();
}

// float8_e5m2
template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
lo_float::float8_e5m2<rm> bit_cast(const uint8_t& src) {
  return lo_float::float8_e5m2<rm>::FromRep(src);
}

template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
uint8_t bit_cast(const lo_float::float8_e5m2<rm>& src) {
  return src.rep();
}

// float6_e3m2
template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
lo_float::float6_e3m2<rm> bit_cast(const uint8_t& src) {
  return lo_float::float6_e3m2<rm>::FromRep(src);
}

template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
uint8_t bit_cast(const lo_float::float6_e3m2<rm>& src) {
  return src.rep();
}

// float6_e2m3
template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
lo_float::float6_e2m3<rm> bit_cast(const uint8_t& src) {
  return lo_float::float6_e2m3<rm>::FromRep(src);
}

template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
uint8_t bit_cast(const lo_float::float6_e2m3<rm>& src) {
  return src.rep();
}

// float4_e2m1
template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
lo_float::float4_e2m1<rm> bit_cast(const uint8_t& src) {
  return lo_float::float4_e2m1<rm>::FromRep(src);
}

template <lo_float::Rounding_Mode rm>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
uint8_t bit_cast(const lo_float::float4_e2m1<rm>& src) {
  return src.rep();
}

template<lo_float::FloatingPointParams Fp, typename UnderlyingType>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
UnderlyingType bit_cast(const lo_float::Templated_Float<Fp>& src) {
  return src.rep();
}


}  // namespace numext

    
}


        

#endif //FLOAT_6_4
  