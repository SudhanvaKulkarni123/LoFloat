/// @author Sudhanva Kulkarni
/* This file contains code for software defined 6 bit and 4 bit floats. It is an extended version of Andrew Fitzgibbon's float8.h
*/
#ifndef ML_DTYPES_FLOAT6_4_H_
#define ML_DTYPES_FLOAT6_4_H_
#define LEN 13  //this is the length  of the bitstring used for stochastic rounding
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
#include "tlapack/base/types.hpp"
#include "tlapack/base/scalar_type_traits.hpp"
#include  "eigen/Eigen/Core"  

#ifdef __has_include
# if __has_include(<version>)
#   include <version>
# endif
#endif

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif





enum Rounding_Mode : uint8_t {
    RoundToNearestEven = 0,
    RoundTowardsZero = 1,
    RoundAwayFromZero = 2,
    StochasticRounding = 3,
    RoundToNearestOdd = 4
};


namespace lo_float {
namespace lo_float_internal {

    static std::uniform_int_distribution<int> distribution(0, (1<< LEN) - 1);
    static std::mt19937 mt(time(nullptr));
    //6 bit floats
    //first two types are as described in OCP. Last one is templated IEEE
    class float6_e3m2;
    class float6_e2m3; 
    template<int p> class float6_p; //
    //8-bit datatypes as from Andrew's float8.h
    class float8_e4m3fn;
    class float8_e4m3fnuz;
    class float8_e4m3b11fnuz;
    class float8_e5m2;
    class float8_e5m2fnuz;
    template <int p> class float8_ieee_p;
    //4 bit types-
    class float4_e2m1; //nan at -0 no infs
    template<int p> class float4_p; //nan at -0, -inf at 0b1111 and +inf at 0b0111


template<typename Derived>
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

    constexpr bool operator==(const Derived& other) const {
        return Compare(derived(), other) == Ordering::kEquivalent;
    }
    constexpr bool operator!=(const Derived& other) const {
        return Compare(derived(), other) != Ordering::kEquivalent;
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
        const Derived& other) const {
        return Compare(derived(), other) == Ordering::kLess;
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
        const Derived& other) const {
        return Compare(derived(), other) <= Ordering::kEquivalent;
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
        const Derived& other) const {
        return Compare(derived(), other) == Ordering::kGreater;
    }
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
        const Derived& other) const {
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
    uint8_t rep_;

    // Helper for compare:
    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::pair<uint8_t, uint8_t>
    SignAndMagnitude(Derived x) {
        const uint8_t x_abs_bits =
            Eigen::numext::bit_cast<uint8_t>(Eigen::numext::abs(x));
        const uint8_t x_bits = Eigen::numext::bit_cast<uint8_t>(x);
        const uint8_t x_sign = x_bits ^ x_abs_bits;
        return {x_sign, x_abs_bits};
    }

    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int8_t
    SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
        return magnitude ^ (static_cast<int8_t>(sign) < 0 ? -1 : 0);
    }

    // Compare function
    EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC friend constexpr Ordering Compare(
        const Derived& lhs, const Derived& rhs) {
        if (Eigen::numext::isnan(lhs) || Eigen::numext::isnan(rhs)) {
            return kUnordered;
        }
        auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
        auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
        if (lhs_mag == 0 && rhs_mag == 0) {
            return kEquivalent;
        }
        int8_t lhs_tc = SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
        int8_t rhs_tc = SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
        if (lhs_tc < rhs_tc) return kLess;
        if (lhs_tc > rhs_tc) return kGreater;
        return kEquivalent;
    }
}; //lo_float_base

template <typename Derived>
class float4_base : public lo_float_base<Derived> {
 
    private :
        friend class lo_float_base<Derived>;
        
        static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int8_t
        SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
            return magnitude ^ (static_cast<int8_t>(sign << 4) < 0 ? -1 : 0);
        }

        
protected:
    using Base = lo_float_base<Derived>;
    using typename Base::ConstructFromRepTag;

    // Protected constructor that calls base
    constexpr float4_base(uint8_t rep, ConstructFromRepTag tag)
        : Base(rep, tag)
    {}

public:
    // Inherit all base constructors
    using Base::Base;

    explicit EIGEN_DEVICE_FUNC operator bool() const {
        return (this->rep() & 0x7) != 0;
    }
    constexpr Derived operator-() const {
        // Flip sign bit (assuming bit 3 = sign)
        return Base::FromRep(static_cast<uint8_t>(this->rep() ^ 0x8));
    }

    
};

      template <typename Derived>
class float6_base : public lo_float_base<Derived> {
      private :

        friend class lo_float_base<Derived>;

   
        static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int8_t
        SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
            return magnitude ^ (static_cast<int8_t>(sign << 2) < 0 ? -1 : 0);
        }

protected:
    using Base = lo_float_base<Derived>;
    using typename Base::ConstructFromRepTag;

    constexpr float6_base(uint8_t rep, ConstructFromRepTag tag)
        : Base(rep, tag)
    {}

public:
    using Base::Base;

    explicit EIGEN_DEVICE_FUNC operator bool() const {
        return (this->rep() & 0x1F) != 0;
    }
    constexpr Derived operator-() const {
        return Base::FromRep(static_cast<uint8_t>(this->rep() ^ 0x20));
    }

};

template <typename Derived>
class float8_base : public lo_float_base<Derived> {
      private :
        friend class lo_float_base<Derived>;
protected:
    using Base = lo_float_base<Derived>;
    using typename Base::ConstructFromRepTag;

    constexpr float8_base(uint8_t rep, ConstructFromRepTag tag)
        : Base(rep, tag)
    {}

public:

    using Base::Base;

    explicit EIGEN_DEVICE_FUNC operator bool() const {
        return (this->rep() & 0x7F) != 0;
    }
    constexpr Derived operator-() const {
        return Base::FromRep(static_cast<uint8_t>(this->rep() ^ 0x80));
    }
};



        
template<Rounding_Mode round_mode>
class float8_e4m3fn : public float8_base<float8_e4m3fn> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by 0bS111'1111.
  // The "fn" suffix is for consistency with the corresponding LLVM/MLIR type,
  // signaling this type is not consistent with IEEE-754.  The "f" indicates
  // it is finite values only. The "n" indicates it includes NaNs, but only
  // at the outer range.
 private:
  using Base = float8_base<float8_e4m3fn>;
  friend class float8_base<float8_e4m3fn>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e5m2& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e4m3b11fnuz& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };
  constexpr bool operator==(const float8_e4m3fn& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }
  
  Rounding_Mode rounding_mode = round_mode;
  

  
      
};

template<Rounding_Mode round_mode>
class float8_e4m3b11fnuz : public float8_base<float8_e4m3b11fnuz> {
  // Exponent: 4, Mantissa: 3, bias: 11.
  // Extended range: no inf, NaN represented by 0b1000'0000.
 private:
  using Base = float8_base<float8_e4m3b11fnuz>;
  friend class float8_base<float8_e4m3b11fnuz>;
  using Base::Base;


 public:
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11fnuz(const float8_e5m2& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11fnuz(const float8_e5m2fnuz& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11fnuz(const float8_e4m3fn& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11fnuz(const float8_e4m3fnuz& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}
  template <int p>
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11fnuz(const float8_ieee_p<p>& f8)
      : float8_e4m3b11fnuz(ConvertFrom(f8)) {}
  
  Rounding_Mode rounding_mode = round_mode;



};

// Legacy name used in XLA (TODO(jewillco): remove).
using float8_e4m3b11 = float8_e4m3b11fnuz;

template<Rounding_Mode round_mode>
class float8_e4m3fnuz : public float8_base<float8_e4m3fnuz> {
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
  using Base = float8_base<float8_e4m3fnuz>;
  friend class float8_base<float8_e4m3fnuz>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e4m3fnuz(const float8_e5m2& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fnuz(const float8_e5m2fnuz& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fnuz(const float8_e4m3b11fnuz& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fnuz(const float8_e4m3fn& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  template <int p>
  explicit EIGEN_DEVICE_FUNC float8_e4m3fnuz(const float8_ieee_p<p>& f8)
      : float8_e4m3fnuz(ConvertFrom(f8)) {}
  
      Rounding_Mode rounding_mode = round_mode;



};

template<Rounding_Mode round_mode>
class float8_e5m2 : public float8_base<float8_e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = float8_base<float8_e5m2>;
  friend class float8_base<float8_e5m2>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3fn f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3fnuz f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3b11fnuz f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e5m2fnuz& f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  template <int p>
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_ieee_p<p> f8)
      : float8_e5m2(ConvertFrom(f8)) {}
    
      Rounding_Mode rounding_mode = round_mode;

};


template<Rounding_Mode round_mode>
class float8_e5m2fnuz : public float8_base<float8_e5m2fnuz> {
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
  using Base = float8_base<float8_e5m2fnuz>;
  friend class float8_base<float8_e5m2fnuz>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e5m2& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3b11fnuz& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3fn& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_e4m3fnuz& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}
  template <int p>
  explicit EIGEN_DEVICE_FUNC float8_e5m2fnuz(const float8_ieee_p<p>& f8)
      : float8_e5m2fnuz(ConvertFrom(f8)) {}
      Rounding_Mode rounding_mode = round_mode;



};

template <int p, Rounding_Mode rounding_mode, int bias = >
class float8_ieee_p : public float8_base<float8_ieee_p<p>> {
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
  typedef float8_ieee_p<p> this_t;
  using Base = float8_base<this_t>;
  friend class float8_base<this_t>;
  using Base::Base;

 public:

  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e5m2& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e5m2fnuz& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3b11fnuz& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3fnuz& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_e4m3fn& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  template <int q>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float8_ieee_p<q>& f8)
      : float8_ieee_p(this->ConvertFrom(f8)) {}
  template<int q>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float6_p<q>& f6)
      : float8_ieee_p(this->ConvertFrom(f6)) {}
  template<int q>
  explicit EIGEN_DEVICE_FUNC float8_ieee_p(const float4_p<q>& f4)
      : float8_ieee_p(this->ConvertFrom(f4)) {}
      Rounding_Mode rounding_mode = round_mode;
  
   constexpr float8_ieee_p<p> operator-() const {
    // TODO: use isnan()
    if ((this->rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  float8_ieee_p<p> operator-(const float8_ieee_p<p>& other) const {
    return Base::operator-(other);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };
  constexpr bool operator==(const float8_ieee_p<p>& other) const {
    return Compare(this->derived(), other) == Ordering::kEquivalent;
  }

  explicit EIGEN_DEVICE_FUNC operator bool() const { return this->rep() != 0; }
};




        template<Rounding_Mode round_mode>
        class float6_e3m2 : public float6_base<float6_e3m2> {
            //1S3E2M, bias = 3, saturated rounding, no Inf or NaN
            private:
             using Base = float6_base<float6_e3m2>;
             friend class float6_base<float6_e3m2>;
             using Base::Base;

             public:
              explicit EIGEN_DEVICE_FUNC float6_e3m2(const float6_e2m3& f6)
                  : float6_e3m2(ConvertFrom(f6)) {}
                  Rounding_Mode rounding_mode = round_mode;

              


        };

        template<Rounding_Mode round_mode>
        class float6_e2m3 : public float6_base<float6_e2m3> {
            //1S3E2M, bias = 1, saturated rounding, no Inf or NaN

            private:
             using Base = float6_base<float6_e2m3>;
             friend class float6_base<float6_e2m3>;
             using Base::Base;

            public:
             explicit EIGEN_DEVICE_FUNC float6_e2m3(const float6_e3m2& f6)
                : float6_e2m3(ConvertFrom(f6)) {}
                Rounding_Mode rounding_mode = round_mode;
        };

        template<int p, Rounding_Mode round_mode>
        class float6_p : public float6_base<float6_p<p>> {
            //1S(6-p)E(p-1)M, bias = 2^(6 - p) - 1 , Inf at 0x3F and 0x1F. NaN at 0x20
            private:
            using Base = float6_base<float6_p<p>>;
            friend class float6_base<float6_p<p>>;
            using Base::Base;

            public:
             explicit EIGEN_DEVICE_FUNC float6_p(const float6_e3m2& f6)
                : float6_p(this->ConvertFrom(f6)) {}
             explicit EIGEN_DEVICE_FUNC float6_p(const float6_e2m3& f6)
                : float6_p(this->ConvertFrom(f6)) {}


            constexpr float6_p<p> operator-() const {
                    // TODO: use isnan()
                    if ((this->rep() & 0x1f) == 0x00) {
                    return *this;
                    }
                    return Base::operator-();
                }
             float6_p<p> operator-(const float6_p<p>& other) const {
                return Base::operator-(other);
             }

             Rounding_Mode rounding_mode = round_mode;
  

        };

        class float4_e2m1 : public float4_base<float4_e2m1> {
            private:
            using Base = float4_base<float4_e2m1>;
            friend class float4_base<float4_e2m1>;
            using Base::Base;




        };

        template<int p>
        class float4_p : public float4_base<float4_p<p>> {
            //1S2E1M, bias = 2^(4 - p) - 1, Inf at 0xF and 0x7, NaN at 0x4
             private:
            using Base = float4_base<float4_p<p>>;
            friend class float4_base<float4_p<p>>;
            using Base::Base;

            public : 
            float4_p<p> operator-(const float4_p<p>& other) const {
                return Base::operator-(other);
            }

            constexpr float4_p<p> operator-() const {
                    // TODO: use isnan()
                    if ((this->rep() & 0x7) == 0x00) {
                    return *this;
                    }
                    return Base::operator-();
                }


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
        private:
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

        // 1.0 * 2^(0b0001 - 7) = 1.0 * 2^-6 = 0.015625
        static constexpr float8_e4m3fn min() {
            return float8_e4m3fn::FromRep(0b0'0001 << kMantissaBits);
        }
        // -(1 + 0b110 * 2^-3) * 2^(0b1111 - 7) = -1.75 * 2^8 = 448
        static constexpr float8_e4m3fn lowest() {
            return float8_e4m3fn::FromRep(0b1'1111'110);
        }
        // (1 + 0b110 * 2^-3) * 2**(0b1111 - 7) = 1.75 * 2^8 = 448
        static constexpr float8_e4m3fn max() {
            return float8_e4m3fn::FromRep(0b0'1111'110);
        }
        // 1.0 * 2^-3 = 0.125
        static constexpr float8_e4m3fn epsilon() {
            return float8_e4m3fn::FromRep((-kMantissaBits + kExponentBias)
                                        << kMantissaBits);
        }
        // 1.0 * 2^-1 = 0.5
        static constexpr float8_e4m3fn round_error() {
            return float8_e4m3fn::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3fn infinity() {
            return float8_e4m3fn::FromRep(0b0'1111'111);
        }
        // NaN.
        static constexpr float8_e4m3fn quiet_NaN() {
            return float8_e4m3fn::FromRep(0b1'0000'000);
        }
        static constexpr float8_e4m3fn signaling_NaN() {
            return float8_e4m3fn::FromRep(0b1'0000'000);
        }
        // 1.0 * 2^(-7 - 3 + 1) = 1.0 * 2^-9 = 0.001953125
        static constexpr float8_e4m3fn denorm_min() {
            return float8_e4m3fn::FromRep(0b0'0000'001);
        }
        };

        struct numeric_limits_float8_e4m3b11fnuz : public numeric_limits_float8_base {
        private:
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

        // 1.0 * 2^(0b0001 - 11) = 1.0 * 2^-10 = 0.0009765625
        static constexpr float8_e4m3b11fnuz min() {
            return float8_e4m3b11fnuz::FromRep(1 << kMantissaBits);
        }
        // -(1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = -1.875 * 2^4 = -30
        static constexpr float8_e4m3b11fnuz lowest() {
            return float8_e4m3b11fnuz::FromRep(0b1'1111'111);
        }
        // (1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = 1.875 * 2^4 = 30
        static constexpr float8_e4m3b11fnuz max() {
            return float8_e4m3b11fnuz::FromRep(0b0'1111'111);
        }
        // 1.0 * 2^-3 = 0.125
        static constexpr float8_e4m3b11fnuz epsilon() {
            return float8_e4m3b11fnuz::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
        }
        // 1.0 * 2^-1 = 0.5
        static constexpr float8_e4m3b11fnuz round_error() {
            return float8_e4m3b11fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3b11fnuz infinity() {
            return float8_e4m3b11fnuz::FromRep(0b0'1111'111);
        }
        // NaN.
        static constexpr float8_e4m3b11fnuz quiet_NaN() {
            return float8_e4m3b11fnuz::FromRep(0b1'0000'000);
        }
        static constexpr float8_e4m3b11fnuz signaling_NaN() {
            return float8_e4m3b11fnuz::FromRep(0b1'0000'000);
        }
        // 1.0 * 2^(-11 - 3 + 1) = 1.0 * 2^-13 = 0.0001220703125
        static constexpr float8_e4m3b11fnuz denorm_min() {
            return float8_e4m3b11fnuz::FromRep(0b0'0000'001);
        }
        };

        struct numeric_limits_float8_e4m3fnuz : public numeric_limits_float8_base {
        private:
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

        static constexpr float8_e4m3fnuz min() {
            return float8_e4m3fnuz::FromRep(0x08);
        }
        static constexpr float8_e4m3fnuz lowest() {
            return float8_e4m3fnuz::FromRep(0xFF);
        }
        static constexpr float8_e4m3fnuz max() {
            return float8_e4m3fnuz::FromRep(0x7F);
        }
        static constexpr float8_e4m3fnuz epsilon() {
            return float8_e4m3fnuz::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
        }
        static constexpr float8_e4m3fnuz round_error() {
            return float8_e4m3fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e4m3fnuz infinity() {
            return float8_e4m3fnuz::FromRep(0x80);
        }  // NaN.
        static constexpr float8_e4m3fnuz quiet_NaN() {
            return float8_e4m3fnuz::FromRep(0x80);
        }
        static constexpr float8_e4m3fnuz signaling_NaN() {
            return float8_e4m3fnuz::FromRep(0x80);
        }
        static constexpr float8_e4m3fnuz denorm_min() {
            return float8_e4m3fnuz::FromRep(0x01);
        }
        };

        struct numeric_limits_float8_e5m2 : public numeric_limits_float8_base {
        private:
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

        // 1.0 * 2^(0b00001 - 15) = 1.0 * 2^-14 = 0.00006103515625
        static constexpr float8_e5m2 min() {
            return float8_e5m2::FromRep(1 << kMantissaBits);
        }
        // -(1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = -1.75 * 2^15 = -57344
        static constexpr float8_e5m2 lowest() {
            return float8_e5m2::FromRep(0b1'11110'11);
        }
        // (1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = 1.75 * 2^15 = 57344
        static constexpr float8_e5m2 max() {
            return float8_e5m2::FromRep(0b0'11110'11);
        }
        // 1.0 * 2^-2 = 0.25
        static constexpr float8_e5m2 epsilon() {
            return float8_e5m2::FromRep((-kMantissaBits + kExponentBias)
                                        << kMantissaBits);
        }
        // 1.0 * 2^-1 = 0.5
        static constexpr float8_e5m2 round_error() {
            return float8_e5m2::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e5m2 infinity() {
            return float8_e5m2::FromRep(0b0'11111'00);
        }
        static constexpr float8_e5m2 quiet_NaN() {
            // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
            // to 0 or 1 and all the bits of the biased exponent field E set to 1
            // (see 3.4). A quiet NaN bit string should be encoded with the first bit
            // (d1) of the trailing significand field T being 1."
            return float8_e5m2::FromRep(0b0'11111'10);
        }
        static constexpr float8_e5m2 signaling_NaN() {
            // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
            // the first bit of the trailing significand field being 0."
            return float8_e5m2::FromRep(0b0'11111'01);
        }
        // 1.0 * 2^(-15 - 2 + 1) = 1.0 * 2^-16 = 0.0000152587890625
        static constexpr float8_e5m2 denorm_min() {
            return float8_e5m2::FromRep(0b0'00000'01);
        }
        };


        struct numeric_limits_float8_e5m2fnuz : public numeric_limits_float8_base {
        private:
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

        static constexpr float8_e5m2fnuz min() {
            return float8_e5m2fnuz::FromRep(0x04);
        }
        static constexpr float8_e5m2fnuz lowest() {
            return float8_e5m2fnuz::FromRep(0xFF);
        }
        static constexpr float8_e5m2fnuz max() {
            return float8_e5m2fnuz::FromRep(0x7F);
        }
        static constexpr float8_e5m2fnuz epsilon() {
            return float8_e5m2fnuz::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
        }
        static constexpr float8_e5m2fnuz round_error() {
            return float8_e5m2fnuz::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_e5m2fnuz infinity() {
            return float8_e5m2fnuz::FromRep(0x80);
        }  // NaN.
        static constexpr float8_e5m2fnuz quiet_NaN() {
            return float8_e5m2fnuz::FromRep(0x80);
        }
        static constexpr float8_e5m2fnuz signaling_NaN() {
            return float8_e5m2fnuz::FromRep(0x80);
        }
        static constexpr float8_e5m2fnuz denorm_min() {
            return float8_e5m2fnuz::FromRep(0x01);
        }
        };

        template <int p>
        struct numeric_limits_float8_ieee_p : public numeric_limits_float8_base {
        private:
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

        static constexpr float8_ieee_p<p> min() {
            return float8_ieee_p<p>::FromRep(1<<(p-1));
        }
        static constexpr float8_ieee_p<p> lowest() {
            return float8_ieee_p<p>::FromRep(0xfe);
        }
        static constexpr float8_ieee_p<p> max() {
            return float8_ieee_p<p>::FromRep(0x7e);
        }
        static constexpr float8_ieee_p<p> epsilon() {
            if constexpr (p < 5) {
            constexpr int expeps = (-kMantissaBits + kExponentBias) << kMantissaBits;
            return float8_ieee_p<p>::FromRep(expeps);
            }
            // p >= 5: eps is subnormal
            return float8_ieee_p<p>::FromRep(uint8_t(1 << (kExponentBias - 1)));
        }
        static constexpr float8_ieee_p<p> round_error() {
            // Return 0.5
            return float8_ieee_p<p>::FromRep((-1 + kExponentBias) << kMantissaBits);
        }
        static constexpr float8_ieee_p<p> infinity() {
            return float8_ieee_p<p>::FromRep(0x7f);
        }
        static constexpr float8_ieee_p<p> quiet_NaN() {
            return float8_ieee_p<p>::FromRep(0x80);
        }
        static constexpr float8_ieee_p<p> signaling_NaN() {
            return float8_ieee_p<p>::FromRep(0x80);
        }
        static constexpr float8_ieee_p<p> denorm_min() {
            return float8_ieee_p<p>::FromRep(0x01);
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
            private :
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

            // 1.0 * 2^(0b00001 - 15) = 1.0 * 2^-14 = 0.00006103515625
            static constexpr float6_e3m2 min() {
                return float6_e3m2::FromRep(1 << kMantissaBits);
            }
            // -(1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = -1.75 * 2^15 = -57344
            static constexpr float6_e3m2 lowest() {
                return float6_e3m2::FromRep(0b1'111'11);
            }
            // (1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = 1.75 * 2^15 = 57344
            static constexpr float6_e3m2 max() {
                return float6_e3m2::FromRep(0b0'111'11);
            }
            // 1.0 * 2^-2 = 0.25
            static constexpr float6_e3m2 epsilon() {
                return float6_e3m2::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
            }
            // 1.0 * 2^-1 = 0.5
            static constexpr float6_e3m2 round_error() {
                return float6_e3m2::FromRep((-1 + kExponentBias) << kMantissaBits);
            }
            static constexpr float6_e3m2 infinity() {
                return float6_e3m2::FromRep(0b0);
            }
            static constexpr float6_e3m2 quiet_NaN() {
                // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
                // to 0 or 1 and all the bits of the biased exponent field E set to 1
                // (see 3.4). A quiet NaN bit string should be encoded with the first bit
                // (d1) of the trailing significand field T being 1."
                return float6_e3m2::FromRep(0b0'11111'10);
            }
            static constexpr float6_e3m2 signaling_NaN() {
                // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
                // the first bit of the trailing significand field being 0."
                return float6_e3m2::FromRep(0b0'111'01);
            }
            // 1.0 * 2^(-15 - 2 + 1) = 1.0 * 2^-16 = 0.0000152587890625
            static constexpr float6_e3m2 denorm_min() {
                return float6_e3m2::FromRep(0b0'000'01);
            }
                    };

        struct numeric_limits_float6_e2m3 : public numeric_limits_float6_base {
            private :
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

            // 1.0 * 2^(0b00001 - 15) = 1.0 * 2^-14 = 0.00006103515625
            static constexpr float6_e2m3 min() {
                return float6_e2m3::FromRep(1 << kMantissaBits);
            }
            // -(1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = -1.75 * 2^15 = -57344
            static constexpr float6_e2m3 lowest() {
                return float6_e2m3::FromRep(0b1'11'111);
            }
            // (1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = 1.75 * 2^15 = 57344
            static constexpr float6_e2m3 max() {
                return float6_e2m3::FromRep(0b0'11'111);
            }
            // 1.0 * 2^-2 = 0.25
            static constexpr float6_e2m3 epsilon() {
                return float6_e2m3::FromRep((-kMantissaBits + kExponentBias)
                                            << kMantissaBits);
            }
            // 1.0 * 2^-1 = 0.5
            static constexpr float6_e2m3 round_error() {
                return float6_e2m3::FromRep((-1 + kExponentBias) << kMantissaBits);
            }
            static constexpr float6_e2m3 infinity() {
                return float6_e2m3::FromRep(0b0);
            }
            static constexpr float6_e2m3 quiet_NaN() {
                // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
                // to 0 or 1 and all the bits of the biased exponent field E set to 1
                // (see 3.4). A quiet NaN bit string should be encoded with the first bit
                // (d1) of the trailing significand field T being 1."
                return float6_e2m3::FromRep(0b0'11'110);
            }
            static constexpr float6_e2m3 signaling_NaN() {
                // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
                // the first bit of the trailing significand field being 0."
                return float6_e2m3::FromRep(0b0'11'001);
            }
            // 1.0 * 2^(-15 - 2 + 1) = 1.0 * 2^-16 = 0.0000152587890625
            static constexpr float6_e2m3 denorm_min() {
                return float6_e2m3::FromRep(0b0'00'001);
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
           

            static constexpr float6_p<p> min() {
                return float6_p<p>::FromRep(1<<(p-1));
            }
            static constexpr float6_p<p> lowest() {
                return float6_p<p>::FromRep(0x3e);
            }
            static constexpr float6_p<p> max() {
                return float6_p<p>::FromRep(0x1e);
            }
            static constexpr float6_p<p> epsilon() {
                if constexpr (p < 4) {
                constexpr int expeps = (-kMantissaBits + kExponentBias) << kMantissaBits;
                return float6_p<p>::FromRep(expeps);
                }
                // p >= 4: eps is subnormal
                return float6_p<p>::FromRep(uint8_t(1 << (kExponentBias - 1)));
            }
            static constexpr float6_p<p> round_error() {
                // Return 0.5
                return float6_p<p>::FromRep((-1 + kExponentBias) << kMantissaBits);
            }
            static constexpr float6_p<p> infinity() {
                return float6_p<p>::FromRep(0x1f);
            }
            static constexpr float6_p<p> quiet_NaN() {
                return float6_p<p>::FromRep(0x20);
            }
            static constexpr float6_p<p> signaling_NaN() {
                return float6_p<p>::FromRep(0x20);
            }
            static constexpr float6_p<p> denorm_min() {
                return float6_p<p>::FromRep(0x01);
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

                    private:
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

            // Minimum positive normalized value: 1.0 * 2^(1 - 1) = 1.0 * 2^0 = 1.0
            static constexpr float4_e2m1 min() {
                return float4_e2m1::FromRep(0b0010); // 0bS E M = 0b0 01 0 = 2
            }

            // Lowest (most negative) value: -(1 + mantissa * 2^-m) * 2^(max_exponent - bias)
            // Here: -(1 + 1 * 2^-1) * 2^(2 - 1) = -1.5 * 2 = -3.0
            static constexpr float4_e2m1 lowest() {
                return float4_e2m1::FromRep(0b1111); // 0b1 11 1 = 15
            }

            // Maximum positive value: (1 + mantissa * 2^-m) * 2^(max_exponent - bias)
            // Here: (1 + 1 * 2^-1) * 2^(2 - 1) = 1.5 * 2 = 3.0
            static constexpr float4_e2m1 max() {
                return float4_e2m1::FromRep(0b0111); // 0b0 11 1 = 7
            }

            // Epsilon: Difference between 1 and the least value greater than 1
            // Here: 1.0 * 2^-1 = 0.5
            static constexpr float4_e2m1 epsilon() {
                return float4_e2m1::FromRep(0b0010); // 0b0 01 0 = 2
            }

            // Round error: Maximum rounding error
            // Here: 1.0 * 2^-1 = 0.5
            static constexpr float4_e2m1 round_error() {
                return float4_e2m1::FromRep(0b0001); // 0b0 00 1 = 1
            }

            // Infinity: Not supported, return zero or a default value
            static constexpr float4_e2m1 infinity() {
                return float4_e2m1::FromRep(0b0000); // 0b0 00 0 = 0
            }

            // Quiet NaN: Exponent all ones, mantissa non-zero
            static constexpr float4_e2m1 quiet_NaN() {
                return float4_e2m1::FromRep(0b1000); // 0b0 11 1 = 7
            }

            // Signaling NaN: Exponent all ones, mantissa zero
            static constexpr float4_e2m1 signaling_NaN() {
                return float4_e2m1::FromRep(0b00); // 0b0 11 0 = 6
            }

            // Denormal minimum: Smallest positive denormalized value: 1.0 * 2^(min_exponent - bias)
            // Here: 1.0 * 2^(1 - 1) = 1.0 * 2^0 = 1.0 (represented differently for denormals)
            static constexpr float4_e2m1 denorm_min() {
                return float4_e2m1::FromRep(0b0001); // 0b0 00 1 = 1
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

            static constexpr float4_p<p> min() {
                return float4_p<p>::FromRep(1<<(p-1));
            }
            static constexpr float4_p<p> lowest() {
                return float4_p<p>::FromRep(0xE);
            }
            static constexpr float4_p<p> max() {
                return float4_p<p>::FromRep(0x6);
            }
            static constexpr float4_p<p> epsilon() {
                if constexpr (p < 3) {
                constexpr int expeps = (-kMantissaBits + kExponentBias) << kMantissaBits;
                return float4_p<p>::FromRep(expeps);
                }
                // p >= 3: eps is subnormal
                return float4_p<p>::FromRep(uint8_t(1 << (kExponentBias - 1)));
            }
            static constexpr float4_p<p> round_error() {
                // Return 0.5
                return float4_p<p>::FromRep((-1 + kExponentBias) << kMantissaBits);
            }
            static constexpr float4_p<p> infinity() {
                return float4_p<p>::FromRep(0x7);
            }
            static constexpr float4_p<p> quiet_NaN() {
                return float4_p<p>::FromRep(0x8);
            }
            static constexpr float4_p<p> signaling_NaN() {
                return float4_p<p>::FromRep(0x8);
            }
            static constexpr float4_p<p> denorm_min() {
                return float4_p<p>::FromRep(0x1);
            }

        };

}
}


namespace std {
// Standard-library overrides.  Note that these are picked up by Eigen as well.
template <>
struct numeric_limits<lo_float::lo_float_internal::float8_e4m3fn>
    : public lo_float::lo_float_internal::numeric_limits_float8_e4m3fn {};

template <>
struct numeric_limits<lo_float::lo_float_internal::float8_e4m3b11fnuz>
    : public lo_float::lo_float_internal::numeric_limits_float8_e4m3b11fnuz {};

template <>
struct numeric_limits<lo_float::lo_float_internal::float8_e4m3fnuz>
    : public lo_float::lo_float_internal::numeric_limits_float8_e4m3fnuz {};

template <>
struct numeric_limits<lo_float::lo_float_internal::float8_e5m2>
    : public lo_float::lo_float_internal::numeric_limits_float8_e5m2 {};


template <>
struct numeric_limits<lo_float::lo_float_internal::float8_e5m2fnuz>
    : public lo_float::lo_float_internal::numeric_limits_float8_e5m2fnuz {};

template <int p>
struct numeric_limits<lo_float::lo_float_internal::float8_ieee_p<p>>
    : public lo_float::lo_float_internal::numeric_limits_float8_ieee_p<p> {};

template <int p>
struct numeric_limits<lo_float::lo_float_internal::float6_p<p>>
    : public lo_float::lo_float_internal::numeric_limits_float6_p<p> {};

template <int p>
struct numeric_limits<lo_float::lo_float_internal::float4_p<p>>
    : public lo_float::lo_float_internal::numeric_limits_float4_p<p> {};


template<>
struct numeric_limits<lo_float::lo_float_internal::float6_e2m3>
    : public lo_float::lo_float_internal::numeric_limits_float6_e2m3 {};


template<>
struct numeric_limits<lo_float::lo_float_internal::float6_e3m2>
    : public lo_float::lo_float_internal::numeric_limits_float6_e3m2 {};


template<>
struct numeric_limits<lo_float::lo_float_internal::float4_e2m1>
    : public lo_float::lo_float_internal::numeric_limits_float4_e2m1 {};



}  // namespace std

namespace lo_float {
    namespace lo_float_internal {


constexpr inline float6_e2m3 abs(const float6_e2m3& a) {
  return float6_e2m3::FromRep(a.rep() & (0b00'0'11'111));
}

inline bool(isnan)(const float6_e2m3& a) {
  return abs(a).rep() == std::numeric_limits<float6_e2m3>::quiet_NaN().rep();
}

constexpr inline float6_e3m2 abs(const float6_e3m2& a) {
  return float6_e3m2::FromRep(a.rep() & (0b00'0'11'111));
}


inline bool(isnan)(const float6_e3m2& a) {
  return a.rep() == std::numeric_limits<float6_e3m2>::quiet_NaN().rep();
}


template<int p>
constexpr inline bool(isnan)(const float6_p<p>& a) {
  return a.rep() == 0x20;
}


template<int p>
constexpr inline float6_p<p> abs(const float6_p<p>& a) {
  return isnan(a) ? a : float6_p<p>::FromRep(a.rep() & (0x1F));
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

constexpr inline float8_e4m3fn abs(const float8_e4m3fn& a) {
  return float8_e4m3fn::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const float8_e4m3fn& a) {
  return abs(a).rep() == std::numeric_limits<float8_e4m3fn>::quiet_NaN().rep();
}

constexpr inline float8_e4m3b11fnuz abs(const float8_e4m3b11fnuz& a) {
  return (a.rep() & 0b0'1111'111) == 0
             ? float8_e4m3b11fnuz::FromRep(a.rep())
             : float8_e4m3b11fnuz::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const float8_e4m3b11fnuz& a) {
  return a.rep() == std::numeric_limits<float8_e4m3b11fnuz>::quiet_NaN().rep();
}

constexpr inline float8_e4m3fnuz abs(const float8_e4m3fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? float8_e4m3fnuz::FromRep(a.rep())
                               : float8_e4m3fnuz::FromRep(a.rep() & 0x7F);
}

constexpr inline bool(isnan)(const float8_e4m3fnuz& a) {
  return abs(a).rep() ==
         std::numeric_limits<float8_e4m3fnuz>::quiet_NaN().rep();
}

constexpr inline float8_e5m2 abs(const float8_e5m2& a) {
  return float8_e5m2::FromRep(a.rep() & 0b0'11111'11);
}

constexpr inline bool(isnan)(const float8_e5m2& a) {
  return abs(a).rep() > std::numeric_limits<float8_e5m2>::infinity().rep();
}


constexpr inline float8_e5m2fnuz abs(const float8_e5m2fnuz& a) {
  return (a.rep() & 0x7F) == 0 ? float8_e5m2fnuz::FromRep(a.rep())
                               : float8_e5m2fnuz::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e5m2fnuz& a) {
  return a.rep() == 0x80;
}

template<int p>
constexpr inline bool(isnan)(const float8_ieee_p<p>& a) {
  return a.rep() == 0x80;
}


template<int p>
constexpr inline float8_ieee_p<p> abs(const float8_ieee_p<p>& a) {
  return isnan(a) ? a : float8_ieee_p<p>::FromRep(a.rep() & 0x7F);
}


inline float4_e2m1 abs(const float4_e2m1& a) {
  return float4_e2m1::FromRep(a.rep() & (0b0000'0'111));
}

inline bool(isnan)(const float4_e2m1& a) {
  return a.rep() == std::numeric_limits<float4_e2m1>::quiet_NaN().rep();
}

template<int p>
inline float4_p<p> abs(const float4_p<p>& a) {
  return float4_p<p>::FromRep(a.rep() & (0b0000'0'111));
}

template<int p>
inline bool(isnan)(const float4_p<p>& a) {
  return a.rep() == std::numeric_limits<float4_p<p>>::quiet_NaN().rep();
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

template <>
struct Traits<float8_e4m3b11fnuz> : public TraitsBase<float8_e4m3b11fnuz> {
  static constexpr int kExponentBias = 11;
};

template <>
struct Traits<float8_e4m3fnuz> : public TraitsBase<float8_e4m3fnuz> {
  using Base = TraitsBase<float8_e4m3fnuz>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <>
struct Traits<float8_e5m2fnuz> : public TraitsBase<float8_e5m2fnuz> {
  using Base = TraitsBase<float8_e5m2fnuz>;
  static constexpr int kExponentBias = Base::kExponentBias + 1;
};

template <int p>
struct Traits<float8_ieee_p<p>> : public TraitsBase<float8_ieee_p<p>> {
  using Base = TraitsBase<float8_ieee_p<p>>;
  static constexpr int kExponentBias = 1 << (7 - p);
};

template <>
struct Traits<float6_e3m2> : public TraitsBase<float6_e3m2> {
    using Base =  TraitsBase<float6_e3m2>;
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 3;
};

template <>
struct Traits<float6_e2m3> : public TraitsBase<float6_e2m3> {
    using Base =  TraitsBase<float6_e2m3>;
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 1;
};

template <int p>
struct Traits<float6_p<p>> : public TraitsBase<float6_p<p>> {
  using Base = TraitsBase<float6_p<p>>;
  static constexpr int kBits = 6;
  static constexpr int kExponentBias = 1 << (5 - p);
};

template <>
struct Traits<float4_e2m1> : public TraitsBase<float4_e2m1> {
    using Base =  TraitsBase<float4_e2m1>;
    static constexpr int kBits = 4;
    static constexpr int kExponentBias = 1;
};


template <int p>
struct Traits<float4_p<p>> : public TraitsBase<float4_p<p>> {
  using Base = TraitsBase<float4_p<p>>;
  static constexpr int kBits = 4;
  static constexpr int kExponentBias = 1 << (3 - p);
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
inline Bits Stochastic_Round(Bits bits, int roundoff) {
  //given pattern FFF...FLRTT...T,rounds stochastically by generating random bits
  // corresponding to  RTT...T and adding the genned number.
  //Then we truncate the mantissa
  auto len = LEN;
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
  static EIGEN_DEVICE_FUNC inline To run(const From& from, Rounding_Mode round_mode) {
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
                case RoundToNearestEven :
                bits = RoundToNe(bits, -kDigitShift);
            } {bits = Stochastic_Round(bits, -kDigitShift); }
            
            else {bits = RoundBitsToNearestEven(bits, -kDigitShift); }
            
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
             if(stochastic){
            rounded_from_bits =
                  Stochastic_Round(rounded_from_bits, exponent_shift);
             } else {
            rounded_from_bits =
                  RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
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
        if(stochastic) {
        rounded_from_bits = Stochastic_Round(from_bits, -kDigitShift);
        } else {
        rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
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



template <typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC Derived lo_float_base<Derived>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from, false);
}

template <typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To lo_float_base<Derived>::ConvertTo(const Derived& from) {

  return ConvertImpl<Derived, To, kSaturate, kTruncate>::run(from, false);

}

template <typename Derived>
template <bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC double lo_float_base<Derived>::ConvertTo(const Derived& from) {

  return ConvertImpl<Derived, double, kSaturate, kTruncate>::run(from, false);



}

template <typename Derived>
template <bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC Derived lo_float_base<Derived>::ConvertFrom(const double& from) {

  return ConvertImpl<double, Derived, kSaturate, kTruncate>::run(from, false);


}










    } //namespace lo_float_internal

    using float8_e4m3fn = lo_float_internal::float8_e4m3fn;
    using float8_e4m3b11 = lo_float_internal::float8_e4m3b11;
    using float8_e4m3fnuz = lo_float_internal::float8_e4m3fnuz;
    using float8_e4m3b11fnuz = lo_float_internal::float8_e4m3b11fnuz;
    using float8_e5m2 = lo_float_internal::float8_e5m2;
    using float8_e5m2fnuz = lo_float_internal::float8_e5m2fnuz;
    template<int p>
    using float8_ieee_p = lo_float_internal::float8_ieee_p<p>;
    

    using float6_e3m2 = lo_float_internal::float6_e3m2;
    using float6_e2m3 = lo_float_internal::float6_e2m3;
    template<int p>
    using float6_p = lo_float_internal::float6_p<p>;


    using float4_e2m1 = lo_float_internal::float4_e2m1;
    template<int p>
    using float4_p = lo_float_internal::float4_p<p>;
    


}   //namespace lo_float


namespace Eigen {
    namespace numext {


template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC lo_float::float8_e4m3fn
bit_cast<lo_float::float8_e4m3fn, uint8_t>(const uint8_t& src) {
  return lo_float::float8_e4m3fn::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, lo_float::float8_e4m3fn>(
    const lo_float::float8_e4m3fn& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC lo_float::float8_e5m2
bit_cast<lo_float::float8_e5m2, uint8_t>(const uint8_t& src) {
  return lo_float::float8_e5m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, lo_float::float8_e5m2>(const lo_float::float8_e5m2& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC lo_float::float6_e3m2
bit_cast<lo_float::float6_e3m2, uint8_t>(const uint8_t& src) {
  return lo_float::float6_e3m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, lo_float::float6_e3m2>(const lo_float::float6_e3m2& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC lo_float::float6_e2m3
bit_cast<lo_float::float6_e2m3, uint8_t>(const uint8_t& src) {
  return lo_float::float6_e2m3::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, lo_float::float6_e2m3>(const lo_float::float6_e2m3& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC lo_float::float4_e2m1
bit_cast<lo_float::float4_e2m1, uint8_t>(const uint8_t& src) {
  return lo_float::float4_e2m1::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, lo_float::float4_e2m1>(const lo_float::float4_e2m1& src) {
  return src.rep();
}





}  // namespace numext

    
}


        

#endif //FLOAT_6_4
  