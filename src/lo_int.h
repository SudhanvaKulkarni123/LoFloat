#ifndef LO_FLOAT_INTN_H_
#define LO_FLOAT_INTN_H_

/*
  Fixed version that preserves original namespace layout and relies on an
  externally‑defined enum class `Signedness` (e.g. in "fp_tools.hpp").
  The implementation supports arbitrary‑bit signed and unsigned integers
  (1 – 128 bits) with proper two's‑complement behaviour, masking on every
  operation, and full constexpr friendliness.
*/

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "fp_tools.hpp"  // provides Signedness and any helper utilities

namespace lo_float {




namespace lo_float_internal {

template <typename T>
constexpr inline T bit_mask(int bits) {
  return bits == int(sizeof(T) * 8) ? T(~T(0)) : (T(1) << bits) - 1;
}

template <typename T>
constexpr inline T mask_to_bits(T x, int bits) {
  return x & bit_mask<T>(bits);
}

// -----------------------------------------------------------------------------
//  Type‑level map from bit‑length → storage type.
// -----------------------------------------------------------------------------
template <int len>
struct get_unsigned_type {
  static_assert(len >= 1 && len <= 128, "len must be in [1,128]");
  using type = std::conditional_t< (len <= 8),  uint8_t,
                std::conditional_t< (len <= 16), uint16_t,
                std::conditional_t< (len <= 32), uint32_t,
                std::conditional_t< (len <= 64), uint64_t,
#if defined(__SIZEOF_INT128__)
                __uint128_t
#else
                void
#endif
                >>>>;
};

template<int len>
using get_unsigned_type_t = typename get_unsigned_type<len>::type;

template<int len>
using get_signed_type_t = typename std::make_signed< get_unsigned_type_t<len> >::type;

// -----------------------------------------------------------------------------
//  i_n  – core arbitrary‑bit integer class (saturating variant)
// -----------------------------------------------------------------------------

template <int len, lo_float::Signedness Sign>
class i_n {
  static_assert(len >= 1 && len <= 128, "len must be in [1,128]");
  using Storage = std::conditional_t< Sign == Signedness::Signed,
                                     get_signed_type_t<len>,
                                     get_unsigned_type_t<len> >;
  static constexpr Storage STORAGE_MASK = bit_mask<Storage>(len);

  // Signed range expressed as 128‑bit integers (to avoid UB on shifts).
  static constexpr __int128_t LOWEST_RAW  = (Sign == Signedness::Signed)
                                            ? -(__int128_t(1) << (len-1))
                                            : 0;
  static constexpr __int128_t HIGHEST_RAW = (Sign == Signedness::Signed)
                                            ? ((__int128_t(1) << (len-1)) - 1)
                                            : ((__int128_t(1) << len) - 1);

  // Sign‑extend masked Storage to full signed 128‑bit value.
  static constexpr __int128_t sign_extend(Storage x) noexcept {
    if constexpr (Sign == Signedness::Signed) {
      const Storage sign_bit = Storage(1) << (len - 1);
      Storage masked = mask_to_bits(x, len);
      return (masked & sign_bit) ? (__int128_t(masked) | ~__int128_t(STORAGE_MASK))
                                 : __int128_t(masked);
    } else {
      return __int128_t(mask_to_bits(x, len));
    }
  }

  // Clamp raw 128‑bit integer into representable range, then mask.
  static constexpr Storage clamp_to_storage(__int128_t raw) noexcept {
    if (raw < LOWEST_RAW)  raw = LOWEST_RAW;
    if (raw > HIGHEST_RAW) raw = HIGHEST_RAW;
    return Storage(mask_to_bits(static_cast<Storage>(raw), len));
  }

public:
  // ------------------------------------------------------------------------
  //  Ctors
  // ------------------------------------------------------------------------
  constexpr i_n() : v_(0) {}
  constexpr i_n(const i_n&)            noexcept = default;
  constexpr i_n(i_n&&)                 noexcept = default;
  constexpr i_n& operator=(const i_n&) noexcept = default;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  explicit constexpr i_n(T x) : v_(clamp_to_storage(__int128_t(x))) {}

  // ------------------------------------------------------------------------
  //  Limits helpers
  // ------------------------------------------------------------------------
  static constexpr i_n lowest()  noexcept { return i_n(LOWEST_RAW);  }
  static constexpr i_n highest() noexcept { return i_n(HIGHEST_RAW); }

  // ------------------------------------------------------------------------
  //  Conversion helpers
  // ------------------------------------------------------------------------
  constexpr __int128_t   int_value() const noexcept { return sign_extend(v_); }
  constexpr Storage      raw_storage() const noexcept { return v_; }

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  explicit constexpr operator T() const { return static_cast<T>(int_value()); }

  constexpr inline operator int() const { return static_cast<int>(int_value()); }
  explicit constexpr operator bool() const { return int_value() != 0; }
  constexpr operator std::optional<int64_t>() const { return static_cast<int64_t>(int_value()); }

  // ------------------------------------------------------------------------
  //  Arithmetic (all saturating via ctor)
  // ------------------------------------------------------------------------
#define LOF_BIN_OP(op)                                                                       \
  constexpr i_n operator op(const i_n& o) const noexcept {                                   \
    return i_n( int_value() op o.int_value() );                                              \
  }
  LOF_BIN_OP(+)
  LOF_BIN_OP(-)
  LOF_BIN_OP(*)
  LOF_BIN_OP(/)   // undefined behav. if div by 0 – same as built‑in ints
  LOF_BIN_OP(%)
#undef LOF_BIN_OP

  // ------------------------------------------------------------------------
  //  Bitwise (masking keeps value in range, saturation not needed)
  // ------------------------------------------------------------------------
#define LOF_BIT_OP(op)                                                                       \
  constexpr i_n operator op(const i_n& o) const noexcept {                                   \
    return i_n( Storage( mask_to_bits(v_ op o.v_, len) ) );                                  \
  }
  LOF_BIT_OP(&)
  LOF_BIT_OP(|)
  LOF_BIT_OP(^)
#undef LOF_BIT_OP

  constexpr i_n operator~() const noexcept { return i_n( Storage(~v_) ); }

  // Shifts: left shift may overflow → saturate (unsigned) or wrap towards sign boundaries (signed)
  constexpr i_n operator<<(int k) const noexcept {
    if (k >= len) return highest();
    return i_n( int_value() << k );
  }
  constexpr i_n operator>>(int k) const noexcept {
    if constexpr (Sign == Signedness::Signed) {
      return i_n( int_value() >> k ); // arithmetic shift already sign‑propagates
    } else {
      return i_n( Storage(v_ >> k) ); // logical shift, already in range
    }
  }

 

  // ------------------------------------------------------------------------
  //  Compound‑assignment
  // ------------------------------------------------------------------------
#define LOF_COMPOUND(op)                                                                     \
  constexpr i_n& operator op##=(const i_n& o) noexcept {                                     \
    *this = *this op o;                                                                      \
    return *this;                                                                            \
  }
  LOF_COMPOUND(+)
  LOF_COMPOUND(-)
  LOF_COMPOUND(*)
  LOF_COMPOUND(/)
  LOF_COMPOUND(%)
  LOF_COMPOUND(&)
  LOF_COMPOUND(|)
  LOF_COMPOUND(^)
  LOF_COMPOUND(<<)
  LOF_COMPOUND(>>)
#undef LOF_COMPOUND

  // ++ / --
  constexpr i_n& operator++()    noexcept { return *this += i_n(1); }
  constexpr i_n  operator++(int) noexcept { i_n t = *this; ++*this; return t; }
  constexpr i_n& operator--()    noexcept { return *this -= i_n(1); }
  constexpr i_n  operator--(int) noexcept { i_n t = *this; --*this; return t; }

  // ------------------------------------------------------------------------
  //  Comparisons (use extended values)
  // ------------------------------------------------------------------------
#define LOF_CMP(op)                                                                          \
  constexpr bool operator op(const i_n& o) const noexcept {                                  \
    return int_value() op o.int_value();                                                     \
  }
  LOF_CMP(==)
  LOF_CMP(!=)
  LOF_CMP(<)
  LOF_CMP(>)
  LOF_CMP(<=)
  LOF_CMP(>=)
#undef LOF_CMP

  // ------------------------------------------------------------------------
  //  String / stream helpers
  // ------------------------------------------------------------------------
  friend std::ostream& operator<<(std::ostream& os, const i_n& x) {
    os << static_cast<int64_t>(x.int_value());
    return os;
  }
  std::string ToString() const {
    std::ostringstream ss; ss << *this; return ss.str();
  }


private:
  Storage v_{};  // Always holds a masked representation (no sign‑extension)
}; // class i_n

//---------------------------------------------------------------------------//
//  Convenience aliases (retain original names)
//---------------------------------------------------------------------------//

template <int len>
using  int_n = i_n<len, Signedness::Signed>;

template <int len>
using uint_n = i_n<len, Signedness::Unsigned>;


} // namespace lo_float_internal

template <int len>
using int_n = lo_float_internal::i_n<len, Signedness::Signed>;

template <int len>
using uint_n = lo_float_internal::i_n<len, Signedness::Unsigned>;


using  int4 = int_n<4>;
using uint4 = uint_n<4>;




//---------------------------------------------------------------------------//
//  numeric_limits specialisations (kept in `lo_float::internal`)
//---------------------------------------------------------------------------//
namespace lo_float_internal {

template <int len, Signedness Sign>
struct intn_numeric_limits_base {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed      = Sign == Signedness::Signed;
  static constexpr bool is_integer     = true;
  static constexpr bool is_exact       = true;
  static constexpr bool has_infinity   = false;
  static constexpr bool has_quiet_NaN  = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr std::float_round_style round_style = std::round_toward_zero;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = !is_signed;
  static constexpr int  radix = 2;
  static constexpr int  digits    = is_signed ? (len - 1) : len;
  static constexpr int  digits10  = 0;
  static constexpr int  max_digits10 = 0;
  static constexpr int  min_exponent = 0, min_exponent10 = 0;
  static constexpr int  max_exponent = 0, max_exponent10 = 0;
  static constexpr bool traps = true;
  static constexpr bool tinyness_before = false;

  static constexpr lo_float_internal::i_n<len, Sign> min()      noexcept { return lo_float_internal::i_n<len, Sign>::lowest(); }
  static constexpr lo_float_internal::i_n<len, Sign> lowest()   noexcept { return lo_float_internal::i_n<len, Sign>::lowest(); }
  static constexpr lo_float_internal::i_n<len, Sign> max()      noexcept { return lo_float_internal::i_n<len, Sign>::highest(); }
  static constexpr lo_float_internal::i_n<len, Sign> epsilon()  noexcept { return lo_float_internal::i_n<len, Sign>(0); }
  static constexpr lo_float_internal::i_n<len, Sign> round_error() noexcept { return lo_float_internal::i_n<len, Sign>(0); }
  static constexpr lo_float_internal::i_n<len, Sign> infinity() noexcept { return lo_float_internal::i_n<len, Sign>(0); }
  static constexpr lo_float_internal::i_n<len, Sign> quiet_NaN() noexcept { return lo_float_internal::i_n<len, Sign>(0); }
  static constexpr lo_float_internal::i_n<len, Sign> signaling_NaN() noexcept { return lo_float_internal::i_n<len, Sign>(0); }
  static constexpr lo_float_internal::i_n<len, Sign> denorm_min() noexcept { return lo_float_internal::i_n<len, Sign>(0); }
};

} // namespace internal




template<typename T>
struct is_integral {
  static constexpr bool val = false;
};

template<int len, Signedness sign>
struct is_integral<lo_float_internal::i_n<len, sign>> {
  static constexpr bool val = true;
};

template<typename T>
inline constexpr bool is_integral_v = is_integral<T>::val || std::is_integral_v<T>;


template<int len, Signedness sign>
using i_n = lo_float_internal::i_n<len, sign>;


template<typename T>
struct get_type_len {
  static constexpr int val = 0;
};

template<int N, Signedness sign>
struct get_type_len<i_n<N, sign>> {
  static constexpr int val = N;
};

template<typename T>
inline constexpr int get_type_len_v = get_type_len<T>::val;


//signed shift function
template<int len, Signedness sign, int offset>
inline constexpr i_n<len, sign> signed_shift(i_n<len, sign> input)
{
  if constexpr (offset < 0) {
    return input >> offset;
  }  
  return input << offset;
} 

} // namespace lo_float

//---------------------------------------------------------------------------//
//  std::numeric_limits specialisation (all bit‑lengths)
//---------------------------------------------------------------------------//
namespace std {

template <int len, lo_float::Signedness Sign>
struct numeric_limits<lo_float::lo_float_internal::i_n<len, Sign>>
    : public lo_float::lo_float_internal::intn_numeric_limits_base<len, Sign> {};

template<int len>
struct make_signed<lo_float::i_n<len, lo_float::Signedness::Unsigned>> {
  using type = lo_float::i_n<len-1, lo_float::Signedness::Signed>;
};




} // namespace std

#endif /* LO_FLOAT_INTN_H_ */
