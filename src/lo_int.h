#ifndef LO_FLOAT_INTN_H_
#define LO_FLOAT_INTN_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include "fp_tools.hpp" 

namespace lo_float {

/* ------------------------------------------------------------------ *
 *  Helper: pick the narrowest unsigned host type that fits LEN bits  *
 * ------------------------------------------------------------------ */
template<int LEN>
struct get_unsigned_type {
  static_assert(LEN >= 1 && LEN <= 64, "LEN must be 1-64");
  using type = std::conditional_t<
      (LEN <= 8),  uint8_t,
      std::conditional_t<
          (LEN <= 16), uint16_t,
          std::conditional_t<
              (LEN <= 32), uint32_t,
              uint64_t>>>;
};
template<int LEN> using get_unsigned_type_t = typename get_unsigned_type<LEN>::type;

template<int LEN, lo_float::Signedness Signedness>
class i_n {
  using Storage = get_unsigned_type_t<LEN>;
  static constexpr Storage MASK = (LEN == 64) ? Storage(-1) : ((Storage(1) << LEN) - 1);

  Storage v_{0};

  static constexpr Storage mask(Storage x) { return x & MASK; }

  static constexpr Storage sign_extend(Storage x) {
    if constexpr (Signedness != lo_float::Signedness::Signed) return mask(x);              // unsigned view
    else {
      const Storage sign_bit = Storage(1) << (LEN - 1);
      return mask(x) ^ sign_bit ? (mask(x) | ~MASK) : mask(x);
    }
  }

  constexpr Storage int_value() const { return sign_extend(v_); }

public:
  /* --------------------- ctors --------------------- */
  constexpr i_n() = default;
  constexpr i_n(const i_n&) noexcept = default;
  constexpr i_n(i_n&&) noexcept = default;
  constexpr i_n& operator=(const i_n&) = default;

  template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  explicit constexpr i_n(T x) : v_(mask(Storage(x))) {}

  /* ---------------- arithmetic / bit-wise ---------------- */
  #define LOF_BINARY_OP(op)                                           \
    constexpr i_n operator op(const i_n& o) const {                   \
      return i_n(int_value() op o.int_value());                       \
    }
  LOF_BINARY_OP(+)
  LOF_BINARY_OP(-)
  LOF_BINARY_OP(*)
  LOF_BINARY_OP(/)
  LOF_BINARY_OP(%)
  LOF_BINARY_OP(&)
  LOF_BINARY_OP(|)
  LOF_BINARY_OP(^)
  #undef LOF_BINARY_OP

  constexpr i_n operator~() const { return i_n(~int_value()); }
  constexpr i_n operator<<(int k) const { return i_n(mask(v_ << k)); }
  constexpr i_n operator>>(int k) const {
    if constexpr (Signedness = lo_float::Signedness::Signed) return i_n(int_value() >> k);   // arithmetic shift
    else                  return i_n(v_ >> k);            // logical shift
  }

  /* --------------- comparisons --------------- */
  #define LOF_CMP(op)                                                 \
    constexpr bool operator op(const i_n& o) const {                  \
      return int_value() op o.int_value();                            \
    }
  LOF_CMP(==) LOF_CMP(!=) LOF_CMP(<) LOF_CMP(>) LOF_CMP(<=) LOF_CMP(>=)
  #undef LOF_CMP

  /* --------------- compound ops --------------- */
  #define LOF_COMPOUND(op)                                            \
    constexpr i_n& operator op##=(const i_n& o) {                     \
      *this = *this op o;                                             \
      return *this;                                                   \
    }
  LOF_COMPOUND(+) LOF_COMPOUND(-) LOF_COMPOUND(*) LOF_COMPOUND(/)
  LOF_COMPOUND(%) LOF_COMPOUND(&) LOF_COMPOUND(|) LOF_COMPOUND(^)
  LOF_COMPOUND(<<) LOF_COMPOUND(>>)
  #undef LOF_COMPOUND

  /* --------------- increment / decrement --------------- */
  constexpr i_n& operator++()              { return *this += i_n(1); }
  constexpr i_n  operator++(int)           { i_n t=*this; ++*this; return t; }
  constexpr i_n& operator--()              { return *this -= i_n(1); }
  constexpr i_n  operator--(int)           { i_n t=*this; --*this; return t; }

  /* --------------- cast helpers --------------- */
  template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  explicit constexpr operator T() const    { return static_cast<T>(int_value()); }

  constexpr operator std::optional<int64_t>() const {
    return static_cast<int64_t>(int_value());
  }

  /* --------------- limits --------------- */
  static constexpr i_n lowest() {
    if constexpr (Signedness = lo_float::Signedness::Signed) return i_n(Storage(1) << (LEN - 1)); // two's-complement min
    else                  return i_n(0);
  }
  static constexpr i_n highest() { return i_n(MASK - (Signedness = lo_float::Signedness::Signed ? (Storage(1) << (LEN - 1)) : 0)); }

  /* --------------- misc --------------- */
  friend std::ostream& operator<<(std::ostream& os, const i_n& x) {
    os << static_cast<int64_t>(x.int_value());
    return os;
  }
  std::string ToString() const {
    std::ostringstream ss; ss << *this; return ss.str();
  }
}; // class i_n

/* -------------------- convenience aliases -------------------- */
template<int LEN> using  int_n = i_n<LEN, true>;
template<int LEN> using uint_n = i_n<LEN, false>;

using  int4 = int_n<4>;
using uint4 = uint_n<4>;
using  int8 = int_n<8>;
using uint8 = uint_n<8>;              // … and so on

/* =============================================================== *
 *          numeric_limits specialisation (partial)                *
 * =============================================================== */
namespace internal {
template<int LEN, lo_float::Signedness signedness>
struct intn_numeric_limits_base {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed      = signedness == lo_float::Signedness::Signed;
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
  static constexpr int  digits    = is_signed ? (LEN - 1) : LEN;
  static constexpr int  digits10  = 0;
  static constexpr int  max_digits10 = 0;
  static constexpr int  min_exponent = 0, min_exponent10 = 0;
  static constexpr int  max_exponent = 0, max_exponent10 = 0;
  static constexpr bool traps = true;
  static constexpr bool tinyness_before = false;

  static constexpr i_n<LEN, signedness> min()      noexcept { return i_n<LEN, signedness>::lowest(); }
  static constexpr i_n<LEN, signedness> lowest()   noexcept { return i_n<LEN, signedness>::lowest(); }
  static constexpr i_n<LEN, signedness> max()      noexcept { return i_n<LEN, signedness>::highest(); }
  static constexpr i_n<LEN, signedness> epsilon()  noexcept { return i_n<LEN, signedness>(0); }
  static constexpr i_n<LEN, signedness> round_error() noexcept { return i_n<LEN, signedness>(0); }
  static constexpr i_n<LEN, signedness> infinity() noexcept { return i_n<LEN, signedness>(0); }
  static constexpr i_n<LEN, signedness> quiet_NaN() noexcept { return i_n<LEN, signedness>(0); }
  static constexpr i_n<LEN, signedness> signaling_NaN() noexcept { return i_n<LEN, signedness>(0); }
  static constexpr i_n<LEN, signedness> denorm_min() noexcept { return i_n<LEN, signedness>(0); }
};
} // namespace internal
} // namespace lo_float

/* -------- std::numeric_limits specialisations (all LEN) -------- */
namespace std {
template<int LEN, lo_float::Signedness Signed>
struct numeric_limits<lo_float::i_n<LEN, Signed>>
    : public lo_float::internal::intn_numeric_limits_base<LEN, Signed> {};
} // namespace std

#endif /* LO_FLOAT_INTN_H_ */
