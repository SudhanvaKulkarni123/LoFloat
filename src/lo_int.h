/* Copyright 2023 The ml_dtypes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ML_DTYPES_INT4_H_
#define ML_DTYPES_INT4_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace ml_dtypes {

// Stores the 4-bit integer value in the low four bits of a byte.  The upper
// four bits are left unspecified and ignored.
template <typename UnderlyingTy>
struct i4 {
 private:
  UnderlyingTy v_;

  static_assert(
      std::is_same_v<UnderlyingTy, uint8_t> ||
          std::is_same_v<UnderlyingTy, int8_t>,
      "The underyling type must be a signed or unsigned 8-bit integer.");

  // Mask the upper four bits.
  static inline constexpr UnderlyingTy Mask(UnderlyingTy v) { return v & 0x0F; }

  // Mask the upper four bits and sign-extend for signed types.
  static inline constexpr UnderlyingTy MaskAndSignExtend(UnderlyingTy v) {
    return std::is_signed_v<UnderlyingTy> ? Mask(v) | ((v & 0x08) ? 0xF0 : 0x00)
                                          : Mask(v);
  }

  // Casts to the corresponding UnderlyingTy value.
  inline constexpr UnderlyingTy IntValue() const {
    return MaskAndSignExtend(v_);
  }

 public:
  constexpr i4() noexcept : v_(0) {}
  constexpr i4(const i4& other) noexcept = default;
  constexpr i4(i4&& other) noexcept = default;
  constexpr i4& operator=(const i4& other) = default;
  constexpr i4& operator=(i4&&) = default;

  explicit constexpr i4(UnderlyingTy val) : v_(Mask(val)) {}
  template <typename T>
  explicit constexpr i4(T t) : i4(static_cast<UnderlyingTy>(t)) {}

  static constexpr i4 lowest() {
    return std::is_signed<UnderlyingTy>::value ? i4(-8) : i4(0);
  }
  static constexpr i4 highest() {
    return std::is_signed<UnderlyingTy>::value ? i4(7) : i4(15);
  }

  template <typename T>
  explicit constexpr operator T() const {
    return static_cast<T>(IntValue());
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator std::optional<int64_t>() const {
    return static_cast<int64_t>(IntValue());
  }

  constexpr i4 operator-() const { return i4(-v_); }
  constexpr i4 operator+(const i4& other) const { return i4(v_ + other.v_); }
  constexpr i4 operator-(const i4& other) const { return i4(v_ - other.v_); }
  constexpr i4 operator*(const i4& other) const { return i4(v_ * other.v_); }
  constexpr i4 operator/(const i4& other) const {
    return i4(IntValue() / other.IntValue());
  }
  constexpr i4 operator%(const i4& other) const {
    return i4((IntValue() % other.IntValue()));
  }

  constexpr i4 operator&(const i4& other) const { return i4(v_ & other.v_); }
  constexpr i4 operator|(const i4& other) const { return i4(v_ | other.v_); }
  constexpr i4 operator^(const i4& other) const { return i4(v_ ^ other.v_); }
  constexpr i4 operator~() const { return i4(~v_); }
  constexpr i4 operator>>(int amount) const { return i4(IntValue() >> amount); }
  constexpr i4 operator<<(int amount) const { return i4(v_ << amount); }

  constexpr bool operator==(const i4& other) const {
    return Mask(v_) == Mask(other.v_);
  }
  constexpr bool operator!=(const i4& other) const {
    return Mask(v_) != Mask(other.v_);
  }
  constexpr bool operator<(const i4& other) const {
    return IntValue() < other.IntValue();
  }
  constexpr bool operator>(const i4& other) const {
    return IntValue() > other.IntValue();
  }
  constexpr bool operator<=(const i4& other) const {
    return IntValue() <= other.IntValue();
  }
  constexpr bool operator>=(const i4& other) const {
    return IntValue() >= other.IntValue();
  }

  constexpr bool operator==(int64_t other) const { return IntValue() == other; }
  constexpr bool operator!=(int64_t other) const { return IntValue() != other; }
  constexpr bool operator<(int64_t other) const { return IntValue() < other; }
  constexpr bool operator>(int64_t other) const { return IntValue() > other; }
  constexpr bool operator<=(int64_t other) const { return IntValue() <= other; }
  constexpr bool operator>=(int64_t other) const { return IntValue() >= other; }

  friend constexpr bool operator==(int64_t a, const i4& b) {
    return a == b.IntValue();
  }
  friend constexpr bool operator!=(int64_t a, const i4& b) {
    return a != b.IntValue();
  }
  friend constexpr bool operator<(int64_t a, const i4& b) {
    return a < b.IntValue();
  }
  friend constexpr bool operator>(int64_t a, const i4& b) {
    return a > b.IntValue();
  }
  friend constexpr bool operator<=(int64_t a, const i4& b) {
    return a <= b.IntValue();
  }
  friend constexpr bool operator>=(int64_t a, const i4& b) {
    return a >= b.IntValue();
  }

  constexpr i4& operator++() {
    v_ = Mask(v_ + 1);
    return *this;
  }

  constexpr i4 operator++(int) {
    i4 orig = *this;
    this->operator++();
    return orig;
  }

  constexpr i4& operator--() {
    v_ = Mask(v_ - 1);
    return *this;
  }

  constexpr i4 operator--(int) {
    i4 orig = *this;
    this->operator--();
    return orig;
  }

  constexpr i4& operator+=(const i4& other) {
    *this = *this + other;
    return *this;
  }
  constexpr i4& operator-=(const i4& other) {
    *this = *this - other;
    return *this;
  }
  constexpr i4& operator*=(const i4& other) {
    *this = *this * other;
    return *this;
  }
  constexpr i4& operator/=(const i4& other) {
    *this = *this / other;
    return *this;
  }
  constexpr i4& operator%=(const i4& other) {
    *this = *this % other;
    return *this;
  }
  constexpr i4& operator&=(const i4& other) {
    *this = *this & other;
    return *this;
  }
  constexpr i4& operator|=(const i4& other) {
    *this = *this | other;
    return *this;
  }
  constexpr i4& operator^=(const i4& other) {
    *this = *this ^ other;
    return *this;
  }
  constexpr i4& operator>>=(int amount) {
    *this = *this >> amount;
    return *this;
  }
  constexpr i4& operator<<=(int amount) {
    *this = *this << amount;
    return *this;
  }

  friend ::std::ostream& operator<<(::std::ostream& os, const i4& num) {
    os << static_cast<int16_t>(num);
    return os;
  }

  std::string ToString() const {
    std::ostringstream os;
    os << static_cast<int16_t>(*this);
    return os.str();
  }
};

using int4 = i4<int8_t>;
using uint4 = i4<uint8_t>;

namespace internal {

struct int4_numeric_limits_base {
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_integer = true;
  static inline constexpr const bool is_exact = true;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_quiet_NaN = false;
  static inline constexpr const bool has_signaling_NaN = false;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_absent;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_toward_zero;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const int max_digits10 = 0;  // Not used for integers.
  static inline constexpr const int radix = 2;
  static inline constexpr const int min_exponent = 0;
  static inline constexpr const int min_exponent10 = 0;
  static inline constexpr const int max_exponent = 0;
  static inline constexpr const int max_exponent10 = 0;
  static inline constexpr const bool traps = true;
  static inline constexpr const bool tinyness_before = false;

  static constexpr ml_dtypes::int4 epsilon() noexcept {
    return ml_dtypes::int4(0);
  }
  static constexpr ml_dtypes::int4 round_error() noexcept {
    return ml_dtypes::int4(0);
  }
  static constexpr ml_dtypes::int4 infinity() noexcept {
    return ml_dtypes::int4(0);
  }
  static constexpr ml_dtypes::int4 quiet_NaN() noexcept {
    return ml_dtypes::int4(0);
  }
  static constexpr ml_dtypes::int4 signaling_NaN() noexcept {
    return ml_dtypes::int4(0);
  }
  static constexpr ml_dtypes::int4 denorm_min() noexcept {
    return ml_dtypes::int4(0);
  }
};

}  // namespace internal

}  // namespace ml_dtypes

namespace std {

template <>
struct numeric_limits<ml_dtypes::int4>
    : public ml_dtypes::internal::int4_numeric_limits_base {
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_modulo = false;
  static inline constexpr const int digits = 3;
  static inline constexpr const int digits10 = 0;  // floor(3 * log10(2))
  static constexpr ml_dtypes::int4 min() noexcept {
    return ml_dtypes::int4::lowest();
  }
  static constexpr ml_dtypes::int4 lowest() noexcept {
    return ml_dtypes::int4::lowest();
  }
  static constexpr ml_dtypes::int4 max() noexcept {
    return ml_dtypes::int4::highest();
  }
};

template <>
struct numeric_limits<ml_dtypes::uint4>
    : public ml_dtypes::internal::int4_numeric_limits_base {
  static inline constexpr const bool is_signed = false;
  static inline constexpr const bool is_modulo = true;
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = 1;  // floor(4 * log10(2))
  static constexpr ml_dtypes::uint4 min() noexcept {
    return ml_dtypes::uint4::lowest();
  }
  static constexpr ml_dtypes::uint4 lowest() noexcept {
    return ml_dtypes::uint4::lowest();
  }
  static constexpr ml_dtypes::uint4 max() noexcept {
    return ml_dtypes::uint4::highest();
  }
};

}  // namespace std

#endif  // ML_DTYPES_INT4_H_