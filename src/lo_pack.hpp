// ===========================================================================
//  lo_pack.hpp  --  bit-packing / unpacking of low-precision floats
// ---------------------------------------------------------------------------
//  An n-bit float (n in [1,16]) wastes 8 - n (or 16 - n) bits when stored in
//  its native uint8_t / uint16_t. This header packs a *group* of such floats
//  tightly so the group lands on a byte boundary with zero waste:
//
//      n in [1,8]   -> a group occupies lcm(n,8)  bits  (align to 8)
//      n in [9,16]  -> a group occupies lcm(n,16) bits  (align to 16)
//
//  so  group_count = group_bits / n   floats fit in   group_bytes = group_bits/8
//  bytes exactly.  e.g. 3-bit floats: lcm(3,8)=24 bits -> 8 floats in 3 bytes.
//
//  The design is two layers:
//   (1) a FORMAT-AGNOSTIC bit-packing layer that moves raw n-bit reps in/out of
//       a byte buffer (runtime n, so it is trivial to bind to Python later);
//   (2) a runtime float<->rep codec (convert_float_to_rep / convert_rep_to_float)
//       built so FloatingPointParams is a RUNTIME argument -- exactly like
//       virtual_round -- rather than a compile-time template parameter. The
//       encode codec is a runtime twin of ConvertImpl::run() in lo_float.h with
//       the SOURCE fixed to float; the decode codec is the runtime inverse with
//       the DEST fixed to float. Both are validated bit-exact against the
//       compile-time Templated_Float<Fp> path in test/test_pack.cpp.
//
//  Public APIs (per the backlog spec):
//    encode:
//      LoF_encode<E>(uint8_t* dst, const float* src, params, count)   round+pack
//      LoF_encode<Fp,E>(uint8_t* dst, const Templated_Float<Fp>* src, count)  packed memcpy
//    decode:
//      LoF_decode<Fp,E>(Templated_Float<Fp>* dst, const uint8_t* src, count)  -> typed
//      LoF_decode<E>(float* dst, const uint8_t* src, params, count)   -> float
//  Endianness E (PackEndian) is a template parameter:
//      Forward  -- element i occupies group-slot i        (front-to-back)
//      Reverse  -- element i occupies group-slot (G-1-i)   (back-to-front)
//  Bit order *within* a byte is LSB-first in both cases.
//
//  GPU array wrappers (LoF_encode_cuda/LoF_decode_cuda, device pointers, one
//  thread per group) mirror the CPU ones.
// ===========================================================================
#pragma once

#include "lo_float.h"
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace lo_float {

// front-to-back vs back-to-front group ordering ("endianness").
enum class PackEndian { Forward, Reverse };

namespace lo_pack_internal {

// ---------------------------------------------------------------------------
//  Layer 1: format-agnostic group geometry + bit moves (runtime n).
// ---------------------------------------------------------------------------
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE constexpr int lof_gcd(int a, int b) {
    return b == 0 ? a : lof_gcd(b, a % b);
}
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE constexpr int lof_lcm(int a, int b) {
    return (a / lof_gcd(a, b)) * b;
}
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE int pack_align_bits(int n) { return n <= 8 ? 8 : 16; }
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE int pack_group_bits(int n)  { return lof_lcm(n, pack_align_bits(n)); }
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE int pack_group_count(int n) { return pack_group_bits(n) / n; }
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE int pack_group_bytes(int n) { return pack_group_bits(n) / 8; }

LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE size_t packed_bytes(int n, size_t count) {
    const int gc = pack_group_count(n);
    return ((count + gc - 1) / gc) * static_cast<size_t>(pack_group_bytes(n));
}

// bit offset of the i-th element of a group (n bits each, group_count slots).
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE
int bit_offset(int i, int n, int group_count, PackEndian e) {
    const int slot = (e == PackEndian::Forward) ? i : (group_count - 1 - i);
    return slot * n;
}

// Write the low `n` bits of `v` into bit buffer `dst` starting at bit `off`
// (LSB-first within each byte). The destination region must be pre-zeroed.
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE
void put_bits(uint8_t* dst, int off, uint32_t v, int n) {
    for (int b = 0; b < n; ++b) {
        const int pos = off + b;
        dst[pos >> 3] |= static_cast<uint8_t>(((v >> b) & 1u) << (pos & 7));
    }
}

// Read `n` bits from bit buffer `src` starting at bit `off` (LSB-first).
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE
uint32_t get_bits(const uint8_t* src, int off, int n) {
    uint32_t v = 0;
    for (int b = 0; b < n; ++b) {
        const int pos = off + b;
        v |= static_cast<uint32_t>((src[pos >> 3] >> (pos & 7)) & 1u) << b;
    }
    return v;
}

// ---------------------------------------------------------------------------
//  Layer 2a: float -> n-bit rep, runtime FloatingPointParams.
//  Runtime twin of ConvertImpl<float, Templated_Float<Fp>>::run() (lo_float.h):
//  the algorithm is identical; the constexpr ToTraits fields and the
//  `if constexpr` branches are replaced by runtime values read from `params`,
//  and the result is returned as the raw rep (low `bitwidth` bits significant)
//  instead of a typed Templated_Float. SOURCE is fixed to float.
// ---------------------------------------------------------------------------
template <typename ToInf, typename ToNaN>
LOFLOAT_HOST_DEVICE uint32_t
convert_float_to_rep(float from, FloatingPointParams<ToInf, ToNaN> params, ProjSpec ps = ProjSpec{}) {
    using lo_float_internal::RoundMantissa;
    using lo_float_internal::countl_zero;
    using lo_float_internal::bit_cast;

    const Rounding_Mode round_mode = ps.rounding_mode;
    const int           stoch_len  = ps.stoch_length;
    const Saturation_Mode sat_mode = ps.saturation_mode;

    // ---- From = float (compile-time) ----
    constexpr int  kFromBits         = 32;
    constexpr int  kFromMantissaBits = 23;
    constexpr int  kFromExponentBias = 127;
    constexpr uint32_t kFromMantissaMask = (1u << kFromMantissaBits) - 1u;
    constexpr int  kFromMinExp = -125;   // std::numeric_limits<float>::min_exponent
    using WideBits = uint64_t;

    // ---- To = params (runtime) ----
    const int  n           = params.bitwidth;
    const int  toMant      = params.mantissa_bits;
    const bool to_signed   = (params.is_signed == Signedness::Signed);
    const int  toExpBits   = to_signed ? n - toMant - 1 : n - toMant;
    const int  toBias      = params.bias;
    const int  kExponentOffset = toBias - kFromExponentBias;
    const int  kDigitShift     = toMant - kFromMantissaBits;          // < 0 here
    const int  toMinExp    = 1 - toBias;                             // numeric_limits min_exponent
    const int  toMaxExp    = (1 << toExpBits) - 1 - toBias;          // numeric_limits max_exponent
    const int  toDigits    = toMant + 1;
    const bool has_inf     = (params.OV_behavior != Inf_Behaviors::Saturating);

    // rep formulas (magnitudes) mirroring numeric_limits<Templated_Float<Fp>>
    const uint32_t signmask = to_signed ? (1u << (n - 1)) : 0u;
    const uint32_t maxFiniteRep = has_inf
        ? (params.IsInf.minPosInf() - 1u)
        : (to_signed ? (((1u << n) - 1u) >> 1) : ((1u << n) - 1u));
    const uint32_t infRep  = has_inf ? params.IsInf.minPosInf() : maxFiniteRep;
    const uint32_t qNaNRep = params.IsNaN.qNanBitPattern();
    const uint32_t sNaNRep = params.IsNaN.sNanBitPattern();

    // Apply sign exactly as run()'s `from_sign_bit ? -to : to` does, i.e. via the
    // negate operator's guard (lo_float.h operator-): negating +0 stays +0 when
    // the sign-bit-only pattern is the format's NaN (no representable -0).
    auto with_sign = [&](uint32_t mag, bool neg) -> uint32_t {
        if (!neg || !to_signed) return mag;
        if (mag == 0 && params.IsNaN(signmask)) return 0u;
        return mag | signmask;
    };

    const bool from_sign_bit = (bit_cast<uint32_t>(from) >> (kFromBits - 1)) != 0;

    // negative value into an unsigned target
    if (!to_signed && from_sign_bit) {
        if (params.unsigned_behavior == Unsigned_behavior::NegtoZero) return 0u;
        if (params.NA_behavior == NaN_Behaviors::_3109) return qNaNRep;
        return sNaNRep;
    }

    const uint32_t from_bits = bit_cast<uint32_t>(from) & 0x7FFFFFFFu;  // |from|

    if (lo_float_internal::isinf(from)) {
        if (sat_mode != Saturation_Mode::SatFinite && has_inf)
            return with_sign(infRep, from_sign_bit);
        return with_sign(maxFiniteRep, from_sign_bit);
    }
    if (lo_float_internal::isnan(from)) return qNaNRep;
    if (from_bits == 0) return with_sign(0u, from_sign_bit);

    const int biased_from_exponent = from_bits >> kFromMantissaBits;

    // ---- branch 1: To has MORE exponents near zero than From (float subnormal in) ----
    if (toMinExp < kFromMinExp) {
        if (biased_from_exponent == 0) {
            WideBits bits = from_bits;
            const int normalization_factor =
                countl_zero<kFromBits, uint32_t, int>(from_bits)
                - (kFromBits - kFromMantissaBits) + 1;
            const int biased_exponent = kExponentOffset - normalization_factor + 1;
            if (biased_exponent <= 0) {
                if (kExponentOffset < 64) bits <<= kExponentOffset;
            } else {
                bits <<= normalization_factor;
                bits &= ~(WideBits{1} << kFromMantissaBits);
                bits |= static_cast<WideBits>(biased_exponent) << kFromMantissaBits;
            }
            if (kDigitShift > 0) bits <<= kDigitShift;
            else { bits = RoundMantissa(bits, -kDigitShift, ps); bits >>= -kDigitShift; }
            return with_sign(static_cast<uint32_t>(bits), from_sign_bit);
        }
    }

    // ---- branch 2: To has FEWER exponents near zero -> To subnormal / zero ----
    if (toMinExp > kFromMinExp) {
        const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
        const int biased_to_exponent = unbiased_exponent + toBias;
        if (biased_to_exponent <= 0) {
            const uint32_t from_has_leading_one = (biased_from_exponent > 0) ? 1u : 0u;
            const int exponent_shift = -kDigitShift - biased_to_exponent + (int)from_has_leading_one;
            uint32_t rounded_from_bits =
                (from_bits & kFromMantissaMask) | (from_has_leading_one << kFromMantissaBits);
            uint32_t bits = 0;
            if (exponent_shift <= kFromMantissaBits + 1) {
                rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift, ps);
                bits = (rounded_from_bits >> exponent_shift);
            } else {
                unsigned long long widened_bits = (unsigned long long)rounded_from_bits;
                switch (round_mode) {
                    case Rounding_Mode::RoundToOdd:
                    case Rounding_Mode::RoundAwayFromZero:
                        bits = (rounded_from_bits > 0 ? 1u : 0u); break;
                    case Rounding_Mode::RoundTowardsZero: bits = 0; break;
                    case Rounding_Mode::RoundUp:
                        bits = (rounded_from_bits && !from_sign_bit > 0 ? 1u : 0u); break;
                    case Rounding_Mode::RoundDown:
                        bits = ((rounded_from_bits) && from_sign_bit > 0 ? 1u : 0u); break;
                    case Rounding_Mode::StochasticRoundingA:
                        widened_bits = lo_float_internal::Stochastic_Round_A(widened_bits, exponent_shift, stoch_len);
                        bits = (widened_bits >> exponent_shift); break;
                    case Rounding_Mode::StochasticRoundingB:
                        widened_bits = lo_float_internal::Stochastic_Round_B(widened_bits, exponent_shift, stoch_len);
                        bits = (widened_bits >> exponent_shift); break;
                    case Rounding_Mode::StochasticRoundingC:
                        widened_bits = lo_float_internal::Stochastic_Round_C(widened_bits, exponent_shift, stoch_len);
                        bits = (widened_bits >> exponent_shift); break;
                    case Rounding_Mode::StochasticRoundingD:
                        widened_bits = lo_float_internal::Stochastic_Round_D(widened_bits, exponent_shift, stoch_len);
                        bits = (widened_bits >> exponent_shift); break;
                    case Rounding_Mode::True_StochasticRounding:
                        widened_bits = lo_float_internal::True_Stochastic_Round(widened_bits, exponent_shift);
                        bits = (widened_bits >> exponent_shift); break;
                    case Rounding_Mode::ProbabilisticRounding:
                        widened_bits = lo_float_internal::Probabilistic_Round(widened_bits, exponent_shift);
                        bits = (widened_bits >> exponent_shift); break;
                    default: bits = 0; break;
                }
            }
            return with_sign(bits, from_sign_bit);
        }
    }

    // ---- normal path ----
    WideBits rounded_from_bits = from_bits;
    if (kDigitShift < 0) {
        rounded_from_bits = RoundMantissa(rounded_from_bits, -kDigitShift, ps);
        rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
    }
    rounded_from_bits += static_cast<WideBits>(kExponentOffset) << kFromMantissaBits;

    uint32_t bits;
    WideBits aligned_highest = maxFiniteRep;
    if (kDigitShift < 0) {
        aligned_highest <<= -kDigitShift;
        bits = static_cast<uint32_t>(rounded_from_bits >> -kDigitShift);
    } else {
        rounded_from_bits <<= kDigitShift;
        bits = static_cast<uint32_t>(rounded_from_bits);
    }

    // overflow: pair(toMaxExp,toDigits) < pair(128,24)  (float max_exponent, digits)
    const bool to_is_narrower = (toMaxExp < 128) || (toMaxExp == 128 && toDigits < 24);
    if (to_is_narrower && rounded_from_bits > aligned_highest) {
        const bool emit_inf = (sat_mode == Saturation_Mode::OvfInf) && has_inf;
        return with_sign(emit_inf ? infRep : maxFiniteRep, from_sign_bit);
    }
    return with_sign(bits, from_sign_bit);
}

// ---------------------------------------------------------------------------
//  Layer 2b: n-bit rep -> float, runtime FloatingPointParams. Runtime inverse
//  of run() with the DEST fixed to float. lowprec->float is an exact (lossless)
//  expanding conversion, so the value is reconstructed directly from the
//  sign/exponent/mantissa fields; NaN/Inf are detected with the format's own
//  checkers, exactly as isnan/isinf do for Templated_Float in lo_float.h.
//  Verified bit-exact against (float)Templated_Float<Fp>::FromRep(rep).
// ---------------------------------------------------------------------------
template <typename FromInf, typename FromNaN>
LOFLOAT_HOST_DEVICE float
convert_rep_to_float(uint32_t rep, FloatingPointParams<FromInf, FromNaN> params) {
    const int  n         = params.bitwidth;
    const int  mant      = params.mantissa_bits;
    const int  bias      = params.bias;
    const bool is_signed = (params.is_signed == Signedness::Signed);
    const bool has_inf   = (params.OV_behavior != Inf_Behaviors::Saturating);

    rep &= (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);

    const bool sign = is_signed ? (((rep >> (n - 1)) & 1u) != 0) : false;
    const uint32_t mag = is_signed ? (rep & ((1u << (n - 1)) - 1u)) : rep;

    // Special values use the format's own checkers (operate on the full rep,
    // mirroring Fp.IsNaN(rep) / Fp.IsInf(rep) in lo_float.h). Inf is checked
    // first, then NaN, matching ConvertImpl::run()'s ordering.
    if (has_inf && params.IsInf(rep))
        return sign ? -std::numeric_limits<float>::infinity()
                    :  std::numeric_limits<float>::infinity();
    if (params.IsNaN(rep))
        return std::numeric_limits<float>::quiet_NaN();

    const uint32_t exp_field  = mag >> mant;
    const uint32_t mant_field = mag & ((1u << mant) - 1u);

    double val;
    if (exp_field == 0)
        val = std::ldexp(static_cast<double>(mant_field), 1 - bias - mant);   // subnormal
    else
        val = std::ldexp(1.0 + static_cast<double>(mant_field) / static_cast<double>(1u << mant),
                         static_cast<int>(exp_field) - bias);                 // normal

    return static_cast<float>(sign ? -val : val);
}

} // namespace lo_pack_internal

// ---------------------------------------------------------------------------
//  Fixed-group primitives (exactly group_count(n) elements <-> group_bytes(n)).
//  encode_group fills dst (which it zeroes first); decode_group reads dst.
// ---------------------------------------------------------------------------

// API 2 encode: pack already-rounded reps from a typed Templated_Float<Fp>.
template <FloatingPointParams Fp, PackEndian E = PackEndian::Forward>
LOFLOAT_HOST_DEVICE int encode_group(uint8_t* dst,
                                     const lo_float_internal::Templated_Float<Fp>* src) {
    namespace P = lo_pack_internal;
    constexpr int n = Fp.bitwidth;
    const int gc = P::pack_group_count(n);
    const int gb = P::pack_group_bytes(n);
    const uint32_t mask = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
    for (int i = 0; i < gb; ++i) dst[i] = 0;
    for (int i = 0; i < gc; ++i)
        P::put_bits(dst, P::bit_offset(i, n, gc, E), static_cast<uint32_t>(src[i].rep()) & mask, n);
    return gb;
}

// API 1 encode: round float -> rep (runtime params), then pack.
template <typename ToInf, typename ToNaN, PackEndian E = PackEndian::Forward>
LOFLOAT_HOST_DEVICE int encode_group(uint8_t* dst, const float* src,
                                     FloatingPointParams<ToInf, ToNaN> params, ProjSpec ps = ProjSpec{}) {
    namespace P = lo_pack_internal;
    const int n  = params.bitwidth;
    const int gc = P::pack_group_count(n);
    const int gb = P::pack_group_bytes(n);
    for (int i = 0; i < gb; ++i) dst[i] = 0;
    for (int i = 0; i < gc; ++i)
        P::put_bits(dst, P::bit_offset(i, n, gc, E), P::convert_float_to_rep(src[i], params, ps), n);
    return gb;
}

// API 1 decode: unpack into typed Templated_Float<Fp> reps.
template <FloatingPointParams Fp, PackEndian E = PackEndian::Forward>
LOFLOAT_HOST_DEVICE int decode_group(lo_float_internal::Templated_Float<Fp>* dst,
                                     const uint8_t* src) {
    namespace P = lo_pack_internal;
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    const int gc = P::pack_group_count(n);
    const int gb = P::pack_group_bytes(n);
    for (int i = 0; i < gc; ++i)
        dst[i] = TF::FromRep(static_cast<UType>(P::get_bits(src, P::bit_offset(i, n, gc, E), n)));
    return gb;
}

// API 2 decode: unpack and convert to float via the runtime codec.
template <typename FromInf, typename FromNaN, PackEndian E = PackEndian::Forward>
LOFLOAT_HOST_DEVICE int decode_group(float* dst, const uint8_t* src,
                                     FloatingPointParams<FromInf, FromNaN> params) {
    namespace P = lo_pack_internal;
    const int n  = params.bitwidth;
    const int gc = P::pack_group_count(n);
    const int gb = P::pack_group_bytes(n);
    for (int i = 0; i < gc; ++i)
        dst[i] = P::convert_rep_to_float(P::get_bits(src, P::bit_offset(i, n, gc, E), n), params);
    return gb;
}

// ---------------------------------------------------------------------------
//  CPU array wrappers.  `count` need not be a multiple of group_count: the
//  final partial group is zero-padded on encode and only `count` elements are
//  produced on decode.  Returns the number of bytes written/consumed.
// ---------------------------------------------------------------------------

// Total bytes a packed buffer needs for `count` elements of an n-bit format.
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE size_t LoF_packed_size(int n, size_t count) {
    return lo_pack_internal::packed_bytes(n, count);
}

// encode: typed source (API 2)
template <FloatingPointParams Fp, PackEndian E = PackEndian::Forward>
int LoF_encode(uint8_t* dst, const lo_float_internal::Templated_Float<Fp>* src, size_t count) {
    using TF = lo_float_internal::Templated_Float<Fp>;
    const int gc = lo_pack_internal::pack_group_count(Fp.bitwidth);
    int written = 0;
    size_t i = 0;
    for (; i + gc <= count; i += gc)
        written += encode_group<Fp, E>(dst + written, src + i);
    if (i < count) {
        TF tmp[64];
        for (int j = 0; j < gc; ++j) tmp[j] = (i + (size_t)j < count) ? src[i + j] : TF{};
        written += encode_group<Fp, E>(dst + written, tmp);
    }
    return written;
}

// encode: float source + runtime params (API 1)
template <PackEndian E = PackEndian::Forward, typename ToInf, typename ToNaN>
int LoF_encode(uint8_t* dst, const float* src, FloatingPointParams<ToInf, ToNaN> params,
               size_t count, ProjSpec ps = ProjSpec{}) {
    const int gc = lo_pack_internal::pack_group_count(params.bitwidth);
    int written = 0;
    size_t i = 0;
    for (; i + gc <= count; i += gc)
        written += encode_group<ToInf, ToNaN, E>(dst + written, src + i, params, ps);
    if (i < count) {
        float tmp[64];
        for (int j = 0; j < gc; ++j) tmp[j] = (i + (size_t)j < count) ? src[i + j] : 0.0f;
        written += encode_group<ToInf, ToNaN, E>(dst + written, tmp, params, ps);
    }
    return written;
}

// decode: typed dest (API 1)
template <FloatingPointParams Fp, PackEndian E = PackEndian::Forward>
int LoF_decode(lo_float_internal::Templated_Float<Fp>* dst, const uint8_t* src, size_t count) {
    using TF = lo_float_internal::Templated_Float<Fp>;
    const int gc = lo_pack_internal::pack_group_count(Fp.bitwidth);
    int consumed = 0;
    size_t i = 0;
    for (; i + gc <= count; i += gc)
        consumed += decode_group<Fp, E>(dst + i, src + consumed);
    if (i < count) {
        TF tmp[64];
        consumed += decode_group<Fp, E>(tmp, src + consumed);
        for (size_t j = 0; i + j < count; ++j) dst[i + j] = tmp[j];
    }
    return consumed;
}

// decode: float dest + runtime params (API 2)
template <PackEndian E = PackEndian::Forward, typename FromInf, typename FromNaN>
int LoF_decode(float* dst, const uint8_t* src, FloatingPointParams<FromInf, FromNaN> params, size_t count) {
    const int gc = lo_pack_internal::pack_group_count(params.bitwidth);
    int consumed = 0;
    size_t i = 0;
    for (; i + gc <= count; i += gc)
        consumed += decode_group<FromInf, FromNaN, E>(dst + i, src + consumed, params);
    if (i < count) {
        float tmp[64];
        consumed += decode_group<FromInf, FromNaN, E>(tmp, src + consumed, params);
        for (size_t j = 0; i + j < count; ++j) dst[i + j] = tmp[j];
    }
    return consumed;
}

// ---------------------------------------------------------------------------
//  GPU array wrappers (one thread per group). Pointers are device pointers.
//  Kernels are templated on TYPES + the endian enum only (never on the
//  class-type FloatingPointParams NTTP, which nvcc cannot put in a launch
//  stub); the runtime params travel as a by-value kernel argument.
// ---------------------------------------------------------------------------
#if defined(USE_CUDA) && defined(__CUDACC__)

namespace lo_pack_internal {

template <typename ToInf, typename ToNaN, PackEndian E>
__global__ void encode_float_kernel(uint8_t* dst, const float* src, size_t count, size_t groups,
                                    FloatingPointParams<ToInf, ToNaN> params, ProjSpec ps) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int gc = pack_group_count(params.bitwidth);
    const int gb = pack_group_bytes(params.bitwidth);
    const size_t base = g * gc;
    uint8_t* d = dst + g * gb;
    float tmp[64];
    for (int j = 0; j < gc; ++j) tmp[j] = (base + (size_t)j < count) ? src[base + j] : 0.0f;
    encode_group<ToInf, ToNaN, E>(d, tmp, params, ps);
}

template <typename FromInf, typename FromNaN, PackEndian E>
__global__ void decode_float_kernel(float* dst, const uint8_t* src, size_t count, size_t groups,
                                    FloatingPointParams<FromInf, FromNaN> params) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int gc = pack_group_count(params.bitwidth);
    const int gb = pack_group_bytes(params.bitwidth);
    const size_t base = g * gc;
    const uint8_t* s = src + g * gb;
    float tmp[64];
    decode_group<FromInf, FromNaN, E>(tmp, s, params);
    for (int j = 0; base + (size_t)j < count && j < gc; ++j) dst[base + j] = tmp[j];
}

} // namespace lo_pack_internal

// encode: float source + runtime params, on device.
template <PackEndian E = PackEndian::Forward, typename ToInf, typename ToNaN>
int LoF_encode_cuda(uint8_t* d_dst, const float* d_src, FloatingPointParams<ToInf, ToNaN> params,
                    size_t count, ProjSpec ps = ProjSpec{}, cudaStream_t stream = 0) {
    const int gc = lo_pack_internal::pack_group_count(params.bitwidth);
    const int gb = lo_pack_internal::pack_group_bytes(params.bitwidth);
    const size_t groups = (count + gc - 1) / gc;
    if (groups == 0) return 0;
    const int threads = 256;
    const size_t blocks = (groups + threads - 1) / threads;
    lo_pack_internal::encode_float_kernel<ToInf, ToNaN, E>
        <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(d_dst, d_src, count, groups, params, ps);
    return static_cast<int>(groups * gb);
}

// decode: float dest + runtime params, on device.
template <PackEndian E = PackEndian::Forward, typename FromInf, typename FromNaN>
int LoF_decode_cuda(float* d_dst, const uint8_t* d_src, FloatingPointParams<FromInf, FromNaN> params,
                    size_t count, cudaStream_t stream = 0) {
    const int gc = lo_pack_internal::pack_group_count(params.bitwidth);
    const int gb = lo_pack_internal::pack_group_bytes(params.bitwidth);
    const size_t groups = (count + gc - 1) / gc;
    if (groups == 0) return 0;
    const int threads = 256;
    const size_t blocks = (groups + threads - 1) / threads;
    lo_pack_internal::decode_float_kernel<FromInf, FromNaN, E>
        <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(d_dst, d_src, count, groups, params);
    return static_cast<int>(groups * gb);
}

#endif // USE_CUDA && __CUDACC__

} // namespace lo_float
