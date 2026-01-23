#include <cmath>
#include <cstdint>
#include <bit>
#include <limits>
#include <algorithm>
#include <random>
#include "lo_float.h"
#include <pybind11/pybind11.h>
#include <xsimd/xsimd.hpp>

namespace py = pybind11;


namespace lo_float {

    namespace lo_float_internal {
struct FloatFormatSpec {
    
    // Format structure
    int exponent_bits;
    int mantissa_bits;
    int total_bits;
    
    // Biases and limits
    int exponent_bias;
    int min_exponent;
    int max_exponent;
    
    // Behaviors
    Signedness signedness;
    NaNBehavior nan_behavior;
    InfBehavior inf_behavior;=
    
    // Derived masks and values
    uint64_t mantissa_mask;
    uint64_t exponent_mask;
    uint64_t sign_bit_mask;
    bool has_infinity;
    bool has_nan;
    
    FloatFormatSpec(int exp_bits, int mant_bits,
                   Signedness sign = Signedness::Signed,
                   InfBehavior inf = InfBehavior::Extended)
        : exponent_bits(exp_bits)
        , mantissa_bits(mant_bits)
        , total_bits(exp_bits + mant_bits + (sign == Signedness::Signed ? 1 : 0))
        , exponent_bias((1 << (exp_bits - 1)) - 1)
        , min_exponent(1 - exponent_bias)
        , max_exponent(exponent_bias)
        , signedness(sign)
        , nan_behavior(nan)
        , inf_behavior(inf)
    {
        mantissa_mask = (1ULL << mantissa_bits) - 1;
        exponent_mask = ((1ULL << exponent_bits) - 1) << mantissa_bits;
        sign_bit_mask = (signedness == Signedness::Signed) ? (1ULL << (total_bits - 1)) : 0;
        
        // Infinity exists if we have special exponent values
        has_infinity = (inf_behavior == InfBehavior::Extended);
        has_nan = (nan_behavior != NaNBehavior::Quiet || nan_behavior != NaNBehavior::Signaling);
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

// Count leading zeros in a 64-bit value
inline int countl_zero_64(uint64_t x) {
    if (x == 0) return 64;
#ifdef __GNUC__
    return __builtin_clzll(x);
#else
    int count = 0;
    for (int i = 63; i >= 0; --i) {
        if (x & (1ULL << i)) break;
        count++;
    }
    return count;
#endif
}

// Random number generator for stochastic rounding
inline std::mt19937_64& get_rng() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng;
}

double construct_max_value(const FloatFormatSpec& spec) {
    // Max value has all exponent bits set (except all-1s if that's reserved for inf/nan)
    // and all mantissa bits set
    
    int max_biased_exp = spec.has_infinity ? 
        ((1 << spec.exponent_bits) - 2) :  // Reserve all-1s for inf/nan
        ((1 << spec.exponent_bits) - 1);   // Use all-1s
    
    int unbiased_exp = max_biased_exp - spec.exponent_bias;
    
    // All mantissa bits set means (1 + (2^mantissa_bits - 1) / 2^mantissa_bits)
    // = 1 + 1 - 2^(-mantissa_bits)
    // = 2 - 2^(-mantissa_bits)
    double mantissa_value = 2.0 - std::ldexp(1.0, -spec.mantissa_bits);
    
    return std::ldexp(mantissa_value, unbiased_exp);
}

// ============================================================================
// Helper: Construct double from custom format bits
// ============================================================================

double reconstruct_double_from_custom(
    uint64_t custom_bits,
    const FloatFormatSpec& spec,
    bool sign_bit)
{
    // Extract exponent and mantissa
    uint64_t exponent_val = (custom_bits >> spec.mantissa_bits) & ((1ULL << spec.exponent_bits) - 1);
    uint64_t mantissa_val = custom_bits & spec.mantissa_mask;
    
    // Handle special cases
    uint64_t max_exponent = (1ULL << spec.exponent_bits) - 1;
    
    // Zero
    if (exponent_val == 0 && mantissa_val == 0) {
        return sign_bit ? -0.0 : 0.0;
    }
    
    // Infinity
    if (spec.has_infinity && exponent_val == max_exponent && mantissa_val == 0) {
        return sign_bit ? -std::numeric_limits<double>::infinity() 
                        : std::numeric_limits<double>::infinity();
    }
    
    // NaN
    if (exponent_val == max_exponent && mantissa_val != 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Subnormal in custom format
    if (exponent_val == 0) {
        // Subnormal: no implicit leading 1
        int unbiased_exp = spec.min_exponent;
        double mantissa_value = static_cast<double>(mantissa_val) / (1ULL << spec.mantissa_bits);
        return sign_bit ? -std::ldexp(mantissa_value, unbiased_exp)
                        : std::ldexp(mantissa_value, unbiased_exp);
    }
    
    // Normal number
    int unbiased_exp = static_cast<int>(exponent_val) - spec.exponent_bias;
    
    // Mantissa: implicit leading 1 + fractional part
    double mantissa_value = 1.0 + static_cast<double>(mantissa_val) / (1ULL << spec.mantissa_bits);
    
    double result = std::ldexp(mantissa_value, unbiased_exp);
    return sign_bit ? -result : result;
}

double convert_double_to_custom_to_double(
    double input,
    const FloatFormatSpec& to_spec,
    RoundingMode rounding_mode = RoundingMode::RoundToNearestEven,
    int stoch_len = 0)
{
    using FromBits = uint64_t;
    
    // Constants for double (IEEE 754 binary64)
    constexpr int kFromBits = 64;
    constexpr int kFromMantissaBits = 52;
    constexpr int kFromExponentBias = 1023;
    constexpr int kFromExponentBits = 11;
    
    // Runtime "To" parameters
    const int kToMantissaBits = to_spec.mantissa_bits;
    const int kToExponentBias = to_spec.exponent_bias;
    const int kToExponentBits = to_spec.exponent_bits;
    
    // Compute shifts
    const int kDigitShift = kToMantissaBits - kFromMantissaBits;
    const int kExponentOffset = kToExponentBias - kFromExponentBias;
    
    // Extract sign bit
    FromBits from_bits_raw = std::bit_cast<FromBits>(input);
    bool from_sign_bit = (from_bits_raw >> 63) & 1;
    
    // Handle unsigned target with negative input
    if (to_spec.signedness == FloatFormatSpec::Signedness::Unsigned && from_sign_bit) {
        // Underflow to zero or NaN
        if (to_spec.nan_behavior == FloatFormatSpec::NaNBehavior::Signaling) {
            return std::numeric_limits<double>::signaling_NaN();
        } else if (to_spec.nan_behavior == FloatFormatSpec::NaNBehavior::Quiet) {
            return std::numeric_limits<double>::quiet_NaN();
        } else {
            return 0.0;  // Default: clamp to zero
        }
    }
    
    // Get absolute value bits
    FromBits from_bits = std::bit_cast<FromBits>(std::abs(input));
    
    // Handle special values
    if (std::isinf(input)) {
        if (to_spec.inf_behavior == FloatFormatSpec::InfBehavior::Saturating) {
            double max_val = construct_max_value(to_spec);
            return from_sign_bit ? -max_val : max_val;
        } else {
            // Extended: preserve infinity
            return input;
        }
    }
    
    if (std::isnan(input)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (from_bits == 0) {
        return from_sign_bit ? -0.0 : 0.0;
    }
    
    // Extract biased exponent from source
    const int biased_from_exponent = (from_bits >> kFromMantissaBits) & 0x7FF;
    
    // ========================================================================
    // Case 1: Target supports MORE exponents near zero (subnormals -> normals)
    // ========================================================================
    
    if (to_spec.min_exponent < (1 - kFromExponentBias)) {
        if (biased_from_exponent == 0) {
            // Source is subnormal
            uint64_t bits = from_bits;
            
            // Count leading zeros to find normalization factor
            const int normalization_factor = 
                countl_zero_64(from_bits << (64 - kFromMantissaBits - 1)) - 
                (64 - kFromMantissaBits - 1);
            
            const int biased_exponent = kExponentOffset - normalization_factor + 1;
            
            if (biased_exponent <= 0) {
                // Result is still subnormal
                if (kExponentOffset < 64) {
                    bits <<= kExponentOffset;
                }
            } else {
                // Result becomes normal
                bits <<= normalization_factor;
                bits &= ~(1ULL << kFromMantissaBits);  // Clear hidden bit
                bits |= static_cast<uint64_t>(biased_exponent) << kFromMantissaBits;
            }
            
            // Round mantissa if shrinking
            if (kDigitShift > 0) {
                bits <<= kDigitShift;
            } else {
                bits = RoundMantissa(bits, -kDigitShift, round_mode, stoch_len);
                bits >>= -kDigitShift;
            }
            
            // Mask to target bit width
            uint64_t target_mask = (1ULL << to_spec.total_bits) - 1;
            bits &= target_mask;
            
            // Convert back to double
            double result = reconstruct_double_from_custom(bits, to_spec, from_sign_bit);
            return result;
        }
    }
    
    // ========================================================================
    // Case 2: Target supports FEWER exponents near zero (normals -> subnormals)
    // ========================================================================
    
    if (to_spec.min_exponent > (1 - kFromExponentBias)) {
        const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
        const int biased_to_exponent = unbiased_exponent + kToExponentBias;
        
        // Check if result becomes subnormal or underflows
        if (biased_to_exponent <= 0) {
            // Will be subnormal or zero
            const bool from_has_leading_one = (biased_from_exponent > 0);
            int exponent_shift = -kDigitShift - biased_to_exponent + (from_has_leading_one ? 1 : 0);
            
            uint64_t rounded_from_bits = (from_bits & ((1ULL << kFromMantissaBits) - 1)) |
                                         (from_has_leading_one ? (1ULL << kFromMantissaBits) : 0);
            
            uint64_t bits = 0;
            
            if (exponent_shift <= kFromMantissaBits + 1) {
                // Round and shift
                rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift,
                                                     rounding_mode, stoch_len);
                bits = rounded_from_bits >> exponent_shift;
            } else {
                // Underflow: very small number, special rounding
                switch (rounding_mode) {
                    case RoundingMode::RoundAwayFromZero:
                        bits = (rounded_from_bits > 0) ? 1 : 0;
                        break;
                    
                    case RoundingMode::RoundTowardsZero:
                        bits = 0;
                        break;
                    
                    case RoundingMode::RoundUp:
                        bits = (rounded_from_bits && !from_sign_bit) ? 1 : 0;
                        break;
                    
                    case RoundingMode::RoundDown:
                        bits = (rounded_from_bits && from_sign_bit) ? 1 : 0;
                        break;
                    
                    case RoundingMode::StochasticRoundingA:
                    case RoundingMode::StochasticRoundingB:
                    case RoundingMode::StochasticRoundingC:
                    case RoundingMode::True_StochasticRounding:
                    case RoundingMode::ProbabilisticRounding: {
                        // Probabilistic: small chance of rounding up
                        std::uniform_real_distribution<double> dist(0.0, 1.0);
                        double prob = std::ldexp(static_cast<double>(rounded_from_bits), -exponent_shift);
                        bits = (dist(get_rng()) < prob) ? 1 : 0;
                        break;
                    }
                    
                    default:
                        bits = 0;
                        break;
                }
            }
            
            // Convert back to double
            double result = reconstruct_double_from_custom(bits, to_spec, from_sign_bit);
            return result;
        }
    }
    
    // ========================================================================
    // Case 3: Normal conversion (no subnormal boundary crossing)
    // ========================================================================
    
    uint64_t rounded_from_bits = from_bits;
    
    // Round mantissa if shrinking
    if (kDigitShift < 0) {
        rounded_from_bits = RoundMantissa(rounded_from_bits, -kDigitShift, rounding_mode, stoch_len);
        // Zero out tail bits
        rounded_from_bits &= ~((1ULL << (-kDigitShift)) - 1);
    }
    
    // Adjust exponent bias
    rounded_from_bits += static_cast<uint64_t>(kExponentOffset) << kFromMantissaBits;
    
    // Shift to target mantissa width
    uint64_t bits;
    if (kDigitShift < 0) {
        bits = rounded_from_bits >> (-kDigitShift);
    } else {
        bits = rounded_from_bits << kDigitShift;
    }
    
    // Check for overflow
    uint64_t max_normal_exp = to_spec.has_infinity ? 
        ((1ULL << kToExponentBits) - 2) :
        ((1ULL << kToExponentBits) - 1);
    
    uint64_t result_exponent = (bits >> kToMantissaBits) & ((1ULL << kToExponentBits) - 1);
    
    if (result_exponent > max_normal_exp) {
        // Overflow
        if (to_spec.has_infinity) {
            return from_sign_bit ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
        } else {
            // Saturate to max
            double max_val = construct_max_value(to_spec);
            return from_sign_bit ? -max_val : max_val;
        }
    }
    
    // Also check by comparing full bit patterns
    uint64_t target_mask = (1ULL << to_spec.total_bits) - 1;
    bits &= target_mask;
    
    // Construct max representable value in target format
    uint64_t max_bits = (max_normal_exp << kToMantissaBits) | to_spec.mantissa_mask;
    
    if (bits > max_bits) {
        // Overflow
        if (to_spec.has_infinity) {
            return from_sign_bit ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
        } else {
            double max_val = construct_max_value(to_spec);
            return from_sign_bit ? -max_val : max_val;
        }
    }
    
    // Convert back to double
    double result = reconstruct_double_from_custom(bits, to_spec, from_sign_bit);
    return result;
}

float reconstruct_float_from_custom(
    uint64_t custom_bits,
    const FloatFormatSpec& spec,
    bool sign_bit)
{
    // Extract exponent and mantissa
    uint64_t exponent_val = (custom_bits >> spec.mantissa_bits) & ((1ULL << spec.exponent_bits) - 1);
    uint64_t mantissa_val = custom_bits & spec.mantissa_mask;
    
    // Handle special cases
    uint64_t max_exponent = (1ULL << spec.exponent_bits) - 1;
    
    // Zero
    if (exponent_val == 0 && mantissa_val == 0) {
        return sign_bit ? -0.0f : 0.0f;
    }
    
    // Infinity
    if (spec.has_infinity && exponent_val == max_exponent && mantissa_val == 0) {
        return sign_bit ? -std::numeric_limits<float>::infinity() 
                        : std::numeric_limits<float>::infinity();
    }
    
    // NaN
    if (exponent_val == max_exponent && mantissa_val != 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    
    // Subnormal in custom format
    if (exponent_val == 0) {
        // Subnormal: no implicit leading 1
        int unbiased_exp = spec.min_exponent;
        float mantissa_value = static_cast<float>(mantissa_val) / (1ULL << spec.mantissa_bits);
        return sign_bit ? -std::ldexp(mantissa_value, unbiased_exp)
                        : std::ldexp(mantissa_value, unbiased_exp);
    }
    
    // Normal number
    int unbiased_exp = static_cast<int>(exponent_val) - spec.exponent_bias;
    
    // Mantissa: implicit leading 1 + fractional part
    float mantissa_value = 1.0f + static_cast<float>(mantissa_val) / (1ULL << spec.mantissa_bits);
    
    float result = std::ldexp(mantissa_value, unbiased_exp);
    return sign_bit ? -result : result;
}

float convert_float_to_custom_to_float(
    float input,
    const FloatFormatSpec& to_spec,
    RoundingMode rounding_mode = RoundingMode::RoundToNearestEven,
    int stoch_len = 0)
{
    using FromBits = uint32_t;
    
    // Constants for float (IEEE 754 binary32)
    constexpr int kFromBits = 32;
    constexpr int kFromMantissaBits = 23;
    constexpr int kFromExponentBias = 127;
    constexpr int kFromExponentBits = 8;
    
    // Runtime "To" parameters
    const int kToMantissaBits = to_spec.mantissa_bits;
    const int kToExponentBias = to_spec.exponent_bias;
    const int kToExponentBits = to_spec.exponent_bits;
    
    // Compute shifts
    const int kDigitShift = kToMantissaBits - kFromMantissaBits;
    const int kExponentOffset = kToExponentBias - kFromExponentBias;
    
    // Extract sign bit
    FromBits from_bits_raw = std::bit_cast<FromBits>(input);
    bool from_sign_bit = (from_bits_raw >> 31) & 1;
    
    // Handle unsigned target with negative input
    if (to_spec.signedness == FloatFormatSpec::Signedness::Unsigned && from_sign_bit) {
        // Underflow to zero or NaN
        if (to_spec.nan_behavior == FloatFormatSpec::NaNBehavior::Signaling) {
            return std::numeric_limits<float>::signaling_NaN();
        } else if (to_spec.nan_behavior == FloatFormatSpec::NaNBehavior::Quiet) {
            return std::numeric_limits<float>::quiet_NaN();
        } else {
            return 0.0f;  // Default: clamp to zero
        }
    }
    
    // Get absolute value bits
    FromBits from_bits = std::bit_cast<FromBits>(std::abs(input));
    
    // Handle special values
    if (std::isinf(input)) {
        if (to_spec.inf_behavior == FloatFormatSpec::InfBehavior::Saturating) {
            float max_val = construct_max_value_float(to_spec);
            return from_sign_bit ? -max_val : max_val;
        } else {
            // Extended: preserve infinity
            return input;
        }
    }
    
    if (std::isnan(input)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    
    if (from_bits == 0) {
        return from_sign_bit ? -0.0f : 0.0f;
    }
    
    // Extract biased exponent from source
    const int biased_from_exponent = (from_bits >> kFromMantissaBits) & 0xFF;
    
    // ========================================================================
    // Case 1: Target supports MORE exponents near zero (subnormals -> normals)
    // ========================================================================
    
    if (to_spec.min_exponent < (1 - kFromExponentBias)) {
        if (biased_from_exponent == 0) {
            // Source is subnormal
            uint32_t bits = from_bits;
            
            // Count leading zeros to find normalization factor
            const int normalization_factor = 
                countl_zero_32(from_bits << (32 - kFromMantissaBits - 1)) - 
                (32 - kFromMantissaBits - 1);
            
            const int biased_exponent = kExponentOffset - normalization_factor + 1;
            
            if (biased_exponent <= 0) {
                // Result is still subnormal
                if (kExponentOffset < 32) {
                    bits <<= kExponentOffset;
                }
            } else {
                // Result becomes normal
                bits <<= normalization_factor;
                bits &= ~(1U << kFromMantissaBits);  // Clear hidden bit
                bits |= static_cast<uint32_t>(biased_exponent) << kFromMantissaBits;
            }
            
            // Round mantissa if shrinking
            if (kDigitShift > 0) {
                bits <<= kDigitShift;
            } else {
                bits = RoundMantissa(bits, -kDigitShift, round_mode, stoch_len);
                bits >>= -kDigitShift;
            }
            
            // Mask to target bit width
            uint64_t target_mask = (1ULL << to_spec.total_bits) - 1;
            bits &= target_mask;
            
            // Convert back to float
            float result = reconstruct_float_from_custom(bits, to_spec, from_sign_bit);
            return result;
        }
    }
    
    // ========================================================================
    // Case 2: Target supports FEWER exponents near zero (normals -> subnormals)
    // ========================================================================
    
    if (to_spec.min_exponent > (1 - kFromExponentBias)) {
        const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
        const int biased_to_exponent = unbiased_exponent + kToExponentBias;
        
        // Check if result becomes subnormal or underflows
        if (biased_to_exponent <= 0) {
            // Will be subnormal or zero
            const bool from_has_leading_one = (biased_from_exponent > 0);
            int exponent_shift = -kDigitShift - biased_to_exponent + (from_has_leading_one ? 1 : 0);
            
            uint32_t rounded_from_bits = (from_bits & ((1U << kFromMantissaBits) - 1)) |
                                         (from_has_leading_one ? (1U << kFromMantissaBits) : 0);
            
            uint32_t bits = 0;
            
            if (exponent_shift <= kFromMantissaBits + 1) {
                // Round and shift
                rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift,
                                                             rounding_mode,
                                                             stoch_len);
                bits = rounded_from_bits >> exponent_shift;
            } else {
                // Underflow: very small number, special rounding
                switch (rounding_mode) {
                    case RoundingMode::RoundAwayFromZero:
                        bits = (rounded_from_bits > 0) ? 1 : 0;
                        break;
                    
                    case RoundingMode::RoundTowardsZero:
                        bits = 0;
                        break;
                    
                    case RoundingMode::RoundUp:
                        bits = (rounded_from_bits && !from_sign_bit) ? 1 : 0;
                        break;
                    
                    case RoundingMode::RoundDown:
                        bits = (rounded_from_bits && from_sign_bit) ? 1 : 0;
                        break;
                    
                    case RoundingMode::StochasticRoundingA:
                    case RoundingMode::StochasticRoundingB:
                    case RoundingMode::StochasticRoundingC:
                    case RoundingMode::True_StochasticRounding:
                    case RoundingMode::ProbabilisticRounding: {
                        // Probabilistic: small chance of rounding up
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        float prob = std::ldexp(static_cast<float>(rounded_from_bits), -exponent_shift);
                        bits = (dist(get_rng()) < prob) ? 1 : 0;
                        break;
                    }
                    
                    default:
                        bits = 0;
                        break;
                }
            }
            
            // Convert back to float
            float result = reconstruct_float_from_custom(bits, to_spec, from_sign_bit);
            return result;
        }
    }
    
    // ========================================================================
    // Case 3: Normal conversion (no subnormal boundary crossing)
    // ========================================================================
    
    uint32_t rounded_from_bits = from_bits;
    
    // Round mantissa if shrinking
    if (kDigitShift < 0) {
        rounded_from_bits = RoundMantissa(rounded_from_bits, -kDigitShift, rounding_mode, stoch_len);
        // Zero out tail bits
        rounded_from_bits &= ~((1U << (-kDigitShift)) - 1);
    }
    
    // Adjust exponent bias
    rounded_from_bits += static_cast<uint32_t>(kExponentOffset) << kFromMantissaBits;
    
    // Shift to target mantissa width
    uint32_t bits;
    if (kDigitShift < 0) {
        bits = rounded_from_bits >> (-kDigitShift);
    } else {
        bits = rounded_from_bits << kDigitShift;
    }
    
    // Check for overflow
    uint64_t max_normal_exp = to_spec.has_infinity ? 
        ((1ULL << kToExponentBits) - 2) :
        ((1ULL << kToExponentBits) - 1);
    
    uint32_t result_exponent = (bits >> kToMantissaBits) & ((1U << kToExponentBits) - 1);
    
    if (result_exponent > max_normal_exp) {
        // Overflow
        if (to_spec.has_infinity) {
            return from_sign_bit ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
        } else {
            // Saturate to max
            float max_val = construct_max_value_float(to_spec);
            return from_sign_bit ? -max_val : max_val;
        }
    }
    
    // Also check by comparing full bit patterns
    uint64_t target_mask = (1ULL << to_spec.total_bits) - 1;
    bits &= target_mask;
    
    // Construct max representable value in target format
    uint64_t max_bits = (max_normal_exp << kToMantissaBits) | to_spec.mantissa_mask;
    
    if (bits > max_bits) {
        // Overflow
        if (to_spec.has_infinity) {
            return from_sign_bit ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
        } else {
            float max_val = construct_max_value_float(to_spec);
            return from_sign_bit ? -max_val : max_val;
        }
    }
    
    // Convert back to float
    float result = reconstruct_float_from_custom(bits, to_spec, from_sign_bit);
    return result;
}

template <class arch = xsimd::default_arch>
float reconstruct_float_from_custom_simd(
    uint32_t custom_bits,
    const FloatFormatSpec& spec,
    bool sign_bit) {


}


} //namespace lo_float_internal
} //namespace lo_float

// ============================================================================
// Pybind11 Module Definition
// ============================================================================

PYBIND11_MODULE(float_converter, m) {
    m.doc() = "Runtime float format converter with arbitrary precision support";
    
    // Expose FloatFormatSpec
    py::class_<FloatFormatSpec>(m, "FloatFormatSpec")
        .def(py::init<int, int, 
                      FloatFormatSpec::Signedness,
                      FloatFormatSpec::NaNBehavior,
                      FloatFormatSpec::InfBehavior>(),
             py::arg("exponent_bits"),
             py::arg("mantissa_bits"),
             py::arg("signedness") = FloatFormatSpec::Signedness::Signed,
             py::arg("nan_behavior") = FloatFormatSpec::NaNBehavior::Quiet,
             py::arg("inf_behavior") = FloatFormatSpec::InfBehavior::Extended,
             "Create a custom floating-point format specification\n\n"
             "Args:\n"
             "    exponent_bits: Number of exponent bits\n"
             "    mantissa_bits: Number of mantissa bits (excluding implicit bit)\n"
             "    signedness: Signed or Unsigned\n"
             "    nan_behavior: Quiet or Signaling NaN behavior\n"
             "    inf_behavior: Saturating (clamp to max) or Extended (support infinity)")
        .def_readonly("exponent_bits", &FloatFormatSpec::exponent_bits)
        .def_readonly("mantissa_bits", &FloatFormatSpec::mantissa_bits)
        .def_readonly("total_bits", &FloatFormatSpec::total_bits)
        .def_readonly("exponent_bias", &FloatFormatSpec::exponent_bias)
        .def_readonly("min_exponent", &FloatFormatSpec::min_exponent)
        .def_readonly("max_exponent", &FloatFormatSpec::max_exponent)
        .def("__repr__", [](const FloatFormatSpec& spec) {
            return "<FloatFormatSpec E" + std::to_string(spec.exponent_bits) +
                   "M" + std::to_string(spec.mantissa_bits) +
                   " (" + std::to_string(spec.total_bits) + " bits total)>";
        });
    
    // Expose enums
    py::enum_<FloatFormatSpec::Signedness>(m, "Signedness")
        .value("Signed", FloatFormatSpec::Signedness::Signed)
        .value("Unsigned", FloatFormatSpec::Signedness::Unsigned)
        .export_values();
    
    py::enum_<FloatFormatSpec::NaNBehavior>(m, "NaNBehavior")
        .value("Quiet", FloatFormatSpec::NaNBehavior::Quiet)
        .value("Signaling", FloatFormatSpec::NaNBehavior::Signaling)
        .export_values();
    
    py::enum_<FloatFormatSpec::InfBehavior>(m, "InfBehavior")
        .value("Saturating", FloatFormatSpec::InfBehavior::Saturating)
        .value("Extended", FloatFormatSpec::InfBehavior::Extended)
        .export_values();
    
    py::enum_<RoundingMode>(m, "RoundingMode")
        .value("RoundToNearestEven", RoundingMode::RoundToNearestEven)
        .value("RoundAwayFromZero", RoundingMode::RoundAwayFromZero)
        .value("RoundTowardsZero", RoundingMode::RoundTowardsZero)
        .value("RoundUp", RoundingMode::RoundUp)
        .value("RoundDown", RoundingMode::RoundDown)
        .value("StochasticRoundingA", RoundingMode::StochasticRoundingA)
        .value("StochasticRoundingB", RoundingMode::StochasticRoundingB)
        .value("StochasticRoundingC", RoundingMode::StochasticRoundingC)
        .value("True_StochasticRounding", RoundingMode::True_StochasticRounding)
        .value("ProbabilisticRounding", RoundingMode::ProbabilisticRounding)
        .export_values();
    
    // Main conversion function
    m.def("convert_scalar", &convert_double_to_custom_to_double,
          py::arg("value"),
          py::arg("to_format"),
          py::arg("rounding_mode") = RoundingMode::RoundToNearestEven,
          py::arg("stoch_len") = 0,
          "Convert a double precision value through a custom float format and back\n\n"
          "This simulates quantization: double -> custom_format -> double\n\n"
          "Args:\n"
          "    value: Input value (Python float/double)\n"
          "    to_format: Target format specification (FloatFormatSpec)\n"
          "    rounding_mode: Rounding behavior (default: RoundToNearestEven)\n"
          "    stoch_len: Parameter for stochastic rounding modes (default: 0)\n\n"
          "Returns:\n"
          "    float: The value after quantization to custom format, represented as double");
    
    // Utility function to get max value
    m.def("get_max_value", &construct_max_value,
          py::arg("format"),
          "Get the maximum representable finite value for a given format");
}