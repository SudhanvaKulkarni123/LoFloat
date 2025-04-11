/// @author Sudhanva Kulkarni
/// @file
/// @brief This header file defines several enums and a struct that describe
///        floating-point parameters (bit width, rounding, infinity/NaN behavior, etc.)
///        used by the lo_float library.

#pragma once

#include <cstdint>
#include <concepts>

namespace lo_float {

/**
 * @enum Rounding_Mode
 * @brief Defines different rounding strategies for floating-point operations. Note that if the length of the stochastic bitstring cannot be set to 0 if the rounding mode is StochasticRounding.
 */
enum Rounding_Mode : uint8_t {
    /// @brief Round to the nearest representable value.  
    /// If equidistant between two representable values, round to the one with an even least significant bit.
    RoundToNearestEven = 0,

    /// @brief Round toward zero (truncate fractional part).
    RoundTowardsZero = 1,

    /// @brief Round away from zero (always go to the numerically larger magnitude).
    RoundAwayFromZero = 2,

    /// @brief Use stochastic rounding to decide how to round the fractional part.
    StochasticRounding = 3,

    /// @brief Round to the nearest representable value.  
    /// If equidistant, pick the one whose least significant bit is odd.
    RoundToNearestOdd = 4,

    /// @brief Round down (toward -∞), sometimes called "floor" for positive values.
    RoundDown = 5,

    /// @brief Round up (toward +∞), sometimes called "ceiling" for positive values.
    RoundUp = 6,

    /// @brief Round ties away from zero.  
    /// If exactly halfway between two representable values, pick the one with larger magnitude.
    RoundTiesToAway = 7
};

/**
 * @enum Signedness
 * @brief Indicates whether a floating-point format is signed or unsigned.
 */
enum Signedness : uint8_t {
    /// @brief The format uses one sign bit (positive/negative).
    Signed = 0,

    /// @brief The format has no sign bit (only non-negative values).
    Unsigned = 1
};

/**
 * @enum Inf_Behaviors
 * @brief Describes how infinities behave or are handled by the format.
 */
enum Inf_Behaviors : uint8_t {
    /// @brief Non-trapping infinities are allowed (like IEEE 754 behavior).
    NonTrappingInf = 0,

    /// @brief Saturate to the maximum representable value instead of producing an infinity.
    Saturating = 1,

    /// @brief Trapping infinities (reserved for future use or custom behavior).
    Trapping = 2
};

/**
 * @enum NaN_Behaviors
 * @brief Describes how NaNs (Not-a-Number) are handled by the format.
 */
enum NaN_Behaviors : uint8_t {
    /// @brief Support quiet NaNs (non-signaling) in the format (like IEEE 754).
    QuietNaN = 0,

    /// @brief No NaN support (e.g., produce some alternate representation or saturate).
    NoNaN = 1,

    /// @brief Use signaling NaNs, which can trigger additional exceptions or checks.
    SignalingNaN = 2
};

// -------------------------------------------------------------------------
// Concepts for checking Infinity and NaN - used in the FloatingPointParams.
// -------------------------------------------------------------------------

/**
 * @concept InfChecker
 * @brief A type that can detect and produce bit patterns for infinities.
 * 
 * A valid InfChecker must:
 * - Be callable with a `uint64_t` returning a `bool` indicating if that bit pattern is infinite.
 * - Provide `infBitPattern()`, `minNegInf()`, and `minPosInf()` that each return a `uint64_t`.
 */
template <typename T>
concept InfChecker = requires(T t, uint64_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
    { t.infBitPattern() } -> std::convertible_to<uint64_t>;
    { t.minNegInf() } -> std::convertible_to<uint64_t>;
    { t.minPosInf() } -> std::convertible_to<uint64_t>;
};

/**
 * @concept NaNChecker
 * @brief A type that can detect and produce bit patterns for NaNs.
 * 
 * A valid NaNChecker must:
 * - Be callable with a `uint64_t` returning a `bool` indicating if that bit pattern is NaN.
 * - Provide `qNanBitPattern()` and `sNanBitPattern()` that each return a `uint64_t`.
 */
template <typename T>
concept NaNChecker = requires(T t, uint64_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
    { t.qNanBitPattern() } -> std::convertible_to<uint64_t>;
    { t.sNanBitPattern() } -> std::convertible_to<uint64_t>;
};

/**
 * @struct FloatingPointParams
 * @brief Encapsulates the parameters and behavior for a floating-point format.
 *
 * @tparam IsInfFunctor A functor conforming to @ref InfChecker
 * @tparam IsNaNFunctor A functor conforming to @ref NaNChecker
 */
template<InfChecker IsInfFunctor, NaNChecker IsNaNFunctor>
struct FloatingPointParams
{
    /// @brief Total bit width of the floating-point number (including sign, exponent, mantissa).
    int bitwidth;

    /// @brief Number of bits in the mantissa (fraction).  
    /// Exponent bits = bitwidth - mantissa_bits (minus sign bit if signed).
    int mantissa_bits;

    /// @brief The exponent bias used by the format.
    int bias;

    /// @brief Specifies how rounding is performed (see @ref Rounding_Mode).
    Rounding_Mode rounding_mode;

    /// @brief Describes how infinities are handled (see @ref Inf_Behaviors).
    Inf_Behaviors OV_behavior;

    /// @brief Describes how NaNs are handled (see @ref NaN_Behaviors).
    NaN_Behaviors NA_behavior;

    /// @brief Indicates whether this format is signed or unsigned (see @ref Signedness).
    Signedness is_signed;

    /// @brief A functor for checking and generating infinite values (must satisfy @ref InfChecker).
    IsInfFunctor IsInf;

    /// @brief A functor for checking and generating NaN values (must satisfy @ref NaNChecker).
    IsNaNFunctor IsNaN;

    /// @brief Number of bits used for stochastic rounding. This value is ignored if the rounding mode is not StochasticRounding. If set to 0 with StochasticRounding, we throw an error
    int StochasticRoundingBits;

    /**
     * @brief Constructs a FloatingPointParams with the specified parameters.
     * @param bw Total bitwidth of the floating-point format.
     * @param mb Number of mantissa (fraction) bits.
     * @param b Exponent bias.
     * @param rm Rounding mode (see @ref Rounding_Mode).
     * @param ovb How infinities are handled (see @ref Inf_Behaviors).
     * @param nab How NaNs are handled (see @ref NaN_Behaviors).
     * @param is_signed Indicates signedness (see @ref Signedness).
     * @param IsInf A functor conforming to @ref InfChecker for inf detection/creation.
     * @param IsNaN A functor conforming to @ref NaNChecker for NaN detection/creation.
     * @param stoch_length Number of bits for stochastic rounding (default=0).
     */
    constexpr FloatingPointParams(
        int bw,
        int mb,
        int b,
        Rounding_Mode rm,
        Inf_Behaviors ovb,
        NaN_Behaviors nab,
        Signedness is_signed,
        IsInfFunctor IsInf,
        IsNaNFunctor IsNaN,
        int stoch_length = 0
    )
      : bitwidth(bw)
      , mantissa_bits(mb)
      , bias(b)
      , rounding_mode(rm)
      , OV_behavior(ovb)
      , NA_behavior(nab)
      , is_signed(is_signed)
      , IsInf(IsInf)
      , IsNaN(IsNaN)
      , StochasticRoundingBits(stoch_length)
    {}
};

/**
 * @struct SingleInfChecker
 * @brief Detects and produces bit patterns for infinities in 32-bit float format.
 *
 * Infinity is indicated by exponent=0xFF and fraction=0.
 */
struct SingleInfChecker {
    bool operator()(uint32_t bits) const {
        static constexpr uint32_t ExponentMask = 0x7F800000;
        static constexpr uint32_t FractionMask = 0x007FFFFF;
        // Infinity => exponent=0xFF, fraction=0
        bool isInf = ((bits & ExponentMask) == ExponentMask) &&
                     ((bits & FractionMask) == 0);
        return isInf;
    }

    uint32_t infBitPattern() const {
        // +∞ => 0x7F800000
        return 0x7F800000;
    }

    uint32_t minNegInf() const {
        // -∞ => 0xFF800000
        return 0xFF800000;
    }

    uint32_t minPosInf() const {
        // +∞ => 0x7F800000
        return 0x7F800000;
    }
};

/**
 * @struct SingleNaNChecker
 * @brief Detects and produces bit patterns for NaNs in 32-bit float format.
 *
 * NaN is indicated by exponent=0xFF and fraction!=0.
 */
struct SingleNaNChecker {
    bool operator()(uint32_t bits) const {
        static constexpr uint32_t ExponentMask = 0x7F800000;
        static constexpr uint32_t FractionMask = 0x007FFFFF;
        // NaN => exponent=0xFF, fraction!=0
        bool isNaN = ((bits & ExponentMask) == ExponentMask) &&
                     ((bits & FractionMask) != 0);
        return isNaN;
    }

    uint32_t qNanBitPattern() const {
        // quiet-NaN => 0x7FC00000
        return 0x7FC00000;
    }

    uint32_t sNanBitPattern() const {
        // signaling-NaN => 0x7F800001
        return 0x7F800001;
    }
};

/**
 * @struct DoubleInfChecker
 * @brief Detects and produces bit patterns for infinities in 64-bit float (double) format.
 *
 * Infinity is indicated by exponent=0x7FF and fraction=0.
 */
struct DoubleInfChecker {
    bool operator()(uint64_t bits) const {
        static constexpr uint64_t ExponentMask = 0x7FF0000000000000ULL;
        static constexpr uint64_t FractionMask = 0x000FFFFFFFFFFFFFULL;
        // Infinity => exponent=0x7FF, fraction=0
        bool isInf = ((bits & ExponentMask) == ExponentMask) &&
                     ((bits & FractionMask) == 0);
        return isInf;
    }

    uint64_t infBitPattern() const {
        // +∞ => 0x7FF0000000000000
        return 0x7FF0000000000000ULL;
    }

    uint64_t minNegInf() const {
        // -∞ => 0xFFF0000000000000
        return 0xFFF0000000000000ULL;
    }

    uint64_t minPosInf() const {
        // +∞ => 0x7FF0000000000000
        return 0x7FF0000000000000ULL;
    }
};

/**
 * @struct DoubleNaNChecker
 * @brief Detects and produces bit patterns for NaNs in 64-bit float (double) format.
 *
 * NaN is indicated by exponent=0x7FF and fraction!=0.
 */
struct DoubleNaNChecker {
    bool operator()(uint64_t bits) const {
        static constexpr uint64_t ExponentMask = 0x7FF0000000000000ULL;
        static constexpr uint64_t FractionMask = 0x000FFFFFFFFFFFFFULL;
        // NaN => exponent=0x7FF, fraction!=0
        bool isNaN = ((bits & ExponentMask) == ExponentMask) &&
                     ((bits & FractionMask) != 0);
        return isNaN;
    }

    uint64_t qNanBitPattern() const {
        // quiet-NaN => 0x7FF8000000000000
        return 0x7FF8000000000000ULL;
    }

    uint64_t sNanBitPattern() const {
        // signaling-NaN => 0x7FF0000000000001
        return 0x7FF0000000000001ULL;
    }
};

/**
 * @brief Predefined parameters for a single-precision (32-bit) IEEE-like float.
 */
inline constexpr FloatingPointParams<SingleInfChecker, SingleNaNChecker> singlePrecisionParams(
    /* bitwidth      */ 32,
    /* mantissa_bits */ 23,
    /* bias          */ 127,
    /* rounding_mode */ RoundToNearestEven,
    /* OV_behavior   */ NonTrappingInf,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ SingleInfChecker{},
    /* IsNaN         */ SingleNaNChecker{}
);

/**
 * @brief Predefined parameters for a double-precision (64-bit) IEEE-like float.
 */
inline constexpr FloatingPointParams<DoubleInfChecker, DoubleNaNChecker> doublePrecisionParams(
    /* bitwidth      */ 64,
    /* mantissa_bits */ 52,
    /* bias          */ 1023,
    /* rounding_mode */ RoundToNearestEven,
    /* OV_behavior   */ NonTrappingInf,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ DoubleInfChecker{},
    /* IsNaN         */ DoubleNaNChecker{}
);

} // namespace lo_float
