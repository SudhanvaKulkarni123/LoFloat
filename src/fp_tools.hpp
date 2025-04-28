/// @author Sudhanva Kulkarni
/// @file
/// @brief This header file defines several enums and a struct that describe
///        floating-point parameters (bit width, rounding, infinity/NaN behavior, etc.)
///        used by the lo_float library.

#pragma once

#include <cstdint>
#include <concepts>

#define rounding_modes \
    X(RoundToNearestEven) \
    X(RoundTowardsZero) \
    X(RoundAwayFromZero) \
    X(StochasticRoundingA) \
    X(RoundToNearestOdd) \
    X(RoundDown) \
    X(RoundUp) \
    X(RoundTiesToAway) \
    X(StochasticRoundingB) \
    X(StochasticRoundingC) \
    X(True_StochasticRounding) \
    X(ProbabilisticRounding) 

#define signednesses \
    X(Signed) \
    X(Unsigned)

#define inf_behaviors \
    X(Extended) \
    X(Saturating)

#define nan_behaviors \
    X(QuietNaN) \
    X(NoNaN) \
    X(SignalingNaN)

#define unsigned_behaviors \
    X(NegtoZero) \
    X(NegtoNaN)


namespace lo_float {

/**
 * @enum Rounding_Mode
 * @brief Defines different rounding strategies for floating-point operations.
 */enum class Rounding_Mode : uint8_t {
    #define X(name) name,
    rounding_modes
    #undef X
};

/**
 * @enum Signedness
 * @brief Indicates whether a floating-point format is signed or unsigned.
 */
enum Signedness : uint8_t {
    #define X(name) name,
    signednesses
    #undef X
};

enum Unsigned_behavior : uint8_t {
    #define X(name) name,
    unsigned_behaviors
    #undef X
};

/**
 * @enum Inf_Behaviors
 * @brief Describes how infinities behave or are handled by the format.
 */
enum Inf_Behaviors : uint8_t {
    #define X(name) name,
    inf_behaviors
    #undef X
};

/**
 * @enum NaN_Behaviors
 * @brief Describes how NaNs (Not-a-Number) are handled by the format.
 */
enum NaN_Behaviors : uint8_t {
    #define X(name) name,
    nan_behaviors
    #undef X
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
    { t.minNegInf() } -> std::convertible_to<uint64_t>;
    { t.minPosInf() } -> std::convertible_to<uint64_t>;
};
/**
 * @concept negativeException
 * @brief Functor to deal with negative numbers in the case of unsigned floats
 */

 template<typename T>
 concept negativeException = requires(T t, uint64_t bits) {
    { t.sendstoNaN() } -> std::convertible_to<bool>;
    { t.sendtoZero() } -> std::convertible_to<bool>;
 };

/**
 * @concept NaNChecker
 * @brief A type that can detect and produce bit patterns for NaNs.
 * 
 * A valid NaNChecker must:
 * - Be callable with a `uint64_t` returning a `bool` indicating if that bit pattern is NaN.
 * - Provide `qNanBitPattern()` or `sNanBitPattern()` that each return a `uint64_t`.
 */
// Subconcept: Has qNaN
template <typename T>
concept HasQNaN = requires(T t) {
    { t.qNanBitPattern() } -> std::convertible_to<uint64_t>;
};

// Subconcept: Has sNaN
template <typename T>
concept HasSNaN = requires(T t) {
    { t.sNanBitPattern() } -> std::convertible_to<uint64_t>;
};

// Main concept: callable with bits AND has at least one NaN type
template <typename T>
concept NaNChecker = requires(T t, uint64_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
} && (HasQNaN<T> || HasSNaN<T>);
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

    /// @brief Number of bits used for stochastic rounding (0 means no stochastic rounding).
    int StochasticRoundingBits;

    ///  @brief enum to deal with how to deak with negatives for unsigned
    Unsigned_behavior unsigned_behavior;

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
        int stoch_length = 0,
        Unsigned_behavior ub = Unsigned_behavior::NegtoZero
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
      , unsigned_behavior(ub)
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
 * @brief Predefined parameters for a single-precision (32-bit) IEEE-like float.
 */
template<Rounding_Mode rm>
inline constexpr FloatingPointParams<SingleInfChecker, SingleNaNChecker> singlePrecisionParams(
    /* bitwidth      */ 32,
    /* mantissa_bits */ 23,
    /* bias          */ 127,
    /* rounding_mode */ rm,
    /* OV_behavior   */ Extended,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ SingleInfChecker{},
    /* IsNaN         */ SingleNaNChecker{}
);


//defintions for half precision and bfloat-
struct HalfInfChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 15) & 0xFF) == 0xFF) && ((bits & 0x7FFF) == 0);
    }
    uint32_t infBitPattern() const { return 0x7C00; }
    uint32_t minNegInf() const { return 0xFC00; } // -∞ => 0xFC00
    uint32_t minPosInf() const { return 0x7C00; } // +∞ => 0x7C00

};

struct HalfNaNChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 15) & 0xFF) == 0xFF) && ((bits & 0x7FFF) != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7E00; } // typical QNaN
    uint32_t sNanBitPattern() const { return 0x7F00; } // some SNaN pattern
};

template<Rounding_Mode rm, int stoch_len = 0>
inline constexpr FloatingPointParams<HalfInfChecker, HalfNaNChecker> halfPrecisionParams(
    /* bitwidth      */ 16,
    /* mantissa_bits */ 10,
    /* bias          */ 15,
    /* rounding_mode */ rm,
    /* OV_behavior   */ Extended,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ HalfInfChecker{},
    /* IsNaN         */ HalfNaNChecker{},
    /* stochastic_rounding_length */ stoch_len
);


struct BFloatInfChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) == 0);
    }
    uint32_t infBitPattern() const { return 0x7F800000; } // +∞ => 0x7F800000
    uint32_t minNegInf() const { return 0xFF800000; } // -∞ => 0xFF800000
    uint32_t minPosInf() const { return 0x7F800000; } // +∞ => 0x7F800000
};


struct BFloatNaNChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; } // typical QNaN
    uint32_t sNanBitPattern() const { return 0x7FA00000; } // some SNaN pattern
};

template<Rounding_Mode rm, int stoch_len = 0>
inline constexpr FloatingPointParams<BFloatInfChecker, BFloatNaNChecker> bfloatPrecisionParams(
    /* bitwidth      */ 16,
    /* mantissa_bits */ 7,
    /* bias          */ 127,
    /* rounding_mode */ rm,
    /* OV_behavior   */ Extended,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ BFloatInfChecker{},
    /* IsNaN         */ BFloatNaNChecker{},
    /* stochastic_rounding_length */ stoch_len
);


//definitions for tf32
struct TF32InfChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) == 0);
    }
    uint32_t infBitPattern() const { return 0x7F800000; } // +∞ => 0x7F800000
    uint32_t minNegInf() const { return 0xFF800000; } // -∞ => 0xFF800000
    uint32_t minPosInf() const { return 0x7F800000; } // +∞ => 0x7F800000
};

struct TF32NaNChecker {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; } // typical QNaN
    uint32_t sNanBitPattern() const { return 0x7FA00000; } // some SNaN pattern
};

template<Rounding_Mode rm>
inline constexpr FloatingPointParams<TF32InfChecker, TF32NaNChecker> tf32PrecisionParams(
    /* bitwidth      */ 19,
    /* mantissa_bits */ 10,
    /* bias          */ 127,
    /* rounding_mode */ rm,
    /* OV_behavior   */ Extended,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* IsInf         */ TF32InfChecker{},
    /* IsNaN         */ TF32NaNChecker{}
);

} // namespace lo_float
