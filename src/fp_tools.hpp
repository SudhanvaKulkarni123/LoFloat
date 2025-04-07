///@author Sudhanva Kulkarni
///header file containing a bunch of useful enum and struct definitions
#include <cstdint>

#include <concepts>
#include <cstdint>



namespace lo_float {    


enum Rounding_Mode : uint8_t {
    RoundToNearestEven = 0,
    RoundTowardsZero = 1,
    RoundAwayFromZero = 2,
    StochasticRounding = 3,
    RoundToNearestOdd = 4,
    RoundDown = 5,
    RoundUp = 6,
    RoundTiesToAway = 7
};

enum Signedness : uint8_t {
    Signed = 0,
    Unsigned = 1
};

enum Inf_Behaviors : uint8_t {
    NonTrappingInf = 0,
    Saturating = 1,
    Trapping = 2        //TODO : come back to this 
};

enum NaN_Behaviors : uint8_t {
    QuietNaN = 0,
    NoNaN = 1,
    SignalingNaN = 2            //TODO : come back to this 
};



// e4m3 a, e4m3 b
// float c = 1
// c += float(a)*float(b)  
// c = fma(a, b, c) -- TODO



// Concept: "InfChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept InfChecker = requires(T t, uint64_t bits) {
    // Must be callable with a uint32_t, returning something convertible to bool
    { t(bits) } -> std::convertible_to<bool>;

    // Must have an infBitPattern() that returns something convertible to uint32_t
    { t.infBitPattern() } -> std::convertible_to<uint64_t>;

    // Must have a minNegInf() method returning something convertible to uint32_t
    { t.minNegInf() } -> std::convertible_to<uint64_t>;

    // Must have a minPosInf() method returning something convertible to uint32_t
    { t.minPosInf() } -> std::convertible_to<uint64_t>;
};

// Concept: "NaNChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept NaNChecker = requires(T t, uint64_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
    { t.qNanBitPattern() } -> std::convertible_to<uint64_t>;
    { t.sNanBitPattern() } -> std::convertible_to<uint64_t>;

};



template<InfChecker IsInfFunctor, NaNChecker IsNaNFunctor>
struct FloatingPointParams
{
    int bitwidth;       //total bitwidth of the floating point number
    int mantissa_bits;  //number of bits in the mantissa. Number of exponent bits is calced as bitwidth - mantissa_bits
    int bias;           //bias for the exponent
    Rounding_Mode rounding_mode;  
    Inf_Behaviors OV_behavior;
    NaN_Behaviors NA_behavior;
    Signedness is_signed;       //toggles whether float is signed or unsigned. If unsigned, the extra bit is given to the exponent.
    IsInfFunctor IsInf;
    IsNaNFunctor IsNaN;
    int StochasticRoundingBits = 0; //number of bits used for stochastic rounding. Set to 0 by default

    constexpr FloatingPointParams(
        int bw,
        int mb,
        int b,
        Rounding_Mode rm,
        Inf_Behaviors ovb,
        NaN_Behaviors nab,
        Signedness is_signed,
        //SubNormal_Support SN_sup,
        IsInfFunctor IsInf,
        IsNaNFunctor IsNaN
    )
      : bitwidth(bw)
      , mantissa_bits(mb)
      , bias(b)
      , rounding_mode(rm)
      , OV_behavior(ovb)
      , NA_behavior(nab)
      , is_signed(is_signed)
      , SN_support(SN_sup)
      , IsInf(IsInf)
      , IsNaN(IsNaN)
    {}
};



// -------------------------
//  32-bit float functors
// -------------------------
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

///defining standard FloatingPointParams structs for single, double, half, etc

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


// --------------------------
//  64-bit double functors
// --------------------------
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


// --------------------------
//  Single-precision constants
// --------------------------
inline constexpr FloatingPointParams<SingleInfChecker, SingleNaNChecker> singlePrecisionParams(
    /* bitwidth      */ 32,
    /* mantissa_bits */ 23,
    /* bias          */ 127,
    /* rounding_mode */ RoundToNearestEven,
    /* OV_behavior   */ NonTrappingInf,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    // you didn't define SubNormal_Support above, so let's assume it's some enum:
    /* SN_support    */ /* SubNormal_Support::Enabled or whichever */,
    /* IsInf         */ SingleInfChecker{},
    /* IsNaN         */ SingleNaNChecker{}
);

// --------------------------
//  Double-precision constants
// --------------------------
inline constexpr FloatingPointParams<DoubleInfChecker, DoubleNaNChecker> doublePrecisionParams(
    /* bitwidth      */ 64,
    /* mantissa_bits */ 52,
    /* bias          */ 1023,
    /* rounding_mode */ RoundToNearestEven,
    /* OV_behavior   */ NonTrappingInf,
    /* NA_behavior   */ QuietNaN,
    /* is_signed     */ Signed,
    /* SN_support    */ /* SubNormal_Support::Enabled or whichever */,
    /* IsInf         */ DoubleInfChecker{},
    /* IsNaN         */ DoubleNaNChecker{}
);



}