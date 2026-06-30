/// @author Sudhanva Kulkarni
#pragma once
#define LEN 13
#include <random>
#include <algorithm>
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>
#include <ctime>
#include <math.h>
#include <complex>
#include <limits>
#include <ostream>
#include <iostream>
#include <type_traits>
#include <climits>
#include "platform_macros.h"
#ifndef USE_CUDA    
#include <xsimd/xsimd.hpp>
#include "simd_helpers.hpp"     //simd helpers
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#endif
#include "fp_tools.hpp"     //structs and concepts to define Floating Point params
#ifdef ENABLE_EXCEPT
#include "f_exceptions.hpp" //global env for exceptions
#endif

#include "lo_int.h"           //custom integer types
#include "template_helpers.h" //helper templataes

#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif

#ifndef USE_CUDA
namespace xs = xsimd;
#endif

namespace lo_float
{

    inline thread_local unsigned int lof_seed = 12345;

    inline void set_seed(unsigned int seed) {
        lof_seed = seed;
    }

#if defined(USE_CUDA) && defined(__CUDACC__)
    // ---- Device-side RNG for stochastic rounding ---------------------------
    // The host stochastic-rounding helpers draw from std::mt19937 seeded by the
    // thread_local `lof_seed`. None of that survives on the GPU: <random> is
    // host-only, `thread_local` is illegal in __device__ code, and a single
    // shared seed would make every thread draw the *same* value (fully
    // correlated SR, which defeats the purpose). Rather than thread a curand
    // state object through virtual_round -> RoundMantissa -> every helper, we
    // derive each sample *statelessly* from a SplitMix64 hash of the element's
    // own bits, a per-call salt, and the global thread index. This needs no RNG
    // state, is decorrelated across threads/elements, and is deterministic for a
    // fixed launch geometry.
    LOFLOAT_DEVICE LOFLOAT_FORCEINLINE uint64_t lof_splitmix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    LOFLOAT_DEVICE LOFLOAT_FORCEINLINE uint64_t lof_device_global_tid() {
        const uint64_t block =
            (uint64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        const uint64_t tinb =
            (uint64_t(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        return block * (uint64_t(blockDim.x) * blockDim.y * blockDim.z) + tinb;
    }

    // Uniform 64-bit sample; `entropy` is normally the element's magnitude bits
    // and `salt` distinguishes the different rounding modes.
    LOFLOAT_DEVICE LOFLOAT_FORCEINLINE uint64_t lof_device_rand64(uint64_t entropy, uint64_t salt) {
        return lof_splitmix64(entropy
                              ^ (lof_device_global_tid() * 0x9E3779B97F4A7C15ULL)
                              ^ (salt * 0xD1B54A32D192ED03ULL)
                              ^ 0x2545F4914F6CDD1DULL);
    }
#endif

    // Concept to constrain to floating-point types

    namespace lo_float_internal
    {

        // declaration of std::abs for ADL
        using std::abs;

        // forward decl of classes
        template <typename Derived, typename UnderlyingType>
        class lo_float_base;

        template <typename Derived, FloatingPointParams Fp>
        class Var_lo_float;

        template <FloatingPointParams Fp>
        class Templated_Float;

        template <int len, lo_float::Signedness Sign>
        class i_n;

   



        // NOTE: the global exception Environment `f_env` and the `signal_if_754`
        // gate helper now live in f_exceptions.hpp (inline => header-safe across TUs).

        // @brief helper struct that picks underlying float that should be used for the simulation. We require 2*mantissa_bits + 1 mantissa bits in the simulation type
        template <int N>
        struct AOpTypeSelector
        {
            using type = std::conditional_t<
                (N < 12),
                float,
                double>;
        };

        // alias for the helper
        template <int mantissa_bits>
        using AOpType = typename AOpTypeSelector<mantissa_bits>::type;

        // Helper sruct to decide underlying type for sqrt. Here we need 2*mantissa_bits + 2 mantissa bits in the simulation type

        template <int mantissa_bits>
        struct SqrtTypeSelector
        {
            using type = std::conditional_t<
                (mantissa_bits < 11),
                float,
                double>;
        };

        // alias
        template <int mantissa_bits>
        using SqrtType = typename SqrtTypeSelector<mantissa_bits>::type;


        //bitcast helper
        // std::bit_cast does NOT work in nvcc device code (it silently yields 0),
        // which is why the rest of the library uses cuda::std::bit_cast on device.
        // Dispatch here so this helper is correct on both host and device.
        template <typename To, typename From>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE To bit_cast(From val) {
        #if defined(USE_CUDA) && defined(__CUDA_ARCH__)
            return cuda::std::bit_cast<To>(val);
        #else
            return std::bit_cast<To>(val);
        #endif
        }

        #ifdef USE_CUDA
        // c10::Half -> uint16_t
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE uint16_t bit_cast(c10::Half val) {
            return val.x;
        }

        // uint16_t -> c10::Half
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE c10::Half bit_cast(uint16_t val) {
            c10::Half h;
            h.x = val;
            return h;
        }
        #endif


        template <typename Derived, typename UnderlyingType = uint8_t>
        class lo_float_base
        {
        protected:
            struct ConstructFromRepTag
            {
            };

            // "Core" constructor storing rep_ in the base
            constexpr lo_float_base(UnderlyingType rep, ConstructFromRepTag)
                : rep_(rep)
            {
            }

            // CRTP friend declaration
            template <typename T, FloatingPointParams Fp>
            friend class Var_lo_float;
            template <FloatingPointParams Fp>
            friend class Templated_Float;

        public:
            LOFLOAT_HOST_DEVICE constexpr lo_float_base() : rep_(0) {}

            LOFLOAT_HOST_DEVICE constexpr UnderlyingType rep() const
            {
                return rep_;
            }

            // Templated constructor
            template <typename T,
                      typename EnableIf = std::enable_if_t<std::is_arithmetic_v<T>>>
            LOFLOAT_HOST_DEVICE explicit lo_float_base(T f)
                : lo_float_base(ConvertFrom(static_cast<float>(f)).rep(),
                                ConstructFromRepTag{}) {}

            LOFLOAT_HOST_DEVICE explicit lo_float_base(double f64)
                : lo_float_base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}

            LOFLOAT_HOST_DEVICE explicit lo_float_base(float f32)
                : lo_float_base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}

            LOFLOAT_HOST_DEVICE explicit lo_float_base(const int i32)
                : lo_float_base(ConvertFrom(static_cast<double>(i32)).rep(), ConstructFromRepTag{}) {}

            template <int len, Signedness Sign>
            LOFLOAT_HOST_DEVICE explicit lo_float_base(i_n<len, Sign> var_int)
                : lo_float_base(ConvertFrom(static_cast<double>((int)var_int)).rep(), ConstructFromRepTag{}) {}

            // CRTP helpers
            LOFLOAT_HOST_DEVICE constexpr const Derived &derived() const
            {
                return *static_cast<const Derived *>(this);
            }
            LOFLOAT_HOST_DEVICE constexpr Derived &derived()
            {
                return *static_cast<Derived *>(this);
            }

            LOFLOAT_HOST_DEVICE static constexpr Derived FromRep(UnderlyingType rep)
            {
                return Derived(rep, ConstructFromRepTag{});
            }

            // -------------------------------------------
            // Declarations for ConvertFrom / ConvertTo
            // -------------------------------------------
            template <typename From>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Derived ConvertFrom(const From &from, ProjSpec ps = ProjSpec{});

            template <typename To>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE To ConvertTo(const Derived &from, ProjSpec ps = ProjSpec{});

            template <typename T,
                      typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
            LOFLOAT_HOST_DEVICE explicit operator T() const
            {
                return static_cast<T>(static_cast<float>(derived()));
            }
            LOFLOAT_HOST_DEVICE explicit operator double() const
            {
                return ConvertTo<double>(derived());
            }
            LOFLOAT_HOST_DEVICE explicit operator float() const
            {
                return ConvertTo<float>(derived());
            }

            LOFLOAT_HOST_DEVICE explicit operator int() const
            {
                const double d = ConvertTo<double>(derived());
#ifdef ENABLE_EXCEPT
                // §7.2(j): converting NaN/Inf or an out-of-range value to an integer
                // format has no usefully definable result -> invalid operation.
                if (std::isnan(d) || std::isinf(d) ||
                    d < double(std::numeric_limits<int>::min()) ||
                    d > double(std::numeric_limits<int>::max()))
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                return (int)(d);
            }

            template <int len, Signedness sign>
            LOFLOAT_HOST_DEVICE explicit operator i_n<len, sign>() const
            {
                const double d = ConvertTo<double>(derived());
#ifdef ENABLE_EXCEPT
                // §7.2(j): NaN/Inf or out-of-range source -> invalid operation.
                if (std::isnan(d) || std::isinf(d) ||
                    d < double((int)i_n<len, sign>::lowest()) ||
                    d > double((int)i_n<len, sign>::highest()))
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                return (i_n<len, sign>)int(d);
            }

            LOFLOAT_HOST_DEVICE explicit operator bool() const
            {
                if constexpr (get_signedness_v<Derived> == Signedness::Signed)
                {
                    return (rep() & 0x7F) != 0;
                }
                else
                {
                    return rep() != 0;
                }
            }

            // define underlying float before defining arithemtic types
            using UnderlyingFloat = AOpType<get_mantissa_bits_v<Derived>>;

            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Derived operator-() const
            {
                // Reconstruct a stateless temporary checker instead of ODR-using the
                // static constexpr IsNaNFunctor object (nvcc emits no device storage).
                using NaNChk = std::remove_cv_t<std::remove_reference_t<decltype(Derived::IsNaNFunctor)>>;
                // check spl case of -0 for nan
                if (rep_ == 0 && NaNChk{}((1 << (get_bitwidth_v<Derived> - 1))))
                {
                    return FromRep(0);
                }
                if (get_signedness_v<Derived> == Signedness::Signed)
                {
                    return FromRep(static_cast<UnderlyingType>(this->rep() ^ (1 << (get_bitwidth_v<Derived> - 1))));
                }
                else
                {
                    if (get_unsigned_behavior_v<Derived> == Unsigned_behavior::NegtoZero)
                    {
                        return FromRep(0);
                    }
                    else
                    {
                        if (get_NaN_Behavior_v<Derived> == NaN_Behaviors::_3109)
                        {
                            return FromRep(NaNChk{}.qNanBitPattern());
                        }
                        else
                        {
                            // Negating an unsigned value produces a NaN: no usefully
                            // definable result -> invalid operation (§7.2).
#ifdef ENABLE_EXCEPT
                            signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                            return FromRep(NaNChk{}.sNanBitPattern());
                        }
                    }
                }
            }

            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator+(const Derived &other) const
            {
                const UnderlyingFloat res = UnderlyingFloat{derived()} + UnderlyingFloat{other};
#ifdef ENABLE_EXCEPT
                // ∞ + (−∞) has no usefully definable result -> invalid (§7.2 d).
                if (std::isnan(res) && !isnan(derived()) && !isnan(other))
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                return Derived{res};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator=(const Derived &other) const
            {
                return Derived{UnderlyingFloat{other}};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator-(const Derived &other) const
            {
                const UnderlyingFloat res = UnderlyingFloat{derived()} - UnderlyingFloat{other};
#ifdef ENABLE_EXCEPT
                // ∞ − ∞ has no usefully definable result -> invalid (§7.2 d).
                if (std::isnan(res) && !isnan(derived()) && !isnan(other))
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                return Derived{res};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator*(const Derived &other) const
            {
                const UnderlyingFloat res = UnderlyingFloat{derived()} * UnderlyingFloat{other};
#ifdef ENABLE_EXCEPT
                // 0 × ∞ (or ∞ × 0) has no usefully definable result -> invalid (§7.2 b).
                if (std::isnan(res) && !isnan(derived()) && !isnan(other))
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
#endif
                return Derived{res};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator/(const Derived &other) const
            {
                const UnderlyingFloat a = UnderlyingFloat{derived()};
                const UnderlyingFloat b = UnderlyingFloat{other};
#ifdef ENABLE_EXCEPT
                // §7.2/§7.3: 0/0 and ∞/∞ are invalid (no usefully definable result);
                // a finite non-zero divided by zero is a true divide-by-zero.
                if ((a == UnderlyingFloat(0) && b == UnderlyingFloat(0)) ||
                    (std::isinf(a) && std::isinf(b)))
                {
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
                }
                else if (b == UnderlyingFloat(0) && std::isfinite(a))
                {
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::DivisionByZero);
                }
#endif
                return Derived{a / b};
            }

            // Example comparison
            enum Ordering : int8_t
            {
                kLess = -1,
                kEquivalent = 0,
                kGreater = 1,
                kUnordered = 2,
            };

            template <typename T>
            constexpr bool operator==(const T &other) const
            {
                return Compare(derived(), other) == Ordering::kEquivalent;
            }

            template <typename T>
            constexpr bool operator!=(const T &other) const
            {
                return Compare(derived(), other) != Ordering::kEquivalent;
            }

            template <typename T>
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  bool operator<(
                const T &other) const
            {
                return Compare(derived(), other) == Ordering::kLess;
            }

            template <typename T>
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  bool operator<=(
                const T &other) const
            {
                return Compare(derived(), other) == Ordering::kEquivalent || Compare(derived(), other) == Ordering::kLess;
            }

            template <typename T>
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  bool operator>(
                const T &other) const
            {
                return Compare(derived(), other) == Ordering::kGreater;
            }

            template <typename T>
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  bool operator>=(
                const T &other) const
            {
                auto ordering = Compare(derived(), other);
                return ordering == kGreater || ordering == kEquivalent;
            }

            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived &operator+=(
                const Derived &other)
            {
                derived() = derived() + other;
                return derived();
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived &operator-=(
                const Derived &other)
            {
                derived() = derived() - other;
                return derived();
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived &operator*=(
                const Derived &other)
            {
                derived() = derived() * other;
                return derived();
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived &operator/=(
                const Derived &other)
            {
#ifdef ENABLE_EXCEPT
                const UnderlyingFloat a = UnderlyingFloat{derived()};
                const UnderlyingFloat b = UnderlyingFloat{other};
                // §7.2/§7.3: 0/0 and ∞/∞ are invalid; finite non-zero / 0 is divide-by-zero.
                if ((a == UnderlyingFloat(0) && b == UnderlyingFloat(0)) ||
                    (std::isinf(a) && std::isinf(b)))
                {
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::InvalidOperation);
                }
                else if (b == UnderlyingFloat(0) && std::isfinite(a))
                {
                    signal_if_754<get_NaN_Behavior_v<Derived>>(LF_exception_flags::DivisionByZero);
                }
#endif
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
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  std::pair<UnderlyingType, UnderlyingType>
            SignAndMagnitude(Derived x)
            {
                const UnderlyingType x_abs_bits =
                    bit_cast<UnderlyingType>(abs(x));
                const UnderlyingType x_bits = bit_cast<UnderlyingType>(x);
                const UnderlyingType x_sign = x_bits ^ x_abs_bits;
                return {x_sign, x_abs_bits};
            }

            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Signed_type
            SignAndMagnitudeToTwosComplement(UnderlyingType sign, UnderlyingType magnitude)
            {
                return magnitude ^ (static_cast<Signed_type>(sign ? -1 : 0));
            }

            template <typename T>
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  friend constexpr Ordering Compare(
                const Derived &lhs, const T &rhs)
            {
                if (isnan(lhs) || isnan(rhs))
                {
                    return kUnordered;
                }
                auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
                auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
                if (lhs_mag == 0 && rhs_mag == 0)
                {
                    return kEquivalent;
                }
                Signed_type lhs_tc = SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
                Signed_type rhs_tc = SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);

                if (lhs_tc < rhs_tc)
                    return kLess;
                if (lhs_tc > rhs_tc)
                    return kGreater;
                return kEquivalent;
            }

        }; // lo_float_base

        // helper template to pick storage format
        template <int Len>
        using Base_repr_select = std::conditional_t<(Len <= 8), uint8_t, std::conditional_t<(Len <= 16), uint16_t, uint32_t>>;

        // rounding functions-

       // ============================================================================
//  Rounding operations — scalar + SIMD (xsimd) implementations
//
//  Pattern: single template per function, `if constexpr` to branch on
//  whether the type is an xsimd batch.  SIMD paths are guarded with
//  #ifndef USE_CUDA.
// ============================================================================

// ----- RoundBitsToNearestEven (unchanged) -----------------------------------

template <typename Bits, typename Roundoff>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundBitsToNearestEven(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    Bits bias;
    #ifndef USE_CUDA
    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        bias = xsimd::select(Bits(roundoff) == Bits{},
            Bits(0),
            (xs::bitwise_rshift(bits, xs::batch_cast<value_type>(roundoff)) & Bits(1))
                + xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff - 1))
                - Bits(1));
    
    }
    else
    #endif
    {
        bias = roundoff == 0
            ? Bits(0)
            : ((bits >> roundoff) & Bits(1)) + (Bits(1) << (roundoff - 1)) - Bits(1);
    }

    return bits + bias;
}

// ----- Probabilistic_Round --------------------------------------------------
// Coin-flip rounding: if any truncated bits are nonzero, round up with p=0.5.

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits Probabilistic_Round(Bits bits, Roundoff roundoff)
{
    using lane_t = pod_type_t<Bits>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;
#ifndef __CUDA_ARCH__
    std::mt19937 mt(lof_seed);
#endif

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        using vt = pod_type_t<Roundoff>;
        Bits mask = xs::bitwise_lshift(Bits(1), xs::batch_cast<lane_t>(roundoff)) - Bits(1);
        Bits truncated = bits & ~mask;
        Bits tail = bits & mask;

        // Per-lane random 0/1
        std::uniform_int_distribution<int> d01(0, 1);
        std::array<lane_t, lanes> r{};
        for (std::size_t i = 0; i < lanes; ++i)
            r[i] = static_cast<lane_t>(d01(mt));

        Bits r01;
        if constexpr (lanes == 1)
            r01 = Bits{r[0]};
        else
            r01 = xsimd::load_unaligned(r.data());

        Bits bump = xsimd::select(tail != Bits(0), r01, Bits(0));
        return truncated + xs::bitwise_lshift(bump, xs::batch_cast<lane_t>(roundoff));
        #endif
    }
    else
    {
        Bits mask = (Bits(1) << roundoff) - Bits(1);
        Bits truncated = bits & ~mask;
        Bits tail = bits & mask;

#ifdef __CUDA_ARCH__
        lane_t r01 = static_cast<lane_t>(lof_device_rand64(static_cast<uint64_t>(bits), 0x01) & 1u);
#else
        std::uniform_int_distribution<int> d01(0, 1);
        lane_t r01 = static_cast<lane_t>(d01(mt));
#endif

        Bits bump = (tail != Bits(0)) ? Bits(r01) : Bits(0);
        return truncated + (bump << roundoff);
    }
}

// ----- Stochastic_Round_A (unchanged) ---------------------------------------

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits Stochastic_Round_A(Bits bits, Roundoff roundoff, const int len = 0)
{
    using lane_t = pod_type_t<Bits>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;

    if (len <= 0) return bits;

#ifdef __CUDA_ARCH__
    const uint64_t maxv = (uint64_t{1} << unsigned(len)) - 1u;
    Bits to_add = static_cast<Bits>(lof_device_rand64(static_cast<uint64_t>(bits), 0x0A) & maxv);
    return bits + (to_add << (roundoff - len));
#else
    std::mt19937 mt(lof_seed);
    const unsigned maxv = (1u << unsigned(len)) - 1u;
    std::uniform_int_distribution<unsigned> dist(0u, maxv);

  #ifndef USE_CUDA
    std::array<lane_t, lanes> samp{};
    for (std::size_t i = 0; i < lanes; ++i)
        samp[i] = static_cast<lane_t>(dist(mt));

    Bits to_add;
    if constexpr (lanes == 1)
        to_add = Bits{samp[0]};
    else
        to_add = xsimd::load_unaligned(samp.data());

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        return bits + xsimd::bitwise_lshift(to_add, (roundoff - lane_t(len)));
    }
    else
    {
        return bits + (to_add << (roundoff - len));
    }
  #else
    Bits to_add = static_cast<Bits>(dist(mt));
    return bits + (to_add << (roundoff - len));
  #endif
#endif
}

// ----- Stochastic_Round_B ---------------------------------------------------
// Bits may be scalar or batch; roundoff and len are always scalar.

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits Stochastic_Round_B(Bits bits, const Roundoff roundoff, const int len = 0)
{
    using lane_t = pod_type_t<Bits>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;
#ifndef __CUDA_ARCH__
    std::mt19937 mt(lof_seed);
#endif

    if constexpr (is_xsimd_batch<Bits>::value)
    {
        #ifndef USE_CUDA
        if (len <= 0) return bits;
        const Bits shift = roundoff - Bits(len);

        const uint64_t max_u64 = (uint64_t{1} << len) - 1u;
        std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(max_u64));

        alignas(64) lane_t samp_arr[lanes];
        for (std::size_t i = 0; i < lanes; ++i)
            samp_arr[i] = static_cast<lane_t>(dist(mt));

        const Bits samp = xsimd::load_aligned(samp_arr);

        const lane_t complement_scalar = (lane_t{1} << len) - lane_t{1};
        const Bits complement(complement_scalar);
        const Bits to_add = (samp & complement) + Bits(lane_t{1});

        const Bits lower_mask_scalar = (Bits{1} << roundoff) - Bits{1};
        const Bits lower = bits & Bits(lower_mask_scalar);

        const auto sum_mask = (lower > Bits(lane_t{0}));

        Bits add = to_add;
        add = add << shift;

        return bits + xsimd::select(sum_mask, add, Bits(lane_t{0}));
        #endif
    }
    else
    {
#ifdef __CUDA_ARCH__
        unsigned int samp = static_cast<unsigned int>(
            lof_device_rand64(static_cast<uint64_t>(bits), 0x0B) & ((uint64_t{1} << len) - 1u));
#else
        std::uniform_int_distribution<unsigned int> distribution(0, (1 << (len)) - 1);
        unsigned int samp = distribution(mt);
#endif
        Bits complement = (Bits{1} << (len)) - 1;
        Bits to_add = static_cast<Bits>(samp & complement) + 1;
        const Bits lower = bits & ((Bits{1} << roundoff) - 1);
        const bool sum_mask = lower > 0;
        Bits to_ret = bits + (sum_mask ? (to_add << (roundoff - len)) : Bits{});
        return to_ret;
    }
}

// ----- Stochastic_Round_C ---------------------------------------------------

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits Stochastic_Round_C(Bits bits, const Roundoff roundoff, const int len = 0)
{
    using lane_t = pod_type_t<Bits>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;
#ifndef __CUDA_ARCH__
    std::mt19937 mt(lof_seed);
#endif

    if constexpr (is_xsimd_batch<Bits>::value)
    {
        #ifndef USE_CUDA
        if (len <= 0) return bits;
        const int effective_len = len;
        const Bits top_shift = roundoff - Bits(effective_len + 1);

        const uint64_t max_u64 = (uint64_t{1} << effective_len) - 1u;
        std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(max_u64));

        alignas(64) lane_t samp_arr[lanes];
        for (std::size_t i = 0; i < lanes; ++i)
            samp_arr[i] = static_cast<lane_t>(dist(mt));

        const Bits samp = xsimd::load_aligned(samp_arr);
        const Bits one(lane_t{1});

        const Bits coin_bit     = samp & one;
        const Bits remaining    = samp >> 1;

        const Bits bottom_mask =
            (Bits(1) << roundoff) - Bits(1);
        const Bits bottom_bits = bits & Bits(bottom_mask);

        Bits top_bits = remaining + coin_bit;
        top_bits = top_bits << top_shift;

        const auto do_add = (bottom_bits != Bits(lane_t{0}));
        return xsimd::select(do_add, bits + top_bits, bits);
        #endif
    }
    else
    {
#ifdef __CUDA_ARCH__
        unsigned int samp = static_cast<unsigned int>(
            lof_device_rand64(static_cast<uint64_t>(bits), 0x0C) & ((uint64_t{1} << len) - 1u));
#else
        std::uniform_int_distribution<unsigned int> distribution(0, (1 << (len)) - 1);
        unsigned int samp = distribution(mt);
#endif
        const unsigned int coin_bit = samp & 1;
        unsigned int remaining_bits = samp >> 1;
        Bits bottom_bits = bits & ((Bits{1} << roundoff) - 1);
        Bits top_bits = static_cast<Bits>((remaining_bits + coin_bit) << (roundoff - len + 1));
        return (len == 0 || bottom_bits == 0) ? bits : bits + (top_bits);
    }
}

// ----- Stochastic_Round_D ---------------------------------------------------
// Coin-flip variant (paper Mode D): draw a len-bit random string R, flip a
// fair coin c in {0,1}, and add (R + c) at the rounding point, then truncate.
// Averaging over the coin gives Pr[up] = 0.5*( floor(q*xf)/q + ceil(q*xf)/q )
// with q = 2^len (Table II). Same add-then-truncate mechanism as Mode A.
template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits Stochastic_Round_D(Bits bits, Roundoff roundoff, const int len = 0)
{
    using lane_t = pod_type_t<Bits>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;

    if (len <= 0) return bits;

#ifdef __CUDA_ARCH__
    const uint64_t maxv = (uint64_t{1} << unsigned(len)) - 1u;
    const uint64_t rnd  = lof_device_rand64(static_cast<uint64_t>(bits), 0x0D);
    // low `len` bits -> R; the next bit -> independent fair coin.
    const Bits to_add = static_cast<Bits>((rnd & maxv) + ((rnd >> len) & 1u));
    return bits + (to_add << (roundoff - len));
#else
    std::mt19937 mt(lof_seed);
    const unsigned maxv = (1u << unsigned(len)) - 1u;
    std::uniform_int_distribution<unsigned> dist(0u, maxv);
    std::uniform_int_distribution<int>      coin(0, 1);

  #ifndef USE_CUDA
    std::array<lane_t, lanes> samp{};
    for (std::size_t i = 0; i < lanes; ++i)
        samp[i] = static_cast<lane_t>(dist(mt) + static_cast<unsigned>(coin(mt)));

    Bits to_add;
    if constexpr (lanes == 1)
        to_add = Bits{samp[0]};
    else
        to_add = xsimd::load_unaligned(samp.data());

    if constexpr (is_xsimd_batch<Roundoff>::value)
        return bits + xsimd::bitwise_lshift(to_add, (roundoff - lane_t(len)));
    else
        return bits + (to_add << (roundoff - len));
  #else
    Bits to_add = static_cast<Bits>(dist(mt) + static_cast<unsigned>(coin(mt)));
    return bits + (to_add << (roundoff - len));
  #endif
#endif
}

// ----- True_Stochastic_Round ------------------------------------------------
// Rounds up with probability  tail / 2^roundoff.
// roundoff is always a uniform scalar; Bits may be scalar or batch.
template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits True_Stochastic_Round(Bits bits, const Roundoff roundoff)
{
    using lane_t = pod_type_t<Bits>;
    using roundoff_lane_t = pod_type_t<Roundoff>;
    constexpr std::size_t lanes = num_lanes_v<Bits>;
#ifndef __CUDA_ARCH__
    std::mt19937 mt(lof_seed);
#endif

    if constexpr (is_xsimd_batch<Bits>::value)
    {
#ifndef USE_CUDA
        alignas(64) lane_t bits_arr[lanes];
        alignas(64) roundoff_lane_t ro_arr[lanes];
        xsimd::store_aligned(bits_arr, bits);

        if constexpr (is_xsimd_batch<Roundoff>::value)
            xsimd::store_aligned(ro_arr, roundoff);
        else
            std::fill_n(ro_arr, lanes, static_cast<roundoff_lane_t>(roundoff));

        alignas(64) lane_t result_arr[lanes];
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (std::size_t i = 0; i < lanes; ++i)
        {
            const auto ro = ro_arr[i];
            const lane_t mask = (lane_t{1} << ro) - lane_t{1};
            const lane_t tail = bits_arr[i] & mask;
            const lane_t truncated = bits_arr[i] & ~mask;
            const double denom = static_cast<double>(lane_t{1} << ro);
            const double prob = static_cast<double>(tail) / denom;

            if (dist(mt) < prob)
            {
                lane_t rounded = truncated + (lane_t{1} << ro);
                result_arr[i] = (rounded < truncated) ? truncated : rounded;
            }
            else
            {
                result_arr[i] = truncated;
            }
        }

        if constexpr (lanes == 1)
            return Bits{result_arr[0]};
        else
            return xsimd::load_aligned(result_arr);
#endif
    }
    else
    {
        const Bits mask = (Bits{1} << roundoff) - 1;
        const Bits tail = bits & mask;
        const Bits truncated = bits & ~mask;
        const double prob = static_cast<double>(tail) / static_cast<double>(Bits{1} << roundoff);
#ifdef __CUDA_ARCH__
        // Uniform double in [0,1): top 53 bits of the hash over 2^53.
        const double samp = static_cast<double>(lof_device_rand64(static_cast<uint64_t>(bits), 0x71) >> 11)
                            * (1.0 / 9007199254740992.0);
#else
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        const double samp = distribution(mt);
#endif

        if (samp < prob)
        {
            Bits rounded = truncated + (Bits{1} << roundoff);
            if (rounded < truncated)
                return truncated;
            return rounded;
        }
        else
        {
            return truncated;
        }
    }
}
// ----- RoundBitsTowardsZero -------------------------------------------------

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundBitsTowardsZero(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        auto mask = ~(xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1));
        return bits & mask;
        #endif
    }
    else
    {
        auto mask = ~((Bits{1} << roundoff) - 1);
        return bits & mask;
    }
}

// ----- RoundTiedBitsTowardsZero ---------------------------------------------
// Round to nearest; break ties towards zero (i.e. truncate on exact tie).
//
//   bias = (1 << (roundoff-1)) - 1
//
// Adding this bias causes a carry past the rounding point when the tail
// *strictly exceeds* the midpoint, but NOT when the tail equals the
// midpoint exactly (the tie case), giving ties-towards-zero behaviour.

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundTiedBitsTowardsZero(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    Bits bias;
    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        bias = xsimd::select(Bits(roundoff) == Bits{},
            Bits(0),
            xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff - 1)) - Bits(1));
        #endif
    }
    else
    {
        bias = roundoff == 0
            ? Bits(0)
            : (Bits{1} << (roundoff - 1)) - Bits(1);
    }

    return bits + bias;
}

// ----- RoundBitsAwayFromZero ------------------------------------------------

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundBitsAwayFromZero(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        // Round the magnitude away from zero, but ONLY when the discarded tail
        // is nonzero — an exactly-representable value must round-trip exactly.
        Bits low_mask  = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1);
        Bits truncated = bits & ~low_mask;
        Bits tail      = bits & low_mask;
        Bits inc       = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff));
        return truncated + xsimd::select(tail != Bits(0), inc, Bits(0));
        #endif
    }
    else
    {
        // Increment by one ULP only when there is a nonzero remainder; otherwise
        // the input is exactly representable and must be returned unchanged.
        Bits low_mask  = (Bits{1} << roundoff) - 1;
        Bits truncated = bits & ~low_mask;
        Bits tail      = bits & low_mask;
        return truncated + (tail != 0 ? (Bits{1} << roundoff) : Bits{0});
    }
}

// ----- RoundBitsToNearestOdd ------------------------------------------------

template <typename Bits, typename Roundoff>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundBitsToNearestOdd(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    Bits bias;
    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        // bias = ((~bits >> roundoff) & 1) + (1 << (roundoff-1)) - 1
        Bits inv_lsb = xs::bitwise_rshift(~bits, xs::batch_cast<value_type>(roundoff)) & Bits(1);
        Bits half     = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff - 1));
        bias = xsimd::select(Bits(roundoff) == Bits{},
            Bits(0),
            inv_lsb + half - Bits(1));
        #endif
    }
    else
    {
        bias = roundoff == 0
            ? 0
            : ((~bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
    }

    return bits + bias;
}

// ----- RoundUp --------------------------------------------------------------
// Positive lanes: truncate + round up if remainder.
// Negative lanes: truncate only.

template <typename Bits, typename Roundoff, typename BoolT>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundUp(Bits bits, Roundoff roundoff, BoolT positive)
{
    using value_type = pod_type_t<Bits>;

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        // `positive` may be a scalar bool (broadcast to every lane) or a
        // per-lane xsimd::batch_bool — the latter lets a caller round each
        // element according to its own sign.
        Bits low_mask  = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1);
        Bits high_mask = ~low_mask;
        Bits truncated = bits & high_mask;

        auto has_remainder = (bits & low_mask) != Bits(0);
        auto pos_mask = [&] {
            if constexpr (std::is_same_v<BoolT, bool>)
                return xsimd::batch_bool<value_type>(positive);
            else
                return positive;
        }();
        auto do_inc = pos_mask & has_remainder;

        Bits inc = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff));
        return truncated + xsimd::select(do_inc, inc, Bits(0));
        #endif
    }
    else
    {
        const Bits low_mask  = (Bits{1} << roundoff) - 1;
        const Bits high_mask = ~low_mask;
        Bits truncated = bits & high_mask;

        truncated += (positive && ((bits & low_mask) != 0)) ? Bits{1} << roundoff : 0;
        return truncated;
    }
}

// Convenience overload: default positive = true
template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundUp(Bits bits, Roundoff roundoff)
{
    #ifndef USE_CUDA
    if constexpr (is_xsimd_batch<Bits>::value)
    {
        using value_type = pod_type_t<Bits>;
        return RoundUp(bits, roundoff, xsimd::batch_bool<value_type>(true));
    }
    else
    #endif
    {
        return RoundUp(bits, roundoff, true);
    }
}

// ----- RoundDown ------------------------------------------------------------
// Positive lanes: truncate only.
// Negative lanes: truncate + round up (increase magnitude) if remainder.

template <typename Bits, typename Roundoff, typename BoolT>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundDown(Bits bits, Roundoff roundoff, BoolT positive)
{
    using value_type = pod_type_t<Bits>;

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        Bits low_mask  = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1);
        Bits high_mask = ~low_mask;
        Bits truncated = bits & high_mask;

        auto has_remainder = (bits & low_mask) != Bits(0);
        // `positive` may be a scalar bool (broadcast) or a per-lane batch_bool.
        auto pos_mask = [&] {
            if constexpr (std::is_same_v<BoolT, bool>)
                return xsimd::batch_bool<value_type>(positive);
            else
                return positive;
        }();
        auto is_negative   = !pos_mask;
        auto do_inc = is_negative & has_remainder;

        Bits inc = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff));
        return truncated + xsimd::select(do_inc, inc, Bits(0));
        #endif
    }
    else
    {
        const Bits low_mask = (Bits{1} << roundoff) - 1;
        Bits truncated = bits & ~low_mask;
        return truncated + ((!positive && (bits & low_mask) != 0) ? Bits{1} << roundoff : 0);
    }
}

// Convenience overload: default positive = true
template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundDown(Bits bits, Roundoff roundoff)
{
    #ifndef USE_CUDA
    if constexpr (is_xsimd_batch<Bits>::value)
    {
        using value_type = pod_type_t<Bits>;
        return RoundDown(bits, roundoff, xsimd::batch_bool<value_type>(true));
    }
    else
    #endif
    {
        return RoundDown(bits, roundoff, true);
    }
}

// ----- RoundTiesToAway ------------------------------------------------------
// Round to nearest; break ties away from zero (if R==1, round up).

template <typename Bits, typename Roundoff>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundTiesToAway(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;

    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        auto mask = ~(xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1));
        Bits truncated = bits & mask;

        // Extract R bit: (bits >> (roundoff-1)) & 1
        Bits r_bit = xs::bitwise_rshift(bits, xs::batch_cast<value_type>(roundoff - 1)) & Bits(1);
        // Add r_bit << roundoff
        return truncated + xs::bitwise_lshift(r_bit, xs::batch_cast<value_type>(roundoff));
        #endif
    }
    else
    {
        auto mask = ~((Bits{1} << roundoff) - 1);
        Bits truncated = bits & mask;
        return truncated + (((bits >> (roundoff - 1)) & 1) << roundoff);
    }
}



template <typename Bits, typename Roundoff>
inline Bits RoundToOdd(Bits bits, Roundoff roundoff)
{
    using value_type = pod_type_t<Bits>;
 
    if constexpr (is_xsimd_batch<Roundoff>::value)
    {
        #ifndef USE_CUDA
        Bits low_mask  = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff)) - Bits(1);
        Bits high_mask = ~low_mask;
        Bits truncated = bits & high_mask;
 
        Bits tail = bits & low_mask;
        Bits sticky = xs::bitwise_lshift(Bits(1), xs::batch_cast<value_type>(roundoff));
        return truncated | xsimd::select(tail != Bits(0), sticky, Bits(0));
        #endif
    }
    else
    {
        Bits low_mask  = (Bits{1} << roundoff) - 1;
        Bits high_mask = ~low_mask;
        Bits truncated = bits & high_mask;
 
        Bits tail = bits & low_mask;
        return truncated | (tail != 0 ? Bits{1} << roundoff : Bits{0});
    }
}
        //#TODO: add sign to list of args for roundUp and RoundDown
        template <typename Bits>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundMantissa(Bits bits, const int roundoff, const ProjSpec ps, const bool pos = false)
        {
            const Rounding_Mode rm = ps.rounding_mode;
            const int len = ps.stoch_length;

            switch (rm)
            {
            case Rounding_Mode::RoundToNearestEven:
                return RoundBitsToNearestEven(bits, roundoff);
            case Rounding_Mode::RoundToNearestOdd:
                return RoundBitsToNearestOdd(bits, roundoff);
            case Rounding_Mode::RoundTowardsZero:
                return RoundBitsTowardsZero(bits, roundoff);
            case Rounding_Mode::RoundAwayFromZero:
                return RoundBitsAwayFromZero(bits, roundoff);
            case Rounding_Mode::RoundUp:
                return RoundUp(bits, roundoff, pos);
            case Rounding_Mode::RoundDown:
                return RoundDown(bits, roundoff, pos);
            case Rounding_Mode::RoundTiesToAway:
                return RoundTiesToAway(bits, roundoff);
            case Rounding_Mode::StochasticRoundingA:
                return Stochastic_Round_A(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingB:
                return Stochastic_Round_B(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingC:
                return Stochastic_Round_C(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingD:
                return Stochastic_Round_D(bits, roundoff, len);
            case Rounding_Mode::True_StochasticRounding:
                return True_Stochastic_Round(bits, roundoff);
            case Rounding_Mode::ProbabilisticRounding:
                return Probabilistic_Round(bits, roundoff);
            case Rounding_Mode::RoundToOdd:
                return RoundToOdd(bits, roundoff);
            case Rounding_Mode::RoundTiesTowardsZero:
                return RoundTiedBitsTowardsZero(bits, roundoff);
            default:
                return bits; // no rounding
            }
        }

        #ifndef USE_CUDA
        // `Pos` defaults to bool (broadcast to all lanes) but may also be an
        // xsimd::batch_bool so callers can supply a per-element sign for the
        // directional RoundUp/RoundDown modes.
        template <typename Bits, typename Roundoff,  class arch = xsimd::default_arch, typename Pos = bool>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE xs::batch<Bits, arch> RoundMantissa(xs::batch<Bits, arch> bits, const xs::batch<Roundoff, arch> roundoff, const ProjSpec ps, const Pos pos = true)
        {
            const Rounding_Mode rm = ps.rounding_mode;
            const int len = ps.stoch_length;
            switch (rm)
            {
            case Rounding_Mode::RoundToNearestEven:
                return RoundBitsToNearestEven(bits, roundoff);
            case Rounding_Mode::RoundToNearestOdd:
                return RoundBitsToNearestOdd(bits, roundoff);
            case Rounding_Mode::RoundTowardsZero:
                return RoundBitsTowardsZero(bits, roundoff);
            case Rounding_Mode::RoundAwayFromZero:
                return RoundBitsAwayFromZero(bits, roundoff);
            case Rounding_Mode::RoundUp:
                return RoundUp(bits, roundoff, pos);
            case Rounding_Mode::RoundDown:
                return RoundDown(bits, roundoff, pos);
            case Rounding_Mode::RoundTiesToAway:
                return RoundTiesToAway(bits, roundoff);
            case Rounding_Mode::StochasticRoundingA:
                return Stochastic_Round_A(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingB:
                return Stochastic_Round_B(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingC:
                return Stochastic_Round_C(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingD:
                return Stochastic_Round_D(bits, roundoff, len);
            case Rounding_Mode::True_StochasticRounding:
                return True_Stochastic_Round(bits, roundoff);
            case Rounding_Mode::ProbabilisticRounding:
                return Probabilistic_Round(bits, roundoff);
            case Rounding_Mode::RoundToOdd:
                return RoundToOdd(bits, roundoff);
            case Rounding_Mode::RoundTiesTowardsZero:
                return RoundTiedBitsTowardsZero(bits, roundoff);
             default:
                return bits; // no rounding
            }
        }
        #endif

        /// varfloat base
        template <typename Derived, FloatingPointParams Fp>
        class Var_lo_float : public lo_float_base<Derived, Base_repr_select<Fp.bitwidth>>
        {
        private:
            using UType = Base_repr_select<Fp.bitwidth>;
            using Base = lo_float_base<Derived, UType>;

            friend class lo_float_base<Derived, UType>;

            friend class Templated_Float<Fp>;

            // Inherit constructors from lo_float_base
            using Base::Base;

            using SType = typename std::make_signed<UType>::type;

            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE SType
            SignAndMagnitudeToTwosComplement(UType sign, UType magnitude)
            {
                return magnitude ^ (static_cast<SType>(sign << Fp.Len) < 0 ? -1 : 0);
            }

        protected:
            using typename Base::ConstructFromRepTag;

            constexpr Var_lo_float(UType rep, ConstructFromRepTag tag)
                : Base(rep, tag)
            {
            }

        public:
            explicit operator bool() const
            {
                if (get_signedness_v<Derived> == Signedness::Signed)
                {
                    return (this->rep() & ((1 << (Fp.bitwidth - 1)) - 1)) != 0;
                }
                else
                {
                    return this->rep() != 0;
                }
            }

            // declare structs/enums from template arg as static fields so that they can be accessed later
            static constexpr NaNChecker auto IsNaNFunctor = Fp.IsNaN;

            static constexpr InfChecker auto IsInfFunctor = Fp.IsInf;

            static constexpr Inf_Behaviors Overflow_behavior = Fp.OV_behavior;
            static constexpr NaN_Behaviors NaN_behavior = Fp.NA_behavior;

            static constexpr int bitwidth = Fp.bitwidth;

            static constexpr Signedness is_signed = Fp.is_signed;

            static constexpr Unsigned_behavior unsigned_behavior = Fp.unsigned_behavior;

            static constexpr int bias = Fp.bias;

            static constexpr int mantissa_bits = Fp.mantissa_bits;
        };

        template <FloatingPointParams Fp>
        class Templated_Float : public Var_lo_float<Templated_Float<Fp>, Fp>
        {
        private:
            using Base = Var_lo_float<Templated_Float<Fp>, Fp>;

        public:
            using Base::Base;

            static constexpr NaNChecker auto IsNaNFunctor = Fp.IsNaN;

            static constexpr InfChecker auto IsInfFunctor = Fp.IsInf;

            static constexpr Inf_Behaviors Overflow_behavior = Fp.OV_behavior;
            static constexpr NaN_Behaviors NaN_behavior = Fp.NA_behavior;

            static constexpr int bitwidth = Fp.bitwidth;

            static constexpr Signedness is_signed = Fp.is_signed;

            static constexpr Unsigned_behavior unsigned_behavior = Fp.unsigned_behavior;

            static constexpr int bias = Fp.bias;

            static constexpr int mantissa_bits = Fp.mantissa_bits;
        };

        struct Trivial_NaNChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t)
            {
                return false;
            }

            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const
            {
                return 0;
            }

            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const
            {
                return 0;
            }
        };

        struct Trivial_InfChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t) const
            {
                return false;
            }

            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const
            {
                return 0;
            }

            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const
            {
                return 0;
            }
        };

        // define FloatingPointParams for float8e4m3_fn
        struct OCP_F8E4M3_NaNChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits) const
            {
                return bits == 0x000000FF;
            }

            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const
            {
                return 0x000000FF;
            } // typical QNaN

            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const
            {
                return 0x000000FF;
            } // some SNaN pattern
        };

        struct OCP_F8E4M3_InfChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits) const
            {
                return false;
            }

            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const
            {
                return 0x0;
            } // -∞ => 0xFF800000

            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const
            {
                return 0x0;
            } // +∞ => 0x7F800000
        };

        template <Signedness signedness = Signedness::Signed>
        struct IEEE_F8_NaNChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits) const
            {
                return bits == (signedness == Signedness::Signed ? 0x00000080 : 0x000000FF);
            }

            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const
            {
                return signedness == Signedness::Signed ? 0x00000080 : 0x000000FF;
            } // typical QNaN

            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const
            {
                return signedness == Signedness::Signed ? 0x00000080 : 0x000000FF;
            } // some SNaN pattern
        };

        struct IEEE_F8_InfChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits) const
            {
                return bits == 0x0000007F || bits == 0x000000FF;
            }

            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const
            {
                return 0xFF;
            } // -∞ => 0xFF800000

            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const
            {
                return 0x7F;
            } // +∞ => 0x7F800000
        };

        constexpr FloatingPointParams param_float8_e4m3fn(
            8,                         // totoal bitwidth
            3,                         // mantissa bits
            7,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float8e4m3b11_fnuz -> f is finite, n is NaN, u unsigned, z zero

        constexpr FloatingPointParams param_float8_e4m3b11fnuz(
            8,                         // totoal bitwidth
            3,                         // mantissa bits
            11,                        // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Unsigned,      // It is unsigned
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor

        );

        // FloatingPointParams for float8e4m3b11_fnuz -> f is finite, n is NaN, u unsigned, z zero
        constexpr FloatingPointParams param_float8_e4m3fnuz(
            8,                         // totoal bitwidth
            3,                         // mantissa bits
            7,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Unsigned,      // It is unsigned
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // NaNChecker for float8e5m2
        struct OCP_F8E5M2_NaNChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits)
            {
                return bits == 0x000000FF || bits == 0x000000FE || bits == 0x000000FD;
            }

            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const
            {
                return 0x000000FF;
            }

            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const
            {
                return 0x000000FF;
            }
        };

        // FloatingPoint params for float8e5m2
        constexpr FloatingPointParams param_float8_e5m2(
            8,                         // totoal bitwidth
            2,                         // mantissa bits
            15,                        // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float8e5m2fnuz -> f is finite, n is NaN, u unsigned, z zero
        constexpr FloatingPointParams param_float8_e5m2fnuz(
            8,                         // totoal bitwidth
            2,                         // mantissa bits
            15,                        // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Unsigned,      // It is unsigned
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float8ieee_p
        template <int p>
        constexpr FloatingPointParams param_float8_ieee_p(
            8,                       // totoal bitwidth
            p - 1,                   // mantissa bits
            (1 << (7 - p)),          // bias
            Inf_Behaviors::Extended, // No infinity
            NaN_Behaviors::_3109, // NaN behavior
            Signedness::Signed,      // It is signed
            IEEE_F8_InfChecker(),    // Inf Functor
            IEEE_F8_NaNChecker()     // NaN Functor
        );

        // FloatingPointParams for float6e3m2
        constexpr FloatingPointParams param_float6_e3m2(
            6,                         // totoal bitwidth
            2,                         // mantissa bits
            3,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float6e2m3

        constexpr FloatingPointParams param_float6_e2m3(
            6,                         // totoal bitwidth
            3,                         // mantissa bits
            1,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float6_p
        template <int p>
        constexpr FloatingPointParams param_float6_p(
            6,                         // totoal bitwidth
            p - 1,                     // mantissa bits
            (1 << (5 - p)),            // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float4_e2m1
        constexpr FloatingPointParams param_float4_e2m1(
            4,                         // totoal bitwidth
            1,                         // mantissa bits
            1,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // FloatingPointParams for float4_p
        template <int p>
        constexpr FloatingPointParams param_float4_p(
            4,                         // totoal bitwidth
            p - 1,                     // mantissa bits
            (1 << (3 - p)),            // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::_3109,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // inf and nan checkers for P_3109 float
        template <int k, Signedness is_signed>
        struct P_3109_NaNChecker
        {
            LOFLOAT_HOST_DEVICE constexpr bool operator()(uint32_t bits) const
            {
                return bits == (is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1);
            }

            #ifndef USE_CUDA
            template <typename Int, class Arch>
            inline xsimd::batch_bool<Int, Arch>
            sim_check(const xsimd::batch<Int, Arch>& bits) const
            {
                constexpr Int target_scalar =
                    (is_signed == Signedness::Signed)
                        ? (Int{1} << (k - 1))
                        : ((Int{1} << k) - 1);

                const auto target = xsimd::broadcast<Int>(target_scalar);
                return bits == target;
            }
            #endif

            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const
            {
                return is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1;
            } // typical QNaN

            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const
            {
                return is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1;
            } 
        };

        template <int k, Signedness is_signed, Inf_Behaviors has_inf>
        struct P_3109_InfChecker
        {
            LOFLOAT_HOST_DEVICE constexpr bool operator()(uint32_t bits) const
            {
                if constexpr (has_inf != Inf_Behaviors::Saturating)
                {
                    return (bits | (1 << (k - 1))) == (is_signed == Signedness::Signed ? (1 << (k)) - 1 : (1 << k) - 2);
                }
                else
                {
                    return false; // No infinity for P_3109
                }
            }

            #ifndef USE_CUDA
            template <typename Int, class Arch>
            inline xsimd::batch_bool<Int, Arch>
            sim_check(const xsimd::batch<Int, Arch>& bits) const
            {
                if constexpr (has_inf != Inf_Behaviors::Saturating)
                {
                    auto target_scalar =
                        is_signed == Signedness::Signed
                            ? (Int{1} << (k)) - 1
                            : (Int{1} << k) - 2;
                    const auto target = xsimd::broadcast<Int>(target_scalar);
                    return (bits | (Int{1} << (k - 1))) == target;
                }
                else
                {
                    return false; // No infinity for P_3109
                }
            }
            #endif

            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const
            {
                return ((is_signed == Signedness::Signed) ? ((1 << (k)) - 1) : 0);
            } // -∞ => 0xFF800000

            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const
            {
                return (is_signed == Signedness::Signed) ? (1 << (k - 1)) - 1 : (1 << (k)) - 2;
            } // +∞ => 0x7F800000
        };

        // params for P_3109 float
        template <int k, int p, Signedness is_signed = Signedness::Signed, Inf_Behaviors has_inf = Inf_Behaviors::Saturating>
        constexpr FloatingPointParams param_float_p_3109(
            k,                                         // totoal bitwidth
            p - 1,                                     // mantissa bits
            (1 << (k - p - 1)),                        // bias
            has_inf,                                   // No infinity
            NaN_Behaviors::_3109,                   // NaN behavior
            is_signed,                                 // It is signed
            P_3109_InfChecker<k, is_signed, has_inf>(), // Inf Functor
            P_3109_NaNChecker<k, is_signed>()           // NaN Functor
        );

        // ------------------------------------------------------------------
        // OCP MX shared-scale format E8M0: 8-bit, unsigned, 8 exponent bits,
        // 0 mantissa bits, bias 127, no Inf, no zero. The all-ones code 0xFF
        // is NaN; X in [0,254] encodes 2^(X-127).
        // ------------------------------------------------------------------
        struct OCP_E8M0_NaNChecker
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t bits) const { return bits == 0xFF; }
            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const { return 0xFF; }
            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const { return 0xFF; }
        };
        struct OCP_E8M0_InfChecker // no infinities (Saturating)
        {
            LOFLOAT_HOST_DEVICE bool operator()(uint32_t) const { return false; }
            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const { return 0; }
            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const { return 0; }
        };
        constexpr FloatingPointParams param_ocp_e8m0(
            8, 0, 127, Inf_Behaviors::Saturating, NaN_Behaviors::_3109, Signedness::Unsigned,
            OCP_E8M0_InfChecker(), OCP_E8M0_NaNChecker());

        // ------------------------------------------------------------------
        // Tesla Dojo CFloat8/CFloat16: configurable float with a PROGRAMMABLE
        // (template) exponent bias and NO Inf/NaN (every encoding is finite).
        // Dedicated always-false checkers — do NOT reuse the P3109 checkers,
        // which reserve encodings for NaN/Inf.
        // ------------------------------------------------------------------
        struct DojoInfChecker
        {
            LOFLOAT_HOST_DEVICE constexpr bool operator()(uint32_t) const { return false; } // no infinities, ever
            LOFLOAT_HOST_DEVICE uint32_t minNegInf() const { return 0; }
            LOFLOAT_HOST_DEVICE uint32_t minPosInf() const { return 0; }
#ifndef USE_CUDA // SIMD path (mirrors P_3109_InfChecker::sim_check) — all-false
            template <typename Int, class Arch>
            inline xsimd::batch_bool<Int, Arch> sim_check(const xsimd::batch<Int, Arch>&) const {
                return xsimd::batch_bool<Int, Arch>(false);
            }
#endif
        };
        struct DojoNaNChecker
        {
            LOFLOAT_HOST_DEVICE constexpr bool operator()(uint32_t) const { return false; } // no NaNs, ever
            LOFLOAT_HOST_DEVICE uint32_t qNanBitPattern() const { return 0; }
            LOFLOAT_HOST_DEVICE uint32_t sNanBitPattern() const { return 0; }
#ifndef USE_CUDA
            template <typename Int, class Arch>
            inline xsimd::batch_bool<Int, Arch> sim_check(const xsimd::batch<Int, Arch>&) const {
                return xsimd::batch_bool<Int, Arch>(false);
            }
#endif
        };
        // P3109-shaped, but bias is an explicit template parameter and the
        // always-false checkers are used (p = mantissa+1).
        template <int k, int p, int bias, Signedness is_signed = Signedness::Signed>
        constexpr FloatingPointParams param_dojo_cfloat(
            k, p - 1, bias,
            Inf_Behaviors::Saturating, NaN_Behaviors::_3109, is_signed,
            DojoInfChecker(), DojoNaNChecker());

        // ------------------------
        // Bit-pattern constants
        // ------------------------
        struct float_bits
        {
            using utype = std::uint32_t;
            static constexpr utype sign = 0x80000000u;
            static constexpr utype exp  = 0x7F800000u;
            static constexpr utype frac = 0x007FFFFFu;

            static constexpr utype pos_inf = 0x7F800000u;
            static constexpr utype neg_inf = 0xFF800000u;

            // One typical quiet/signaling NaN payloads (not required, but handy)
            static constexpr utype qnan = 0x7FC00000u;
            static constexpr utype snan = 0x7FA00000u;
        };

        struct double_bits
        {
            using utype = std::uint64_t;
            static constexpr utype sign = 0x8000000000000000ull;
            static constexpr utype exp  = 0x7FF0000000000000ull;
            static constexpr utype frac = 0x000FFFFFFFFFFFFFull;

            static constexpr utype pos_inf = 0x7FF0000000000000ull;
            static constexpr utype neg_inf = 0xFFF0000000000000ull;

            // Typical qNaN/sNaN-ish examples
            static constexpr utype qnan = 0x7FF8000000000000ull;
            static constexpr utype snan = 0x7FF4000000000000ull;
        };

        // ------------------------
        // NaN checker (float/double)
        // ------------------------
        template <class FP> struct native_NaNChecker;

        template <>
        struct native_NaNChecker<float>
        {
            using bits_t = float_bits::utype;

            LOFLOAT_HOST_DEVICE constexpr bool operator()(bits_t bits) const
            {
                return ((bits & float_bits::exp) == float_bits::exp) &&
                    ((bits & float_bits::frac) != 0u);
            }

            LOFLOAT_HOST_DEVICE static constexpr bits_t qnan() { return float_bits::qnan; }
            LOFLOAT_HOST_DEVICE static constexpr bits_t snan() { return float_bits::snan; }

            #ifndef USE_CUDA
            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(float_bits::exp))
                                == xs::batch<bits_t, Arch>(float_bits::exp);
                auto frac_nonzero = (bits & xs::batch<bits_t, Arch>(float_bits::frac))
                                != xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_nonzero;
            }
            #endif
        };

        template <>
        struct native_NaNChecker<double>
        {
            using bits_t = double_bits::utype;

            LOFLOAT_HOST_DEVICE constexpr bool operator()(bits_t bits) const
            {
                return ((bits & double_bits::exp) == double_bits::exp) &&
                    ((bits & double_bits::frac) != 0ull);
            }

            LOFLOAT_HOST_DEVICE static constexpr bits_t qnan() { return double_bits::qnan; }
            LOFLOAT_HOST_DEVICE static constexpr bits_t snan() { return double_bits::snan; }

            #ifndef USE_CUDA
            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(double_bits::exp))
                                == xs::batch<bits_t, Arch>(double_bits::exp);
                auto frac_nonzero = (bits & xs::batch<bits_t, Arch>(double_bits::frac))
                                != xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_nonzero;
            }
            #endif
        };

        // ------------------------
        // Inf checker (float/double)
        // ------------------------
        template <class FP> struct InfChecker;

        template <>
        struct InfChecker<float>
        {
            using bits_t = float_bits::utype;

            LOFLOAT_HOST_DEVICE constexpr bool operator()(bits_t bits) const
            {
                return ((bits & float_bits::exp) == float_bits::exp) &&
                    ((bits & float_bits::frac) == 0u);
            }

            LOFLOAT_HOST_DEVICE static constexpr bits_t posinf() { return float_bits::pos_inf; }
            LOFLOAT_HOST_DEVICE static constexpr bits_t neginf() { return float_bits::neg_inf; }

            #ifndef USE_CUDA
            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(float_bits::exp))
                                == xs::batch<bits_t, Arch>(float_bits::exp);
                auto frac_zero    = (bits & xs::batch<bits_t, Arch>(float_bits::frac))
                                == xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_zero;
            }
            #endif
        };

        template <>
        struct InfChecker<double>
        {
            using bits_t = double_bits::utype;

            LOFLOAT_HOST_DEVICE constexpr bool operator()(bits_t bits) const
            {
                return ((bits & double_bits::exp) == double_bits::exp) &&
                    ((bits & double_bits::frac) == 0ull);
            }

            LOFLOAT_HOST_DEVICE static constexpr bits_t posinf() { return double_bits::pos_inf; }
            LOFLOAT_HOST_DEVICE static constexpr bits_t neginf() { return double_bits::neg_inf; }

            #ifndef USE_CUDA
            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(double_bits::exp))
                                == xs::batch<bits_t, Arch>(double_bits::exp);
                auto frac_zero    = (bits & xs::batch<bits_t, Arch>(double_bits::frac))
                                == xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_zero;
            }
            #endif
        };

    } // namepsace lo_float_internal

    // now define the types using the previously defined FloatingPointParams

    using float8_e4m3_fn = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3fn>;

    using float8_e4m3b11_fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3b11fnuz>;

    using float8_e4m3_fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3fnuz>;

    using float8_e5m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e5m2>;

    using float8_e5m2fnuz = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e5m2fnuz>;

    template <int p>
    using float8_ieee_p = lo_float_internal::Templated_Float<lo_float_internal::param_float8_ieee_p<p>>;

    using float6_e3m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float6_e3m2>;

    using float6_e2m3 = lo_float_internal::Templated_Float<lo_float_internal::param_float6_e2m3>;

    template <int p>
    using float6_p = lo_float_internal::Templated_Float<lo_float_internal::param_float6_p<p>>;

    template <int p>
    using float4_e2m1 = lo_float_internal::Templated_Float<lo_float_internal::param_float4_e2m1>;

    template <int p>
    using float4_p = lo_float_internal::Templated_Float<lo_float_internal::param_float4_p<p>>;

    template <int k, int p, lo_float::Signedness is_signed = lo_float::Signedness::Signed, lo_float::Inf_Behaviors has_inf = lo_float::Inf_Behaviors::Saturating>
    using P_3109_float = lo_float_internal::Templated_Float<lo_float_internal::param_float_p_3109<k, p, is_signed, has_inf>>;

    // ----------------------------------------------------------------------
    // Public format-preset aliases (F4)
    // ----------------------------------------------------------------------
    // IEEE / ML presets (reuse existing *PrecisionParams)
    using half     = lo_float_internal::Templated_Float<halfPrecisionParams>;
    using bfloat16 = lo_float_internal::Templated_Float<bfloatPrecisionParams>;
    using tf32     = lo_float_internal::Templated_Float<tf32PrecisionParams>;
    using single   = lo_float_internal::Templated_Float<singlePrecisionParams>;
    using float16  = half;   // bit-width-explicit synonyms
    using float32  = single;

    // OCP MX element presets
    using ocp_e4m3 = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e4m3fn>;
    using ocp_e5m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float8_e5m2>;
    using ocp_e3m2 = lo_float_internal::Templated_Float<lo_float_internal::param_float6_e3m2>;
    using ocp_e2m1 = lo_float_internal::Templated_Float<lo_float_internal::param_float4_e2m1>;
    using ocp_e8m0 = lo_float_internal::Templated_Float<lo_float_internal::param_ocp_e8m0>;

    // Tesla Dojo configurable presets (programmable exponent bias; no Inf/NaN).
    // Default bias = IEEE-style (1<<(exp_bits-1))-1; p = mantissa+1.
    template <int bias = 7>   using dojo_cfloat8_1_4_3  = lo_float_internal::Templated_Float<lo_float_internal::param_dojo_cfloat<8, 4, bias>>;
    template <int bias = 15>  using dojo_cfloat8_1_5_2  = lo_float_internal::Templated_Float<lo_float_internal::param_dojo_cfloat<8, 3, bias>>;
    template <int bias = 127> using dojo_cfloat16_1_8_7 = lo_float_internal::Templated_Float<lo_float_internal::param_dojo_cfloat<16, 8, bias>>;
    template <int bias = 31>  using dojo_cfloat16_1_6_9 = lo_float_internal::Templated_Float<lo_float_internal::param_dojo_cfloat<16, 10, bias>>;

    template <typename T>
    constexpr T ConstexprAbs(T x) { return x < T{0.0} ? -x : x; }

    template <typename T>
    constexpr T ConstexprCeil(T x)
    {
        constexpr T kIntegerThreshold =
            uint64_t{1} << (std::numeric_limits<T>::digits - 1);
        // Too big or NaN inputs get returned unchanged.
        if (!(ConstexprAbs(x) < kIntegerThreshold))
        {
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
    constexpr int Digits10FromDigits(int digits)
    {
        return static_cast<int>(ConstexprFloor((digits - 1) * kLog10Of2));
    }

    // C17 5.2.4.2.2p11:
    // "number of decimal digits, n, such that any floating-point number with p
    // radix b digits can be rounded to a floating-point number with n decimal
    // digits and back again without change to the value"
    // ceil(1 + p * log10(2));
    constexpr int MaxDigits10FromDigits(int digits)
    {
        return static_cast<int>(ConstexprCeil(1.0 + (digits * kLog10Of2)));
    }

    // C17 5.2.4.2.2p11:
    // "minimum negative integer such that 10 raised to that power is in the range
    // of normalized floating-point numbers"
    // TODO: https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent10 says "representable"
    // ceil(log10(2**(emin - 1))) == ceil((emin - 1) * log10(2));
    constexpr int MinExponent10FromMinExponent(int min_exponent)
    {
        return static_cast<int>(ConstexprCeil((min_exponent - 1) * kLog10Of2));
    }

    // C17 5.2.4.2.2p11:
    // "maximum integer such that 10 raised to that power is in the range of
    // representable finite floating-point numbers"
    // floor(log10((1 - 2**-p) * 2**emax)) == floor(log10(1 - 2**-p) +
    // emax * log10(2))
    constexpr int MaxExponent10FromMaxExponentAndDigits(int max_exponent,
                                                        int digits)
    {
        // We only support digits in {3,4}. This table would grow if we wanted to
        // handle more values.
        constexpr double kLog10OfOnePredecessor[] = {
            // log10(1 - 2**-1)
            -0.3010299956639812,
            // log10(1 - 2**-2)
            -0.12493873660829993,
            // log10(1 - 2**-3)
            -0.057991946977686754,
            // log10(1 - 2**-4)
            -0.028028723600243537,
            // log10(1 - 2**-5)
            -0.013788284485633295,
            // log10(1 - 2**-6)
            -0.006931471805599453,
            // log10(1 - 2**-7)
            -0.003415655202992898,
            // log10(1 - 2**-8)
            -0.0017077547810594657};
        //
        return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 1] +
                                               max_exponent * kLog10Of2));
    }

    namespace lo_float_internal
    {

        template <FloatingPointParams Fp>
        struct numeric_limits_flexible
        {

            static constexpr bool is_specialized = true;
            static constexpr bool is_signed = Fp.is_signed == lo_float::Signedness::Signed;
            static constexpr bool is_integer = false;
            static constexpr bool is_exact = false;
            static constexpr bool has_quiet_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::_3109;
            static constexpr bool has_signaling_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::_754;
            static constexpr bool has_denorm = true;
            static constexpr bool has_denorm_loss = false;
            static constexpr bool round_style = std::round_indeterminate;
            static constexpr bool is_iec559 = false;
            static constexpr int radix = std::numeric_limits<float>::radix;
            static constexpr bool traps = false;
            static constexpr bool tinyness_before = true;
            static constexpr int kExponentBias = Fp.bias;
            static constexpr int kMantissaBits = Fp.mantissa_bits;
            static constexpr int digits = Fp.mantissa_bits + 1;
            static constexpr int digits10 = Digits10FromDigits(digits);
            static constexpr int max_digits10 = MaxDigits10FromDigits(digits);
            static constexpr int min_exponent = (1 - kExponentBias);
            static constexpr int min_exponent10 =
                MinExponent10FromMinExponent(min_exponent);
            static constexpr int max_exponent = is_signed ? (1 << (Fp.bitwidth - Fp.mantissa_bits - 1)) - 1 - Fp.bias : (1 << (Fp.bitwidth - Fp.mantissa_bits)) - 1 - Fp.bias;
            static constexpr int max_exponent10 =
                MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
            static constexpr bool has_infinity = Fp.OV_behavior != lo_float::Inf_Behaviors::Saturating;

            using Base_Type = Base_repr_select<Fp.bitwidth>;
            static constexpr Templated_Float<Fp> min()
            {
                return Templated_Float<Fp>::FromRep((1 << (Fp.mantissa_bits)));
            }
            static constexpr Templated_Float<Fp> lowest()
            {
                if (Fp.is_signed == lo_float::Signedness::Signed)
                {
                    if (Fp.OV_behavior == lo_float::Inf_Behaviors::Saturating)
                    {
                        return Templated_Float<Fp>::FromRep(((1 << (Fp.bitwidth)) - 1) >> 1);
                    }
                    return Templated_Float<Fp>::FromRep(Fp.IsInf.minNegInf() - 1);
                }
                else
                {
                    return Templated_Float<Fp>::FromRep(0);
                }
            }
            static constexpr Templated_Float<Fp> max()
            {
                // if Extended return inf - 1, else return fromrep(1111...1)
                if (Fp.OV_behavior == lo_float::Inf_Behaviors::Saturating)
                {
                    if (Fp.is_signed == lo_float::Signedness::Signed)
                    {
                        return Templated_Float<Fp>::FromRep(((1 << (Fp.bitwidth)) - 1) >> 1);
                    }
                    else
                    {
                        return Templated_Float<Fp>::FromRep((1 << (Fp.bitwidth)) - 1);
                    }
                }
                return Templated_Float<Fp>::FromRep(Fp.IsInf.minPosInf() - 1);
            }
            static constexpr Templated_Float<Fp> epsilon()
            {
                return Templated_Float<Fp>::FromRep(
                    static_cast<Base_Type>(1ULL << (kMantissaBits - 1)));
            }
            static constexpr Templated_Float<Fp> round_error()
            {
                return Templated_Float<Fp>::FromRep(
                    ((-1 + kExponentBias) << kMantissaBits));
            }
            static constexpr Templated_Float<Fp> infinity()
            {
                if (Fp.OV_behavior == lo_float::Inf_Behaviors::Saturating)
                    return max();
                return Templated_Float<Fp>::FromRep(Fp.IsInf.minPosInf());
            }
            static constexpr Templated_Float<Fp> quiet_NaN()
            {
                return Templated_Float<Fp>::FromRep(Fp.IsNaN.qNanBitPattern());
            }
            static constexpr Templated_Float<Fp> signaling_NaN()
            {
                return Templated_Float<Fp>::FromRep(Fp.IsNaN.sNanBitPattern());
            }
            static constexpr Templated_Float<Fp> denorm_min()
            {
                return Templated_Float<Fp>::FromRep(0x1);
            }
        };

    } // namespace lo_float_internal

    template <FloatingPointParams Fp>
    using Templated_Float = lo_float_internal::Templated_Float<Fp>;


}

namespace lo_float
{
    namespace lo_float_internal
    {

        // dont need to change this for signed vs unsigned
        template <FloatingPointParams Fp>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Templated_Float<Fp> abs(const Templated_Float<Fp> &a)
        {
            if constexpr (get_signedness_v<Templated_Float<Fp>> == Signedness::Signed)
            {
                return Templated_Float<Fp>::FromRep(a.rep() & ((1 << (Fp.bitwidth - 1)) - 1));
            }
            return a;
        }

        template <FloatingPointParams Fp>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isnan(const Templated_Float<Fp> &a)
        {
            return Fp.IsNaN(a.rep());
        }

        //the simd version takes in the rep for now
        #ifndef USE_CUDA
        template <NaNChecker IsNaN,
          typename Int,
          class Arch = xsimd::default_arch>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE
        xsimd::batch_bool<Int, Arch>
        isnan_simd(const xsimd::batch<Int, Arch>& a, IsNaN IsNaNFunc)
        {
            static_assert(std::is_unsigned_v<Int>,
                        "isnan_simd requires an unsigned integer SIMD type");

            // Call the SIMD checker directly
            return IsNaNFunc.sim_check(a);
        }
        #endif

        template <FloatingPointParams Fp>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isinf(const Templated_Float<Fp> &a)
        {
            return Fp.IsInf(a.rep()) && Fp.OV_behavior != Inf_Behaviors::Saturating;
        }

#ifdef USE_CUDA
        // nvcc cannot deduce the class-type NTTP `Fp` from a Templated_Float<Fp>
        // argument, so the FloatingPointParams-templated abs/isnan/isinf overloads
        // above are not viable candidates on the device. These plain-type (T)
        // overloads ARE deducible by nvcc; they are SFINAE-constrained to the
        // Templated_Float family via its static IsNaNFunctor member (so they never
        // collide with the float/double/c10::Half/i_n overloads) and delegate to the
        // type's own static members instead of a deduced Fp. Guarded by USE_CUDA so
        // the host build keeps using the generic NTTP overloads unchanged.
        template <class T, typename = decltype(T::IsNaNFunctor)>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE T abs(const T &a)
        {
            if constexpr (get_signedness_v<T> == Signedness::Signed)
                return T::FromRep(a.rep() & ((1 << (T::bitwidth - 1)) - 1));
            return a;
        }
        template <class T, typename = decltype(T::IsNaNFunctor)>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isnan(const T &a)
        {
            // Reconstruct a stateless temporary instead of ODR-using the static
            // constexpr IsNaNFunctor object (nvcc emits no device storage for it).
            using Chk = std::remove_cv_t<std::remove_reference_t<decltype(T::IsNaNFunctor)>>;
            return Chk{}(a.rep());
        }
        template <class T, typename = decltype(T::IsInfFunctor)>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isinf(const T &a)
        {
            using Chk = std::remove_cv_t<std::remove_reference_t<decltype(T::IsInfFunctor)>>;
            return Chk{}(a.rep()) && T::Overflow_behavior != Inf_Behaviors::Saturating;
        }
#endif

        #ifndef USE_CUDA
        template <bool OV_behavior,
        typename IsInf,
          typename Int,
          class Arch = xsimd::default_arch>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE
        xsimd::batch_bool<Int, Arch>
        isinf_simd(const xsimd::batch<Int, Arch>& a, IsInf inf_func)
        {
            static_assert(std::is_unsigned_v<Int>,
                        "isinf_simd requires an unsigned integer SIMD type");

            if constexpr (OV_behavior == Inf_Behaviors::Saturating)
            {
                // No infinity in this mode: return all-false mask
                return xsimd::batch_bool<Int, Arch>(false);
            }
            else
            {
                return inf_func.sim_check(a);
            }
        }
        #endif


    }
}

namespace std
{

    template <lo_float::FloatingPointParams Fp>
    struct numeric_limits<lo_float::Templated_Float<Fp>>
        : lo_float::lo_float_internal::numeric_limits_flexible<Fp>
    {
    };

    // abs override
    template <lo_float::FloatingPointParams Fp>
    lo_float::Templated_Float<Fp> abs(
        const lo_float::Templated_Float<Fp> &a)
    {
        return lo_float::lo_float_internal::abs(a);
    }



    // isnan overrides
    template <lo_float::FloatingPointParams Fp>
    bool isnan(const lo_float::Templated_Float<Fp> &a)
    {
        return lo_float::lo_float_internal::isnan(a);
    }

    #ifndef USE_CUDA
    template <lo_float::NaNChecker IsNaN, typename Int, class Arch>
    xsimd::batch_bool<Int, Arch>
    isnan_simd(const xsimd::batch<Int, Arch>& a, IsNaN isnan_func)
    {
        return lo_float::lo_float_internal::isnan_simd<IsNaN>(a, isnan_func);
    }
    #endif

    // isinf override
    template <lo_float::FloatingPointParams Fp>
    bool isinf(const lo_float::Templated_Float<Fp> &a)
    {
        return lo_float::lo_float_internal::isinf(a);
    }


    #ifndef USE_CUDA
    template <lo_float::InfChecker IsInf, typename Int, bool OV_behavior, class Arch>
    xsimd::batch_bool<Int, Arch>
    isinf_simd(const xsimd::batch<Int, Arch>& a, IsInf inf_func)
    {
        return lo_float::lo_float_internal::isinf_simd<IsInf, OV_behavior>(a, inf_func);
    }
    #endif

} // namespace std

namespace lo_float
{
    namespace lo_float_internal
    {

       template <int le_size, class UInt, class ret_type>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE ret_type countl_zero(UInt x)
        {
            static_assert(le_size >= 1, "le_size must be positive");

            using ret_scalar = pod_type_t<ret_type>;
            using uint_scalar = pod_type_t<UInt>;

            // Work in ret_type width (so small inputs are handled correctly)
            ret_type xw;
            #ifndef USE_CUDA
            if constexpr (xsimd::is_batch<UInt>::value)
                xw = widen_low_lanes<ret_scalar, uint_scalar>(x);
            else
                xw = static_cast<ret_type>(x);
            #else
            xw = static_cast<ret_type>(x);
            #endif

            ret_type zeroes = ret_type(le_size - 4);

            auto step = [&](int k) {
                #ifndef USE_CUDA
                if constexpr (xsimd::is_batch<ret_type>::value)
                {
                    using M = typename ret_type::batch_bool_type;
                    M m = (xw >> k) != ret_type(0);
                    zeroes = xsimd::select(m, zeroes - ret_type(k), zeroes);
                    xw     = xsimd::select(m, xw >> k, xw);
                }
                else
                {
                    if (xw >> k) { zeroes -= ret_type(k); xw >>= k; }
                }
                #else
                if (xw >> k) { zeroes -= ret_type(k); xw >>= k; }
                #endif
            };

            if constexpr (le_size > 64) step(64);
            if constexpr (le_size > 32) step(32);
            if constexpr (le_size > 16) step(16);
            if constexpr (le_size > 8)  step(8);
            if constexpr (le_size > 4)  step(4);

            // Use a 32-bit LUT so gather returns 32-bit lanes and adds cleanly.
            alignas(64) static constexpr uint32_t clz4_lut32[16] = {
                4,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0
            };

            // Safe nibble index
            auto idx = xw & ret_type(0xF);

            #ifndef USE_CUDA
            if constexpr (xsimd::is_batch<ret_type>::value)
            {
                using scalar_t = typename ret_type::value_type;
                constexpr std::size_t lanes = ret_type::size;

                scalar_t idx_a[lanes];
                scalar_t zero_a[lanes];
                scalar_t out_a[lanes];

                // If idx/zeroes are xsimd batches:
                idx.store_unaligned(idx_a);
                zeroes.store_unaligned(zero_a);

                #pragma unroll
                for (std::size_t i = 0; i < lanes; ++i)
                {
                    const auto lut = clz4_lut32[static_cast<std::uint32_t>(idx_a[i])];
                    out_a[i] = static_cast<scalar_t>(lut) + zero_a[i];
                }

                // Put values back into the xsimd register
                xw = ret_type::load_unaligned(out_a);
                return xw;
            }
            else
            {
                return ret_type(clz4_lut32[static_cast<uint32_t>(idx)]) + zeroes;
            }
            #else
            return ret_type(clz4_lut32[static_cast<uint32_t>(idx)]) + zeroes;
            #endif


        }



        // template <int kNumBytes>
        // using GetUnsignedInteger =
        //     typename std::get_integer_by_size<kNumBytes>::unsigned_type;

        template <int a, typename Enable = void>
        struct IntegerBySize;

        template <int a>
        struct IntegerBySize<a, std::enable_if_t<a == 1>>
        {
            using unsigned_type = uint8_t;
            using signed_type = int8_t;
        };

        template <int a>
        struct IntegerBySize<a, std::enable_if_t<a == 2>>
        {
            using unsigned_type = uint16_t;
            using signed_type = int16_t;
        };

        template <int a>
        struct IntegerBySize<a, std::enable_if_t<(a > 2 && a <= 4)>>
        {
            using unsigned_type = uint32_t;
            using signed_type = int32_t;
        };

        template <int a>
        struct IntegerBySize<a, std::enable_if_t<(a > 4 && a <= 8)>>
        {
            using unsigned_type = uint64_t;
            using signed_type = int64_t;
        };

        template <int a>
        struct IntegerBySize<a, std::enable_if_t<(a > 8)>>
        {
            using unsigned_type = unsigned long long;
            using signed_type = long long;
        };

        // Alias to get the unsigned type directly
        template <int kNumBytes>
        using GetUnsignedInteger = typename IntegerBySize<kNumBytes>::unsigned_type;

        // Converts between two floating-point types.
        template <typename From, typename To,
                  typename EnableIf = void>
        struct ConvertImpl;

        // Convert to same type.  We need explicit specializations for all combinations
        // of template parameters to avoid ambiguities.
        template <typename Scalar>
        struct IdentityConversion
        {
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Scalar run(const Scalar &from, ProjSpec ps = ProjSpec{})
            {
                return from;
            }

            #ifndef USE_CUDA
            template <class arch = xsimd::default_arch>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE void run(const Scalar* from, Scalar* to, int n, ProjSpec ps = ProjSpec{})
            {
                if (to != from) {
                    memcpy(to, from, n * sizeof(Scalar));
                }
                return;
            }
            #endif
        };

        template <typename Scalar>
        struct ConvertImpl<Scalar, Scalar> : public IdentityConversion<Scalar>
        {
        };



        template <typename Float>
        struct TraitsBase
        {
            using BitsType = GetUnsignedInteger<sizeof(Float)>;
            static constexpr int kBits = sizeof(Float) * CHAR_BIT;
            static constexpr int kMantissaBits = std::numeric_limits<Float>::digits - 1;
            static constexpr int kExponentBits = get_signedness_v<Float> == Signedness::Signed
                                                     ? get_bitwidth_v<Float> - get_mantissa_bits_v<Float> - 1
                                                     : get_bitwidth_v<Float> - get_mantissa_bits_v<Float>;
            static constexpr BitsType kExponentMask = get_signedness_v<Float> == Signedness::Signed
                                                          ? (BitsType{1} << (kExponentBits - 1)) - 1
                                                          : (BitsType{1} << kExponentBits) - 1;
            static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
            // kExponentBias lives here (not only in the Traits<Templated_Float<Fp>>
            // partial specialization) so it is still available when nvcc fails to
            // match that specialization for a Templated_Float type and falls back to
            // the primary Traits<Float> : TraitsBase<Float>. Uses the same get_*_v
            // accessor machinery as the other fields above.
            static constexpr int kExponentBias = get_bias_v<Float>;
        };

        template <int len, Signedness sign>
        struct TraitsBase<i_n<len, sign>>
        {
            using Int = i_n<len, sign>;
            using BitsType = GetUnsignedInteger<sizeof(i_n<len, sign>)>;
            static constexpr int kBits = sizeof(i_n<len, sign>) * CHAR_BIT;
            static constexpr int kMantissaBits = sign == Signedness::Signed ? len - 1 : len;
            static constexpr int kExponentBits = 0;
            static constexpr BitsType kExponentMask = 0;
            static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
            static constexpr int kExponentBias = 0;
        };


        template <typename Float>
        struct Traits : public TraitsBase<Float>
        {
        };

        template <FloatingPointParams Fp>
        struct Traits<Templated_Float<Fp>> : public TraitsBase<Templated_Float<Fp>>
        {
            using Base = TraitsBase<Templated_Float<Fp>>;
            static constexpr int kBits = Fp.bitwidth;
            static constexpr int kMantissaBits = Fp.mantissa_bits;
            static constexpr int kExponentBits = get_signedness_v<Templated_Float<Fp>> == Signedness::Signed
                                                     ? Fp.bitwidth - Fp.mantissa_bits - 1
                                                     : Fp.bitwidth - Fp.mantissa_bits;
            static constexpr int kExponentBias = Fp.bias;
        };

        template <int len, Signedness sign>
        struct Traits<i_n<len, sign>> : public TraitsBase<i_n<len, sign>>
        {
            using Base = TraitsBase<i_n<len, sign>>;
        };


        //FP16

        #ifdef USE_CUDA
        template <>
        struct Traits<c10::Half> : public TraitsBase<c10::Half> {
            using BitsType = uint16_t;
            static constexpr int kBits = 16;
            static constexpr int kExponentBits = 5;
            static constexpr int kMantissaBits = 10;
            static constexpr int kExponentBias = 15;  // (1 << 4) - 1
            static constexpr BitsType kExponentMask = (BitsType{0x1F} << 10);   // 0x7C00
            static constexpr BitsType kMantissaMask = (BitsType{1} << 10) - 1;  // 0x03FF
            static constexpr BitsType kSignBit = BitsType{1} << 15;             // 0x8000
            static constexpr int kMinExponent = 1 - kExponentBias;              // -14
            static constexpr int kMaxExponent = (1 << kExponentBits) - 2 - kExponentBias; // 16
        };
        #endif
        // float (FP32)
template <>
struct Traits<float> : public TraitsBase<float> {
    using BitsType = uint32_t;
    static constexpr int kBits = sizeof(float) * CHAR_BIT;
    static constexpr int kExponentBits = 8;
    static constexpr int kMantissaBits = 23;
    static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
    static constexpr BitsType kExponentMask = (BitsType{0xFF} << 23);
    static constexpr BitsType kMantissaMask = (BitsType{1} << 23) - 1;
    static constexpr BitsType kSignBit = BitsType{1} << 31;
    static constexpr int kMinExponent = 1 - kExponentBias;
    static constexpr int kMaxExponent = (1 << kExponentBits) - 2 - kExponentBias;
};

// double (FP64)
template <>
struct Traits<double> : public TraitsBase<double> {
    using BitsType = uint64_t;
    static constexpr int kBits = sizeof(double) * CHAR_BIT;
    static constexpr int kExponentBits = 11;
    static constexpr int kMantissaBits = 52;
    static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
    static constexpr BitsType kExponentMask = (BitsType{0x7FF} << 52);
    static constexpr BitsType kMantissaMask = (BitsType{1} << 52) - 1;
    static constexpr BitsType kSignBit = BitsType{1} << 63;
    static constexpr int kMinExponent = 1 - kExponentBias;
    static constexpr int kMaxExponent = (1 << kExponentBits) - 2 - kExponentBias;
};

        template <>
        struct Traits<int> : public TraitsBase<int> {
            using BitsType = uint32_t;
            static constexpr int kBits = sizeof(int) * CHAR_BIT;
            static constexpr int kExponentBits = 0;
            static constexpr int kMantissaBits = sizeof(int) * CHAR_BIT - 1;
            static constexpr int kExponentBias = 0;
            static constexpr BitsType kExponentMask = 0;
            static constexpr BitsType kMantissaMask = (BitsType{1} << (sizeof(int) * CHAR_BIT - 1)) - 1;
        };



template <typename T>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE T from_double(double val) {
    return static_cast<T>(val);
}

template <typename T>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE float get_float(T val) {
    return static_cast<float>(val);
}

// Magnitude helper that is correct on BOTH host and device. An unqualified
// abs(value) inside virtual_round() falls back to `int abs(int)` on the DEVICE
// for native float/double (std::abs(float) is host-only there), truncating e.g.
// 1.62f -> 1 and corrupting the exponent path so the result underflows to 0.
// (A plain abs(float) overload here can't be used because lo_float_internal
// already pulls in std::abs, which would make it ambiguous.) Use a uniquely
// named helper instead.
template <typename T>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE T lof_abs(const T& v) {
    if constexpr (std::is_same_v<T, float>)       return ::fabsf(v);
    else if constexpr (std::is_same_v<T, double>) return ::fabs(v);
    else                                          return abs(v);  // Templated_Float / c10::Half
}

#ifdef USE_CUDA
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE c10::Half abs(c10::Half val) {
    val.x &= 0x7FFF;  // clear sign bit
    return val;
}   
#endif


template<typename From>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE From virtual_round(const From &from, int ToMantissaBits, ProjSpec ps = ProjSpec{}) {

    [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
    [[maybe_unused]] const int stoch_len = ps.stoch_length;
    using FromBits = typename Traits<From>::BitsType;
static constexpr int kFromBits = Traits<From>::kBits;
int kDigitShift = ToMantissaBits - Traits<From>::kMantissaBits;
static constexpr int kFromMantissaBits = Traits<From>::kMantissaBits;
     #ifdef USE_CUDA
     const auto sign_bit = cuda::std::bit_cast<FromBits>(from) & (FromBits{1} << (kFromBits - 1));
     #else
     const auto sign_bit = std::bit_cast<FromBits>(from) & (FromBits{1} << (kFromBits - 1));
     #endif

#ifdef USE_CUDA
    FromBits from_bits = cuda::std::bit_cast<FromBits>(from)
                         & ~(FromBits{1} << (kFromBits - 1));
#else
    FromBits from_bits = std::bit_cast<FromBits>(from)
                         & ~(FromBits{1} << (kFromBits - 1));
#endif
// {
     from_bits = RoundMantissa(from_bits, -kDigitShift, ps, sign_bit == 0);

    
     from_bits &= ~((FromBits{1} << (-kDigitShift)) - 1);
// }

#ifdef USE_CUDA
if constexpr (std::is_same_v<From, c10::Half>) {
    c10::Half ret;
    ret.x = static_cast<uint16_t>(from_bits | sign_bit);
        return ret;
} else {
return cuda::std::bit_cast<From>(from_bits | sign_bit);
}
#else
return std::bit_cast<From>(from_bits | sign_bit);
#endif
    
}

#ifndef USE_CUDA
template<typename From, class arch = xs::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE void virtual_round(From* values, From* results, int n, int ToMantissaBits, ProjSpec ps = ProjSpec{}) {

    [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
    [[maybe_unused]] const int stoch_len = ps.stoch_length;
    using FromBits = typename Traits<From>::BitsType;
    using FromBitsSIMD = xs::batch<FromBits, arch>;
    using FromSIMD = xs::batch<From, arch>;
    using IntSIMD = std::conditional_t<std::is_same_v<From, float>, xs::batch<int, arch>, xs::batch<int64_t, arch>>;

    static constexpr int kFromBits = Traits<From>::kBits;
    int kDigitShift = ToMantissaBits - Traits<From>::kMantissaBits;
    static constexpr int kFromMantissaBits = Traits<From>::kMantissaBits;

    static constexpr std::size_t step = FromBitsSIMD::size;

    auto load_from = [&](int i) -> FromBitsSIMD {
        return FromBitsSIMD::load_unaligned(&values[i]);
    };

    auto store_from = [&](int i, const FromBitsSIMD& v) {
        v.store_unaligned(&results[i]);
    };

    #ifdef _LOFOPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n - (n % step); i += step) {
        // Load SIMD batch
        FromSIMD from = FromSIMD::load_unaligned(&values[i]);
        
        // Get absolute value and bit-cast to bits
        FromSIMD abs_from = xs::abs(from);
        FromBitsSIMD from_bits = xs::bit_cast<FromBitsSIMD>(abs_from);
        
        {
            from_bits = RoundMantissa(from_bits, FromBitsSIMD(-kDigitShift), ps);
            FromBitsSIMD mask = ~((FromBitsSIMD(FromBits{1}) << (-kDigitShift)) - FromBitsSIMD(FromBits{1}));
            from_bits = from_bits & mask;
        }
        
        // Bit-cast back and store
        FromSIMD result = xs::bit_cast<FromSIMD>(from_bits);
        result.store_unaligned(&results[i]);
    }

    // Handle remainder
    for (int i = n - (n % step); i < n; ++i) {
        auto from = values[i];
        FromBits from_bits = bit_cast<FromBits>(abs(from));
        {
            from_bits = RoundMantissa(from_bits, -kDigitShift, ps);
            from_bits &= ~((FromBits{1} << (-kDigitShift)) - 1);
        }
        results[i] = bit_cast<From>(from_bits);
    }
    
}
#endif


// HOST_DEVICE (not DEVICE-only): the FloatingPointParams virtual_round calls
// isnan, and it must work on the host side of a CUDA (nvcc) translation unit too
// — otherwise nvcc compiles a host stub that calls exit() at runtime.
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isnan(float value) {
    return ::isnan(value);
}
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isnan(double value) {
    return ::isnan(value);
}
// Native-float isinf overloads paralleling isnan above: lets the conversion
// code call unqualified isinf()/isnan() and resolve to a device-safe overload
// for every From type (native float/double here, Templated_Float at isinf/isnan
// above) instead of std::isinf, which has no overload for the custom types.
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isinf(float value) {
    return ::isinf(value);
}
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isinf(double value) {
    return ::isinf(value);
}




template<typename From_p, typename ToInf, typename ToNaN>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE From_p virtual_round(From_p& value, FloatingPointParams<ToInf, ToNaN> ToFp, ProjSpec ps = ProjSpec{}) {
    [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
    [[maybe_unused]] const int stoch_len = ps.stoch_length;
    //need this for const correctness
    using From = std::remove_cv_t<From_p>;
    using FromBits = typename Traits<From>::BitsType;
    using SignedFromBits = std::make_signed_t<FromBits>;
    static constexpr int kFromBits = Traits<From>::kBits;
    int kDigitShift = ToFp.mantissa_bits - Traits<From>::kMantissaBits;
    static constexpr int kExponentBias = Traits<From>::kExponentBias;
    static constexpr int kFromMantissaBits = Traits<From>::kMantissaBits;
    FromBits sign_bit = bit_cast<FromBits>(value) & (FromBits{1} << (kFromBits - 1));

    const int ToMax_exp = ToFp.is_signed == Signedness::Signed
                                    ? (1 << (ToFp.bitwidth - ToFp.mantissa_bits - 1)) - 1 - ToFp.bias
                                    : (1 << (ToFp.bitwidth - ToFp.mantissa_bits)) - 1 - ToFp.bias;
    const int ToMin_exp = 1 - ToFp.bias;
    const float To_min_val = std::pow(2.0, ToMin_exp)*std::pow(2.0, -ToFp.mantissa_bits);

    // lof_abs (not abs): device-correct magnitude for native float/double; see note.
    FromBits from_bits = bit_cast<FromBits>(lof_abs(value));
    int from_exp = (from_bits >> kFromMantissaBits) - kExponentBias;



    if (isnan(value)) return std::numeric_limits<From>::quiet_NaN();

    const float abs_val = get_float(lof_abs(value));
    if (abs_val < To_min_val) {
        // Below the smallest representable magnitude (denorm_min): the only
        // representable outcomes are 0 and +/-denorm_min, chosen per the
        // rounding mode (a blanket flush to 0 would round directed modes the
        // wrong way -- e.g. RoundAwayFromZero of a tiny value must reach
        // denorm_min, not 0). The mantissa path can't run here: -kDigitShift
        // would exceed the source bit width. Mirrors the typed run() switch.
        const bool pos = (sign_bit == 0);
        bool to_min;                                    // true => +/-denorm_min
        switch (ps.rounding_mode) {
            case Rounding_Mode::RoundTowardsZero:  to_min = false; break;
            // RoundToOdd (sticky): truncating to the grid gives 0 (even); a nonzero
            // tail sets the LSB -> denorm_min (odd). So any nonzero tiny value reaches
            // denorm_min, exactly like away-from-zero in this regime.
            case Rounding_Mode::RoundToOdd:
            case Rounding_Mode::RoundAwayFromZero: to_min = (abs_val > 0.0f); break;
            case Rounding_Mode::RoundUp:           to_min =  pos && (abs_val > 0.0f); break;
            case Rounding_Mode::RoundDown:         to_min = !pos && (abs_val > 0.0f); break;
            case Rounding_Mode::RoundToNearestEven:
            case Rounding_Mode::RoundToNearestOdd:
            case Rounding_Mode::RoundTiesTowardsZero:
            case Rounding_Mode::RoundTiesToAway: {
                const float half = To_min_val * 0.5f;
                if      (abs_val < half) to_min = false;
                else if (abs_val > half) to_min = true;
                // tie: RNE->0 (even); RoundTiesTowardsZero->0; others (RNO/TiesToAway)->denorm_min
                else to_min = (ps.rounding_mode == Rounding_Mode::RoundToNearestOdd
                            || ps.rounding_mode == Rounding_Mode::RoundTiesToAway);
                break;
            }
            default: to_min = false; break;             // stochastic etc.: flush to 0
        }
        const FromBits mag_bits = to_min
            ? bit_cast<FromBits>(from_double<From>((double)To_min_val))
            : FromBits{0};
        return bit_cast<From>(static_cast<FromBits>(mag_bits | sign_bit));
    }
    // Subnormal target range [denorm_min, min_normal): the grid is uniform with
    // spacing denorm_min, so round in the TARGET domain. Make the source's
    // implicit leading 1 explicit, then RoundMantissa keeps the TARGET-mantissa
    // LSB. Rounding the raw source bits instead keys RNE/RNO tie-to-even/odd off
    // the SOURCE exponent LSB, which picks the wrong neighbour when the tie
    // straddles a power of two (e.g. 1.5*denorm_min). Mirrors the typed run().
    if (from_exp < ToMin_exp) {
        const int shift = -kDigitShift - (from_exp - ToMin_exp);   // bits to drop to reach the grid
        FromBits sig = (from_bits & ((FromBits{1} << kFromMantissaBits) - 1))
                       | (FromBits{1} << kFromMantissaBits);        // explicit leading 1
        sig = RoundMantissa(sig, shift, ps, sign_bit == 0);
        const FromBits k = static_cast<FromBits>(sig >> shift);     // integer multiple of denorm_min
        const FromBits mag_bits = bit_cast<FromBits>(from_double<From>((double)k * (double)To_min_val));
        return bit_cast<From>(static_cast<FromBits>(mag_bits | sign_bit));
    }

    if (kDigitShift < 0)
    {
        // pos (sign) is required for the directional modes RoundUp/RoundDown,
        // which operate on the magnitude `from_bits`; without it a positive
        // value is treated as negative and rounds the wrong way (matches the
        // int-ToMantissaBits overload above, ~line 2503).
        from_bits = RoundMantissa(from_bits, -kDigitShift, ps, sign_bit == 0);
        from_bits &= ~((FromBits{1} << (-kDigitShift)) - 1);
    }

    from_exp = (from_bits >> kFromMantissaBits) - kExponentBias;
    if (from_exp > ToMax_exp) return ToFp.OV_behavior == Inf_Behaviors::Saturating ? value : std::numeric_limits<From>::infinity();

    return bit_cast<From>(static_cast<FromBits>(from_bits | sign_bit));
}

#ifndef USE_CUDA
template<typename From, typename ToInf, typename ToNaN,  class arch = xs::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE void virtual_round(From* values, From* results, int n, FloatingPointParams<ToInf, ToNaN> ToFp, ProjSpec ps = ProjSpec{}) {
   [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
   [[maybe_unused]] const int stoch_len = ps.stoch_length;
   using FromBits = typename Traits<From>::BitsType;
using FromBitsSIMD = xs::batch<FromBits, arch>;
using FromSIMD = xs::batch<From, arch>;
using IntSIMD = std::conditional_t<std::is_same_v<From, float>, xs::batch<int, arch>, xs::batch<int64_t, arch>>;
using BoolBitsSIMD = xs::batch_bool<FromBits, arch>;
using BoolFromSIMD = xs::batch_bool<From, arch>;

static constexpr int kFromBits = Traits<From>::kBits;
int kDigitShift = ToFp.mantissa_bits - Traits<From>::kMantissaBits;
static constexpr int kFromMantissaBits = Traits<From>::kMantissaBits;
static constexpr int kExponentBias = Traits<From>::kExponentBias;

static const int ToMax_exp = (ToFp.is_signed == Signedness::Signed
                                   ? (1 << (ToFp.bitwidth - ToFp.mantissa_bits - 1)) - 1 - ToFp.bias
                                   : (1 << (ToFp.bitwidth - ToFp.mantissa_bits)) - 1 - ToFp.bias) + kExponentBias;
static const int ToMin_exp = 1 - ToFp.bias + kExponentBias;
static constexpr std::size_t step = FromBitsSIMD::size;
static constexpr std::size_t unroll = 4;
static constexpr std::size_t block_size = step * unroll;

#ifdef _LOFOPENMP
#pragma omp parallel for
#endif
for (int i = 0; i < n - (n % block_size); i += block_size) {
    FromSIMD result[unroll];
    
    FromBitsSIMD mask = ~((FromBitsSIMD(FromBits{1}) << (-kDigitShift)) - FromBitsSIMD(FromBits{1}));
    
    for (std::size_t u = 0; u < unroll; ++u) {
        FromSIMD from_vals = FromSIMD::load_unaligned(&values[i + u * step]);
        FromSIMD abs_from = xs::abs(from_vals);
        FromBitsSIMD from_bits = xs::bit_cast<FromBitsSIMD>(abs_from);
        
        FromBitsSIMD sign_bit;
        if constexpr (get_signedness_v<From> == Signedness::Signed) {
            sign_bit = from_bits >> (kFromBits - 1);
        } else {
            sign_bit = FromBitsSIMD(FromBits{0});
        }
        
        FromBitsSIMD from_exp = from_bits >> kFromMantissaBits;
        BoolBitsSIMD underflow = from_exp < FromBitsSIMD(FromBits(ToMin_exp));
        BoolFromSIMD is_nan = xs::isnan(from_vals);
        BoolBitsSIMD early_exit = underflow || xs::batch_bool_cast<FromBits>(is_nan);
        
        result[u] = FromSIMD(From{0});
        result[u] = xs::select(is_nan, FromSIMD(std::numeric_limits<From>::quiet_NaN()), result[u]);
        
        FromBitsSIMD processed_bits = RoundMantissa(from_bits, FromBitsSIMD(-kDigitShift), ps);
        processed_bits = processed_bits & mask;
        
        FromBitsSIMD new_exp = processed_bits >> kFromMantissaBits;
        BoolBitsSIMD overflow = new_exp > FromBitsSIMD(FromBits(ToMax_exp));
        
        FromBitsSIMD final_bits = processed_bits | (sign_bit << (kFromBits - 1));
        FromSIMD normal_result = xs::bit_cast<FromSIMD>(final_bits);
        
        FromSIMD overflow_result;
        if (ToFp.OV_behavior == Inf_Behaviors::Saturating) {
            overflow_result = normal_result;
        } else {
            overflow_result = FromSIMD(std::numeric_limits<From>::infinity());
        }
        
        FromSIMD processed_result = xs::select(xs::batch_bool_cast<From>(overflow), overflow_result, normal_result);
        result[u] = xs::select(xs::batch_bool_cast<From>(early_exit), result[u], processed_result);
    }
    
    for (std::size_t u = 0; u < unroll; ++u) {
        result[u].store_unaligned(&results[i + u * step]);
    }
}
}
#endif

        template <typename From, typename To>
        struct ConvertImpl<From, To,
                           std::enable_if_t<!std::is_same_v<From, To>>>
        {
            using FromTraits = Traits<From>;
            using FromBits = typename FromTraits::BitsType;
            using SignedFromBits = std::make_signed_t<FromBits>;
            static constexpr int kFromBits = FromTraits::kBits;
            static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
            static constexpr int kFromExponentBits = FromTraits::kExponentBits;
            static constexpr int kFromExponentBias = FromTraits::kExponentBias;
            static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;
            // none are void

            using ToTraits = Traits<To>;
            using ToBits = typename ToTraits::BitsType;
            static constexpr int kToBits = ToTraits::kBits;
            static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
            static constexpr int kToExponentBits = ToTraits::kExponentBits;
            static constexpr int kToExponentBias = ToTraits::kExponentBias;
            static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;
            // none are void
            // `WideBits` is wide enough to accommodate the largest exponent and mantissa
            // in either `From` or `To`.
            static constexpr int kWideBits =
                (std::max(kToMantissaBits, kFromMantissaBits)) + // Max significand.
                (std::max(kToExponentBits, kFromExponentBits));  // Max exponent.
            static constexpr int kWideBytes = (kWideBits + (CHAR_BIT - 1)) / CHAR_BIT;
            using WideBits = GetUnsignedInteger<kWideBytes>;
            // static_assert(std::is_same_v<WideBits, unsigned long long>, "WideBits<8> must be uint64_t");
            // using WideBits = u_int32_t;
            static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
            static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;

            // set exception flags for overflow and underflow  here
            // current implementation cant deal with round ups near +zero and round downs near -0
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE To run(const From &from, ProjSpec ps = ProjSpec{})
            {
                const Rounding_Mode round_mode = ps.rounding_mode;
                const int stoch_len = ps.stoch_length;
                const Saturation_Mode sat_mode = ps.saturation_mode;
                // Shift bits to destination type, without sign bit.

                const bool from_sign_bit = (get_signedness_v<From> == Signedness::Unsigned) ? false : bit_cast<FromBits>(from) >> (kFromBits - 1);

                if (get_signedness_v<To> == Signedness::Unsigned && from_sign_bit)
                {
                    // set underflow flag

                    // a negative value collapsing to 0 (or NaN) in an unsigned format
                    // is treated as underflow here.
#ifdef ENABLE_EXCEPT
                    signal_if_754<get_NaN_Behavior_v<To>>(LF_exception_flags::Underflow);
#endif
                    if (get_unsigned_behavior_v<To> == Unsigned_behavior::NegtoZero)
                    {
                        return To{};
                    }
                    else
                    {
                        if (get_NaN_Behavior_v<To> == NaN_Behaviors::_754)
                        {
                            return std::numeric_limits<To>::signaling_NaN();
                        }
                        else if (get_NaN_Behavior_v<To> == NaN_Behaviors::_3109)
                        {
                            return std::numeric_limits<To>::quiet_NaN();
                        }
                        else
                        {
// trapping NaN - call trap
#ifdef ENABLE_EXCEPT
                            signal_if_754<get_NaN_Behavior_v<To>>(LF_exception_flags::InvalidOperation);
#endif
                            return std::numeric_limits<To>::signaling_NaN();
                        }
                    }
                }

                const FromBits from_bits =
                    bit_cast<FromBits>(lof_abs(from));

                // Special values, preserving sign.

                // Infinite input. The Saturation_Mode (ps.saturation_mode) decides whether the
                // infinity is preserved or clamped to the finite range:
                //   OvfInf      -> propagate +/-inf when the target has infinity, else clamp to max.
                //   SatPropagate-> propagate +/-inf when the target has infinity, else clamp to max.
                //   SatFinite   -> always clamp to the finite max (no value is ever infinite).
                if (isinf(from))
                {
                    const bool to_has_inf = (get_overflow_behavior_v<To> != Inf_Behaviors::Saturating);
                    if (sat_mode != Saturation_Mode::SatFinite && to_has_inf)
                    {
                        return from_sign_bit ? -std::numeric_limits<To>::infinity()
                                             : std::numeric_limits<To>::infinity();
                    }
                    return from_sign_bit ? -std::numeric_limits<To>::max()
                                         : std::numeric_limits<To>::max();
                }
                if (isnan(from))
                {
                    return std::numeric_limits<To>::quiet_NaN();
                }
                if (from_bits == 0)
                {
                    return from_sign_bit ? -To{} : To{};
                }

                const int biased_from_exponent = from_bits >> kFromMantissaBits; // check if number is subnormal

                // `To` supports more exponents near zero which means that some subnormal
                // values in `From` may become normal.
                if constexpr (std::numeric_limits<To>::min_exponent <
                              std::numeric_limits<From>::min_exponent)
                {
                    if (biased_from_exponent == 0)
                    {
                        // Subnormals.
                        WideBits bits = from_bits;

                        // normalization factor can be represnted safely in the same format as FromBits. Say FromBits is uint8, then leading zeros is at most 8, kFromBits - kMantissaBits is also at most 8, so we are bounded by 17.
                        const int normalization_factor =
                            countl_zero<kFromBits, FromBits, int>(from_bits) - (kFromBits - kFromMantissaBits) + 1;
                        const int biased_exponent = kExponentOffset - normalization_factor + 1;
                        if (biased_exponent <= 0)
                        {

                            // Result is subnormal.  Adjust the subnormal bits to account for
                            // the difference in exponent bias.
                            if constexpr (kExponentOffset < (kWideBits))
                            {
                                bits <<= kExponentOffset;
                            }
                        }
                        else
                        {
                            // Result is normal. Shift the mantissa to account for the number of
                            // leading zero digits, and clear the hidden bit.
                            bits <<= normalization_factor;
                            bits &= ~(WideBits{1} << kFromMantissaBits);
                            // Insert the exponent bits.
                            bits |= static_cast<WideBits>(biased_exponent) << kFromMantissaBits;
                        }

                        // Truncate/round mantissa if necessary.
                        if constexpr (kDigitShift > 0)
                        {

                            bits <<= kDigitShift;
                        }
                        else
                        {
                            bits = RoundMantissa(bits, -kDigitShift, ps);
                            bits >>= -kDigitShift;
                        }

                        To to = bit_cast<To>(static_cast<ToBits>(bits));

                        return from_sign_bit ? -to : to;
                    }
                }
                // `To` supports fewer exponents near zero which means that some values in
                // `From` may become subnormal.

                // Enter this case for round from double to fp8
                if constexpr (std::numeric_limits<To>::min_exponent >
                              std::numeric_limits<From>::min_exponent)
                {
                    const int unbiased_exponent = biased_from_exponent - kFromExponentBias;

                    const int biased_to_exponent = unbiased_exponent + kToExponentBias;

                    // Subnormals and zero.
                    if (biased_to_exponent <= 0)
                    {
                        // Round and shift mantissa down.
                        // unbiased exp = -8
                        // biased_to_exp = -8 + 8 = 0
                        // kdigitShft = 3 - 52 = -49

                        const FromBits from_has_leading_one = (biased_from_exponent > 0) ? 1 : 0;
                        int exponent_shift =
                            -kDigitShift - biased_to_exponent + from_has_leading_one;
                        FromBits rounded_from_bits =
                            (from_bits & FromTraits::kMantissaMask) |
                            (from_has_leading_one << (kFromMantissaBits));
                        ToBits bits = 0;

                        if (exponent_shift <= kFromMantissaBits + 1)
                        {
                            // NOTE: we need to round again from the original from_bits,
                            // otherwise the lower precision bits may already be lost.  There is
                            // an edge-case where rounding to a normalized value would normally
                            // round down, but for a subnormal, we need to round up.
                            rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift, ps);

                            bits = (rounded_from_bits >> exponent_shift);
                        }
                        else
                        {

                            unsigned long long widened_bits = (unsigned long long)(rounded_from_bits);
                            switch (round_mode)
                            {
                                // RoundToOdd (sticky): a nonzero significand below the
                                // rounding point sets the LSB -> denorm_min, same as
                                // away-from-zero in this deep-underflow regime.
                                case Rounding_Mode::RoundToOdd:
                                case Rounding_Mode::RoundAwayFromZero:
                                    rounded_from_bits = (rounded_from_bits > 0 ? 1 : 0);
                                    bits = rounded_from_bits;
                                    break;

                                case Rounding_Mode::RoundTowardsZero:
                                    bits = 0;
                                    break;

                                case Rounding_Mode::RoundUp:
                                    rounded_from_bits = (rounded_from_bits && !from_sign_bit > 0 ? 1 : 0);
                                    bits = rounded_from_bits;
                                    break;

                                case Rounding_Mode::RoundDown:
                                    rounded_from_bits = ((rounded_from_bits) && from_sign_bit > 0 ? 1 : 0);
                                    bits = rounded_from_bits;
                                    break;

                                case Rounding_Mode::StochasticRoundingA:
                                    widened_bits = Stochastic_Round_A(widened_bits, exponent_shift, stoch_len);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                case Rounding_Mode::StochasticRoundingB:
                                    widened_bits = Stochastic_Round_B(widened_bits, exponent_shift, stoch_len);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                case Rounding_Mode::StochasticRoundingC:
                                    widened_bits = Stochastic_Round_C(widened_bits, exponent_shift, stoch_len);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                case Rounding_Mode::StochasticRoundingD:
                                    widened_bits = Stochastic_Round_D(widened_bits, exponent_shift, stoch_len);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                case Rounding_Mode::True_StochasticRounding:
                                    widened_bits = True_Stochastic_Round(widened_bits, exponent_shift);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                case Rounding_Mode::ProbabilisticRounding:
                                    widened_bits = Probabilistic_Round(widened_bits, exponent_shift);
                                    bits = (widened_bits >> exponent_shift);
                                    break;

                                default:
                                    // Round to nearest even
                                    widened_bits = 0;
                                    bits = (widened_bits >> exponent_shift);
                                    break;
                            }

                        }
#ifdef ENABLE_EXCEPT
                        // A non-zero input landed in the target's subnormal range: a tiny
                        // non-zero result was detected -> underflow (§7.5).
                        if (bits != 0)
                            signal_if_754<get_NaN_Behavior_v<To>>(LF_exception_flags::Underflow);
#endif
                        // Insert sign and return.
                        return from_sign_bit ? -bit_cast<To>(bits) : bit_cast<To>(bits);
                    }
                }

                // Round the mantissa if it is shrinking.
                WideBits rounded_from_bits = from_bits;

                if constexpr (kDigitShift < 0)
                {
                    // need some logic to add leading 1 if normalized
                    rounded_from_bits = RoundMantissa(rounded_from_bits, -kDigitShift, ps);
                    // Zero-out tail bits.
                    rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
                }

                rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                                     << kFromMantissaBits;

                ToBits bits;
                // Check for overflows by aligning the significands. We always align the
                // narrower significand to the wider significand.
                const WideBits kToHighestRep =
                    bit_cast<ToBits>(std::numeric_limits<To>::max());
                WideBits aligned_highest{kToHighestRep};
                if constexpr (kDigitShift < 0)
                {
                    aligned_highest <<= -kDigitShift;
                    bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
                }
                else if constexpr (kDigitShift >= 0)
                {
                    rounded_from_bits <<= kDigitShift;
                    bits = ToBits{static_cast<ToBits>(rounded_from_bits)};
                }

                To to = bit_cast<To>(bits);
                if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                             std::numeric_limits<To>::digits) <
                              std::make_pair(std::numeric_limits<From>::max_exponent,
                                             std::numeric_limits<From>::digits))
                {
                    if (rounded_from_bits > aligned_highest)
                    {

                    #ifdef ENABLE_EXCEPT
                            signal_if_754<get_NaN_Behavior_v<To>>(LF_exception_flags::Overflow);
                    #endif
                        // A FINITE input overflowed the target's finite range.
                        //   OvfInf      -> +/-inf if the target has infinity (IEEE 754), else max.
                        //   SatFinite   -> always the finite max.
                        //   SatPropagate-> finite values clamp to max (only true infinities, handled
                        //                  in the inf-input branch above, are propagated as inf).
                        const bool emit_inf = (sat_mode == Saturation_Mode::OvfInf) &&
                                              std::numeric_limits<To>::has_infinity;
                        if (emit_inf)
                        {
                            to = from_sign_bit ? -std::numeric_limits<To>::infinity()
                                               : std::numeric_limits<To>::infinity();
                        }
                        else
                        {
                            to = from_sign_bit ? -std::numeric_limits<To>::max()
                                               : std::numeric_limits<To>::max();
                            //: static_cast<To>(1.0);
                        }
                        return to;
                    }
                }

                return from_sign_bit ? -to : to;
            }

            // SIMD round for fast - round array of length n
 // SIMD round for fast - round array of length n
// Branch A helper function
#ifndef USE_CUDA
template <typename WideBitsSIMD, typename SignedWideBitsSIMD, 
          typename WideBits, typename SignedWideBits, typename arch>
__attribute__((noinline))
static WideBitsSIMD handle_expanding_conversion(
    WideBitsSIMD from_bits,
    ProjSpec ps)
{
    [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
    [[maybe_unused]] const int stoch_len = ps.stoch_length;
    auto input_exp = xs::batch_cast<SignedWideBits>(from_bits >> kFromMantissaBits);
    auto is_zero = (input_exp == SignedWideBitsSIMD(0));
    
    auto normalization_factor = 
        countl_zero<kFromBits, WideBitsSIMD, WideBitsSIMD>(from_bits) -
        (WideBitsSIMD(kFromBits - kFromMantissaBits) + WideBitsSIMD(1));
    
    auto biased_exponent = SignedWideBitsSIMD(kExponentOffset) - [&]() {
        SignedWideBitsSIMD result;
        result.data = reinterpret_cast<decltype(result.data)>(normalization_factor.data);
        return result;
    }() + SignedWideBitsSIMD(1);
    
    auto is_lezero_exp = (biased_exponent <= SignedWideBitsSIMD(0));
    
    auto bits_path1 = from_bits;
    if constexpr (kExponentOffset < sizeof(WideBits) * 8) {
        bits_path1 = bits_path1 << xs::batch_cast<WideBits>(kExponentOffset);
    }
    
    auto bits_path2 = (from_bits << normalization_factor) & 
                     ~(WideBitsSIMD(WideBits{1}) << kFromMantissaBits) |
                     (xs::batch_cast<WideBits>(biased_exponent) << WideBits(kFromMantissaBits));
    
    auto result = xs::select(
        xs::batch_bool<WideBits, arch>(is_lezero_exp) & 
        xs::batch_bool<WideBits, arch>(is_zero), 
        bits_path1, 
        bits_path2);
    
    if constexpr (kDigitShift > 0) {
        result = result << WideBitsSIMD(kDigitShift);
    } else if constexpr (kDigitShift < 0) {
        result = RoundMantissa(result, WideBitsSIMD(-kDigitShift), ps);
        result = result >> WideBitsSIMD(-kDigitShift);
    }
    
    return xs::select(xs::batch_bool<WideBits, arch>(is_zero), 
                     WideBitsSIMD(0), 
                     result);
}
#endif
// Branch B helper function
/* register break down-
x0 stores from_bits
v0 stores input_exp_local
v31 store sthe mask


*/

#ifndef USE_CUDA

template <typename FromTraits, typename WideBitsSIMD, 
          typename SignedWideBitsSIMD, typename WideBits, typename SignedWideBits, typename arch>
LOFLOAT_FORCEINLINE
static WideBitsSIMD handle_shrinking_conversion(
    WideBitsSIMD from_bits,
    ProjSpec ps)
{
    [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
    [[maybe_unused]] const int stoch_len = ps.stoch_length;
    // Start normal path early for ILP - no dependency on subnormal checks
    WideBitsSIMD normal_result;
    if constexpr (kDigitShift < 0) {
        auto mod_digitshift = WideBitsSIMD(-kDigitShift);
        normal_result = RoundMantissa(from_bits, mod_digitshift, ps);
        normal_result = normal_result & ~((WideBits{1} << mod_digitshift) - 1);
        normal_result = (normal_result +
            WideBitsSIMD((static_cast<WideBits>(kExponentOffset) << kFromMantissaBits)))
            >> mod_digitshift;
    } else {
        normal_result = (from_bits +
            WideBitsSIMD((static_cast<WideBits>(kExponentOffset) << kFromMantissaBits)))
            << WideBitsSIMD(kDigitShift);
    }

    auto input_exp = (from_bits >> kFromMantissaBits);
    SignedWideBitsSIMD input_exp_signed;
    input_exp_signed.data = reinterpret_cast<decltype(input_exp_signed.data)>(input_exp.data);

    auto biased_to_exp = input_exp_signed - SignedWideBitsSIMD(kFromExponentBias) +
        SignedWideBitsSIMD(kToExponentBias);
    auto is_subnormal = (biased_to_exp <= SignedWideBitsSIMD(0));

    // Fast path: skip subnormal handling entirely when no lanes are subnormal
    if (!xs::any(is_subnormal)) {
        return normal_result;
    }

    auto is_zero_signed = (input_exp_signed == SignedWideBitsSIMD(0));
    auto threshold = SignedWideBitsSIMD(kFromMantissaBits + 1);
    auto s_exponent_shift = SignedWideBitsSIMD(-kDigitShift) - biased_to_exp +
        xs::select(is_zero_signed, SignedWideBitsSIMD(0), SignedWideBitsSIMD(1));
    auto needs_shift = is_subnormal && (s_exponent_shift <= threshold);

    WideBitsSIMD exponent_shift_unsigned;
    exponent_shift_unsigned.data = reinterpret_cast<decltype(exponent_shift_unsigned.data)>(s_exponent_shift.data);

    auto is_zero_unsigned = xs::batch_bool<WideBits, arch>(is_zero_signed);
    auto leading_one = xs::select(is_zero_unsigned,
        WideBitsSIMD(0),
        WideBitsSIMD(1) << WideBits(kFromMantissaBits));
    auto mantissa = (from_bits & WideBitsSIMD(static_cast<WideBits>(FromTraits::kMantissaMask))) | leading_one;
    mantissa = RoundMantissa(mantissa, exponent_shift_unsigned, ps);

    auto needs_shift_unsigned = xs::batch_bool<WideBits, arch>(needs_shift);
    auto subnormal_result = xs::select(needs_shift_unsigned,
                                       mantissa >> exponent_shift_unsigned,
                                       WideBitsSIMD(0));

    auto is_subnormal_unsigned = xs::batch_bool<WideBits, arch>(is_subnormal);
    return xs::select(is_subnormal_unsigned, subnormal_result, normal_result);
}       

#endif

// Main run function
#ifndef USE_CUDA
template <class arch = xs::default_arch>
static LOFLOAT_HOST void run(const From* from,
                                                To* to,
                                                int n,
                                                ProjSpec ps = ProjSpec{})
{
                [[maybe_unused]] const Rounding_Mode round_mode = ps.rounding_mode;
                [[maybe_unused]] const int stoch_len = ps.stoch_length;
                using FromBitsSIMD = xs::batch<FromBits, arch>;
                using ToBitsSIMD   = xs::batch<ToBits,   arch>;
                using WideBitsSIMD = xs::batch<WideBits, arch>;
                using SignedWideBits = std::make_signed_t<WideBits>;
                using SignedWideBitsSIMD = xs::batch<std::make_signed_t<WideBits>, arch>;
                using IntSIMD      = SignedWideBitsSIMD;
                using SignedFromBitsSIMD = xs::batch<SignedFromBits, arch>;
                constexpr bool is_signed_type = (get_signedness_v<From> == Signedness::Signed);
                
                constexpr int from_lanes = FromBitsSIMD::size;
                constexpr int to_lanes   = ToBitsSIMD::size;
                constexpr int step       = (from_lanes < to_lanes) ? from_lanes : to_lanes;
                static constexpr WideBits kFromSignBit = (WideBits(1) << (kFromBits - 1));
    static constexpr WideBits kToSignBit = (WideBits(1) << (kToBits - 1));
  
    const FromBits* from_bits_ptr = reinterpret_cast<const FromBits*>(from);
    ToBits* to_bits_ptr           = reinterpret_cast<ToBits*>(to);

    auto load_from_widened = [&](int i) -> WideBitsSIMD {
        if constexpr (sizeof(FromBits) == sizeof(WideBits)) {
            return xs::bit_cast<WideBitsSIMD>(
                FromBitsSIMD::load_aligned(&from_bits_ptr[i])
            );
        } else {
            WideBits tmp[step];
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((tmp[Is] = static_cast<WideBits>(from_bits_ptr[i + Is])), ...);
            }(std::make_index_sequence<step>{});
            return WideBitsSIMD::load_aligned(tmp);
        }
    };

    auto store_to_full = [&](int i, const WideBitsSIMD& v) {
        if constexpr (sizeof(ToBits) == sizeof(WideBits)) {
            xs::bit_cast<ToBitsSIMD>(v).store_aligned(&to_bits_ptr[i]);
        } else {
            alignas(64) WideBits tmp[step];
            v.store_aligned(tmp);
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((to_bits_ptr[i + Is] = static_cast<ToBits>(tmp[Is])), ...);
            }(std::make_index_sequence<step>{});
        }
    };

    // Process loop body for a single iteration
    auto process_iteration = [&](int i) -> WideBitsSIMD {
   WideBitsSIMD from_bits_wide = load_from_widened(i);
    
    xs::batch_bool<WideBits, arch> signed_mask_wide =
        is_signed_type
            ? ((from_bits_wide & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
            : xs::batch_bool<WideBits, arch>(false);
    
    WideBitsSIMD from_bits = from_bits_wide & ~WideBitsSIMD(kFromSignBit);
    
        WideBitsSIMD finite_out;

    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent)
    {
       finite_out = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                     WideBits, SignedWideBits, arch>(
                from_bits, ps);
    }
        else if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent)
    {
         finite_out = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                     SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                from_bits, ps);
    }
    else 
    {
        finite_out = from_bits;
    }
    
        finite_out = xs::select(
            signed_mask_wide,
            finite_out | WideBitsSIMD(kToSignBit),
            finite_out & ~WideBitsSIMD(kToSignBit)
        );
        
        return finite_out;
    };

    int i = 0;
    
    #ifdef _LOFOPENMP
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int idx = 0; idx <= n - step*2; idx += step*2)
        {
            // Increased prefetch distance
            constexpr int kPrefetchDistance = 2 * step;
            if (idx + kPrefetchDistance < n) {
                __builtin_prefetch(&from_bits_ptr[idx + kPrefetchDistance], 0, 3);
                __builtin_prefetch(&to_bits_ptr[idx + kPrefetchDistance], 1, 3);
            }
            
            // Unroll 4x manually - load all first to start memory fetches early
            WideBitsSIMD from_bits_wide_0 = load_from_widened(idx);
            WideBitsSIMD from_bits_wide_1 = load_from_widened(idx + step);

            
            // Process iteration 0
            xs::batch_bool<WideBits, arch> signed_mask_wide_0 =
                is_signed_type
                    ? ((from_bits_wide_0 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                    : xs::batch_bool<WideBits, arch>(false);
            WideBitsSIMD from_bits_0 = from_bits_wide_0 & ~WideBitsSIMD(kFromSignBit);
            
            // Process iteration 1
            xs::batch_bool<WideBits, arch> signed_mask_wide_1 =
                is_signed_type
                    ? ((from_bits_wide_1 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                    : xs::batch_bool<WideBits, arch>(false);
            WideBitsSIMD from_bits_1 = from_bits_wide_1 & ~WideBitsSIMD(kFromSignBit);
  
            // Handle conversions for all 4 iterations
            WideBitsSIMD finite_out_0, finite_out_1;
            
            if constexpr (std::numeric_limits<To>::min_exponent <
                          std::numeric_limits<From>::min_exponent)
            {
                finite_out_0 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                         WideBits, SignedWideBits, arch>(
                    from_bits_0, ps);
                finite_out_1 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                         WideBits, SignedWideBits, arch>(
                    from_bits_1, ps);

            }
            else if constexpr (std::numeric_limits<To>::min_exponent >
                               std::numeric_limits<From>::min_exponent)
            {
                finite_out_0 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                         SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                    from_bits_0, ps);
                finite_out_1 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                         SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                    from_bits_1, ps);
            }
            else
            {
                finite_out_0 = from_bits_0;
                finite_out_1 = from_bits_1;
            }
            
            // Apply sign bits
            finite_out_0 = xs::select(
                signed_mask_wide_0,
                finite_out_0 | WideBitsSIMD(kToSignBit),
                finite_out_0 & ~WideBitsSIMD(kToSignBit)
            );
            finite_out_1 = xs::select(
                signed_mask_wide_1,
                finite_out_1 | WideBitsSIMD(kToSignBit),
                finite_out_1 & ~WideBitsSIMD(kToSignBit)
            );

            // Store all results
            store_to_full(idx, finite_out_0);
            store_to_full(idx + step, finite_out_1);
        }
        
        // Remaining iterations (0-3 iterations)
        #pragma omp for schedule(static)
        for (int idx = (n / (step*2)) * (step*2); idx <= n - step; idx += step)
        {
            WideBitsSIMD result = process_iteration(idx);
            store_to_full(idx, result);
        }
    }
    #else
    // Single-threaded version with unrolling
    for (i = 0; i <= n - step*4; i += step*4)
    {
        constexpr int kPrefetchDistance = 64 * step;
        if (i + kPrefetchDistance < n) {
            __builtin_prefetch(&from_bits_ptr[i + kPrefetchDistance], 0, 3);
            __builtin_prefetch(&to_bits_ptr[i + kPrefetchDistance], 1, 3);
        }
        
        // Load all 4 iterations early
        WideBitsSIMD from_bits_wide_0 = load_from_widened(i);
        WideBitsSIMD from_bits_wide_1 = load_from_widened(i + step);
        WideBitsSIMD from_bits_wide_2 = load_from_widened(i + step*2);
        WideBitsSIMD from_bits_wide_3 = load_from_widened(i + step*3);
        
        // Process all 4 (same as OpenMP version above)
        xs::batch_bool<WideBits, arch> signed_mask_wide_0 =
            is_signed_type
                ? ((from_bits_wide_0 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                : xs::batch_bool<WideBits, arch>(false);
        WideBitsSIMD from_bits_0 = from_bits_wide_0 & ~WideBitsSIMD(kFromSignBit);
        
        xs::batch_bool<WideBits, arch> signed_mask_wide_1 =
            is_signed_type
                ? ((from_bits_wide_1 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                : xs::batch_bool<WideBits, arch>(false);
        WideBitsSIMD from_bits_1 = from_bits_wide_1 & ~WideBitsSIMD(kFromSignBit);
        
        xs::batch_bool<WideBits, arch> signed_mask_wide_2 =
            is_signed_type
                ? ((from_bits_wide_2 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                : xs::batch_bool<WideBits, arch>(false);
        WideBitsSIMD from_bits_2 = from_bits_wide_2 & ~WideBitsSIMD(kFromSignBit);
        
        xs::batch_bool<WideBits, arch> signed_mask_wide_3 =
            is_signed_type
                ? ((from_bits_wide_3 & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
                : xs::batch_bool<WideBits, arch>(false);
        WideBitsSIMD from_bits_3 = from_bits_wide_3 & ~WideBitsSIMD(kFromSignBit);
        
        WideBitsSIMD finite_out_0, finite_out_1, finite_out_2, finite_out_3;
        
        if constexpr (std::numeric_limits<To>::min_exponent <
                      std::numeric_limits<From>::min_exponent)
        {
            finite_out_0 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                     WideBits, SignedWideBits, arch>(
                from_bits_0, ps);
            finite_out_1 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                     WideBits, SignedWideBits, arch>(
                from_bits_1, ps);
            finite_out_2 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                     WideBits, SignedWideBits, arch>(
                from_bits_2, ps);
            finite_out_3 = handle_expanding_conversion<WideBitsSIMD, SignedWideBitsSIMD, 
                                                     WideBits, SignedWideBits, arch>(
                from_bits_3, ps);
        }
        else if constexpr (std::numeric_limits<To>::min_exponent >
                           std::numeric_limits<From>::min_exponent)
        {
            finite_out_0 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                     SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                from_bits_0, ps);
            finite_out_1 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                     SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                from_bits_1, ps);
            finite_out_2 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                     SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                from_bits_2, ps);
            finite_out_3 = handle_shrinking_conversion<FromTraits, WideBitsSIMD, 
                                                     SignedWideBitsSIMD, WideBits, SignedWideBits, arch>(
                from_bits_3, ps);
        }
        else
        {
            finite_out_0 = from_bits_0;
            finite_out_1 = from_bits_1;
            finite_out_2 = from_bits_2;
            finite_out_3 = from_bits_3;
        }
        
        finite_out_0 = xs::select(
            signed_mask_wide_0,
            finite_out_0 | WideBitsSIMD(kToSignBit),
            finite_out_0 & ~WideBitsSIMD(kToSignBit)
        );
        finite_out_1 = xs::select(
            signed_mask_wide_1,
            finite_out_1 | WideBitsSIMD(kToSignBit),
            finite_out_1 & ~WideBitsSIMD(kToSignBit)
        );
        finite_out_2 = xs::select(
            signed_mask_wide_2,
            finite_out_2 | WideBitsSIMD(kToSignBit),
            finite_out_2 & ~WideBitsSIMD(kToSignBit)
        );
        finite_out_3 = xs::select(
            signed_mask_wide_3,
            finite_out_3 | WideBitsSIMD(kToSignBit),
            finite_out_3 & ~WideBitsSIMD(kToSignBit)
        );
    
        store_to_full(i, finite_out_0);
        store_to_full(i + step, finite_out_1);
        store_to_full(i + step*2, finite_out_2);
        store_to_full(i + step*3, finite_out_3);
}
    
    // Remaining iterations
    for (; i <= n - step; i += step)
    {
        WideBitsSIMD result = process_iteration(i);
        store_to_full(i, result);
    }
    #endif
}
#endif                         
};

        template <typename Derived, typename UnderlyingType>
        template <typename From>
        Derived lo_float_base<Derived, UnderlyingType>::ConvertFrom(const From &from, ProjSpec ps)
        {
            return ConvertImpl<From, Derived>::run(from, ps);
        }

        template <typename Derived, typename UnderlyingType>
        template <typename To>
        To lo_float_base<Derived, UnderlyingType>::ConvertTo(const Derived &from, ProjSpec ps)
        {
            return ConvertImpl<Derived, To>::run(from, ps);
        }

       // ---- Proxy types ----

template <typename In>
struct ProjectProxy {
    In x;
    ProjSpec ps;

    template <typename Out>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE operator Out() const noexcept {
        return ConvertImpl<In, Out>::run(x, ps);
        }
};

        #ifndef USE_CUDA
template <typename In>
struct RoundProxy {
    In* x;
    int n;
    ProjSpec ps;

    template <typename Out>
    constexpr LOFLOAT_HOST LOFLOAT_FORCEINLINE operator Out() const noexcept {
        Out* y = nullptr; // or however you want to handle this
        return ConvertImpl<In, Out>::run(x, y, n, ps);
        }
};
        #endif

template <typename In1, typename In2, typename Op>
struct BinOpProxy {
    In1 x;
    In2 y;
    ProjSpec ps;

    template <typename Out>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE operator Out() const noexcept {
        return ConvertImpl<float, Out>::run(
            Op{}(static_cast<float>(x), static_cast<float>(y)), ps);
    }
};

struct OpAdd { constexpr float operator()(float a, float b) const noexcept { return a + b; } };
struct OpSub { constexpr float operator()(float a, float b) const noexcept { return a - b; } };
struct OpMul { constexpr float operator()(float a, float b) const noexcept { return a * b; } };
struct OpDiv { constexpr float operator()(float a, float b) const noexcept { return a / b; } };

// ---- Deduced versions (no explicit out) ----

template <typename In>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE auto Project(In x, ProjSpec ps = ProjSpec{}) noexcept {
    return ProjectProxy<In>{x, ps};
}

template <typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE auto Add(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return BinOpProxy<In1, In2, OpAdd>{x, y, ps};
}

template <typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE auto Sub(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return BinOpProxy<In1, In2, OpSub>{x, y, ps};
}

template <typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE auto Mul(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return BinOpProxy<In1, In2, OpMul>{x, y, ps};
}

template <typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE auto Div(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return BinOpProxy<In1, In2, OpDiv>{x, y, ps};
}

// ---- Explicit out versions (Mul<out>(x, y) style) ----

template <typename Out, typename In>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Out Project(In x, ProjSpec ps = ProjSpec{}) noexcept {
    return ConvertImpl<In, Out>::run(x, ps);
}

template <typename Out, typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Out Add(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return ConvertImpl<float, Out>::run(static_cast<float>(x) + static_cast<float>(y), ps);
        }

template <typename Out, typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Out Sub(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return ConvertImpl<float, Out>::run(static_cast<float>(x) - static_cast<float>(y), ps);
        }

template <typename Out, typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Out Mul(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return ConvertImpl<float, Out>::run(static_cast<float>(x) * static_cast<float>(y), ps);
        }

template <typename Out, typename In1, typename In2>
constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Out Div(In1 x, In2 y, ProjSpec ps = ProjSpec{}) noexcept {
    return ConvertImpl<float, Out>::run(static_cast<float>(x) / static_cast<float>(y), ps);
        }

        template <typename Float>
        struct float2_batch
        {

            Float a;
            Float b;

            // Constructor to initialize all elements to a specific value
            float2_batch(Float value1 = Float{}, Float value2 = Float{}) : a(value1), b(value2) {}
        };

        template <typename Float>
        struct float_3batch
        {
            Float a;
            Float b;
            Float c;

            // Constructor to initialize all elements to a specific value
            float_3batch(Float value = Float{}) : a(value), b(value), c(value) {}
        };

        template <typename Float>
        struct float_4batch
        {
            Float a;
            Float b;
            Float c;
            Float d;

            // Constructor to initialize all elements to a specific value
            float_4batch(Float value = Float{}) : a(value), b(value), c(value), d(value) {}
        };

    } // namespace lo_float_internal

    template <FloatingPointParams Fp>
    using Templated_Float = lo_float_internal::Templated_Float<Fp>;

    template <class T>
    struct is_lo_float
    {
        static constexpr bool value = false;
    };

    template <FloatingPointParams Fp>
    struct is_lo_float<Templated_Float<Fp>>
    {
        static constexpr bool value = true;
    };

    template <typename T>
    constexpr bool is_floating_point_v = is_lo_float<T>::value || std::is_floating_point_v<T>;

    template <typename T>
    using float4_batch = lo_float_internal::float_4batch<T>;

    template <typename T>
    using float3_batch = lo_float_internal::float_3batch<T>;

    template <typename T>
    using float2_batch = lo_float_internal::float2_batch<T>;

    template <typename out, typename in>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Project(in x, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Project<out, in>(x, ps);
    }

    template <typename out, typename in>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Round(in x, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Project<out, in>(x, ps);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Add(in1 x, in2 y, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Add<out, in1, in2>(x, y, ps);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Sub(in1 x, in2 y, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Sub<out, in1, in2>(x, y, ps);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Mul(in1 x, in2 y, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Mul<out, in1, in2>(x, y, ps);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Div(in1 x, in2 y, ProjSpec ps = ProjSpec{}) noexcept
    {
        return lo_float_internal::Div<in1, in2, out>(x, y, ps);
    }

    #ifndef USE_CUDA
    template <typename out, typename in, int unroll_len = 0, class arch = xsimd::default_arch>
    LOFLOAT_HOST void Project(in* x, out *y, int n, ProjSpec ps = ProjSpec{}) noexcept
    {
        using Converter = lo_float_internal::ConvertImpl<in, out>;
        Converter::template run<arch>(x, y, n, ps);
        return;
    }
    #endif

    #ifndef USE_CUDA
 template <typename Out, typename In1, typename In2, typename In3, class arch = xsimd::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE
void fma_vec(const In1* LOFLOAT_RESTRICT x,
            const In2* LOFLOAT_RESTRICT y,
            const In3* LOFLOAT_RESTRICT z,
            Out* LOFLOAT_RESTRICT out,
            int n,
            ProjSpec ps = ProjSpec{}) noexcept
{
    static_assert(std::is_trivially_copyable_v<In1> && std::is_trivially_copyable_v<In2> &&
                std::is_trivially_copyable_v<In3> && std::is_trivially_copyable_v<Out>,
                "fma_vec expects trivially copyable element types.");

    if (!x || !y || !z || !out || n <= 0) return;

    using xs_arch = arch;
    using f_batch = xsimd::batch<float, xs_arch>;
    
    // Query the number of lanes for float32
    constexpr std::size_t W = f_batch::size;

    const int vec_end = (n / static_cast<int>(W)) * static_cast<int>(W);

    // Helper: microkernel on [i, i+W)
    auto microkernel = [&](int i) {
        // Allocate fp32 buffers for conversion
        alignas(xs_arch::alignment()) float x_fp32[W];
        alignas(xs_arch::alignment()) float y_fp32[W];
        alignas(xs_arch::alignment()) float z_fp32[W];
        alignas(xs_arch::alignment()) float result_fp32[W];
        
        // Convert inputs to fp32 scalar-by-scalar
        for (std::size_t lane = 0; lane < W; ++lane) {
            x_fp32[lane] = static_cast<float>(x[i + lane]);
            y_fp32[lane] = static_cast<float>(y[i + lane]);
            z_fp32[lane] = static_cast<float>(z[i + lane]);
        }
        
        // Load into SIMD registers
        const f_batch fx = xsimd::load_aligned(x_fp32);
        const f_batch fy = xsimd::load_aligned(y_fp32);
        const f_batch fz = xsimd::load_aligned(z_fp32);
        
        // Compute FMA: x * y + z
        const f_batch fs = xsimd::fma(fx, fy, fz);
        
        // Store result back to fp32 buffer
        xsimd::store_aligned(result_fp32, fs);
        
        // Round scalar-by-scalar to output type
        for (std::size_t lane = 0; lane < W; ++lane) {
            out[i + lane] = lo_float::Round<Out>(result_fp32[lane], ps);
        }
    };

#if defined(_LOFOPENMP)
    #pragma omp parallel for 
    for (int i = 0; i < vec_end; i += static_cast<int>(W)) {
        microkernel(i);
    }
#else
    for (int i = 0; i < vec_end; i += static_cast<int>(W)) {
        microkernel(i);
    }
#endif

    // Tail - process remaining elements
    for (int i = vec_end; i < n; ++i) {
        const float x_f = static_cast<float>(x[i]);
        const float y_f = static_cast<float>(y[i]);
        const float z_f = static_cast<float>(z[i]);
        const float result = x_f * y_f + z_f;
        out[i] = lo_float::Round<Out>(result, ps);
    }
}
#endif

#ifndef USE_CUDA
template <typename Out, typename In1, typename In2, class arch = xsimd::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE
void add_vec(const In1* LOFLOAT_RESTRICT x,
             const In2* LOFLOAT_RESTRICT y,
             Out* LOFLOAT_RESTRICT out,
             int n,
             ProjSpec ps = ProjSpec{}) noexcept
{
    static_assert(std::is_trivially_copyable_v<In1> && std::is_trivially_copyable_v<In2> &&
                  std::is_trivially_copyable_v<Out>,
                  "add_vec expects trivially copyable element types.");
    if (!x || !y || !out || n <= 0) return;

    using xs_arch = arch;
    using f_batch = xsimd::batch<float, xs_arch>;
    constexpr std::size_t W = f_batch::size;
    const int vec_end = (n / static_cast<int>(W)) * static_cast<int>(W);

#if defined(_LOFOPENMP)
    #pragma omp parallel
    {
        alignas(xs_arch::alignment()) float x_fp32[W];
        alignas(xs_arch::alignment()) float y_fp32[W];
        alignas(xs_arch::alignment()) float result_fp32[W];

        #pragma omp for
        for (int i = 0; i < vec_end; i += static_cast<int>(W)) {
            for (std::size_t lane = 0; lane < W; ++lane) {
                x_fp32[lane] = static_cast<float>(x[i + lane]);
                y_fp32[lane] = static_cast<float>(y[i + lane]);
            }

            const f_batch fs = xsimd::load_aligned(x_fp32) + xsimd::load_aligned(y_fp32);
            xsimd::store_aligned(result_fp32, fs);

            for (std::size_t lane = 0; lane < W; ++lane) {
                out[i + lane] = lo_float::Round<Out>(result_fp32[lane], ps);
            }
        }
    }
#else
    alignas(xs_arch::alignment()) float x_fp32[W];
    alignas(xs_arch::alignment()) float y_fp32[W];
    alignas(xs_arch::alignment()) float result_fp32[W];

    for (int i = 0; i < vec_end; i += static_cast<int>(W)) {
        for (std::size_t lane = 0; lane < W; ++lane) {
            x_fp32[lane] = static_cast<float>(x[i + lane]);
            y_fp32[lane] = static_cast<float>(y[i + lane]);
        }

        const f_batch fs = xsimd::load_aligned(x_fp32) + xsimd::load_aligned(y_fp32);
        xsimd::store_aligned(result_fp32, fs);

        for (std::size_t lane = 0; lane < W; ++lane) {
            out[i + lane] = lo_float::Round<Out>(result_fp32[lane], ps);
        }
    }
#endif

    for (int i = vec_end; i < n; ++i) {
        const float result = static_cast<float>(x[i]) + static_cast<float>(y[i]);
        out[i] = lo_float::Round<Out>(result, ps);
    }
}
#endif




#ifndef USE_CUDA
template <typename TA, typename TX, typename TY, class arch = xsimd::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE
void axpy(const int n,
          const TA* LOFLOAT_RESTRICT a,
          const TX* LOFLOAT_RESTRICT x,
          const int incx,
          TY* LOFLOAT_RESTRICT y,
          const int incy,
          ProjSpec ps = ProjSpec{}) noexcept
{
    static_assert(std::is_trivially_copyable_v<TA> &&
                  std::is_trivially_copyable_v<TX> &&
                  std::is_trivially_copyable_v<TY>,
                  "axpy expects trivially copyable element types.");

    if (!a || !x || !y || n <= 0) return;

    using xs_arch = arch;
    using f_batch = xsimd::batch<float, xs_arch>;
    constexpr std::size_t W = f_batch::size;

    const float a_f = static_cast<float>(*a);
    const f_batch fa(a_f);

    // Only vectorize when contiguous
    if (incx == 1 && incy == 1)
    {
        const int vec_end = (n / static_cast<int>(W)) * static_cast<int>(W);

        auto microkernel = [&](int i) {
            alignas(xs_arch::alignment()) float x_fp32[W];
            alignas(xs_arch::alignment()) float y_fp32[W];
            alignas(xs_arch::alignment()) float out_fp32[W];

            for (std::size_t lane = 0; lane < W; ++lane) {
                x_fp32[lane] = static_cast<float>(x[i + lane]);
                y_fp32[lane] = static_cast<float>(y[i + lane]);
            }

            const f_batch fx = xsimd::load_aligned(x_fp32);
            const f_batch fy = xsimd::load_aligned(y_fp32);

            // y = a*x + y
            const f_batch fs = xsimd::fma(fa, fx, fy);

            xsimd::store_aligned(out_fp32, fs);

            for (std::size_t lane = 0; lane < W; ++lane) {
                y[i + lane] = lo_float::Round<TY>(out_fp32[lane], ps);
            }
        };

    #if defined(_LOFOPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < vec_end; i += static_cast<int>(W)) {
        microkernel(i);
    }

    // Tail
    
    for (int i = vec_end; i < n; ++i) {
        const float xf = static_cast<float>(x[i]);
        const float yf = static_cast<float>(y[i]);
        const float result = a_f * xf + yf;
        y[i] = lo_float::Round<TY>(result, ps);
    }
}
else
{
    // strided fallback
    int ix = 0, iy = 0;
    for (int i = 0; i < n; ++i) {
        const float xf = static_cast<float>(x[ix]);
        const float yf = static_cast<float>(y[iy]);
        const float result = a_f * xf + yf;
        y[iy] = lo_float::Round<TY>(result, ps);
        ix += incx;
        iy += incy;
    }
}

}
#endif

template <typename From>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE From virtual_round(const From &from, int ToMantissaBits, ProjSpec ps = ProjSpec{}) {
    return lo_float_internal::virtual_round(from, ToMantissaBits, ps);
}

#ifndef USE_CUDA
template<typename From>
LOFLOAT_HOST LOFLOAT_FORCEINLINE void virtual_round(From* values, From* results, int ToMantissaBits, int n, ProjSpec ps = ProjSpec{}) {
    return lo_float_internal::virtual_round(values, results, ToMantissaBits, n, ps);
}
#endif

template <typename From, typename ToInf, typename ToNaN>
LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE From virtual_round(const From &from, FloatingPointParams<ToInf, ToNaN> ToFp, ProjSpec ps = ProjSpec{}) {
    return lo_float_internal::virtual_round(from, ToFp, ps);
}

#ifndef USE_CUDA
template<typename From, typename ToInf, typename ToNaN>
LOFLOAT_HOST LOFLOAT_FORCEINLINE void virtual_round(From* values, From* results, int n, FloatingPointParams<ToInf, ToNaN> ToFp, ProjSpec ps = ProjSpec{}) {
    return lo_float_internal::virtual_round(values, results, n, ToFp, ps);
}
#endif

#ifndef USE_CUDA
// ---------------------------------------------------------------------------
//  virtual_mx_round  (CPU / xsimd) — microscaling (MX) block quantization.
//
//  CPU counterpart of the virtual_mx_round CUDA kernel in Lof_kernel.cu. The
//  input is partitioned into contiguous blocks of `block_size` elements; each
//  block shares one scale. For each block:
//
//    amax  = max |x_i|                              (xsimd reduction)
//    scale = virtual_round( amax / priv_max_normal, // share-exponent for the
//                           params_public,          // block, rounded into the
//                           round_mode_public )     // public (scale) format
//    x_i  := virtual_round( x_i / scale,            // round each element into
//                           params_private,         // the private (element)
//                           round_mode_private,     // format, then rescale
//                           stoch_len ) * scale
//
//  "Virtual": the result is written back in `From` (e.g. fp32) as the rescaled
//  rounded value — no separate low-precision storage. This is the CPU analog of
//  the GPU path used by fake_mx_quantize_tensor.
//
//  priv_max_normal is the largest finite normal of the private format, derived
//  from its runtime params exactly as the CUDA kernel does.
//
//  Uses xsimd for the per-block amax reduction and the divide/multiply; the
//  per-element rounding goes through the same scalar virtual_round the rest of
//  the library is tested against (so the result matches that reference exactly).
// ---------------------------------------------------------------------------
template<typename From,
         typename PubInf, typename PubNaN,
         typename PrivInf, typename PrivNaN,
         class arch = xs::default_arch>
LOFLOAT_HOST void virtual_mx_round(
    From* inout, int n, int block_size,
    FloatingPointParams<PubInf, PubNaN>   params_public,
    FloatingPointParams<PrivInf, PrivNaN> params_private,
    ProjSpec ps_public  = ProjSpec{},
    ProjSpec ps_private = ProjSpec{})
{
    using batch = xs::batch<From, arch>;
    constexpr int W = static_cast<int>(batch::size);

    if (block_size <= 0 || n <= 0) return;

    // Largest finite normal of the private format (same derivation as the
    // CUDA kernel): (2 - 2^-mant) * 2^max_exp.
    const int priv_max_exp = params_private.is_signed == Signedness::Signed
        ? (1 << (params_private.bitwidth - params_private.mantissa_bits - 1)) - 1 - params_private.bias
        : (1 << (params_private.bitwidth - params_private.mantissa_bits))     - 1 - params_private.bias;
    const From priv_max_normal = static_cast<From>(
        std::ldexp(1.0, priv_max_exp) * (2.0 - std::ldexp(1.0, -params_private.mantissa_bits)));

    for (int b0 = 0; b0 < n; b0 += block_size)
    {
        const int len  = std::min(block_size, n - b0);
        From*     blk  = inout + b0;

        // 1) amax over the block (xsimd full-width reduction + scalar tail).
        From amax = From(0);
        {
            int i = 0;
            if (len >= W) {
                batch vmax = xs::abs(batch::load_unaligned(blk));
                i = W;
                for (; i + W <= len; i += W)
                    vmax = xs::max(vmax, xs::abs(batch::load_unaligned(blk + i)));
                amax = xs::reduce_max(vmax);
            }
            for (; i < len; ++i)
                amax = std::max(amax, static_cast<From>(std::fabs(static_cast<double>(blk[i]))));
        }

        // 2) shared scale = round_public(amax / priv_max_normal).
        From scale = lo_float::virtual_round(
            static_cast<From>(amax / priv_max_normal), params_public,
            ProjSpec{ps_public.rounding_mode, ps_public.saturation_mode, 0});

        if (!(scale > From(0))) {
            // amax underflowed to a zero scale: the block is (numerically) zero.
            for (int i = 0; i < len; ++i) blk[i] = From(0);
            continue;
        }

        // 3) per element: round(x / scale) into the private format, then rescale.
        const batch vscale(scale);
        int i = 0;
        for (; i + W <= len; i += W)
            (batch::load_unaligned(blk + i) / vscale).store_unaligned(blk + i);   // divide
        for (; i < len; ++i)
            blk[i] = blk[i] / scale;

        for (int j = 0; j < len; ++j)                                             // round (private)
            blk[j] = lo_float::virtual_round(blk[j], params_private, ps_private);

        i = 0;
        for (; i + W <= len; i += W)
            (batch::load_unaligned(blk + i) * vscale).store_unaligned(blk + i);   // rescale
        for (; i < len; ++i)
            blk[i] = blk[i] * scale;
    }
}
#endif

struct DeviceInfChecker {
    uint64_t exp_mask, mant_mask, neg_inf, pos_inf;
    LOFLOAT_HOST_DEVICE bool operator()(uint64_t bits) const { return (bits & exp_mask) == exp_mask && (bits & mant_mask) == 0; }
    LOFLOAT_HOST_DEVICE uint64_t minNegInf() const { return neg_inf; }
    LOFLOAT_HOST_DEVICE uint64_t minPosInf() const { return pos_inf; }
};

struct DeviceNaNChecker {
    uint64_t exp_mask, mant_mask, qnan, snan;
    LOFLOAT_HOST_DEVICE bool operator()(uint64_t bits) const { return (bits & exp_mask) == exp_mask && (bits & mant_mask) != 0; }
    LOFLOAT_HOST_DEVICE uint64_t qNanBitPattern() const { return qnan; }
    LOFLOAT_HOST_DEVICE uint64_t sNanBitPattern() const { return snan; }
};



    template <typename T>
    concept Float = lo_float::is_floating_point_v<T>;

    template <typename T>
    concept Int = lo_float::is_integral_v<T>;

} // namespace lo_float


