/// @author Sudhanva Kulkarni
#ifndef LO_FLOAT_H
#define LO_FLOAT_H
// #define ENABLE_EXCEPT

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
#include <math.h>
#include <complex>
#include <limits>
#include <ostream>
#include <type_traits>
#include <climits>
#include "platform_macros.h"
#include <xsimd/xsimd.hpp>
#include "fp_tools.hpp"     //structs and concepts to define Floating Point params
#include "f_exceptions.hpp" //global env for exceptions

#include "lo_int.h"           //custom integer types
#include "template_helpers.h" //helper templataes
#include "simd_helpers.hpp"     //simd helpers

#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif

namespace xs = xsimd;

namespace lo_float
{

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

        static std::mt19937 mt(static_cast<int>(time(nullptr)));

        // global function to set seed
        void set_seed(int a)
        {
            mt.seed(a);
        }

#ifdef ENABLE_EXCEPT
        Environment f_env{};
#endif

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
            constexpr lo_float_base() : rep_(0) {}

            constexpr UnderlyingType rep() const
            {
                return rep_;
            }

            // Templated constructor
            template <typename T,
                      typename EnableIf = std::enable_if_t<std::is_arithmetic_v<T>>>
            explicit lo_float_base(T f)
                : lo_float_base(ConvertFrom(static_cast<float>(f)).rep(),
                                ConstructFromRepTag{}) {}

            explicit lo_float_base(double f64)
                : lo_float_base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}

            explicit lo_float_base(float f32)
                : lo_float_base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}

            explicit lo_float_base(const int i32)
                : lo_float_base(ConvertFrom(static_cast<double>(i32)).rep(), ConstructFromRepTag{}) {}

            template <int len, Signedness Sign>
            explicit lo_float_base(i_n<len, Sign> var_int)
                : lo_float_base(ConvertFrom(static_cast<double>((int)var_int)).rep(), ConstructFromRepTag{}) {}

            // CRTP helpers
            constexpr const Derived &derived() const
            {
                return *static_cast<const Derived *>(this);
            }
            constexpr Derived &derived()
            {
                return *static_cast<Derived *>(this);
            }

            static constexpr Derived FromRep(UnderlyingType rep)
            {
                return Derived(rep, ConstructFromRepTag{});
            }

            // -------------------------------------------
            // Declarations for ConvertFrom / ConvertTo
            // -------------------------------------------
            template <typename From>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Derived ConvertFrom(const From &from, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0);

            template <typename To>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE To ConvertTo(const Derived &from, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0);

            template <typename T,
                      typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
            explicit operator T() const
            {
                return static_cast<T>(static_cast<float>(derived()));
            }
            explicit operator double() const
            {
                return ConvertTo<double>(derived());
            }
            explicit operator float() const
            {
                return ConvertTo<float>(derived());
            }

            explicit operator int() const
            {
                return (int)(ConvertTo<double>(derived()));
            }

            template <int len, Signedness sign>
            explicit operator i_n<len, sign>() const
            {
                return (i_n<len, sign>)int(ConvertTo<double>(derived()));
            }

            explicit operator bool() const
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
                // check spl case of -0 for nan
                if (rep_ == 0 && Derived::IsNaNFunctor((1 << (get_bitwidth_v<Derived> - 1))))
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
                        if (get_NaN_Behavior_v<Derived> == NaN_Behaviors::QuietNaN)
                        {
                            return FromRep(Derived::IsNaNFunctor.qNanBitPattern());
                        }
                        else
                        {
                            // need to signal exception here
                            return FromRep(Derived::IsNaNFunctor.sNanBitPattern());
                        }
                    }
                }
            }

            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator+(const Derived &other) const
            {
                return Derived{UnderlyingFloat{derived()} + UnderlyingFloat{other}};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator=(const Derived &other) const
            {
                return Derived{UnderlyingFloat{other}};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator-(const Derived &other) const
            {
                return Derived{UnderlyingFloat{derived()} - UnderlyingFloat{other}};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator*(const Derived &other) const
            {
                return Derived{UnderlyingFloat{derived()} * UnderlyingFloat{other}};
            }
            LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE  Derived
            operator/(const Derived &other) const
            {
#ifdef ENABLE_EXCEPT
                if (!other)
                {
                    f_env.set_exception_flag(LF_exception_flags::DivisionByZero);
                    if (!derived())
                    {
                        f_env.set_exception_flag(LF_exception_flags::InvalidOperation);
                    }
                }
#endif
                return Derived{UnderlyingFloat{derived()} / UnderlyingFloat{other}};
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
                if constexpr (!other)
                {
                    f_env.set_exception_flag(LF_exception_flags::DivisionByZero);
                    if constexpr (!derived())
                    {
                        f_env.set_exception_flag(LF_exception_flags::InvalidOperation);
                    }
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
                    std::bit_cast<UnderlyingType>(abs(x));
                const UnderlyingType x_bits = std::bit_cast<UnderlyingType>(x);
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

        template <typename Bits, typename Roundoff>
        constexpr inline Bits RoundBitsToNearestEven(Bits bits, Roundoff roundoff)
        {
            using value_type = pod_type_t<Bits>;

            Bits bias;
            if constexpr (xsimd::is_batch<Roundoff>::value)
            {
                // SIMD roundoff (fast path)
                bias = xsimd::select(xs::batch_cast<value_type>(roundoff) == Bits(0),
                    Bits(0),
                    (xs::bitwise_rshift(bits, xs::batch_cast<value_type>(roundoff)) & Bits(1)) + xs::bitwise_lshift(Bits(1),xs::batch_cast<value_type>(roundoff - 1)) - Bits(1));
            }
            else
            {
                // Scalar roundoff (fast path)
                bias = roundoff == 0
                    ? Bits(0)
                    : ((bits >> roundoff) & Bits(1)) + (Bits(1) << (roundoff - 1)) - Bits(1);
            }

            return bits + bias;
        }


        template <typename Bits, typename Roundoff>
        inline Bits Probabilistic_Round(Bits bits, Roundoff roundoff)
        {
            // using lane_t = pod_type_t<Bits>;
            // constexpr std::size_t lanes = num_lanes_v<Bits>;

            // Bits mask = (Bits(1) << roundoff) - Bits(1);
            // Bits truncated = bits & ~mask;
            // Bits tail = bits & mask;

            // // Per-lane random 0/1 (from the same global mt stream)
            // std::array<lane_t, lanes> r{};
            // std::uniform_int_distribution<int> d01(0, 1);
            // for (std::size_t i = 0; i < lanes; ++i) r[i] = static_cast<lane_t>(d01(mt));
            // Bits r01;
            // if constexpr (lanes == 1)
            //     r01 = Bits{r[0]};
            // else
            //     r01 = xsimd::load_unaligned(r.data());

            // // if constexpr (xsimd::is_batch<Roundoff>::value)
            // // {
            // //     auto tail_nz = (tail != Bits(0));
            // //     Bits bump = xsimd::select(tail_nz, r01, Bits(0));
            // //     return truncated + (bump << roundoff);
            // // }
            // // else
            // {
            //     Bits bump = (tail != Bits(0)) * r01;
            //     return truncated + (bump << roundoff);
            // }
            return bits;
        }

        template <typename Bits, typename Roundoff>
        inline Bits Stochastic_Round_A(Bits bits, Roundoff roundoff, const int len = 0)
        {
         using lane_t = pod_type_t<Bits>;

        constexpr std::size_t lanes = num_lanes_v<Bits>;


            if (len <= 0) return bits;

            // Per-lane random integer in [0, 2^len - 1]
            // Assumes len < 32 (and < bitwidth(lane_t)); adjust if you need larger.
            const unsigned maxv = (1u << unsigned(len)) - 1u;
            std::uniform_int_distribution<unsigned> dist(0u, maxv);

            std::array<lane_t, lanes> samp{};
            for (std::size_t i = 0; i < lanes; ++i) samp[i] = static_cast<lane_t>(dist(mt));
            Bits to_add;
            if constexpr (lanes == 1)
                to_add = Bits{samp[0]};
            else
                to_add = xsimd::load_unaligned(samp.data());

            if constexpr (xsimd::is_batch<Roundoff>::value)
            {
                return bits + xsimd::bitwise_lshift(to_add , (roundoff - lane_t(len)));
            }
            else
            {
                return bits + (to_add << (roundoff - len));
            }
        }


        template <typename Bits>
        inline Bits Stochastic_Round_B(Bits bits, const int roundoff, const int len = 0)
        {
            std::uniform_int_distribution<unsigned int> distribution(0, (1 << (len)) - 1);
            // auto len = 2;
            int samp = distribution(mt); // Generate a random integer
            Bits complement = (Bits{1} << (len)) - 1;
            Bits to_add = static_cast<Bits>(samp & complement) + 1;
            const Bits lower = bits & ((Bits{1} << roundoff) - 1); // Get the lower bits to be rounded off
            const bool sum_mask = lower > 0;
            Bits to_ret = bits + (sum_mask ? (to_add << (roundoff - len)) : Bits{}); // Add random bits to the input bits
            return to_ret;
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xsimd::batch<Bits, arch>
        // Stochastic_Round_B(xsimd::batch<Bits, arch> bits, int roundoff, int len = 0)
        // {
        //     static_assert(std::is_integral_v<Bits>, "Bits must be an integral type.");
        //     using B = xsimd::batch<Bits, arch>;
        //     constexpr std::size_t lanes = B::size;

        //     // Match scalar behavior expectations: these are "fast exits"
        //     if (len <= 0 || roundoff <= 0) return bits;

        //     // Scalar code assumes len <= roundoff (otherwise shift is negative UB)
        //     if (len > roundoff) len = roundoff;

        //     const int shift = roundoff - len;  // >= 0

        //     // --- RNG per lane: uniform in [0, 2^len - 1] ---
        //     // Use uint64_t to avoid UB for shifts; for valid ranges it matches scalar.
        //     const uint64_t max_u64 = (uint64_t{1} << len) - 1u;
        //     std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(max_u64));

        //     alignas(B::alignment()) Bits samp_arr[lanes];
        //     for (std::size_t i = 0; i < lanes; ++i)
        //         samp_arr[i] = static_cast<Bits>(dist(mt));

        //     const B samp = xsimd::load_aligned(samp_arr);

        //     // complement = (1<<len)-1
        //     const Bits complement_scalar = (Bits{1} << len) - Bits{1};
        //     const B complement(complement_scalar);

        //     // to_add = (samp & complement) + 1
        //     const B to_add = (samp & complement) + B(Bits{1});

        //     // lower = bits & ((1<<roundoff)-1)
        //     // (Assumes roundoff < bitwidth, same as scalar's valid range.)
        //     const Bits lower_mask_scalar = (Bits{1} << roundoff) - Bits{1};
        //     const B lower = bits & B(lower_mask_scalar);

        //     // sum_mask = lower > 0
        //     const auto sum_mask = (lower > B(Bits{0}));

        //     // add = to_add << shift
        //     B add = to_add;
        //     if (shift > 0) add = add << shift;

        //     return bits + xsimd::select(sum_mask, add, B(Bits{0}));
        // }



        template <typename Bits>
        inline Bits Stochastic_Round_C(Bits bits, const int roundoff, const int len = 0)
        {
            // given pattern FFF...FLRTT...T,rounds stochastically by generating random bits
            //  corresponding to  RTT...T and adding the genned number.
            // Then we truncate the mantissa
            // auto len = 2;
            std::uniform_int_distribution<unsigned int> distribution(0, (1 << (len)) - 1);
            int samp = distribution(mt); // Generate a random integer of length len, next get top "roundoff" bits
            // if RTTTT != 0, add a coin flip to samp
            const unsigned int coin_bit = samp & 1;
            unsigned int remaining_bits = samp >> 1; // Get the remaining bits after the coin flip
            Bits bottom_bits = bits & ((Bits{1} << roundoff) - 1);
            Bits top_bits = static_cast<Bits>((remaining_bits + coin_bit) << (roundoff - len + 1));
            return (len == 0 || bottom_bits == 0) ? bits : bits + (top_bits);
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xs::batch<Bits, arch> Stochastic_Round_C(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff, int len = 0)
        // {
        //     static_assert(std::is_integral_v<Bits>, "Bits must be an integral type.");
        //     using batch_bits = xs::batch<Bits, arch>;
        //     constexpr std::size_t lanes = num_lanes_v<batch_bits>;

        //     // Fast exits + avoid UB
        //     if (len <= 0 || roundoff <= 0) return bits;

        //     // This shift must be >= 0 for your formula
        //     // top_bits shift = roundoff - len + 1
        //     if (len > roundoff + 1) len = roundoff + 1;
        //     const int top_shift = roundoff - len + 1; // now >= 0

        //     // ----- RNG per lane (scalar) -----
        //     // Compute max = 2^len - 1 without UB for len==32
        //     const uint64_t max_u64 = (uint64_t{1} << len) - 1u;
        //     std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(max_u64));

        //     alignas(64) Bits samp_arr[lanes];
        //     #pragma unroll
        //     for (std::size_t i = 0; i < lanes; ++i)
        //         samp_arr[i] = static_cast<Bits>(dist(mt));

        //     const batch_bits samp = xs::load_aligned(samp_arr);

        //     // ----- SIMD logic -----
        //     const batch_bits one(Bits{1});

        //     // coin_bit = samp & 1
        //     const batch_bits coin_bit = samp & one;

        //     // remaining_bits = samp >> 1
        //     const batch_bits remaining_bits = samp >> 1;

        //     // bottom_bits = bits & ((1<<roundoff)-1)
        //     const Bits bottom_mask_scalar =
        //         (roundoff >= int(sizeof(Bits) * 8)) ? ~Bits{0} : (Bits{1} << roundoff) - 1;
        //     const batch_bits bottom_mask(bottom_mask_scalar);
        //     const batch_bits bottom_bits = bits & bottom_mask;

        //     // top_bits = (remaining_bits + coin_bit) << (roundoff - len + 1)
        //     batch_bits top_bits = remaining_bits + coin_bit;
        //     if (top_shift > 0) top_bits = top_bits << top_shift;

        //     // return (bottom_bits == 0) ? bits : bits + top_bits
        //     const auto do_add = (bottom_bits != batch_bits(Bits{0}));
        //     return xs::select(do_add, bits + top_bits, bits);
        // }

        template <typename Bits>
        inline Bits True_Stochastic_Round(Bits bits, const int roundoff)
        {
            // true stoch round rounds up with prob RTT...T / 2^roundoff
            const Bits mask = (Bits{1} << roundoff) - 1;
            const Bits tail = bits & mask;       // bits to be discarded
            const Bits truncated = bits & ~mask; // keep the rest

            // Compute probability of rounding up
            const double prob = static_cast<double>(tail) / static_cast<double>(Bits{1} << roundoff);

            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            const double samp = distribution(mt);

            // Round up with probability equal to tail / 2^roundoff
            if (samp < prob)
            {
                Bits rounded = truncated + (Bits{1} << roundoff);
                if (rounded < truncated)
                {
                    return truncated;
                }
                return rounded;
            }
            else
            {
                return truncated;
            }
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xs::batch<Bits, arch> True_Stochastic_Round(xs::batch<Bits, arch> bits, const int roundoff)
        // {
        //     const Bits mask = (Bits{1} << roundoff) - 1;
        //     const Bits tail = bits & mask;       // bits to be discarded
        //     const Bits truncated = bits & ~mask; // keep the rest

        //     //get num lanes
        //     constexpr size_t num_lanes = xs::batch<Bits, arch>::size;
        //     double prob[num_lanes];
        //     double samp[num_lanes];

        //     // Compute probability of rounding up
        //     #pragma unroll
        //     for (size_t i = 0; i < num_lanes; i++)
        //     {
        //         const double prob = static_cast<double>(tail) / static_cast<double>(Bits{1} << roundoff);
        //     }

        //     std::uniform_real_distribution<double> distribution(0.0, 1.0);
        //     #pragma unroll
        //     for (size_t i = 0; i < num_lanes; i++)
        //     {
        //         samp[i] = distribution(mt);
        //     }
        //     // Round up with probability equal to tail / 2^roundoff
        //     #pragma unroll
        //     for (size_t i = 0; i < num_lanes; i++)
        //     {
        //     if (samp[i] < prob[i])
        //     {
        //         Bits rounded = truncated + (Bits{1} << roundoff);
        //         if (rounded < truncated)
        //         {
        //             return truncated;
        //         }
        //         return rounded;
        //     }
        //     else
        //     {
        //         return truncated;
        //     }
        //     }

        // }


        template <typename Bits>
        inline Bits RoundBitsTowardsZero(Bits bits, int roundoff)
        {
            // Round towards zero by just truncating the bits
            // in bits FFF...FLRTT....T RTT....T needs to be rounded off, so just set  RTT..T to be 0
            auto mask = ~((Bits{1} << (roundoff)) - 1);

            return bits & mask;
        }

        template <typename Bits>
        inline Bits RoundTiedBitsTowardsZero(Bits bits, int roundoff)
        {
            // Round towards zero by just truncating the bits
            // in bits FFF...FLRTT....T RTT....T needs to be rounded off, so just set  RTT..T to be 0
            auto mask = ~((Bits{1} << (roundoff)) - 1);

            return bits & mask;
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xs::batch<Bits, arch> RoundBitsTowardsZero(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff)
        // {
        //     // Round towards zero by just truncating the bits
        //     // in bits FFF...FLRTT....T RTT....T needs to be rounded off, so just set  RTT..T to be 0
        //     auto mask = ~((xs::batch<Bits, arch>(1) << (roundoff)) - xs::batch<Bits, arch>(1));
        //     return bits & mask;
        // }

        template <typename Bits>
        inline Bits RoundBitsAwayFromZero(Bits bits, int roundoff)
        {
            // Round away from Zero by truncating bits and adding one to the remaining bit pattern if RTT...T > 0
            //  in bits FFF...FRTT...T, set RTT...T to be zero and add 1 to FFF...F
            auto mask = ~((Bits{1} << roundoff) - 1);
            Bits truncated = bits & mask;
            return truncated + (bits > 0 ? Bits{1} << roundoff : 0);
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xs::batch<Bits, arch> RoundBitsAwayFromZero(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff)
        // {
        //     // Round away from Zero by truncating bits and adding one to the remaining bit pattern if RTT...T > 0
        //     //  in bits FFF...FRTT...T, set RTT...T to be zero and add 1 to FFF...F
        //     auto mask = ~((xs::batch<Bits, arch>(1) << roundoff) - xs::batch<Bits, arch>(1));
        //     xs::batch<Bits, arch> truncated = bits & mask;
        //     return truncated + (bits > xs::batch<Bits, arch>(0) ? xs::batch<Bits, arch>(1) << roundoff : xs::batch<Bits, arch>(0));
        // }

        template <typename Bits>
        constexpr inline Bits RoundBitsToNearestOdd(Bits bits, int roundoff)
        {
            // Round to nearest odd by adding a bias term.
            // Consider a bit pattern:
            //   FFF...FLRTT...T,
            // where bits RTT...T need to be rounded-off. We add a bias term to the
            // bit pattern such that a carry is introduced to round up only if
            // - L is 0, R is 1, and any T is one, OR
            // - L is 0, R is 1
            // This ensures the final result is odd.
            Bits bias = roundoff == 0
                            ? 0
                            : ((~bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
            return bits + bias;
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // constexpr inline xs::batch<Bits, arch> RoundBitsToNearestOdd(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff)
        // {
        //     xs::batch<Bits, arch> bias = roundoff == xs::batch<int, arch>(0)
        //                                      ? xs::batch<Bits, arch>(0)
        //                                      : (((~bits) >> roundoff) & xs::batch<Bits, arch>(1)) + (xs::batch<Bits, arch>(1) << (roundoff - 1)) - xs::batch<Bits, arch>(1);
        //     return bits + bias;
        // }

        template <typename Bits>
        inline Bits RoundUp(Bits bits, int roundoff, bool positive = true)
        {
            // round bit pattern up by adding 1 to the bit pattern if positive, truncate if negative
            // in bits FFF...FLRTT...T, set RTT...T to be 0 and add 1 to FFF...F
            const Bits low_mask = (Bits{1} << roundoff) - 1; // 000…0111…1
            const Bits high_mask = ~low_mask;                // 111…1000…0
            Bits truncated = bits & high_mask;               // keep the high part

            truncated += (positive && ((bits & low_mask) != 0)) ? Bits{1} << roundoff : 0;

            return truncated;
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xs::batch<Bits, arch> RoundUp(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff, xs::batch_bool<Bits, arch> positive = true)
        // {
        //     const xs::batch<Bits, arch> low_mask = (xs::batch<Bits, arch>(1) << roundoff) - xs::batch<Bits, arch>(1); // 000…0111…1
        //     const xs::batch<Bits, arch> high_mask = ~low_mask;                                                        // 111…1000…0
        //     xs::batch<Bits, arch> truncated = bits & high_mask;                                                       // keep the high part

        //     truncated += (positive & ((bits & low_mask) != xs::batch<Bits, arch>(0))) ? xs::batch<Bits, arch>(1) << roundoff : xs::batch<Bits, arch>(0);

        //     return truncated;
        // }

        template <typename Bits>
        inline Bits RoundDown(Bits bits, int roundoff, bool positive = true)
        {
            // just truncate the bits
            // in bits FFF...FLRTT...T, set RTT...T to be 0 if positive, add 1 if negative
            auto mask = ~((Bits{1} << roundoff) - 1);
            Bits truncated = bits & mask;
            return truncated + (!positive ? (bits > 0 ? Bits{1} << roundoff : 0) : 0);
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // inline xsimd::batch<Bits, arch>
        // RoundDown(xsimd::batch<Bits, arch> bits,
        //         xsimd::batch<int, arch> roundoff,
        //         xsimd::batch_bool<Bits, arch> positive = xsimd::batch_bool<Bits, arch>(true))
        // {
        //     using B = xsimd::batch<Bits, arch>;
        //     using M = xsimd::batch_bool<Bits, arch>;

        //     if (roundoff <= 0) return bits;
        //     if (roundoff >= int(sizeof(Bits) * 8)) return B(Bits{0}); // or return bits; pick your policy

        //     const B one(Bits{1});
        //     const B lowmask  = (one << roundoff) - one;  // low `roundoff` bits = 1
        //     const B keepmask = ~lowmask;                 // keep upper bits

        //     const B truncated = bits & keepmask;

        //     // Any truncated-off bits?
        //     const M has_remainder = (bits & lowmask) != B(Bits{0});

        //     // For negative lanes (positive == false), rounding down means:
        //     // if remainder exists, increase magnitude by 1<<roundoff
        //     const M is_negative = !positive;
        //     const M do_inc = is_negative & has_remainder;

        //     const B inc = (one << roundoff);
        //     return truncated + xsimd::select(do_inc, inc, B(Bits{0}));
        // }


        template <typename Bits>
        inline Bits RoundTiesToAway(Bits bits, int roundoff)
        {
            // given LLLRTT...T, round to nearest and if tie, round away from zero by adding 1
            // so if R = 1, add 1 to the bit pattern, else trunctate
            auto mask = ~((Bits{1} << roundoff) - 1);
            Bits truncated = bits & mask;
            return truncated + (((bits >> (roundoff - 1)) & 1) << roundoff);
        }

        // template <typename Bits, class arch = xsimd::default_arch>
        // LOFLOAT_HOST LOFLOAT_FORCEINLINE xs::batch<Bits, arch> RoundTiesToAway(xs::batch<Bits, arch> bits, xs::batch<int, arch> roundoff)
        // {
        //     // given LLLRTT...T, round to nearest and if tie, round away from zero by adding 1
        //     // so if R = 1, add 1 to the bit pattern, else trunctate
        //     auto mask = ~((xs::batch<Bits, arch>(1) << roundoff) - xs::batch<Bits, arch>(1));
        //     xs::batch<Bits, arch> truncated = bits & mask;
        //     return truncated + (((bits >> (roundoff - 1)) & xs::batch<Bits, arch>(1)) << roundoff);
        // }

        //#TODO: add sign to list of args for roundUp and RoundDown
        template <typename Bits>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Bits RoundMantissa(Bits bits, const int roundoff, const Rounding_Mode rm, const int len = 0)
        {
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
                return RoundUp(bits, roundoff);
            case Rounding_Mode::RoundDown:
                return RoundDown(bits, roundoff);
            case Rounding_Mode::RoundTiesToAway:
                return RoundTiesToAway(bits, roundoff);
            case Rounding_Mode::StochasticRoundingA:
                return Stochastic_Round_A(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingB:
                return Stochastic_Round_B(bits, roundoff, len);
            case Rounding_Mode::StochasticRoundingC:
                return Stochastic_Round_C(bits, roundoff, len);
            case Rounding_Mode::True_StochasticRounding:
                return True_Stochastic_Round(bits, roundoff);
            default:
                return bits; // no rounding
            }
        }

        template <typename Bits, typename Roundoff,  class arch = xsimd::default_arch>
        LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE xs::batch<Bits, arch> RoundMantissa(xs::batch<Bits, arch> bits, const xs::batch<Roundoff, arch> roundoff, const Rounding_Mode rm, const int len = 0)
        {
            switch (rm)
            {
            case Rounding_Mode::RoundToNearestEven:
                return RoundBitsToNearestEven(bits, roundoff);
            // case Rounding_Mode::RoundToNearestOdd:
            //     return RoundBitsToNearestOdd(bits, roundoff);
            // case Rounding_Mode::RoundTowardsZero:
            //     return RoundBitsTowardsZero(bits, roundoff);
            // case Rounding_Mode::RoundAwayFromZero:
            //     return RoundBitsAwayFromZero(bits, roundoff);
            // case Rounding_Mode::RoundUp:
            //     return RoundUp(bits, roundoff);
            // case Rounding_Mode::RoundDown:
            //     return RoundDown(bits, roundoff);
            // case Rounding_Mode::RoundTiesToAway:
            //     return RoundTiesToAway(bits, roundoff);
            // case Rounding_Mode::StochasticRoundingA:
            //     return Stochastic_Round_A(bits, roundoff, len);
            // case Rounding_Mode::StochasticRoundingB:
            //     return Stochastic_Round_B(bits, roundoff, len);
            // case Rounding_Mode::StochasticRoundingC:
            //     return Stochastic_Round_C(bits, roundoff, len);
            // case Rounding_Mode::True_StochasticRounding:
            //     return True_Stochastic_Round(bits, roundoff);
             default:
                return bits; // no rounding
            }
        }

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

        // define FloatingPointParams for float8e4m3_fn
        struct OCP_F8E4M3_NaNChecker
        {
            bool operator()(uint32_t bits) const
            {
                return bits == 0x000000FF;
            }

            uint32_t qNanBitPattern() const
            {
                return 0x000000FF;
            } // typical QNaN

            uint32_t sNanBitPattern() const
            {
                return 0x000000FF;
            } // some SNaN pattern
        };

        struct OCP_F8E4M3_InfChecker
        {
            bool operator()(uint32_t bits) const
            {
                return false;
            }

            uint32_t minNegInf() const
            {
                return 0x0;
            } // -∞ => 0xFF800000

            uint32_t minPosInf() const
            {
                return 0x0;
            } // +∞ => 0x7F800000
        };

        template <Signedness signedness = Signedness::Signed>
        struct IEEE_F8_NaNChecker
        {
            bool operator()(uint32_t bits) const
            {
                return bits == (signedness == Signedness::Signed ? 0x00000080 : 0x000000FF);
            }

            uint32_t qNanBitPattern() const
            {
                return signedness == Signedness::Signed ? 0x00000080 : 0x000000FF;
            } // typical QNaN

            uint32_t sNanBitPattern() const
            {
                return signedness == Signedness::Signed ? 0x00000080 : 0x000000FF;
            } // some SNaN pattern
        };

        struct IEEE_F8_InfChecker
        {
            bool operator()(uint32_t bits) const
            {
                return bits == 0x0000007F || bits == 0x000000FF;
            }

            uint32_t minNegInf() const
            {
                return 0xFF;
            } // -∞ => 0xFF800000

            uint32_t minPosInf() const
            {
                return 0x7F;
            } // +∞ => 0x7F800000
        };

        constexpr FloatingPointParams param_float8_e4m3fn(
            8,                         // totoal bitwidth
            3,                         // mantissa bits
            7,                         // bias
            Inf_Behaviors::Saturating, // No infinity
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
            Signedness::Unsigned,      // It is unsigned
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // NaNChecker for float8e5m2
        struct OCP_F8E5M2_NaNChecker
        {
            bool operator()(uint32_t bits)
            {
                return bits == 0x000000FF || bits == 0x000000FE || bits == 0x000000FD;
            }

            uint32_t qNanBitPattern() const
            {
                return 0x000000FF;
            }

            uint32_t sNanBitPattern() const
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN, // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
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
            NaN_Behaviors::QuietNaN,   // NaN behavior
            Signedness::Signed,        // It is signed
            OCP_F8E4M3_InfChecker(),   // Inf Functor
            OCP_F8E4M3_NaNChecker()    // NaN Functor
        );

        // inf and nan checkers for P3109 float
        template <int k, Signedness is_signed>
        struct P3109_NaNChecker
        {
            constexpr bool operator()(uint32_t bits) const
            {
                return bits == (is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1);
            }

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

            uint32_t qNanBitPattern() const
            {
                return is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1;
            } // typical QNaN

            uint32_t sNanBitPattern() const
            {
                return is_signed == Signedness::Signed ? (1 << (k - 1)) : (1 << k) - 1;
            } 
        };

        template <int k, Signedness is_signed, Inf_Behaviors has_inf>
        struct P3109_InfChecker
        {
            constexpr bool operator()(uint32_t bits) const
            {
                if constexpr (has_inf != Inf_Behaviors::Saturating)
                {
                    return (bits | (1 << (k - 1))) == (is_signed == Signedness::Signed ? (1 << (k)) - 1 : (1 << k) - 2);
                }
                else
                {
                    return false; // No infinity for P3109
                }
            }

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
                    return false; // No infinity for P3109
                }
            }

            uint32_t minNegInf() const
            {
                return ((is_signed == Signedness::Signed) ? ((1 << (k)) - 1) : 0);
            } // -∞ => 0xFF800000

            uint32_t minPosInf() const
            {
                return (is_signed == Signedness::Signed) ? (1 << (k - 1)) - 1 : (1 << (k)) - 2;
            } // +∞ => 0x7F800000
        };

        // params for P3109 float
        template <int k, int p, Signedness is_signed = Signedness::Signed, Inf_Behaviors has_inf = Inf_Behaviors::Saturating>
        constexpr FloatingPointParams param_float_p3109(
            k,                                         // totoal bitwidth
            p - 1,                                     // mantissa bits
            (1 << (k - p - 1)),                        // bias
            has_inf,                                   // No infinity
            NaN_Behaviors::QuietNaN,                   // NaN behavior
            is_signed,                                 // It is signed
            P3109_InfChecker<k, is_signed, has_inf>(), // Inf Functor
            P3109_NaNChecker<k, is_signed>()           // NaN Functor
        );

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

            constexpr bool operator()(bits_t bits) const
            {
                return ((bits & float_bits::exp) == float_bits::exp) &&
                    ((bits & float_bits::frac) != 0u);
            }

            static constexpr bits_t qnan() { return float_bits::qnan; }
            static constexpr bits_t snan() { return float_bits::snan; }

            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(float_bits::exp))
                                == xs::batch<bits_t, Arch>(float_bits::exp);
                auto frac_nonzero = (bits & xs::batch<bits_t, Arch>(float_bits::frac))
                                != xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_nonzero;
            }
        };

        template <>
        struct native_NaNChecker<double>
        {
            using bits_t = double_bits::utype;

            constexpr bool operator()(bits_t bits) const
            {
                return ((bits & double_bits::exp) == double_bits::exp) &&
                    ((bits & double_bits::frac) != 0ull);
            }

            static constexpr bits_t qnan() { return double_bits::qnan; }
            static constexpr bits_t snan() { return double_bits::snan; }

            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(double_bits::exp))
                                == xs::batch<bits_t, Arch>(double_bits::exp);
                auto frac_nonzero = (bits & xs::batch<bits_t, Arch>(double_bits::frac))
                                != xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_nonzero;
            }
        };

        // ------------------------
        // Inf checker (float/double)
        // ------------------------
        template <class FP> struct InfChecker;

        template <>
        struct InfChecker<float>
        {
            using bits_t = float_bits::utype;

            constexpr bool operator()(bits_t bits) const
            {
                return ((bits & float_bits::exp) == float_bits::exp) &&
                    ((bits & float_bits::frac) == 0u);
            }

            static constexpr bits_t posinf() { return float_bits::pos_inf; }
            static constexpr bits_t neginf() { return float_bits::neg_inf; }

            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(float_bits::exp))
                                == xs::batch<bits_t, Arch>(float_bits::exp);
                auto frac_zero    = (bits & xs::batch<bits_t, Arch>(float_bits::frac))
                                == xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_zero;
            }
        };

        template <>
        struct InfChecker<double>
        {
            using bits_t = double_bits::utype;

            constexpr bool operator()(bits_t bits) const
            {
                return ((bits & double_bits::exp) == double_bits::exp) &&
                    ((bits & double_bits::frac) == 0ull);
            }

            static constexpr bits_t posinf() { return double_bits::pos_inf; }
            static constexpr bits_t neginf() { return double_bits::neg_inf; }

            template <class Arch>
            xs::batch_bool<bits_t, Arch> sim_check(const xs::batch<bits_t, Arch>& bits) const
            {
                auto exp_all_ones = (bits & xs::batch<bits_t, Arch>(double_bits::exp))
                                == xs::batch<bits_t, Arch>(double_bits::exp);
                auto frac_zero    = (bits & xs::batch<bits_t, Arch>(double_bits::frac))
                                == xs::batch<bits_t, Arch>(bits_t{0});
                return exp_all_ones & frac_zero;
            }
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
    using P3109_float = lo_float_internal::Templated_Float<lo_float_internal::param_float_p3109<k, p, is_signed, has_inf>>;

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
            static constexpr bool has_quiet_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::QuietNaN;
            static constexpr bool has_signaling_NaN = Fp.NA_behavior == lo_float::NaN_Behaviors::SignalingNaN;
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

    void set_seed(unsigned int seedVal)
    {
        // Just forward to the internal function or directly:
        lo_float_internal::mt.seed(seedVal);

        // or lo_float_internal::set_seed_internal(seedVal);
    }
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

        template <FloatingPointParams Fp>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE bool isinf(const Templated_Float<Fp> &a)
        {
            return Fp.IsInf(a.rep()) && Fp.OV_behavior != Inf_Behaviors::Saturating;
        }

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

    template <lo_float::NaNChecker IsNaN, typename Int, class Arch>
    xsimd::batch_bool<Int, Arch>
    isnan_simd(const xsimd::batch<Int, Arch>& a, IsNaN isnan_func)
    {
        return lo_float::lo_float_internal::isnan_simd<IsNaN>(a, isnan_func);
    }

    // isinf override
    template <lo_float::FloatingPointParams Fp>
    bool isinf(const lo_float::Templated_Float<Fp> &a)
    {
        return lo_float::lo_float_internal::isinf(a);
    }


    template <lo_float::InfChecker IsInf, typename Int, bool OV_behavior, class Arch>
    xsimd::batch_bool<Int, Arch>
    isinf_simd(const xsimd::batch<Int, Arch>& a, IsInf inf_func)
    {
        return lo_float::lo_float_internal::isinf_simd<IsInf, OV_behavior>(a, inf_func);
    }

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
            if constexpr (xsimd::is_batch<UInt>::value)
                xw = widen_low_lanes<ret_scalar, uint_scalar>(x);
            else
                xw = static_cast<ret_type>(x);

            ret_type zeroes = ret_type(le_size - 4);

            auto step = [&](int k) {
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
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE Scalar run(const Scalar &from, Rounding_Mode rm = Rounding_Mode::RoundAwayFromZero, int stoch_len = 0)
            {
                return from;
            }

            template <class arch = xsimd::default_arch>
            static LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE void run(const Scalar* from, Scalar* to, int n, Rounding_Mode rm = Rounding_Mode::RoundAwayFromZero, int stoch_len = 0)
            {
                if (to != from) {
                    memcpy(to, from, n * sizeof(Scalar));
                }
                return;
            }
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
            static constexpr int kExponentBias = std::numeric_limits<Float>::kExponentBias;
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
            static LOFLOAT_HOST_DEVICE __attribute__((noinline)) To run(const From &from, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven, int stoch_len = 0)
            {
                // Shift bits to destination type, without sign bit.

                const bool from_sign_bit = (get_signedness_v<From> == Signedness::Unsigned) ? false : std::bit_cast<FromBits>(from) >> (kFromBits - 1);

                if (get_signedness_v<To> == Signedness::Unsigned && from_sign_bit)
                {
                    // set underflow flag

#ifdef ENABLE_EXCEPT
                    f_env.set_exception_flag(LF_exception_flags::Underflow);
#endif
                    if (get_unsigned_behavior_v<To> == Unsigned_behavior::NegtoZero)
                    {
                        return To{};
                    }
                    else
                    {
                        if (get_NaN_Behavior_v<To> == NaN_Behaviors::SignalingNaN)
                        {
                            return std::numeric_limits<To>::signaling_NaN();
                        }
                        else if (get_NaN_Behavior_v<To> == NaN_Behaviors::QuietNaN)
                        {
                            return std::numeric_limits<To>::quiet_NaN();
                        }
                        else
                        {
// trapping NaN - call trap
#ifdef ENABLE_EXCEPT
                            f_env.set_exception_flag(LF_exception_flags::InvalidOperation);
#endif
                            return std::numeric_limits<To>::signaling_NaN();
                        }
                    }
                }

                const FromBits from_bits =
                    std::bit_cast<FromBits>(abs(from));

                // Special values, preserving sign.

                if (std::isinf(from) && get_overflow_behavior_v<To> != Inf_Behaviors::Saturating)
                {
                    return from_sign_bit ? -std::numeric_limits<To>::infinity()
                                         : std::numeric_limits<To>::infinity();
                }
                if (std::isnan(from))
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
                            bits = RoundMantissa(bits, -kDigitShift, round_mode, stoch_len);
                            bits >>= -kDigitShift;
                        }

                        To to = std::bit_cast<To>(static_cast<ToBits>(bits));

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
                            rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift, round_mode, stoch_len);

                            bits = (rounded_from_bits >> exponent_shift);
                        }
                        else
                        {

                            unsigned long long widened_bits = (unsigned long long)(rounded_from_bits);
                            switch (round_mode)
                            {
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
                        // Insert sign and return.
                        return from_sign_bit ? -std::bit_cast<To>(bits) : std::bit_cast<To>(bits);
                    }
                }

                // Round the mantissa if it is shrinking.
                WideBits rounded_from_bits = from_bits;

                if constexpr (kDigitShift < 0)
                {
                    // need some logic to add leading 1 if normalized
                    rounded_from_bits = RoundMantissa(rounded_from_bits, -kDigitShift, round_mode, stoch_len);
                    // Zero-out tail bits.
                    rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
                }

                rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                                     << kFromMantissaBits;

                ToBits bits;
                // Check for overflows by aligning the significands. We always align the
                // narrower significand to the wider significand.
                const WideBits kToHighestRep =
                    std::bit_cast<ToBits>(std::numeric_limits<To>::max());
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

                To to = std::bit_cast<To>(bits);
                if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                             std::numeric_limits<To>::digits) <
                              std::make_pair(std::numeric_limits<From>::max_exponent,
                                             std::numeric_limits<From>::digits))
                {
                    if (rounded_from_bits > aligned_highest)
                    {

                    #ifdef ENABLE_EXCEPT
                            f_env.set_exception_flag(LF_exception_flags::Overflow);
                    #endif
                        if (std::numeric_limits<To>::has_infinity)
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
template <typename WideBitsSIMD, typename SignedWideBitsSIMD, 
          typename WideBits, typename SignedWideBits, typename arch>
__attribute__((noinline))
static WideBitsSIMD handle_expanding_conversion(
    WideBitsSIMD from_bits,
    Rounding_Mode round_mode,
    int stoch_len)
{
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
        result = RoundMantissa(result, WideBitsSIMD(-kDigitShift), round_mode, stoch_len);
        result = result >> WideBitsSIMD(-kDigitShift);
    }
    
    return xs::select(xs::batch_bool<WideBits, arch>(is_zero), 
                     WideBitsSIMD(0), 
                     result);
}

// Branch B helper function
/* register break down-
x0 stores from_bits
v0 stores input_exp_local
v31 store sthe mask


*/



template <typename FromTraits, typename WideBitsSIMD, 
          typename SignedWideBitsSIMD, typename WideBits, typename SignedWideBits, typename arch>
__attribute__((noinline))
static WideBitsSIMD handle_shrinking_conversion(
    WideBitsSIMD from_bits,
    Rounding_Mode round_mode,
    int stoch_len)
{


// Now use it:
auto input_exp = (from_bits >> kFromMantissaBits); 

// Zero-cost reinterpret (no function call, no spill)
auto input_exp_signed = [&]() {
    SignedWideBitsSIMD result;
    result.data = reinterpret_cast<decltype(result.data)>(input_exp.data);
    return result;
}();

auto biased_to_exp = input_exp_signed - SignedWideBitsSIMD(kFromExponentBias) + 
                     SignedWideBitsSIMD(kToExponentBias);

auto is_subnormal = (biased_to_exp <= SignedWideBitsSIMD(0));
auto is_zero_signed = (input_exp_signed == SignedWideBitsSIMD(0));

auto threshold = SignedWideBitsSIMD(kFromMantissaBits + 1);
auto s_exponent_shift = SignedWideBitsSIMD(-kDigitShift) - biased_to_exp +
                        xs::select(is_zero_signed, SignedWideBitsSIMD(0), SignedWideBitsSIMD(1));

auto needs_shift = is_subnormal && (s_exponent_shift <= threshold);
auto becomes_zero = is_subnormal && (s_exponent_shift > threshold);

// Zero-cost reinterpret back to unsigned
auto exponent_shift_unsigned = [&]() {
    WideBitsSIMD result;
    result.data = reinterpret_cast<decltype(result.data)>(s_exponent_shift.data);
    return result;
}();


// Use batch_bool CONSTRUCTOR (not bitwise_cast) to convert mask types
auto is_zero_unsigned = xs::batch_bool<WideBits, arch>(is_zero_signed);

auto leading_one = xs::select(is_zero_unsigned,
                              WideBitsSIMD(0), 
                              WideBitsSIMD(1) << WideBits(kFromMantissaBits));

auto mantissa = (from_bits & WideBitsSIMD(static_cast<WideBits>(FromTraits::kMantissaMask))) | leading_one;

mantissa = RoundMantissa(mantissa, exponent_shift_unsigned, round_mode, stoch_len);

// Use batch_bool CONSTRUCTOR for mask conversion
auto needs_shift_unsigned = xs::batch_bool<WideBits, arch>(needs_shift);
auto subnormal_result = xs::select(needs_shift_unsigned,
                                   mantissa >> exponent_shift_unsigned,
                                   WideBitsSIMD(0));

// Normal path
WideBitsSIMD normal_result;
if constexpr (kDigitShift < 0) {
    auto mod_digitshift = WideBitsSIMD(-kDigitShift);
    normal_result = RoundMantissa(from_bits, mod_digitshift, round_mode, stoch_len);
    normal_result = normal_result & ~((WideBits{1} << mod_digitshift) - 1);
    normal_result = (normal_result + 
                    WideBitsSIMD((static_cast<WideBits>(kExponentOffset) << kFromMantissaBits)))
                    >> mod_digitshift;
} else {
    normal_result = (from_bits + 
                    WideBitsSIMD((static_cast<WideBits>(kExponentOffset) << kFromMantissaBits))) 
                    << WideBitsSIMD(kDigitShift);
}

auto is_subnormal_unsigned = xs::batch_bool<WideBits, arch>(is_subnormal);
auto result = xs::select(is_subnormal_unsigned, 
                        subnormal_result, 
                        normal_result);

auto becomes_zero_unsigned = xs::batch_bool<WideBits, arch>(becomes_zero);
return xs::select(becomes_zero_unsigned, 
                 WideBitsSIMD(0), 
                 result);
                }

// Main run function
template <class arch = xs::default_arch>
static LOFLOAT_HOST LOFLOAT_FORCEINLINE void run(const From* from,
                                                To* to,
                                                int n,
                                                Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven,
                                                int stoch_len = 0)
{
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
  


                // Correct pointer casts (do NOT take &from / &to)
    const FromBits* from_bits_ptr = reinterpret_cast<const FromBits*>(from);
    ToBits* to_bits_ptr           = reinterpret_cast<ToBits*>(to);


    auto load_from_widened = [&](int i) -> WideBitsSIMD {
            using FromBitsSIMD = xs::batch<FromBits, arch>;
            WideBits tmp[step];
            #pragma unroll
            for (int lane = 0; lane < step; ++lane)
            {
                tmp[lane] = static_cast<WideBits>(from_bits_ptr[i + lane]);
            }

            return WideBitsSIMD::load_unaligned(tmp);
    };

    auto store_to_full = [&](int i, const WideBitsSIMD& v) {
        #pragma unroll
        for (int lane = 0; lane < step; ++lane)
        {
            to_bits_ptr[i + lane] = static_cast<ToBits>(v.get(lane));
        }
    };

 int i;
    #ifdef _LOFOPENMP
    #pragma omp parallel for
    #endif
for (i = 0; i <= n - step; i += step)
{
   WideBitsSIMD from_bits_wide = load_from_widened(i);
    
    xs::batch_bool<WideBits, arch> signed_mask_wide =
        is_signed_type
            ? ((from_bits_wide & WideBitsSIMD(kFromSignBit)) != WideBitsSIMD(0))
            : xs::batch_bool<WideBits, arch>(false);
    
    WideBitsSIMD from_bits = from_bits_wide & ~WideBitsSIMD(kFromSignBit);
    
    xs::batch_bool<WideBits, arch> is_zero_from_exp;
    {
        SignedWideBitsSIMD biased_from_exponent = xs::batch_cast<SignedWideBits>(WideBitsSIMD(from_bits >> kFromMantissaBits));    
        xs::batch_bool<SignedWideBits, arch> is_zero_from_exp_signed = (biased_from_exponent == SignedWideBitsSIMD(0));
        is_zero_from_exp = xs::batch_bool<WideBits, arch>(is_zero_from_exp_signed);
    }
    
    WideBitsSIMD bits;
    WideBitsSIMD underflow_bits;
    batch_bool<WideBits, arch> completed_elems;
    batch_bool<WideBits, arch> res_is_zero = batch_bool<WideBits, arch>(false);

    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent)
    {
        WideBitsSIMD normalization_factor = xs::select(
            is_zero_from_exp,
            WideBitsSIMD(0),
            countl_zero<kFromBits, WideBitsSIMD, WideBitsSIMD>(from_bits) -
            (WideBitsSIMD(kFromBits - kFromMantissaBits) + WideBitsSIMD(1)));
        
        SignedWideBitsSIMD biased_exponent =
            SignedWideBitsSIMD(kExponentOffset) - SignedWideBitsSIMD(normalization_factor) + SignedWideBitsSIMD(1);
        
        xs::batch_bool<SignedWideBits, arch> is_lezero_exp = (biased_exponent <= SignedWideBitsSIMD(0));
        normalization_factor = xs::select(
            xs::batch_bool<WideBits, arch>(!is_lezero_exp), 
            WideBitsSIMD(0), 
            normalization_factor);
        
        bits = from_bits;
        
        if constexpr (kExponentOffset < kWideBits)
        {
            bits = xs::select(
                batch_bool<WideBits, arch>(is_lezero_exp) & is_zero_from_exp,
                bits << xs::batch_cast<WideBits>(kExponentOffset),
                bits);
        }
        
        {
            auto aux_bits = bits << normalization_factor;
            aux_bits = aux_bits & ~(WideBitsSIMD(WideBits{1}) << kFromMantissaBits);
            aux_bits = aux_bits | (batch_cast<WideBits>(biased_exponent) << WideBits(kFromMantissaBits));
            bits = xs::select(batch_bool<WideBits, arch>(is_lezero_exp), bits, aux_bits);
        }
        
        if constexpr (kDigitShift > 0)
        {
            bits = bits << WideBitsSIMD(kDigitShift);
        } else {
            bits = RoundMantissa(bits, -IntSIMD(kDigitShift), round_mode, stoch_len);
            bits = bits >> WideBitsSIMD(-kDigitShift);
        }
        
        completed_elems = is_zero_from_exp;
        underflow_bits = bits;
    }
    
    if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent)
    {
        SignedWideBitsSIMD biased_from_exponent = xs::batch_cast<SignedWideBits>(WideBitsSIMD(from_bits >> kFromMantissaBits));
        xs::batch_bool<SignedWideBits, arch> is_zero_from_exp_signed = (biased_from_exponent == SignedWideBitsSIMD(0));
        
        SignedWideBitsSIMD unbiased_exp = biased_from_exponent - SignedWideBitsSIMD(kFromExponentBias);
        SignedWideBitsSIMD biased_to_exp = unbiased_exp + SignedWideBitsSIMD(kToExponentBias);
        
        xs::batch_bool<SignedWideBits, arch> signed_res_is_subnormal_mask_wide =
            (biased_to_exp <= SignedWideBitsSIMD(0));
        xs::batch_bool<WideBits, arch> res_is_subnormal_mask_wide =
            xs::batch_bool<WideBits, arch>(signed_res_is_subnormal_mask_wide);
        
        SignedWideBitsSIMD from_has_leading_one = xs::select(
            is_zero_from_exp_signed,
            SignedWideBitsSIMD(0),
            SignedWideBitsSIMD(1));
        
        SignedWideBitsSIMD s_exponent_shift = SignedWideBitsSIMD(-kDigitShift) -
                                                     biased_to_exp +
                                                     from_has_leading_one;
        
        xs::batch_bool<SignedWideBits, arch> needs_shift = 
            (signed_res_is_subnormal_mask_wide) && (s_exponent_shift <= SignedWideBitsSIMD(kFromMantissaBits + 1));
        
        res_is_zero = batch_bool<WideBits, arch>(
            (signed_res_is_subnormal_mask_wide) && 
            (s_exponent_shift > SignedWideBitsSIMD(kFromMantissaBits + 1)));
        
        WideBitsSIMD exponent_shift = xs::select(
            batch_bool<WideBits, arch>(needs_shift),
            xs::batch_cast<WideBits>(s_exponent_shift),
            WideBitsSIMD(0));
        
        WideBitsSIMD rounded_from_bits =
            (from_bits & WideBitsSIMD(static_cast<WideBits>(FromTraits::kMantissaMask))) |
            (batch_cast<WideBits>(from_has_leading_one) << WideBits(kFromMantissaBits));
        
        rounded_from_bits = RoundMantissa(rounded_from_bits, exponent_shift, round_mode, stoch_len);
        bits = rounded_from_bits >> exponent_shift;
        
        completed_elems = res_is_subnormal_mask_wide;
        underflow_bits = bits;
    }
    
    WideBitsSIMD finite_out;
    {
        WideBitsSIMD rounded_from_bits = from_bits;
        
        if constexpr (kDigitShift < 0) {
            WideBitsSIMD mod_digitshift = xs::select(
                completed_elems,
                WideBitsSIMD(0),
                WideBitsSIMD(-kDigitShift));
            
            rounded_from_bits = RoundMantissa(rounded_from_bits, mod_digitshift, round_mode, stoch_len);
            rounded_from_bits = rounded_from_bits & ~((WideBits{1} << (mod_digitshift)) - 1);
        }
        
        rounded_from_bits = rounded_from_bits + xs::select(
            completed_elems,
            WideBitsSIMD(0),
            WideBitsSIMD((batch_cast<WideBits>(kExponentOffset) << kFromMantissaBits)));
        
        WideBitsSIMD kToHighestRep_simd =
            WideBitsSIMD(static_cast<WideBits>(std::bit_cast<ToBits>(std::numeric_limits<To>::max())));
        WideBitsSIMD aligned_highest = kToHighestRep_simd;
        
        if constexpr (kDigitShift < 0)
        {
            aligned_highest = aligned_highest << WideBitsSIMD(-kDigitShift);
            WideBitsSIMD mod_digitshift = xs::select(
                completed_elems,
                WideBitsSIMD(0),
                WideBitsSIMD(-kDigitShift));
            bits = rounded_from_bits >> mod_digitshift;
        } else {
            bits = rounded_from_bits << WideBitsSIMD(kDigitShift);
        }
        
        finite_out = xs::select(completed_elems, underflow_bits, bits);
        finite_out = xs::select(res_is_zero, WideBitsSIMD(0), finite_out);
    }
    
    finite_out = xs::select(
        signed_mask_wide,
        finite_out | WideBitsSIMD(kToSignBit),
        finite_out & ~WideBitsSIMD(kToSignBit));
    
    store_to_full(i, finite_out);
}
}
                         };

        template <typename Derived, typename UnderlyingType>
        template <typename From>
        Derived lo_float_base<Derived, UnderlyingType>::ConvertFrom(const From &from, Rounding_Mode rm, int stoch_len)
        {
            return ConvertImpl<From, Derived>::run(from, rm, stoch_len);
        }

        template <typename Derived, typename UnderlyingType>
        template <typename To>
        To lo_float_base<Derived, UnderlyingType>::ConvertTo(const Derived &from, Rounding_Mode rm, int stoch_len)
        {
            return ConvertImpl<Derived, To>::run(from, rm, stoch_len);
        }

        template <typename out, typename in>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Project(in x, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<in, out>::run(x, rm, stoch_len);
        }

        template <typename out, typename in>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Round(in* x, out* y, int n, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<in, out>::run(x, y, n, rm, stoch_len);
        }

        template <typename out, typename in1, typename in2>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out add(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<float, out>::run(static_cast<float>(x) + static_cast<float>(y), rm, stoch_len);
        }

        template <typename out, typename in1, typename in2>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out sub(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<float, out>::run(static_cast<float>(x) - static_cast<float>(y), rm, stoch_len);
        }

        template <typename out, typename in1, typename in2>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out mul(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<float, out>::run(static_cast<float>(x) * static_cast<float>(y), rm, stoch_len);
        }

        template <typename out, typename in1, typename in2>
        constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out div(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
        {
            return ConvertImpl<float, out>::run(static_cast<float>(x) / static_cast<float>(y), rm, stoch_len);
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
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Project(in x, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::Project<out, in>(x,rm, stoch_len);
    }

    template <typename out, typename in>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out Round(in x, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::Project<out, in>(x,rm, stoch_len);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out add(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::add<out, in1, in2>(x, y, rm, stoch_len);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out sub(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::sub<out, in1, in2>(x, y, rm, stoch_len);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out mul(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::mul<out, in1, in2>(x, y, rm, stoch_len);
    }

    template <typename out, typename in1, typename in2>
    constexpr LOFLOAT_HOST_DEVICE LOFLOAT_FORCEINLINE out div(in1 x, in2 y, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        return lo_float_internal::div<in1, in2, out>(x, y, rm, stoch_len);
    }

    template <typename out, typename in, int unroll_len = 0, class arch = xsimd::default_arch>
    LOFLOAT_HOST void Project(in* x, out *y, int n, Rounding_Mode rm = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) noexcept
    {
        using Converter = lo_float_internal::ConvertImpl<in, out>;
        Converter::template run<arch>(x, y, n, rm, stoch_len);
        return;
    }

 template <typename Out, typename In1, typename In2, typename In3, class arch = xsimd::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE
void fma_vec(const In1* LOFLOAT_RESTRICT x,
            const In2* LOFLOAT_RESTRICT y,
            const In3* LOFLOAT_RESTRICT z,
            Out* LOFLOAT_RESTRICT out,
            int n,
            Rounding_Mode rm = Rounding_Mode::RoundToNearestEven,
            int stoch_len = 0) noexcept
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
            out[i + lane] = lo_float::Round<Out>(result_fp32[lane], rm, stoch_len);
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
        out[i] = lo_float::Round<Out>(result, rm, stoch_len);
    }
}





template <typename TA, typename TX, typename TY, class arch = xsimd::default_arch>
LOFLOAT_HOST LOFLOAT_FORCEINLINE
void axpy(const int n,
          const TA* LOFLOAT_RESTRICT a,
          const TX* LOFLOAT_RESTRICT x,
          const int incx,
          TY* LOFLOAT_RESTRICT y,
          const int incy,
          Rounding_Mode rm = Rounding_Mode::RoundToNearestEven,
          int stoch_len = 0) noexcept
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
                y[i + lane] = lo_float::Round<TY>(out_fp32[lane], rm, stoch_len);
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
        y[i] = lo_float::Round<TY>(result, rm, stoch_len);
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
        y[iy] = lo_float::Round<TY>(result, rm, stoch_len);
        ix += incx;
        iy += incy;
    }
}

}




    template <typename T>
    concept Float = lo_float::is_floating_point_v<T>;

    template <typename T>
    concept Int = lo_float::is_integral_v<T>;

} // namespace lo_float

#endif // FLOAT_6_4

