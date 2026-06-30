// -------------------------------------------------------------
//  test_pack.cpp  ―  CPU tests for lo_pack.hpp bit-packing
// -------------------------------------------------------------
//  Per loop/NUMERICAL_TESTING.md:
//   * The runtime float<->rep codec twins are validated BIT-EXACT against the
//     compile-time Templated_Float<Fp> path, which is the ground-truth oracle
//     (§1 oracle, §2 exact round-trip, §7 exhaustive small formats):
//       decode:  convert_rep_to_float(rep, Fp)  == (float)TF::FromRep(rep)   (all 2^n reps)
//       encode:  convert_float_to_rep(x, Fp)    == TF(x).rep()               (probe floats:
//                exact grid values, midpoints/ties, perturbations, overflow, tiny)
//   * Packing itself is a lossless bijection on reps: encode->decode reproduces
//     every rep identically; byte counts are checked against an independent lcm
//     oracle; both endianness orders and partial (zero-padded) groups exercised.
//   * §11 0 errors = pass, loud on failure.
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "lo_float.h"
#include "lo_pack.hpp"

using namespace lo_float;

// independent oracle for the packed-byte geometry
static int oracle_gcd(int a, int b) { return b == 0 ? a : oracle_gcd(b, a % b); }
static int oracle_lcm(int a, int b) { return (a / oracle_gcd(a, b)) * b; }
static int oracle_group_bytes(int n) { int al = (n <= 8) ? 8 : 16; return oracle_lcm(n, al) / 8; }
static int oracle_group_count(int n) { int al = (n <= 8) ? 8 : 16; return oracle_lcm(n, al) / n; }
static size_t oracle_packed_bytes(int n, size_t c) {
    int gc = oracle_group_count(n);
    return ((c + gc - 1) / gc) * oracle_group_bytes(n);
}

static bool bit_eq_f(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// ---- (A) decode twin vs oracle: exhaustive over all reps ----
template <FloatingPointParams Fp>
int test_decode_codec(const char* tag) {
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    const size_t N = (size_t)1u << n;
    int errors = 0;
    for (size_t r = 0; r < N; ++r) {
        float got = lo_pack_internal::convert_rep_to_float((uint32_t)r, Fp);
        float exp = (float)TF::FromRep((UType)r);
        if (!bit_eq_f(got, exp)) {
            if (errors < 8)
                std::cout << "[" << tag << "] FAIL decode rep=" << r
                          << " got=" << got << " oracle=" << exp << "\n";
            ++errors;
        }
    }
    return errors;
}

// ---- (B) encode twin vs oracle: probe floats derived from the grid ----
template <FloatingPointParams Fp>
int test_encode_codec(const char* tag) {
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    const size_t N = (size_t)1u << n;
    int errors = 0;

    auto check = [&](float x) {
        uint32_t got = lo_pack_internal::convert_float_to_rep(x, Fp);
        uint32_t exp = (uint32_t)TF(x).rep();
        // NaN reps are not unique; require the DUT to agree the result is NaN.
        if (Fp.IsNaN(exp)) { if (!Fp.IsNaN(got)) { ++errors; } return; }
        if (got != exp) {
            if (errors < 12)
                std::cout << "[" << tag << "] FAIL encode x=" << x
                          << " got=" << got << " oracle=" << exp << "\n";
            ++errors;
        }
    };

    // every representable grid value + its two fp32 neighbours (perturbations)
    std::vector<float> fin;
    fin.reserve(N);
    for (size_t r = 0; r < N; ++r) {
        float x = (float)TF::FromRep((UType)r);
        if (std::isnan(x) || std::isinf(x)) continue;
        check(x);
        check(std::nextafterf(x, +INFINITY));
        check(std::nextafterf(x, -INFINITY));
        fin.push_back(x);
    }
    // exact midpoints between adjacent representable values -> real RNE/RNO/ties cases
    std::sort(fin.begin(), fin.end());
    for (size_t i = 1; i < fin.size(); ++i)
        if (fin[i] != fin[i - 1]) check(0.5f * (fin[i] + fin[i - 1]));
    // explicit extras: zero, signed zero, overflow, deep underflow, float subnormal
    check(0.0f); check(-0.0f);
    check(1e30f); check(-1e30f); check(INFINITY); check(-INFINITY);
    check(1e-40f); check(-1e-40f);            // float subnormal -> branch 1
    check(std::numeric_limits<float>::denorm_min());
    return errors;
}

// ---- (C) packing round-trip via the public API ----
template <FloatingPointParams Fp, PackEndian E>
int test_typed_roundtrip(const char* tag) {
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    const size_t N = (size_t)1u << n;
    int errors = 0;

    std::vector<TF> src(N), dec(N);
    for (size_t r = 0; r < N; ++r) src[r] = TF::FromRep((UType)r);

    std::vector<uint8_t> buf(LoF_packed_size(n, N), 0xAB);
    int wrote = LoF_encode<Fp, E>(buf.data(), src.data(), N);
    if ((size_t)wrote != oracle_packed_bytes(n, N)) {
        std::cout << "[" << tag << "] FAIL typed byte count got " << wrote
                  << " expected " << oracle_packed_bytes(n, N) << "\n"; ++errors;
    }
    int read = LoF_decode<Fp, E>(dec.data(), buf.data(), N);
    if (read != wrote) { std::cout << "[" << tag << "] FAIL consumed != wrote\n"; ++errors; }
    for (size_t r = 0; r < N; ++r)
        if (dec[r].rep() != src[r].rep()) {
            if (errors < 8) std::cout << "[" << tag << "] FAIL typed rep[" << r << "]\n";
            ++errors;
        }
    return errors;
}

// float-source API: round->pack->unpack; decoded == oracle (float)TF(x)
template <FloatingPointParams Fp, PackEndian E>
int test_float_roundtrip(const char* tag) {
    using TF = lo_float_internal::Templated_Float<Fp>;
    constexpr int n = Fp.bitwidth;
    int errors = 0;

    std::vector<float> in;
    for (double v = -8.0; v <= 8.0; v += 0.125) in.push_back((float)v);
    in.push_back(0.0f); in.push_back(-0.0f); in.push_back(1e30f); in.push_back(-1e30f);
    const size_t count = in.size();

    std::vector<float> expect(count), dec(count);
    for (size_t i = 0; i < count; ++i) expect[i] = (float)TF(in[i]);  // oracle: round then read back

    std::vector<uint8_t> buf(LoF_packed_size(n, count));
    int wrote = LoF_encode<E>(buf.data(), in.data(), Fp, count);
    if ((size_t)wrote != oracle_packed_bytes(n, count)) {
        std::cout << "[" << tag << "] FAIL float byte count\n"; ++errors;
    }
    LoF_decode<E>(dec.data(), buf.data(), Fp, count);
    for (size_t i = 0; i < count; ++i)
        if (!bit_eq_f(dec[i], expect[i])) {
            if (errors < 8)
                std::cout << "[" << tag << "] FAIL float[" << i << "] in=" << in[i]
                          << " got=" << dec[i] << " expected=" << expect[i] << "\n";
            ++errors;
        }
    return errors;
}

template <FloatingPointParams Fp>
int test_endianness_differs(const char* tag) {
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    const size_t N = (size_t)1u << n;
    int errors = 0;
    std::vector<TF> src(N);
    for (size_t r = 0; r < N; ++r) src[r] = TF::FromRep((UType)r);
    std::vector<uint8_t> fwd(LoF_packed_size(n, N)), rev(LoF_packed_size(n, N));
    LoF_encode<Fp, PackEndian::Forward>(fwd.data(), src.data(), N);
    LoF_encode<Fp, PackEndian::Reverse>(rev.data(), src.data(), N);
    if (oracle_group_count(n) > 1 && fwd == rev) {
        std::cout << "[" << tag << "] FAIL endianness produced identical bytes\n"; ++errors;
    }
    if (oracle_group_count(n) == 1 && fwd != rev) {
        std::cout << "[" << tag << "] FAIL single-element group not endian-invariant\n"; ++errors;
    }
    return errors;
}

template <FloatingPointParams Fp, PackEndian E>
int test_partial(const char* tag) {
    using TF    = lo_float_internal::Templated_Float<Fp>;
    using UType = lo_float_internal::Base_repr_select<Fp.bitwidth>;
    constexpr int n = Fp.bitwidth;
    int errors = 0;
    int gc = oracle_group_count(n);
    size_t count = (gc > 1) ? (size_t)(2 * gc + 1) : 5;
    std::vector<TF> src(count), dec(count);
    for (size_t i = 0; i < count; ++i) src[i] = TF::FromRep((UType)((i * 7 + 1) & ((1u << n) - 1u)));
    std::vector<uint8_t> buf(LoF_packed_size(n, count));
    int wrote = LoF_encode<Fp, E>(buf.data(), src.data(), count);
    if ((size_t)wrote != oracle_packed_bytes(n, count)) {
        std::cout << "[" << tag << "] FAIL partial byte count\n"; ++errors;
    }
    LoF_decode<Fp, E>(dec.data(), buf.data(), count);
    for (size_t i = 0; i < count; ++i)
        if (dec[i].rep() != src[i].rep()) { if (errors < 8) std::cout << "[" << tag << "] FAIL partial[" << i << "]\n"; ++errors; }
    return errors;
}

template <FloatingPointParams Fp>
int test_format(const char* tag) {
    int e = 0;
    e += test_decode_codec<Fp>(tag);
    e += test_encode_codec<Fp>(tag);
    e += test_typed_roundtrip<Fp, PackEndian::Forward>(tag);
    e += test_typed_roundtrip<Fp, PackEndian::Reverse>(tag);
    e += test_float_roundtrip<Fp, PackEndian::Forward>(tag);
    e += test_float_roundtrip<Fp, PackEndian::Reverse>(tag);
    e += test_endianness_differs<Fp>(tag);
    e += test_partial<Fp, PackEndian::Forward>(tag);
    if (e == 0) std::cout << "[" << tag << "] ok\n";
    return e;
}

int main() {
    namespace I = lo_float::lo_float_internal;
    int errors = 0;

    // 8-bit-aligned path (widths 1..8): odd widths + named presets
    errors += test_format<I::param_float_p_3109<3, 2>>("p3109<3,2>");
    errors += test_format<I::param_float4_e2m1>("e2m1(4)");
    errors += test_format<I::param_float_p_3109<5, 2>>("p3109<5,2>");
    errors += test_format<I::param_float6_e3m2>("e3m2(6)");
    errors += test_format<I::param_float_p_3109<7, 3>>("p3109<7,3>");
    errors += test_format<I::param_float8_e5m2>("e5m2(8)");
    errors += test_format<I::param_float8_e4m3fn>("e4m3(8)");
    // 16-bit-aligned path (widths 9..16)
    errors += test_format<I::param_float_p_3109<9, 3>>("p3109<9,3>");
    errors += test_format<I::param_float_p_3109<11, 5>>("p3109<11,5>");
    errors += test_format<I::param_float_p_3109<12, 8>>("p3109<12,8>");
    errors += test_format<halfPrecisionParams>("half(16)");

    if (errors == 0) std::cout << "ALL PACK TESTS PASSED\n";
    else             std::cout << "PACK TESTS FAILED: " << errors << " errors\n";
    return errors == 0 ? 0 : 1;
}
