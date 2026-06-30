// -------------------------------------------------------------
//  test_pack_cuda.cu  ―  GPU tests for lo_pack.hpp bit-packing
// -------------------------------------------------------------
//  Per loop/NUMERICAL_TESTING.md §8: a deterministic path must agree
//  BIT-EXACT between CPU and GPU. We run the runtime float<->rep pack/unpack on
//  the device (LoF_encode_cuda / LoF_decode_cuda, FloatingPointParams passed as
//  a runtime kernel argument) and compare the device-produced packed bytes and
//  decoded floats byte-for-byte / bit-for-bit against the host (CPU) path.
//  0 errors = pass.
// -------------------------------------------------------------
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "lo_float.h"
#include "lo_pack.hpp"

using namespace lo_float;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cout << "CUDA error " << cudaGetErrorString(e) << " at " << __LINE__ << "\n"; \
    std::exit(2); } } while (0)

static bool bit_eq_f(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// Run float-source pack+unpack on the device; compare bit-exact to the host path.
template <FloatingPointParams Fp, PackEndian E>
int gpu_float_roundtrip(const char* tag) {
    const int n = Fp.bitwidth;
    int errors = 0;

    std::vector<float> in;
    for (double v = -8.0; v <= 8.0; v += 0.0625) in.push_back((float)v);
    in.push_back(0.0f); in.push_back(-0.0f);
    in.push_back(1e30f); in.push_back(-1e30f);
    const size_t count = in.size();
    const size_t nbytes = LoF_packed_size(n, count);

    // host reference
    std::vector<uint8_t> h_packed(nbytes);
    std::vector<float>   h_dec(count);
    LoF_encode<E>(h_packed.data(), in.data(), Fp, count);
    LoF_decode<E>(h_dec.data(), h_packed.data(), Fp, count);

    // device
    float*   d_in;  uint8_t* d_packed;  float* d_dec;
    CUDA_CHECK(cudaMalloc(&d_in,     count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_packed, nbytes));
    CUDA_CHECK(cudaMalloc(&d_dec,    count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    int wrote = LoF_encode_cuda<E>(d_packed, d_in, Fp, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    int read  = LoF_decode_cuda<E>(d_dec, d_packed, Fp, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> g_packed(nbytes);
    std::vector<float>   g_dec(count);
    CUDA_CHECK(cudaMemcpy(g_packed.data(), d_packed, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_dec.data(),    d_dec,    count * sizeof(float), cudaMemcpyDeviceToHost));

    if ((size_t)wrote != nbytes || (size_t)read != nbytes) {
        std::cout << "[" << tag << "] FAIL gpu byte count wrote=" << wrote
                  << " read=" << read << " expect=" << nbytes << "\n"; ++errors;
    }
    for (size_t b = 0; b < nbytes; ++b)
        if (g_packed[b] != h_packed[b]) {
            if (errors < 8)
                std::cout << "[" << tag << "] FAIL packed byte[" << b << "] gpu="
                          << (int)g_packed[b] << " cpu=" << (int)h_packed[b] << "\n";
            ++errors;
        }
    for (size_t i = 0; i < count; ++i)
        if (!bit_eq_f(g_dec[i], h_dec[i])) {
            if (errors < 8)
                std::cout << "[" << tag << "] FAIL dec[" << i << "] gpu="
                          << g_dec[i] << " cpu=" << h_dec[i] << "\n";
            ++errors;
        }

    cudaFree(d_in); cudaFree(d_packed); cudaFree(d_dec);
    return errors;
}

template <FloatingPointParams Fp>
int test_format(const char* tag) {
    int e = 0;
    e += gpu_float_roundtrip<Fp, PackEndian::Forward>(tag);
    e += gpu_float_roundtrip<Fp, PackEndian::Reverse>(tag);
    if (e == 0) std::cout << "[" << tag << "] ok (CPU==GPU)\n";
    return e;
}

int main() {
    namespace I = lo_float::lo_float_internal;
    int errors = 0;

    errors += test_format<I::param_float_p_3109<3, 2>>("p3109<3,2>");
    errors += test_format<I::param_float4_e2m1>("e2m1(4)");
    errors += test_format<I::param_float_p_3109<5, 2>>("p3109<5,2>");
    errors += test_format<I::param_float6_e3m2>("e3m2(6)");
    errors += test_format<I::param_float_p_3109<7, 3>>("p3109<7,3>");
    errors += test_format<I::param_float8_e5m2>("e5m2(8)");
    errors += test_format<I::param_float8_e4m3fn>("e4m3(8)");
    errors += test_format<I::param_float_p_3109<9, 3>>("p3109<9,3>");
    errors += test_format<I::param_float_p_3109<12, 8>>("p3109<12,8>");
    errors += test_format<halfPrecisionParams>("half(16)");

    if (errors == 0) std::cout << "ALL PACK GPU TESTS PASSED\n";
    else             std::cout << "PACK GPU TESTS FAILED: " << errors << " errors\n";
    return errors == 0 ? 0 : 1;
}
