// ============================================================================
// test_virtual_mx_round_cuda.cu  --  GPU tests for lo_float::virtual_mx_round
// (the MX block-quantization kernel in src/mx_round.cuh).
//
// Checks the GPU result BIT-EXACT against a host (CPU) oracle that recomputes
// the same MX algorithm with the trusted scalar virtual_round — i.e. the exact
// same check the CPU test (test_virtual_mx_round.cpp) makes, now across the
// device. Deterministic rounding must agree bit-for-bit on CPU and GPU
// (NUMERICAL_TESTING.md rule 8).
//
//   per block:  amax  = max|x|
//               scale = virtual_round(amax/priv_max_normal, params_pub, rm_pub)
//               x    := virtual_round(x/scale, params_priv, rm_priv) * scal
// e
//
// Coverage: all THREE kernel code paths are exercised by choosing
// (num_threads_per_cta, MX, NX) so block_size = MX*NX lands in each case:
//   CASE 1 warp   block_size <= 32
//   CASE 2 cta    32 < block_size <= num_threads_per_cta
//   CASE 3 cub    block_size  > num_threads_per_cta
// plus several array lengths (multiples and non-multiples of block_size),
// private/scale formats, and rounding modes.
//
// Build:  USE_CUDA=1 make test_virtual_mx_round_cuda
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "lo_float.h"
#include "mx_round.cuh"

using namespace lo_float;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} } while(0)

using Pub  = FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>;
using Priv = FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>;

// Inf/NaN checkers are unused by virtual_round's body; zero-fill them.
static const DeviceInfChecker kInf{0,0,0,0};
static const DeviceNaNChecker kNaN{0,0,0,0};

static Pub make_params(int bw, int mb, int bias, Signedness sgn,
                       Inf_Behaviors ov = Inf_Behaviors::Saturating) {
    return Pub(bw, mb, bias, ov, NaN_Behaviors::_754, sgn, kInf, kNaN);
}

static double priv_max_normal_d(const Priv& p) {
    const int me = p.is_signed == Signedness::Signed
        ? (1 << (p.bitwidth - p.mantissa_bits - 1)) - 1 - p.bias
        : (1 << (p.bitwidth - p.mantissa_bits))     - 1 - p.bias;
    return std::ldexp(1.0, me) * (2.0 - std::ldexp(1.0, -p.mantissa_bits));
}

static bool bit_eq(float a, float b) {
    uint32_t x,y; std::memcpy(&x,&a,4); std::memcpy(&y,&b,4);
    return x == y || (std::isnan(a) && std::isnan(b));
}

static float rnd_wide() {
    float u = ((float)std::rand() + 1.0f) / ((float)RAND_MAX + 2.0f);  // (0,1), never 0
    int   e = (std::rand() % 21) - 10;
    float s = (std::rand() & 1) ? 1.0f : -1.0f;
    return s * std::ldexp(u, e);
}

// Host oracle for one block (uses the trusted scalar virtual_round).
static void oracle_block(const float* in, float* out, int len,
                         const Pub& ppub, const Priv& ppriv,
                         Rounding_Mode rm_pub, Rounding_Mode rm_priv) {
    double amax = 0.0;
    for (int i = 0; i < len; ++i) amax = std::max(amax, std::fabs((double)in[i]));
    float scale = lo_float::virtual_round((float)(amax / priv_max_normal_d(ppriv)), ppub, ProjSpec{rm_pub});
    if (!(scale > 0.0f)) { for (int i = 0; i < len; ++i) out[i] = 0.0f; return; }
    for (int i = 0; i < len; ++i)
        out[i] = lo_float::virtual_round(in[i] / scale, ppriv, ProjSpec{rm_priv}) * scale;
}

// Launch one config (compile-time NTPC/MX/NX) and compare GPU vs CPU oracle.
template<int NTPC, int MX, int NX>
static int test_config(const char* tag, const Pub& ppub, const Priv& ppriv,
                       Rounding_Mode rm_pub, Rounding_Mode rm_priv) {
    constexpr int bs = MX * NX;
    int err = 0;
    const int lens[] = { bs, bs*3, bs*3 + 1, 1000, 4096, 5003 };

    for (int n : lens) {
        std::vector<float> h_in(n), h_gpu(n), h_exp(n);
        for (auto& x : h_in) x = rnd_wide();

        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h_in.data(), n*sizeof(float), cudaMemcpyHostToDevice));

        const int64_t num_blocks = (n + bs - 1) / bs;
        int grid = (int)std::min<int64_t>(65535, std::max<int64_t>(1, num_blocks));
        virtual_mx_round<float, NTPC, MX, NX><<<grid, NTPC>>>(
            d, n, nullptr, ppub, ppriv, ProjSpec{rm_pub}, ProjSpec{rm_priv});
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));

        for (int b0 = 0; b0 < n; b0 += bs)
            oracle_block(h_in.data()+b0, h_exp.data()+b0, std::min(bs, n-b0),
                         ppub, ppriv, rm_pub, rm_priv);

        for (int i = 0; i < n; ++i)
            if (!bit_eq(h_gpu[i], h_exp[i])) {
                if (++err <= 6)
                    printf("  [%s n=%d] mismatch @%d in=%g gpu=%g cpu=%g\n",
                           tag, n, i, h_in[i], h_gpu[i], h_exp[i]);
            }
    }
    printf("  config %-26s bs=%-4d NTPC=%-4d : %s\n", tag, bs, NTPC, err ? "FAIL" : "ok");
    return err;
}

int main() {
    std::srand(1234);
    int total = 0;

    const Pub  e8m0  = make_params(8, 0, 127, Signedness::Unsigned, Inf_Behaviors::Extended);
    const Pub  e5m2s = make_params(8, 2, 15,  Signedness::Signed);   // alt scale format
    const Priv e4m3  = make_params(8, 3, 7,   Signedness::Signed);
    const Priv e5m2  = make_params(8, 2, 15,  Signedness::Signed);

    const Rounding_Mode RNE = Rounding_Mode::RoundToNearestEven;

    printf("== CASE 1 (warp, block_size<=32) ==\n");
    total += test_config<256, 1, 32>("warp e4m3/e8m0",  e8m0,  e4m3, RNE, RNE);
    total += test_config<256, 1,  8>("warp e4m3/e8m0",  e8m0,  e4m3, RNE, Rounding_Mode::RoundDown);
    total += test_config<256, 1, 16>("warp e5m2/e8m0",  e8m0,  e5m2, RNE, Rounding_Mode::RoundUp);

    printf("== CASE 2 (cta, 32<block_size<=NTPC) ==\n");
    total += test_config<256, 4, 32>("cta  e4m3/e8m0",  e8m0,  e4m3, RNE, RNE);
    total += test_config<256, 2, 32>("cta  e4m3/e5m2sc", e5m2s, e4m3, RNE, Rounding_Mode::RoundTowardsZero);
    total += test_config<128, 4, 32>("cta  e5m2/e8m0",  e8m0,  e5m2, RNE, RNE);

    printf("== CASE 3 (cub, block_size>NTPC) ==\n");
    total += test_config<64,  4, 32>("cub  e4m3/e8m0",  e8m0,  e4m3, RNE, RNE);
    total += test_config<128, 8, 32>("cub  e4m3/e8m0",  e8m0,  e4m3, RNE, Rounding_Mode::RoundDown);
    total += test_config<64,  8, 32>("cub  e5m2/e8m0",  e8m0,  e5m2, RNE, RNE);

    
    // All-zero block must round to zero (scale==0 guard), not NaN.
    {
        const int bs = 32, n = 64;            // block 0 = zeros, block 1 = nonzero
        std::vector<float> h(n, 0.0f);
        for (int i = bs; i < n; ++i) h[i] = rnd_wide();
        float* d; CUDA_CHECK(cudaMalloc(&d, n*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
        virtual_mx_round<float,256,1,32><<<1,256>>>(d, n, nullptr, e8m0, e4m3, ProjSpec{RNE}, ProjSpec{RNE});
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> g(n);
        CUDA_CHECK(cudaMemcpy(g.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
        bool ok = true;
        for (int i = 0; i < bs; ++i) if (g[i] != 0.0f) ok = false;          // zero block -> 0
        for (int i = 0; i < n; ++i)  if (!std::isfinite(g[i])) ok = false;   // no NaN/Inf
        printf("  all-zero block -> zero (no NaN) : %s\n", ok ? "ok" : "FAIL");
        if (!ok) ++total;
    }

    printf("\n=== TOTAL ERRORS: %d %s ===\n", total, total ? "(FAIL)" : "(ALL PASS)");
    return total ? 1 : 0;
}
