#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "platform_macros.h"

static constexpr int N = 1024;

// round_to_odd_fma: same semantics as in cutlass_gemms.cuh but self-contained
// here so we don't drag in the full CUTLASS dependency for this unit test.
template <typename From>
LOFLOAT_DEVICE LOFLOAT_FORCEINLINE From round_to_odd_fma(From a, From b, From c) {
    if constexpr (std::is_same_v<From, float>) {
        float rd = __fmaf_rd(a, b, c);
        float ru = __fmaf_ru(a, b, c);
        uint32_t inexact = (ru != rd) ? 1u : 0u;
        float rz = (rd >= 0.0f) ? rd : ru;
        return __uint_as_float(__float_as_uint(rz) | inexact);
    } else {
        double rd = __fma_rd(a, b, c);
        double ru = __fma_ru(a, b, c);
        long long inexact = (ru != rd) ? 1LL : 0LL;
        double rz = (rd >= 0.0) ? rd : ru;
        return __longlong_as_double(__double_as_longlong(rz) | inexact);
    }
}

// Each thread computes one element.
// true_fma[i]  = fmaf(a[i], b[i], c[i])  (hardware single-precision FMA)
// sim_fma[i]   = (float) round_to_odd_fma in double, then cast to float.
// The two must be bitwise equal (no double-rounding error).
__global__ void rto_test_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ true_fma,
    float* __restrict__ sim_fma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        true_fma[i] = __fmaf_rn(a[i], b[i], c[i]);
        sim_fma[i]  = (float)round_to_odd_fma((double)a[i], (double)b[i], (double)c[i]);
    }
}

// Half-precision case:
// true_fma[i]  = __hfma(a[i], b[i], c[i])   (hardware FP16 FMA)
// sim_fma[i]   = __float2half_rn(round_to_odd_fma<float>(...))
//                  step 1: FP32 FMA with round-to-odd
//                  step 2: round the FP32 result to FP16
// The two must be bitwise equal.
__global__ void rto_half_test_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const __half* __restrict__ c,
    __half* __restrict__ true_fma,
    __half* __restrict__ sim_fma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        true_fma[i] = __hfma(a[i], b[i], c[i]);
        float rto   = round_to_odd_fma((float)a[i], (float)b[i], (float)c[i]);
        sim_fma[i]  = __float2half_rn(rto);
    }
}

int main() {
    float ha[N], hb[N], hc[N], h_true[N], h_sim[N];
    srand(12345);
    for (int i = 0; i < N; i++) {
        ha[i] = (float)((double)rand() / RAND_MAX);
        hb[i] = (float)((double)rand() / RAND_MAX);
        hc[i] = (float)((double)rand() / RAND_MAX);
    }

    float *da, *db, *dc, *d_true, *d_sim;
    cudaMalloc(&da,     N * sizeof(float));
    cudaMalloc(&db,     N * sizeof(float));
    cudaMalloc(&dc,     N * sizeof(float));
    cudaMalloc(&d_true, N * sizeof(float));
    cudaMalloc(&d_sim,  N * sizeof(float));

    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, N * sizeof(float), cudaMemcpyHostToDevice);

    rto_test_kernel<<<1, N>>>(da, db, dc, d_true, d_sim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_true, d_true, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sim,  d_sim,  N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 4 elements:\n");
    for (int i = 0; i < 4; i++) {
        uint32_t t, s;
        memcpy(&t, &h_true[i], 4);
        memcpy(&s, &h_sim[i],  4);
        printf("  [%d] a=%.8e b=%.8e c=%.8e  true=%.8e(0x%08x)  sim=%.8e(0x%08x)\n",
               i, ha[i], hb[i], hc[i], h_true[i], t, h_sim[i], s);
    }

    int all_match = 1;
    for (int i = 0; i < N; i++) {
        uint32_t t, s;
        memcpy(&t, &h_true[i], 4);
        memcpy(&s, &h_sim[i],  4);
        if (t != s) {
            printf("MISMATCH at i=%d: true=0x%08x sim=0x%08x\n", i, t, s);
            all_match = 0;
        }
    }
    printf("%s (float): %d/%d elements bitwise equal.\n",
           all_match ? "PASS" : "FAIL", all_match ? N : 0, N);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cudaFree(d_true); cudaFree(d_sim);

    // ── Half-precision test ───────────────────────────────────────────────
    // Inputs are random FP16 values in [-1, 1].

    __half ha_h[N], hb_h[N], hc_h[N], h_true_h[N], h_sim_h[N];
    for (int i = 0; i < N; i++) {
        ha_h[i] = __float2half((float)((double)rand() / RAND_MAX) * 2.0f - 1.0f);
        hb_h[i] = __float2half((float)((double)rand() / RAND_MAX) * 2.0f - 1.0f);
        hc_h[i] = __float2half((float)((double)rand() / RAND_MAX) * 2.0f - 1.0f);
    }

    __half *dah, *dbh, *dch, *d_true_h, *d_sim_h;
    cudaMalloc(&dah,      N * sizeof(__half));
    cudaMalloc(&dbh,      N * sizeof(__half));
    cudaMalloc(&dch,      N * sizeof(__half));
    cudaMalloc(&d_true_h, N * sizeof(__half));
    cudaMalloc(&d_sim_h,  N * sizeof(__half));

    cudaMemcpy(dah, ha_h, N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dbh, hb_h, N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dch, hc_h, N * sizeof(__half), cudaMemcpyHostToDevice);

    rto_half_test_kernel<<<1, N>>>(dah, dbh, dch, d_true_h, d_sim_h);
    cudaDeviceSynchronize();

    cudaMemcpy(h_true_h, d_true_h, N * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sim_h,  d_sim_h,  N * sizeof(__half), cudaMemcpyDeviceToHost);

    printf("\nFirst 4 elements (half):\n");
    for (int i = 0; i < 4; i++) {
        uint16_t t, s;
        memcpy(&t, &h_true_h[i], 2);
        memcpy(&s, &h_sim_h[i],  2);
        printf("  [%d] a=%.4e b=%.4e c=%.4e  true=%.4e(0x%04x)  sim=%.4e(0x%04x)\n",
               i,
               (float)ha_h[i], (float)hb_h[i], (float)hc_h[i],
               (float)h_true_h[i], t, (float)h_sim_h[i], s);
    }

    int all_match_h = 1;
    for (int i = 0; i < N; i++) {
        uint16_t t, s;
        memcpy(&t, &h_true_h[i], 2);
        memcpy(&s, &h_sim_h[i],  2);
        if (t != s) {
            printf("MISMATCH at i=%d: true=0x%04x sim=0x%04x\n", i, t, s);
            all_match_h = 0;
        }
    }
    printf("%s (half): %d/%d elements bitwise equal.\n",
           all_match_h ? "PASS" : "FAIL", all_match_h ? N : 0, N);

    cudaFree(dah); cudaFree(dbh); cudaFree(dch);
    cudaFree(d_true_h); cudaFree(d_sim_h);
    return (all_match && all_match_h) ? 0 : 1;
}
