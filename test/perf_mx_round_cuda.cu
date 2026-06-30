// ============================================================================
// perf_mx_round_cuda.cu — perf of the GPU MX block-quantization kernel vs a
// regular (elementwise) GPU virtual_round, for block sizes 32 and 128 across n.
//
// Baseline "regular virtual_round" = one virtual_round per element, no MX
// scaling/reduction. The MX kernel additionally does a per-block amax reduction,
// a shared scale, and a divide/round/multiply — so it is expected to be somewhat
// slower; this quantifies that overhead.
//
// Build: USE_CUDA=1 make perf_mx_round_cuda
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include "lo_float.h"
#include "mx_round.cuh"

using namespace lo_float;
using P = FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>;

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)

// Regular elementwise virtual_round (the baseline; no MX blocking).
__global__ void regular_round(float* io, int64_t n, P params, Rounding_Mode rm) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) io[i] = lo_float::virtual_round(io[i], params, lo_float::ProjSpec{rm});
}

static float time_ms(int reps, std::function<void()> launch) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());                 // warmup
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

template<int NTPC, int MX, int NX>
static float time_mx(float* d, int64_t n, P pub, P priv, int reps) {
    const int64_t nb = (n + (MX*NX) - 1) / (MX*NX);
    int grid = (int)std::min<int64_t>(65535, std::max<int64_t>(1, nb));
    return time_ms(reps, [=]{
        virtual_mx_round<float, NTPC, MX, NX><<<grid, NTPC>>>(
            d, n, nullptr, pub, priv, lo_float::ProjSpec{Rounding_Mode::RoundToNearestEven},
            lo_float::ProjSpec{Rounding_Mode::RoundToNearestEven});
    });
}

int main() {
    const P e8m0(8,0,127,Inf_Behaviors::Extended,NaN_Behaviors::_754,Signedness::Unsigned,{0,0,0,0},{0,0,0,0});
    const P e4m3(8,3,7,Inf_Behaviors::Saturating,NaN_Behaviors::_754,Signedness::Signed,{0,0,0,0},{0,0,0,0});
    const int reps = 50;
    const int64_t ns[] = { 1<<16, 1<<18, 1<<20, 1<<22, 1<<24 };

    printf("GPU virtual_mx_round perf (e4m3 priv / e8m0 scale, RNE), best avg of %d reps\n", reps);
    printf("%-12s %14s %14s %14s   %10s %10s\n",
           "n", "regular(ms)", "mx bs=32(ms)", "mx bs=128(ms)", "x32", "x128");

    for (int64_t n : ns) {
        std::vector<float> h(n);
        for (auto& x : h) x = (float)rand()/RAND_MAX - 0.5f;
        float* d; CK(cudaMalloc(&d, n*sizeof(float)));

        auto reload = [&]{ CK(cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice)); };

        reload();
        int blk = 256, grid = (int)std::min<int64_t>(65535,(n+blk-1)/blk);
        float t_reg = time_ms(reps, [=]{ regular_round<<<grid,blk>>>(d, n, e4m3, Rounding_Mode::RoundToNearestEven); });

        reload(); float t32  = time_mx<256,1,32>(d, n, e8m0, e4m3, reps);   // block_size 32
        reload(); float t128 = time_mx<256,4,32>(d, n, e8m0, e4m3, reps);   // block_size 128

        printf("%-12lld %14.4f %14.4f %14.4f   %10.2f %10.2f\n",
               (long long)n, t_reg, t32, t128, t32/t_reg, t128/t_reg);
        CK(cudaFree(d));
    }
    return 0;
}
