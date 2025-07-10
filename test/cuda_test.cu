#include <iostream>
#include <cuda_runtime.h>
#include "lo_float.h" // Assumed header with FloatingPointParams and Templated_Float

// Type aliases

using namespace lo_float;
using rne8 = float8_ieee_p<4, Rounding_Mode::RoundToNearestEven>; // round-to-nearest-even
using sr8  = float8_ieee_p<4, Rounding_Mode::StochasticRoundingA, 4>;  // stochastic rounding

__global__ void dot_kernel(const sr8* a, const sr8* b, float* result, int n) {
    __shared__ float partial_sum[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    partial_sum[lane] = sum;
    __syncthreads();

    // Reduction within block
    if (lane < 128) partial_sum[lane] += partial_sum[lane + 128]; __syncthreads();
    if (lane < 64) partial_sum[lane] += partial_sum[lane + 64]; __syncthreads();
    if (lane < 32) {
        volatile float* vsmem = partial_sum;
        vsmem[lane] += vsmem[lane + 32];
        vsmem[lane] += vsmem[lane + 16];
        vsmem[lane] += vsmem[lane + 8];
        vsmem[lane] += vsmem[lane + 4];
        vsmem[lane] += vsmem[lane + 2];
        vsmem[lane] += vsmem[lane + 1];
    }

    if (lane == 0) atomicAdd(result, partial_sum[0]);
}

int main() {
    constexpr int N = 1024;
    sr8 *d_a, *d_b;
    float *d_result, h_result = 0.0f;

    sr8 h_a[N], h_b[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = sr8::from_float(static_cast<float>(i % 8));
        h_b[i] = sr8::from_float(static_cast<float>((N - i) % 8));
    }

    cudaMalloc(&d_a, N * sizeof(sr8));
    cudaMalloc(&d_b, N * sizeof(sr8));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(sr8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(sr8), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    dot_kernel<<<1, 256>>>(d_a, d_b, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Dot product result (sr8): " << h_result << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    return 0;
}
