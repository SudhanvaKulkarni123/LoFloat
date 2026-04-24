#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include "lo_float.h"
#include "Lof_kernels.h"
#include "cutlass_gemms.cuh"
#include "LUT_apprx.h"


namespace cg = cooperative_groups;
namespace lo_float {
template <typename From>
__global__ void round_mantissa_kernel(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = virtual_round(in[idx], mantissa_bits, round_mode, stoch_len);
}
template <typename From>
void round_mantissa(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len) {
    round_mantissa_kernel<<<(n + 255) / 256, 256>>>(in, out, n, mantissa_bits, round_mode, stoch_len);
}
template <typename From>
__global__ void round_fp_params_kernel(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) inout[idx] = virtual_round(inout[idx], params, round_mode, stoch_len);
}
template <typename From>
void round_fp_params(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len) {
    round_fp_params_kernel<<<(n + 255) / 256, 256>>>(inout, n, params, round_mode, stoch_len);
}

torch::Tensor LoF_gemm(
    torch::Tensor A,
    torch::Tensor B,
    int accum_mant_bits,
    Rounding_Mode round_mode,
    int stochastic_rounding_bits)
{
    // ── Dimension checks ────────────────────────────────────────────────
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "LoF_gemm: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0),
        "LoF_gemm: A.cols (", A.size(1), ") != B.rows (", B.size(0), ")");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
        "LoF_gemm: only float32 inputs are supported");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
            "LoF_gemm requires CUDA tensors (A on ", A.device(),
            ", B on ", B.device(), ")");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto C = torch::zeros({M, N}, A.options());

    // ── Raw device pointers ─────────────────────────────────────────────
    float* d_A = A.data_ptr<float>();
    float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();
    float* d_D = C.data_ptr<float>();

    using MatA = lo_float::Matrix<float, int, lo_float::RowMajor>;
    using MatB = lo_float::Matrix<float, int, lo_float::RowMajor>;
    using MatC = lo_float::Matrix<float, int, lo_float::RowMajor>;
    using MatD = lo_float::Matrix<float, int, lo_float::RowMajor>;

    MatA matA(d_A, M, K, /*ld=*/K);
    MatB matB(d_B, K, N, /*ld=*/N);   // RowMajor: ld = number of columns
    MatC matC(d_C, M, N, /*ld=*/N);
    MatD matD(d_D, M, N, /*ld=*/N);

    // ── Construct and launch ─────────────────────────────────────────────
    lo_float::Gemm<MatA, MatB, MatC, MatD> gemm(
        accum_mant_bits,
        round_mode,
        stochastic_rounding_bits);

    auto status = gemm(/*alpha=*/1.0f, /*beta=*/0.0f, matA, matB, matC, matD);
    // TORCH_CHECK(status == lo_float::GemmStatus::Success,
    //     "LoF_gemm: kernel launch failed (status=", static_cast<int>(status), ")");

    cudaError_t sync_err = cudaDeviceSynchronize();
    TORCH_CHECK(sync_err == cudaSuccess,
    "LoF_gemm sync: ", cudaGetErrorString(sync_err));

    return C;
}


template <typename T, class Reducer, class ApproxFn>
__global__ void silu_kernel(const T* __restrict__ in,
                            T* __restrict__ out,
                            int64_t n,
                            FuncApprox<Reducer, ApproxFn> approx) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = approx(in[idx]);
}

template <typename T>
void silu_lut(const T* in, T* out, int64_t n) {
    silu_kernel<<<(n + 255) / 256, 256>>>(in, out, n, g_silu);
}

template void silu_lut<float>    (const float*,     float*,     int64_t);
template void silu_lut<double>   (const double*,    double*,    int64_t);
template void silu_lut<c10::Half>(const c10::Half*, c10::Half*, int64_t);


template void round_mantissa<c10::Half>(const c10::Half*, c10::Half*, int64_t, int, Rounding_Mode, int);
template void round_mantissa<float>(const float*, float*, int64_t, int, Rounding_Mode, int);
template void round_mantissa<double>(const double*, double*, int64_t, int, Rounding_Mode, int);
template void round_fp_params<c10::Half>(c10::Half*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
template void round_fp_params<float>(float*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
template void round_fp_params<double>(double*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
}