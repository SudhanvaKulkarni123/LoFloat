#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <cub/cub.cuh>
#include <cuda/functional>   // cuda::maximum (replaces removed cub::Max)
#include "lo_float.h"
#include "Lof_kernels.h"
#include "cutlass_gemms.cuh"
#include "cutlass_conv.cuh"
#include "LUT_apprx.h"
#include "mx_round.cuh"   // lo_float::virtual_mx_round (also used by the GPU test)

  struct ContiguousLayout {
      int64_t n, block_size;
      __device__ int64_t num_blocks()   const { return n / block_size; }
      __device__ int     block_elems()  const { return block_size; }
      __device__ int64_t offset(int64_t blk, int e) const { return blk*block_size + e; }
      __device__ bool    valid (int64_t blk, int e) const { return blk*block_size + e < n; }
  };

  struct RectLayout {            // row-major, leading dim = ld
      int M, N, ld, BR, BC;
      __device__ int64_t num_blocks()  const { return (int64_t)(M/BR) * (N/BC); }
      __device__ int     block_elems() const { return BR*BC; }
      __device__ int64_t offset(int64_t blk, int e) const {
          int nbc = N/BC;
          int bi = blk / nbc, bj = blk % nbc;     // block coords
          int lr = e / BC,    lc = e % BC;         // intra-block coords
          return (int64_t)(bi*BR + lr)*ld + (bj*BC + lc);
      }
      __device__ bool valid(int64_t blk, int e) const { return offset(blk, e) < (int64_t)M*ld; }
  };

namespace cg = cooperative_groups;
namespace lo_float {
template <typename From>
__global__ void round_mantissa_kernel(const From* in, From* out, int64_t n, int mantissa_bits, ProjSpec ps, From scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        From v = (scale == From(1)) ? in[idx] : From(scale * in[idx]);
        out[idx] = virtual_round(v, mantissa_bits, ps);
    }
}
template <typename From>
void round_mantissa(const From* in, From* out, int64_t n, int mantissa_bits, ProjSpec ps, From scale) {
    round_mantissa_kernel<<<(n + 255) / 256, 256>>>(in, out, n, mantissa_bits, ps, scale);
}
template <typename From>
__global__ void round_fp_params_kernel(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, ProjSpec ps, From scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        From v = (scale == From(1)) ? inout[idx] : From(scale * inout[idx]);
        inout[idx] = virtual_round(v, params, ps);
    }
}
template <typename From>
void round_fp_params(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, ProjSpec ps, From scale) {
    round_fp_params_kernel<<<(n + 255) / 256, 256>>>(inout, n, params, ps, scale);
}


// virtual_mx_round now lives in mx_round.cuh (included above) so the GPU test
// can use it without pulling in torch/cutlass. The discard-the-rounding bug
// (virtual_round results were not assigned back) is fixed there.

// Host launcher: dispatch a runtime block_size to the matching compile-time
// kernel instantiation (num_threads_per_cta = 256). float only. Called from the
// Python binding for CUDA tensors.
void run_virtual_mx_round(
    float* inout, int64_t n, int block_size,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_public,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_private,
    ProjSpec ps_public, ProjSpec ps_private)
{
    constexpr int NTPC = 256;
    const int64_t nb = (n + block_size - 1) / block_size;
    int grid = (int)(nb < 1 ? 1 : (nb > 65535 ? 65535 : nb));
    #define LOF_MX_LAUNCH(MX, NX)                                              \
        virtual_mx_round<float, NTPC, MX, NX><<<grid, NTPC>>>(                \
            inout, n, nullptr, params_public, params_private,                \
            ps_public, ps_private)
    switch (block_size) {
        case 16:  LOF_MX_LAUNCH(1, 16);  break;
        case 32:  LOF_MX_LAUNCH(1, 32);  break;
        case 64:  LOF_MX_LAUNCH(2, 32);  break;
        case 128: LOF_MX_LAUNCH(4, 32);  break;
        case 256: LOF_MX_LAUNCH(8, 32);  break;
        default:
            throw std::runtime_error(
                "virtual_mx_round: unsupported block_size (use 16/32/64/128/256)");
    }
    #undef LOF_MX_LAUNCH
}

torch::Tensor LoF_gemm(
    torch::Tensor A,
    torch::Tensor B,
    int accum_mant_bits,
    Rounding_Mode round_mode,
    int stochastic_rounding_bits,
    double scale_a,
    double scale_b)
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
        lo_float::ProjSpec{round_mode, lo_float::Saturation_Mode::OvfInf, stochastic_rounding_bits});

    auto status = gemm.run_scaled(
        /*scale_a=*/static_cast<float>(scale_a),
        /*scale_b=*/static_cast<float>(scale_b),
        matA, matB, matC, matD);
    // TORCH_CHECK(status == lo_float::GemmStatus::Success,
    //     "LoF_gemm: kernel launch failed (status=", static_cast<int>(status), ")");

    cudaError_t sync_err = cudaDeviceSynchronize();
    TORCH_CHECK(sync_err == cudaSuccess,
    "LoF_gemm sync: ", cudaGetErrorString(sync_err));

    return C;
}

torch::Tensor LoF_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w,
    int accum_mant_bits,
    Rounding_Mode round_mode,
    int stochastic_rounding_bits,
    double weight_scale,
    double input_scale)
{
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
        "LoF_conv2d: input and weight must be 4-D (NCHW / OIHW)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
        "LoF_conv2d: only float32 inputs are supported");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
        "LoF_conv2d requires CUDA tensors (input on ", input.device(),
        ", weight on ", weight.device(), ")");
    TORCH_CHECK(input.size(1) == weight.size(1),
        "LoF_conv2d: input C_in (", input.size(1), ") != weight C_in (", weight.size(1), ")");

    // NCHW -> NHWC, OIHW -> KRSC (CUTLASS conv's native layouts).
    auto x = input.permute({0, 2, 3, 1}).contiguous();
    auto w = weight.permute({0, 2, 3, 1}).contiguous();

    const int N = static_cast<int>(x.size(0));
    const int H = static_cast<int>(x.size(1));
    const int W = static_cast<int>(x.size(2));
    const int C = static_cast<int>(x.size(3));
    const int K = static_cast<int>(w.size(0));
    const int R = static_cast<int>(w.size(1));
    const int S = static_cast<int>(w.size(2));

    const int ph = static_cast<int>(pad_h), pw = static_cast<int>(pad_w);
    const int sh = static_cast<int>(stride_h), sw = static_cast<int>(stride_w);
    const int dh = static_cast<int>(dilation_h), dw = static_cast<int>(dilation_w);

    const int P = (H + 2 * ph - dh * (R - 1) - 1) / sh + 1;
    const int Q = (W + 2 * pw - dw * (S - 1) - 1) / sw + 1;
    TORCH_CHECK(P > 0 && Q > 0,
        "LoF_conv2d: non-positive output spatial size (P=", P, ", Q=", Q, ")");

    auto out_nhwc = torch::zeros({N, P, Q, K}, x.options());

    cutlass::conv::Conv2dProblemSize problem_size(
        N, H, W, C, K, R, S, P, Q,
        ph, pw, sh, sw, dh, dw,
        cutlass::conv::Mode::kCrossCorrelation);

    lo_float::Conv2d<> conv(accum_mant_bits,
        lo_float::ProjSpec{round_mode, lo_float::Saturation_Mode::OvfInf, stochastic_rounding_bits});

    // Rescale back to the unscaled domain, same convention as LoF_gemm's
    // scale_a/scale_b (either scale being 0 is treated as 1, no-op).
    double denom = weight_scale * input_scale;
    float alpha = (denom == 0.0) ? 1.0f : static_cast<float>(1.0 / denom);

    auto status = conv(
        problem_size,
        x.data_ptr<float>(), w.data_ptr<float>(),
        out_nhwc.data_ptr<float>(), out_nhwc.data_ptr<float>(),
        alpha, 0.0f);

    TORCH_CHECK(status == decltype(conv)::Status::kSuccess,
        "LoF_conv2d: kernel launch/init failed (status=", static_cast<int>(status), ")");

    cudaError_t sync_err = cudaDeviceSynchronize();
    TORCH_CHECK(sync_err == cudaSuccess, "LoF_conv2d sync: ", cudaGetErrorString(sync_err));

    // NHWC -> NCHW
    return out_nhwc.permute({0, 3, 1, 2}).contiguous();
}


// Piecewise-linear SiLU. `lut` holds N+1 uniform knots over [-R, R] (built in
// Python by PWLSiLU). For x in [-R, R] we interpolate the table; outside we
// fall back to the asymptote relu(x) (== x for x>R, 0 for x<-R), matching
// PWLSiLU.forward exactly. The table is read from global memory by pointer, so
// N is bounded by L2/cache residency rather than the 4KB/32KB kernel-arg limit.
//
// N (the segment count, always a power of 2) is a compile-time template
// parameter so the bound `N - 1` and `inv_step` fold to constants; the runtime
// length is dispatched to the matching instantiation in LoPy_bind.cpp.
template <typename T, int N>
__global__ void pwl_silu_kernel(const T* __restrict__ in,
                                T* __restrict__ out,
                                const float* __restrict__ lut,
                                int64_t n, float R, float inv_step) {
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float x  = static_cast<float>(in[k]);
    float xc = fminf(fmaxf(x, -R), R);
    float u  = (xc + R) * inv_step;          // in [0, N]
    int   i  = static_cast<int>(u);          // floor (u >= 0)
    if (i > N - 1) i = N - 1;
    float frac = u - static_cast<float>(i);
    float y0 = lut[i];
    float y1 = lut[i + 1];
    float y  = fmaf(frac, y1 - y0, y0);
    if (fabsf(x) > R) y = x > 0.0f ? x : 0.0f;   // relu asymptote
    out[k] = static_cast<T>(y);
}

template <typename T, int N>
void pwl_silu(const T* in, T* out, const float* lut, int64_t n, float R) {
    if (n == 0) return;
    float inv_step = static_cast<float>(N) / (2.0f * R);
    pwl_silu_kernel<T, N><<<(n + 255) / 256, 256>>>(in, out, lut, n, R, inv_step);
}

// Instantiate for every supported power-of-2 length (single source of truth in
// Lof_kernels.h) across the three element types virtual_round also supports.
#define LOF_PWL_SILU_INST(NV) \
    template void pwl_silu<float,     NV>(const float*,     float*,     const float*, int64_t, float); \
    template void pwl_silu<double,    NV>(const double*,    double*,    const float*, int64_t, float); \
    template void pwl_silu<c10::Half, NV>(const c10::Half*, c10::Half*, const float*, int64_t, float);
LOF_PWL_SILU_LENGTHS(LOF_PWL_SILU_INST)
#undef LOF_PWL_SILU_INST


template void round_mantissa<c10::Half>(const c10::Half*, c10::Half*, int64_t, int, ProjSpec, c10::Half);
template void round_mantissa<float>(const float*, float*, int64_t, int, ProjSpec, float);
template void round_mantissa<double>(const double*, double*, int64_t, int, ProjSpec, double);
template void round_fp_params<c10::Half>(c10::Half*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, ProjSpec, c10::Half);
template void round_fp_params<float>(float*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, ProjSpec, float);
template void round_fp_params<double>(double*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, ProjSpec, double);
}