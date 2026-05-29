#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <cub/cub.cuh>
#include "lo_float.h"
#include "Lof_kernels.h"
#include "cutlass_gemms.cuh"
#include "LUT_apprx.h"


namespace cg = cooperative_groups;
namespace lo_float {
template <typename From>
__global__ void round_mantissa_kernel(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len, From scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        From v = (scale == From(1)) ? in[idx] : From(scale * in[idx]);
        out[idx] = virtual_round(v, mantissa_bits, round_mode, stoch_len);
    }
}
template <typename From>
void round_mantissa(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len, From scale) {
    round_mantissa_kernel<<<(n + 255) / 256, 256>>>(in, out, n, mantissa_bits, round_mode, stoch_len, scale);
}
template <typename From>
__global__ void round_fp_params_kernel(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len, From scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        From v = (scale == From(1)) ? inout[idx] : From(scale * inout[idx]);
        inout[idx] = virtual_round(v, params, round_mode, stoch_len);
    }
}
template <typename From>
void round_fp_params(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len, From scale) {
    round_fp_params_kernel<<<(n + 255) / 256, 256>>>(inout, n, params, round_mode, stoch_len, scale);
}



template<typename From, int num_threads_per_cta = 256, int block_size = 32>
__global__ void virtual_mx_round(
    From* inout, int64_t n, From* /*scales (unused: virtual MX)*/,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_public,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_private,
    Rounding_Mode round_mode_public, Rounding_Mode round_mode_private, int stoch_len)
{
    static_assert((block_size & (block_size - 1)) == 0,
                  "block_size must be a power of 2");
    static_assert((num_threads_per_cta & (num_threads_per_cta - 1)) == 0,
                  "num_threads_per_cta must be a power of 2");

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int64_t num_mx_blocks = n / block_size;

    // Compute the rounded public scale from the block's amax.
    // stoch_len = 0 because we never want stochastic rounding on the scale.
    auto compute_scale = [&] (float amax) -> From {
        From s = static_cast<From>(amax * (1.0f / params_private.max_normal));
        virtual_round(s, params_public, round_mode_public, /*stoch_len=*/0);
        return s;
    };

    // =======================================================================
    // CASE 1:  block_size <= 32        (multiple MX blocks live in one warp)
    // =======================================================================
    if constexpr (block_size <= 32) {
        static_assert(num_threads_per_cta % block_size == 0,
                      "num_threads_per_cta must be divisible by block_size");
        constexpr int blocks_per_cta = num_threads_per_cta / block_size;

        const int local_block_idx = tid / block_size;
        const int local_lane      = tid & (block_size - 1);

        for (int64_t cta_iter = blockIdx.x;
             cta_iter * blocks_per_cta < num_mx_blocks;
             cta_iter += gridDim.x)
        {
            const int64_t block_idx = cta_iter * blocks_per_cta + local_block_idx;
            const bool    valid     = block_idx < num_mx_blocks;
            const int64_t idx       = block_idx * block_size + local_lane;

            From  val  = valid ? inout[idx] : From(0);
            float aval = fabsf(static_cast<float>(val));

            // Subgroup amax via shfl_xor (offsets < block_size stay in-block).
            #pragma unroll
            for (int offset = block_size / 2; offset > 0; offset >>= 1) {
                aval = fmaxf(aval, __shfl_xor_sync(0xFFFFFFFFu, aval, offset));
            }
            const From scale = compute_scale(aval);

            if (valid) {
                // Divide by scale, round into private format, then re-multiply.
                From scaled = static_cast<From>(static_cast<float>(val) /
                                                static_cast<float>(scale));
                virtual_round(scaled, params_private, round_mode_private, stoch_len);
                inout[idx] = static_cast<From>(static_cast<float>(scaled) *
                                               static_cast<float>(scale));
            }
        }
    }
    // =======================================================================
    // CASE 2:  32 < block_size <= num_threads_per_cta
    // =======================================================================
    else if constexpr (block_size <= num_threads_per_cta) {
        static_assert(num_threads_per_cta % block_size == 0,
                      "num_threads_per_cta must be divisible by block_size");
        constexpr int warps_per_block = block_size / 32;
        constexpr int blocks_per_cta  = num_threads_per_cta / block_size;

        __shared__ float smem_partial[blocks_per_cta][warps_per_block];
        __shared__ From  smem_scale  [blocks_per_cta];

        const int local_block_idx = warp_id / warps_per_block;
        const int warp_in_block   = warp_id % warps_per_block;
        const int local_lane      = warp_in_block * 32 + lane_id;

        for (int64_t cta_iter = blockIdx.x;
             cta_iter * blocks_per_cta < num_mx_blocks;
             cta_iter += gridDim.x)
        {
            const int64_t block_idx = cta_iter * blocks_per_cta + local_block_idx;
            const bool    valid     = block_idx < num_mx_blocks;
            const int64_t idx       = block_idx * block_size + local_lane;

            From  val  = valid ? inout[idx] : From(0);
            float aval = fabsf(static_cast<float>(val));

            // Intra-warp reduction.
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                aval = fmaxf(aval, __shfl_xor_sync(0xFFFFFFFFu, aval, offset));
            }
            if (lane_id == 0)
                smem_partial[local_block_idx][warp_in_block] = aval;
            __syncthreads();

            // First warp of each MX block finishes the reduction & makes scale.
            if (warp_in_block == 0) {
                float v = (lane_id < warps_per_block)
                            ? smem_partial[local_block_idx][lane_id]
                            : 0.0f;
                #pragma unroll
                for (int offset = warps_per_block / 2; offset > 0; offset >>= 1) {
                    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
                }
                if (lane_id == 0)
                    smem_scale[local_block_idx] = compute_scale(v);
            }
            __syncthreads();

            const From scale = smem_scale[local_block_idx];
            if (valid) {
                From scaled = static_cast<From>(static_cast<float>(val) /
                                                static_cast<float>(scale));
                virtual_round(scaled, params_private, round_mode_private, stoch_len);
                inout[idx] = static_cast<From>(static_cast<float>(scaled) *
                                               static_cast<float>(scale));
            }
            __syncthreads();   // protect smem before next iteration
        }
    }
    // =======================================================================
    // CASE 3:  block_size > num_threads_per_cta   (two-pass, CUB reduce)
    // =======================================================================
    else {
        static_assert(block_size % num_threads_per_cta == 0,
                      "block_size must be divisible by num_threads_per_cta");
        constexpr int chunks_per_block = block_size / num_threads_per_cta;

        using BlockReduce = cub::BlockReduce<float, num_threads_per_cta>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ From  smem_scale;

        for (int64_t block_idx = blockIdx.x;
             block_idx < num_mx_blocks;
             block_idx += gridDim.x)
        {
            const int64_t base = block_idx * block_size;

            // Pass 1: amax across the entire MX block.
            float thread_max = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
                const int64_t idx = base + chunk * num_threads_per_cta + tid;
                thread_max = fmaxf(thread_max,
                                   fabsf(static_cast<float>(inout[idx])));
            }
            float amax = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

            if (tid == 0) smem_scale = compute_scale(amax);
            __syncthreads();
            const From scale = smem_scale;

            // Pass 2: divide, round private, re-multiply.
            #pragma unroll
            for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
                const int64_t idx = base + chunk * num_threads_per_cta + tid;
                From val    = inout[idx];
                From scaled = static_cast<From>(static_cast<float>(val) /
                                                static_cast<float>(scale));
                virtual_round(scaled, params_private, round_mode_private, stoch_len);
                inout[idx]  = static_cast<From>(static_cast<float>(scaled) *
                                                static_cast<float>(scale));
            }
            __syncthreads();   // before reusing temp_storage / smem_scale
        }
    }
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
        round_mode,
        stochastic_rounding_bits);

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


template void round_mantissa<c10::Half>(const c10::Half*, c10::Half*, int64_t, int, Rounding_Mode, int, c10::Half);
template void round_mantissa<float>(const float*, float*, int64_t, int, Rounding_Mode, int, float);
template void round_mantissa<double>(const double*, double*, int64_t, int, Rounding_Mode, int, double);
template void round_fp_params<c10::Half>(c10::Half*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int, c10::Half);
template void round_fp_params<float>(float*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int, float);
template void round_fp_params<double>(double*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int, double);
}