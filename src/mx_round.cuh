#pragma once
// ============================================================================
//  mx_round.cuh — GPU microscaling (MX) block-quantization kernel.
//
//  Extracted from Lof_kernel.cu so it can be unit-tested without pulling in
//  torch / cutlass (which Lof_kernel.cu needs for its other kernels). The host
//  CPU counterpart is lo_float::virtual_mx_round in lo_float.h.
//
//  Each contiguous block of `block_size = MX*NX` elements shares one scale:
//      amax  = max |x_i|
//      scale = virtual_round(amax / priv_max_normal, params_public, rm_public)
//      x_i  := virtual_round(x_i / scale, params_private, rm_private, stoch) * scale
//
//  Three code paths by block_size vs num_threads_per_cta:
//    CASE 1  block_size <= 32                       — warp shfl reduction
//    CASE 2  32 < block_size <= num_threads_per_cta — shared-mem two-level
//    CASE 3  block_size  > num_threads_per_cta      — CUB block-reduce, two-pass
// ============================================================================

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/functional>   // cuda::maximum
#include "lo_float.h"

namespace lo_float {

template<typename From, int num_threads_per_cta = 256, int MX = 1, int NX = 32>
__global__ void virtual_mx_round(
    From* inout, int64_t n, From* /*scales (unused: virtual MX)*/,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_public,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_private,
    ProjSpec ps_public, ProjSpec ps_private)
{
    int constexpr block_size = MX * NX;
    static_assert((block_size & (block_size - 1)) == 0,
                  "block_size must be a power of 2");
    static_assert((num_threads_per_cta & (num_threads_per_cta - 1)) == 0,
                  "num_threads_per_cta must be a power of 2");

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    // ceil: include the trailing partial block when n is not a multiple of
    // block_size. Out-of-range lanes are masked (idx < n) everywhere below.
    const int64_t num_mx_blocks = (n + block_size - 1) / block_size;

    // Compute the rounded public scale from the block's amax.
    // stoch_len = 0 because we never want stochastic rounding on the scale.
    // Largest finite normal of the private format. Derived from the runtime
    // params the same way virtual_round() computes ToMax_exp: the all-ones
    // mantissa value (2 - 2^-mant) scaled by 2^max_exp.
    const int priv_max_exp = params_private.is_signed == Signedness::Signed
        ? (1 << (params_private.bitwidth - params_private.mantissa_bits - 1)) - 1 - params_private.bias
        : (1 << (params_private.bitwidth - params_private.mantissa_bits)) - 1 - params_private.bias;
    const float priv_max_normal =
        ldexpf(1.0f, priv_max_exp) * (2.0f - ldexpf(1.0f, -params_private.mantissa_bits));

    auto compute_scale = [&] (float amax) -> From {
        From s = static_cast<From>(amax * (1.0f / priv_max_normal));
        // NB: virtual_round RETURNS the rounded value (it does not write through
        // the reference), so the result must be assigned back.
        s = virtual_round(s, params_public,
                          ProjSpec{ps_public.rounding_mode, ps_public.saturation_mode, /*stoch_len=*/0});
        return s;
    };

    // Round one element: round(val/scale) into the private format, rescaled.
    // An all-zero (or underflowed) block has scale == 0 -> the element is 0 (a
    // guard, else val/scale would be 0/0 = NaN). Matches the CPU path.
    auto quant_elem = [&] (From val, From scale) -> From {
        if (!(static_cast<float>(scale) > 0.0f)) return From(0);
        From scaled = static_cast<From>(static_cast<float>(val) / static_cast<float>(scale));
        scaled = virtual_round(scaled, params_private, ps_private);
        return static_cast<From>(static_cast<float>(scaled) * static_cast<float>(scale));
    };

    if constexpr (MX != 1 || NX != 1) {

    }

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
            const int64_t idx       = block_idx * block_size + local_lane;
            const bool    valid     = idx < n;

            From  val  = idx < n ? inout[idx] : From(0);
            float aval = fabsf(static_cast<float>(val));

            // Subgroup amax via shfl_xor (offsets < block_size stay in-block).
            #pragma unroll
            for (int offset = block_size / 2; offset > 0; offset >>= 1) {
                aval = fmaxf(aval, __shfl_xor_sync(0xFFFFFFFFu, aval, offset));
            }
            const From scale = compute_scale(aval);

            if (valid)
                inout[idx] = quant_elem(val, scale);
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

            const int64_t idx       = block_idx * block_size + local_lane;
            const bool    valid     = idx < n;

            From  val  = idx < n ? inout[idx] : From(0);
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
            if (valid)
                inout[idx] = quant_elem(val, scale);
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
                if (idx < n)
                    thread_max = fmaxf(thread_max,
                                       fabsf(static_cast<float>(inout[idx])));
            }
            float amax = BlockReduce(temp_storage).Reduce(thread_max, ::cuda::maximum<float>{});

            if (tid == 0) smem_scale = compute_scale(amax);
            __syncthreads();
            const From scale = smem_scale;

            // Pass 2: divide, round private, re-multiply.
            #pragma unroll
            for (int chunk = 0; chunk < chunks_per_block; ++chunk) {
                const int64_t idx = base + chunk * num_threads_per_cta + tid;
                if (idx < n)
                    inout[idx] = quant_elem(inout[idx], scale);
            }
            __syncthreads();   // before reusing temp_storage / smem_scale
        }
    }
}

}  // namespace lo_float
