#pragma once
#include <torch/types.h>
#include "lo_float.h"  // for Rounding_Mode

// Compile-time LUT segment counts supported by pwl_silu — always powers of 2.
// Single source of truth shared by the .cu instantiations and the .cpp runtime
// dispatch switch, so the two lists cannot drift apart. Covers every lut_size
// (= 1 << lut_bits) PWLSiLU can produce from lut_bits 1..12.
#define LOF_PWL_SILU_LENGTHS(X) \
    X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256) X(512) X(1024) X(2048) X(4096)

#ifdef USE_CUDA
namespace lo_float{
torch::Tensor LoF_gemm(
    torch::Tensor A,
    torch::Tensor B,
    int accum_mant_bits,
    Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven,
    int stochastic_rounding_bits = 0,
    double scale_a = 1.0,
    double scale_b = 1.0);

// Low-precision Conv2d forward (float32 only, groups=1) via CUTLASS implicit
// GEMM. `input` is NCHW, `weight` is (C_out, C_in, kH, kW) — both converted to
// NHWC/KRSC internally. Output is divided by (weight_scale * input_scale) to
// rescale back to the unscaled domain, same convention as LoF_gemm's
// scale_a/scale_b.
torch::Tensor LoF_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w,
    int accum_mant_bits,
    Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven,
    int stochastic_rounding_bits = 0,
    double weight_scale = 1.0,
    double input_scale = 1.0);

template <typename T, int N>
void pwl_silu(const T* in, T* out, const float* lut, int64_t n, float R);

// Launches the GPU MX block-quantization kernel (dispatches block_size). float only.
void run_virtual_mx_round(
    float* inout, int64_t n, int block_size,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_public,
    FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params_private,
    ProjSpec ps_public, ProjSpec ps_private);
}
#endif