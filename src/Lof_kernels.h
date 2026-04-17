#pragma once
#include <torch/types.h>
#include "lo_float.h"  // for Rounding_Mode
using namespace lo_float;
#ifdef USE_CUDA
torch::Tensor LoF_gemm(
    torch::Tensor A,
    torch::Tensor B,
    int accum_mant_bits,
    Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven,
    int stochastic_rounding_bits = 0);
#endif