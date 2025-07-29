///@author Sudhanva Kulkarni
/// This file defines teh cuda version of the vanilla quantization functions supported in Torch_overload.cpp
#include "lo_int.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <assert.h>
#include <random>
#include <tuple>
#include "lo_float.h"


using namespace lo_float;


template<int k, int p, Signedness sign, Inf_Behaviors has_inf>
__global__  void fake_quantize_tensor(const torch::Tensor& input_tensor, torch::Tensor& output_tensor, float scale, float zero_point) {
    // Check if the tensor is contiguous

    using out_type = P3109_float<k, p, sign, has_inf>;

    // Create an empty tensor for the quantized output
    float* input_ptr = tensor.data_ptr<float>();
    float* output_ptr = quantized_tensor.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        output_ptr[i] = static_cast<float>(static_cast<out_type>(std::round(input_ptr[i] / scale + zero_point)));
    }

    return quantized_tensor;
}