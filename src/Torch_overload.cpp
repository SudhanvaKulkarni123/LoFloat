#include "lo_float.h"
#include "lo_int.h"
#include <torch/torch.h>
#include <assert.h>
#include <random>
#include <tuple>
#include "pybind_instantiations.hpp"



using namespace lo_float;


template<int k, int p, Signedness sign, Inf_Behaviors has_inf>
inline auto fake_quantize_tensor(const torch::Tensor& tensor, float scale, float zero_point) {
    // Check if the tensor is contiguous

    using out_type = P3109_float<k, p, sign, has_inf>;

    // Create an empty tensor for the quantized output
    auto quantized_tensor = torch::empty_like(tensor, torch::kFloat32);

    #pragma omp parallel for
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        // Apply quantization
        quantized_tensor[i] = static_cast<float>(static_cast<out_type>(std::round(tensor[i].item<float>() / scale) + zero_point));
    }

    return quantized_tensor;
}


template<int k>
struct get_next_uint {
    using uint_type = std::conditional_t<k <= 8, uint8_t, int16_t>;
};

template<typename T>
struct get_specifier {
    static constexpr at::ScalarType value = std::is_same<T, uint8_t>::value ? torch::kUInt8 : torch::kInt16 ;            
};

template<>
struct get_specifier<float> {
    static constexpr at::ScalarType value = at::kFloat;      // float32
};

template <at::ScalarType dtype>
struct type_from_specifier;

template <> struct type_from_specifier<at::kByte>  { using type = uint8_t; };
template <> struct type_from_specifier<at::kChar>  { using type = int8_t; };
template <> struct type_from_specifier<at::kShort> { using type = int16_t; };


template<int k, int p, Signedness sign, Inf_Behaviors has_inf>
inline auto real_quantize_tensor(const torch::Tensor& tensor, float scale, float zero_point) {
    // Check if the tensor is contiguous


    using out_type = get_next_uint<k>::uint_type;
    using P3109_type = P3109_float<k, p, sign, has_inf>;
    using specifier = get_specifier<out_type>;

    // Create an empty tensor for the quantized output
    auto quantized_tensor = torch::empty_like(tensor, specifier::value);

    #pragma omp parallel for
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        // Apply quantization
        quantized_tensor[i] = std::bit_cast<out_type>(static_cast<P3109_type>((tensor[i].item<float>() / scale) + zero_point));
    }

    return quantized_tensor;
}

template<int k, int p, Signedness sign, Inf_Behaviors has_inf, typename out_type>
inline auto dequantize_tensor(const torch::Tensor& tensor, float scale, float zero_point) {
    // Check if the tensor is contiguous


    constexpr at::ScalarType torch_type = get_specifier<out_type>::value;
    // Create an empty tensor for the dequantized output
    auto dequantized_tensor = torch::empty_like(tensor, torch_type);

    using P3109_type = P3109_float<k, p, sign, has_inf>;

    #pragma omp parallel for
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        // Apply dequantization
        dequantized_tensor[i] = static_cast<out_type>(std::bit_cast<P3109_type>(tensor[i].item<uint8_t>())) * scale + zero_point;
    }

    return dequantized_tensor;
}

#define BIND_QUANTIZATION_FUNCTIONS(k, p, sign, has_inf) \
    m.def("fake_quantize_tensor", \
          [](const torch::Tensor& t, float s, float zp) { \
              return fake_quantize_tensor<k, p, sign, has_inf>(t, s, zp); \
          }, \
          "Fake quantization function for tensors"); \
    m.def("real_quantize_tensor", \
          [](const torch::Tensor& t, float s, float zp) { \
              return real_quantize_tensor<k, p, sign, has_inf>(t, s, zp); \
          }, \
          "Real quantization function for tensors"); \
    m.def("dequantize_tensor", \
          [](const torch::Tensor& t, float s, float zp) { \
              return dequantize_tensor<k, p, sign, has_inf, float>(t, s, zp); \
          }, \
          "Dequantization function for tensors");


PYBIND11_MODULE(LoFloat, m) {
    // Bind the quantization functions for various configurations
    BIND_FLOAT(8, 2, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 2, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 2, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 2, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 3, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 3, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 3, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 3, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 4, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 4, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 4, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 4, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 5, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 5, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 5, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 5, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 6, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 6, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 6, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 6, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_QUANTIZATION_FUNCTIONS(8, 3, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_QUANTIZATION_FUNCTIONS(8, 3, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_QUANTIZATION_FUNCTIONS(8, 3, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_QUANTIZATION_FUNCTIONS(8, 3, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_QUANTIZATION_FUNCTIONS(8, 4, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_QUANTIZATION_FUNCTIONS(8, 4, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_QUANTIZATION_FUNCTIONS(8, 4, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_QUANTIZATION_FUNCTIONS(8, 4, Signedness::Unsigned, Inf_Behaviors::Saturating);
}

