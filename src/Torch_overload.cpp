#include "lo_float.h"
#include "Gemms.hpp"
#include <torch/torch.h>
#include <torch/extension.h>
#include <assert.h>
#include <random>
#include <tuple>
#include "pybind_instantiations.hpp"


using namespace lo_float;
using namespace Lo_Gemm;

#define LOOP_CAST(out_type) \
    #pragma omp parallel for    \
        for (int64_t i = 0; i < tensor.numel(); ++i) {  \
            output_ptr[i] = static_cast<float>(static_cast<out_type>(std::round(input_ptr[i] / scale + zero_point)));   \
        }   
#define MX_LOOP_CAST(out_type, scale_type) \
    #pragma omp parallel for    \
        for (int64_t i = 0; i < tensor)

enum precision_handles : uint8_t {
    F8_P3 = 0,
    F8_P4 = 1,
    F32 = 2,
    F16 = 3
};

//add descriptors to this enum whenever a new type is instantiated
enum LoPy_types {
    binary8p3se,
    binary8p4se,
    binary8p5se,
    binary8p3sf,
    binary8p4sf,
    binary8p5sf,
    binary6p1ue,
    binary6p2sf,
    binaery6p3sf,
    binary6p4sf,
    binary4p1ue,
    binary4p2sf,
    binary4p3sf,
    ocpe4m3,
    ocpe5m2,
    ocpe3m2,
    ocpe2m1,
    ocpe8m0,
    Half,
    Bfloat16,
    Tf32
};

enum Mx_LoPy_types {
    mxocpe2m1_ocpe8m0,
    mxocpe3m2_ocpe8m0,
    mxocpe4m3_ocpe8m0,
    mxocpe5m2_ocpe8m0,
    mxbinary8p3se_binary8p1ue,
    mxbinary8p4se_binary8p1ue,
    mxbinary8p5se_binary8p1ue,
    mxbinary8p3sf_binary8p1ue,
    mxbinary8p4sf_binary8p1ue,
    mxbinary8p5sf_binary8p1ue,
    mxbinary6p1ue_binary8p1ue,
    mxbinary6p2sf_binary8p1ue,
    mxbinaery6p3sf_binary8p1ue,
    mxbinary6p4sf_binary8p1ue
};


//template<int k, int p, Signedness sign, Inf_Behaviors has_inf>
auto fake_quantize_tensor(const torch::Tensor& tensor, float scale, float zero_point, LoPy_types type) {
    // Check if the tensor is contiguous

    // Create an empty tensor for the quantized output
    auto quantized_tensor = torch::empty_like(tensor, torch::kFloat32);
    float* input_ptr = tensor.data_ptr<float>();
    float* output_ptr = quantized_tensor.data_ptr<float>();

    switch(type) {
        case LoPy_types::binary8p3se: {
            LOOP_CAST(P3109_float<8, 3, Signedness::Signed, Inf_Behaviors::Extended>);
            break;
        }
        case LoPy_types::binary8p4se: {
            LOOP_CAST(P3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>);
            break;
        }
        case LoPy_types::binary8p5se: {
            LOOP_CAST(P3109_float<8, 5, Signedness::Signed, Inf_Behaviors::Extended>);
            break;
        }
        case LoPy_types::binary6p1ue: {
            LOOP_CAST(P3109_float<6, 1, Signedness::Unsigned, Inf_Behaviors::Extended>);
            break;
        }
        case LoPy_types::binary6p2sf: {
            LOOP_CAST(P3109_float<6, 2, Signedness::Signed, Inf_Behaviors::Saturating>);
            break;
        }
        case LoPy_types::binary6p3sf: {
            LOOP_CAST(P3109_float<6, 3, Signedness::Signed, Inf_Behaviors::Saturating>);
            break;
        }
        case LoPy_types::binary6p4sf: {
            LOOP_CAST(P3109_float<6, 4, Signedness::Signed, Inf_Behaviors::Saturating>);
            break;
        }
        case LoPy_types::binary4p1ue: {
            LOOP_CAST(P3109_float<4, 1, Signedness::Unsigned, Inf_Behaviors::Extended>);
            break;
        }
        case LoPy_types::binary4p2sf: {
            LOOP_CAST(P3109_float<4, 2, Signedness::Signed, Inf_Behaviors::Saturating>);
            break;
        }

        default:
            TORCH_CHECK(false, "Unsupported quantization type. Please add type to enum and instantiate in file Torch_overload.cpp");
    }


    return quantized_tensor;
}

auto mx_fake_quantize_tensor(const torch::Tensor& tensor, float scale, float zero_point, Mx_LoPy_types type, int block_size) {

    auto quantized_tensor = torch::empty_like(tensor, torch::kFloat32);

    float* input_ptr = tensor.data_ptr<float>();
    float* output_ptr = quantized_tensor.data_ptr<float>();
    

    switch(type) {

        case Mx_LoPy_types::mxocpe2m1_ocpe8m0: {
            OCP_e2m1* tmp = (OCP_e2m1*) malloc(tensor.numel());
            OCP_e8m0* tmp_scal = (OCP_e8m0*) malloc(tensor.numel()/block_size);
            #pragma omp parallel for    
            for (int64_t i = 0; i < tensor.numel(); ++i) {  
                output_ptr[i] = static_cast<float>(static_cast<out_type>(std::round(input_ptr[i] / scale + zero_point)));   
            }
            break;
        }

    }
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



torch::Tensor quantized_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias, 
    ) {

    // Ensure input tensor is contiguous
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // Ensure input and weight dimensions are compatible
    TORCH_CHECK(input.size(1) == weight.size(1),
                "Input size must match weight size. Got ",
                input.size(1), " and ", weight.size(1));

    // Perform matrix multiplication (e.g., input @ weight.T)
    // weight should be transposed: [out_features, in_features]
    torch::Tensor output = at::matmul(input, weight.t());

    // Add bias if defined
    if (bias.defined()) {
        // Bias is expected to be 1D with size equal to output's last dim
        TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
        TORCH_CHECK(bias.size(0) == output.size(1),
                    "Bias size must match output size");

        output += bias;
    }

    return output;
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
    BIND_FLOAT(8, 5, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 5, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 5, Signedness::Unsigned, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 6, Signedness::Signed, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 6, Signedness::Signed, Inf_Behaviors::Saturating);
    BIND_FLOAT(8, 6, Signedness::Unsigned, Inf_Behaviors::Extended);
    BIND_FLOAT(8, 6, Signedness::Unsigned, Inf_Behaviors::Saturating);

    BIND_QUANTIZATION_FUNCTIONS(8, 4, Signedness::Signed, Inf_Behaviors::Saturating);

}

