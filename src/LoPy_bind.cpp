// ============================================================================
// LoPy_bind.cpp - PyTorch Bindings for Runtime Converter
// ============================================================================

#include <torch/extension.h>
#include "lo_float.h"
#include <stdexcept>

namespace py = pybind11;

using namespace lo_float;
struct PyInfCheckerAdapter {
    py::object checker;
    
    bool operator()(uint64_t bits) const {
        py::gil_scoped_acquire acquire;
        return checker(py::int_(bits)).cast<bool>();
    }
    
    uint64_t minNegInf() const {
        py::gil_scoped_acquire acquire;
        return checker.attr("minNegInf")().cast<uint64_t>();
    }
    
    uint64_t minPosInf() const {
        py::gil_scoped_acquire acquire;
        return checker.attr("minPosInf")().cast<uint64_t>();
    }
};

struct PyNaNCheckerAdapter {
    py::object checker;
    
    bool operator()(uint64_t bits) const {
        py::gil_scoped_acquire acquire;
        return checker(py::int_(bits)).cast<bool>();
    }
    
    uint64_t qNanBitPattern() const {
        py::gil_scoped_acquire acquire;
        if (py::hasattr(checker, "qNanBitPattern")) {
            return checker.attr("qNanBitPattern")().cast<uint64_t>();
        }
        throw std::runtime_error("NaN checker missing qNanBitPattern method");
    }
    
    uint64_t sNanBitPattern() const {
        py::gil_scoped_acquire acquire;
        if (py::hasattr(checker, "sNanBitPattern")) {
            return checker.attr("sNanBitPattern")().cast<uint64_t>();
        }
        throw std::runtime_error("NaN checker missing sNanBitPattern method");
    }
};

struct FloatingPointParamsPy {
    int bitwidth;
    int mantissa_bits;
    int bias;
    Inf_Behaviors OV_behavior;
    NaN_Behaviors NA_behavior;
    Signedness is_signed;
    py::object IsInf;  // Python object
    py::object IsNaN;  // Python object

    private:
    static void validate_inf_checker(const py::object &checker) {
    if (!py::hasattr(checker, "__call__")) {
            throw std::invalid_argument(
                "inf_checker must be callable (have __call__ method)"
            );
        }
        if (!py::hasattr(checker, "minNegInf")) {
            throw std::invalid_argument(
                "inf_checker must have method: minNegInf() -> int"
            );
        }
        
        if (!py::hasattr(checker, "minPosInf")) {
            throw std::invalid_argument(
                "inf_checker must have method: minPosInf() -> int"
            );
        }
    }

    static void validate_nan_checker(const py::object& checker) {
        if (!py::hasattr(checker, "__call__")) {
            throw std::invalid_argument(
                "nan_checker must be callable (have __call__ method)"
            );
        }
        
        bool has_qnan = py::hasattr(checker, "qNanBitPattern");
        bool has_snan = py::hasattr(checker, "sNanBitPattern");
        
        if (!has_qnan && !has_snan) {
            throw std::invalid_argument(
                "nan_checker must have at least one of: qNanBitPattern() or sNanBitPattern()"
            );
        }
        
        if (has_qnan && !py::hasattr(checker.attr("qNanBitPattern"), "__call__")) {
            throw std::invalid_argument("qNanBitPattern must be callable");
        }
        
        if (has_snan && !py::hasattr(checker.attr("sNanBitPattern"), "__call__")) {
            throw std::invalid_argument("sNanBitPattern must be callable");
        }
    }

    
    public:
    FloatingPointParamsPy(
        int bw, int mb, int b,
        Inf_Behaviors ov, NaN_Behaviors na,
        Signedness sign,
        py::object inf_checker, py::object nan_checker
    ) : bitwidth(bw), mantissa_bits(mb), bias(b),
        OV_behavior(ov), NA_behavior(na),
        is_signed(sign),
        IsInf(inf_checker), IsNaN(nan_checker)
    {
        validate_inf_checker(inf_checker);
        validate_nan_checker(nan_checker);
    }
};

// Wrappers for torch::Tensor
torch::Tensor virtual_round_mantissa(
    const torch::Tensor &input,
    int to_mantissa_bits,
    Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven,
    int stoch_len = 0
) {
    auto output = torch::empty_like(input);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_mantissa", ([&] {
        auto* in_ptr = input.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();
        virtual_round(in_ptr, out_ptr, to_mantissa_bits, input.numel(), round_mode, stoch_len);
    }));
    
    return output;
}

torch::Tensor virtual_round_params(
    const torch::Tensor &input,
    const FloatingPointParamsPy &params,
    Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven,
    int stoch_len = 0
) {
    auto output = torch::empty_like(input);
    
    PyInfCheckerAdapter inf_adapter{params.IsInf};
    PyNaNCheckerAdapter nan_adapter{params.IsNaN};
    
    auto cpp_params = FloatingPointParams<PyInfCheckerAdapter, PyNaNCheckerAdapter>{
        params.bitwidth, params.mantissa_bits, params.bias,
        params.OV_behavior, params.NA_behavior, params.is_signed,
        inf_adapter, nan_adapter, Unsigned_behavior::NegtoZero
    };
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_params", ([&] {
        auto* in_ptr = input.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();
        virtual_round(in_ptr, out_ptr, input.numel(), cpp_params, round_mode, stoch_len);
    }));
    
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Runtime float format converter with custom quantization";
    
    // Bind enums
    py::enum_<Signedness>(m, "Signedness")
        .value("Signed", Signedness::Signed)
        .value("Unsigned", Signedness::Unsigned)
        .export_values();
    
    py::enum_<Inf_Behaviors>(m, "InfBehavior")
        .value("Extended", Inf_Behaviors::Extended)
        .value("Saturating", Inf_Behaviors::Saturating)
        .export_values();
    
    py::enum_<Rounding_Mode>(m, "RoundingMode")
        .value("RoundToNearestEven", Rounding_Mode::RoundToNearestEven)
        .value("RoundTowardsZero", Rounding_Mode::RoundTowardsZero)
        .value("RoundUp", Rounding_Mode::RoundUp)
        .value("RoundDown", Rounding_Mode::RoundDown)
        .value("RoundAwayFromZero", Rounding_Mode::RoundAwayFromZero)
        .value("StochasticRoundingA", Rounding_Mode::StochasticRoundingA)
        .value("StochasticRoundingB", Rounding_Mode::StochasticRoundingB)
        .value("StochasticRoundingC", Rounding_Mode::StochasticRoundingC)
        .value("True_StochasticRounding", Rounding_Mode::True_StochasticRounding)
        .value("ProbabilisticRounding", Rounding_Mode::ProbabilisticRounding)
        .export_values();
    
    py::enum_<NaN_Behaviors>(m, "NaNBehavior")
        .value("QuietNaN", NaN_Behaviors::QuietNaN)
        .value("SignalingNaN", NaN_Behaviors::SignalingNaN)
        .export_values();
    
    // Bind FloatFormatDescriptor structure
    py::class_<FloatingPointParamsPy>(m, "FloatFormatDescriptor")
        .def(py::init<int, int, int, Inf_Behaviors, NaN_Behaviors, Signedness, py::object, py::object>(),
             py::arg("total_bits"),
             py::arg("exponent_bits"),
             py::arg("mantissa_bits"),
             py::arg("inf_behavior"),
             py::arg("nan_behavior"),
             py::arg("signedness"),
             py::arg("is_inf_checker"),
             py::arg("is_nan_checker"),
             "Create a custom float format descriptor");

    m.def("virtual_round", &virtual_round_mantissa,
          py::arg("input"),
          py::arg("to_mantissa_bits"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven,
          py::arg("stoch_len") = 0,
          "Round tensor to specified mantissa bits");
    
    m.def("virtual_round", &virtual_round_params,
          py::arg("input"),
          py::arg("params"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven,
          py::arg("stoch_len") = 0,
          "Round tensor using FloatingPointParams");
        
    
}