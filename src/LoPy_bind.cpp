#include <torch/extension.h>
#include "lo_float.h"
#include <stdexcept>

namespace py = pybind11;
using namespace lo_float;

#ifdef USE_CUDA
namespace lo_float {
template <typename T>
void round_mantissa(const T* in, T* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len);

template <typename T>
void round_fp_params(T* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len);
}
#endif

struct PyInfCheckerAdapter {
    py::object checker;
    bool operator()(uint64_t bits) const { py::gil_scoped_acquire a; return checker(py::int_(bits)).cast<bool>(); }
    uint64_t minNegInf() const { py::gil_scoped_acquire a; return checker.attr("minNegInf")().cast<uint64_t>(); }
    uint64_t minPosInf() const { py::gil_scoped_acquire a; return checker.attr("minPosInf")().cast<uint64_t>(); }
};

struct PyNaNCheckerAdapter {
    py::object checker;
    bool operator()(uint64_t bits) const { py::gil_scoped_acquire a; return checker(py::int_(bits)).cast<bool>(); }
    uint64_t qNanBitPattern() const {
        py::gil_scoped_acquire a;
        if (py::hasattr(checker, "qNanBitPattern")) return checker.attr("qNanBitPattern")().cast<uint64_t>();
        throw std::runtime_error("NaN checker missing qNanBitPattern method");
    }
    uint64_t sNanBitPattern() const {
        py::gil_scoped_acquire a;
        if (py::hasattr(checker, "sNanBitPattern")) return checker.attr("sNanBitPattern")().cast<uint64_t>();
        throw std::runtime_error("NaN checker missing sNanBitPattern method");
    }
};

struct FloatingPointParamsPy {
    int bitwidth, mantissa_bits, bias;
    Inf_Behaviors OV_behavior;
    NaN_Behaviors NA_behavior;
    Signedness is_signed;
    py::object IsInf, IsNaN;

private:
    static void validate_inf_checker(const py::object &checker) {
        if (!py::hasattr(checker, "__call__")) throw std::invalid_argument("inf_checker must be callable");
        if (!py::hasattr(checker, "minNegInf")) throw std::invalid_argument("inf_checker must have minNegInf()");
        if (!py::hasattr(checker, "minPosInf")) throw std::invalid_argument("inf_checker must have minPosInf()");
    }
    static void validate_nan_checker(const py::object &checker) {
        if (!py::hasattr(checker, "__call__")) throw std::invalid_argument("nan_checker must be callable");
        if (!py::hasattr(checker, "qNanBitPattern") && !py::hasattr(checker, "sNanBitPattern"))
            throw std::invalid_argument("nan_checker must have qNanBitPattern() or sNanBitPattern()");
    }

public:
    FloatingPointParamsPy(int bw, int mb, int b, Inf_Behaviors ov, NaN_Behaviors na, Signedness sign, py::object inf_checker, py::object nan_checker)
        : bitwidth(bw), mantissa_bits(mb), bias(b), OV_behavior(ov), NA_behavior(na), is_signed(sign), IsInf(inf_checker), IsNaN(nan_checker)
    { validate_inf_checker(inf_checker); validate_nan_checker(nan_checker); }
};

static DeviceInfChecker make_device_inf(const FloatingPointParamsPy &p) {
    uint64_t exp_mask  = ((1ULL << (p.bitwidth - p.mantissa_bits - 1)) - 1) << p.mantissa_bits;
    uint64_t mant_mask = (1ULL << p.mantissa_bits) - 1;
    return { exp_mask, mant_mask, p.IsInf.attr("minNegInf")().cast<uint64_t>(), p.IsInf.attr("minPosInf")().cast<uint64_t>() };
}

static DeviceNaNChecker make_device_nan(const FloatingPointParamsPy &p, const DeviceInfChecker &inf) {
    return { inf.exp_mask, inf.mant_mask, p.IsNaN.attr("qNanBitPattern")().cast<uint64_t>(), p.IsNaN.attr("sNanBitPattern")().cast<uint64_t>() };
}

torch::Tensor virtual_round_mantissa(const torch::Tensor &input, int to_mantissa_bits, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) {
    auto output = torch::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_mantissa", ([&] {
        auto* in_ptr  = input.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();
        if (input.is_cuda()) {
#ifdef USE_CUDA
            round_mantissa(in_ptr, out_ptr, input.numel(), to_mantissa_bits, round_mode, stoch_len);
#else
            throw std::runtime_error("CUDA not available in this build");
#endif
        } else {
#ifndef USE_CUDA
            virtual_round(in_ptr, out_ptr, to_mantissa_bits, input.numel(), round_mode, stoch_len);
#endif
        }
    }));
    return output;
}

torch::Tensor virtual_round_params(const torch::Tensor &input, const FloatingPointParamsPy &params, Rounding_Mode round_mode = Rounding_Mode::RoundToNearestEven, int stoch_len = 0) {
    auto output = torch::empty_like(input);
    if (input.is_cuda()) {
        #ifdef USE_CUDA
        auto inf_dev = make_device_inf(params);
        auto nan_dev = make_device_nan(params, inf_dev);
        auto cpp_params = FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>{
            params.bitwidth, params.mantissa_bits, params.bias,
            params.OV_behavior, params.NA_behavior, params.is_signed,
            inf_dev, nan_dev, Unsigned_behavior::NegtoZero
        };
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_params_cuda", ([&] {
            output = input.clone();
            round_fp_params(output.data_ptr<scalar_t>(), output.numel(), cpp_params, round_mode, stoch_len);
        }));
        #else 
        throw std::runtime_error("CUDA not available in this build");
        #endif
    } else {
        #ifndef USE_CUDA
        PyInfCheckerAdapter inf_adapter{params.IsInf};
        PyNaNCheckerAdapter nan_adapter{params.IsNaN};
        auto cpp_params = FloatingPointParams<PyInfCheckerAdapter, PyNaNCheckerAdapter>{
            params.bitwidth, params.mantissa_bits, params.bias,
            params.OV_behavior, params.NA_behavior, params.is_signed,
            inf_adapter, nan_adapter, Unsigned_behavior::NegtoZero
        };
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_params_cpu", ([&] {
            virtual_round(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.numel(), cpp_params, round_mode, stoch_len);
        }));
        #endif
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Runtime float format converter with custom quantization";

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

    py::class_<FloatingPointParamsPy>(m, "FloatFormatDescriptor")
        .def(py::init<int, int, int, Inf_Behaviors, NaN_Behaviors, Signedness, py::object, py::object>(),
             py::arg("total_bits"), py::arg("exponent_bits"), py::arg("mantissa_bits"),
             py::arg("inf_behavior"), py::arg("nan_behavior"), py::arg("signedness"),
             py::arg("is_inf_checker"), py::arg("is_nan_checker"))
        .def_readonly("total_bits",    &FloatingPointParamsPy::bitwidth)
        .def_readonly("mantissa_bits", &FloatingPointParamsPy::mantissa_bits);

    m.def("virtual_round", &virtual_round_mantissa,
          py::arg("input"), py::arg("to_mantissa_bits"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven, py::arg("stoch_len") = 0);

    m.def("virtual_round", &virtual_round_params,
          py::arg("input"), py::arg("params"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven, py::arg("stoch_len") = 0);
}