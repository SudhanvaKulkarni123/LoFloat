#include <torch/extension.h>
#include "lo_float.h"
#include "Lof_kernels.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using namespace lo_float;


#ifdef USE_CUDA
namespace lo_float {
template <typename T>
void round_mantissa(const T* in, T* out, int64_t n, int mantissa_bits, ProjSpec ps, T scale);

template <typename T>
void round_fp_params(T* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, ProjSpec ps, T scale);
// pwl_silu<T, N> is declared in Lof_kernels.h (one decl per supported length).
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

torch::Tensor virtual_round_mantissa(const torch::Tensor &input, int to_mantissa_bits, lo_float::Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven, int stoch_len = 0, double scale = 1.0) {
    auto output = torch::empty_like(input);
    #ifdef USE_CUDA
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "virtual_round_mantissa", ([&] {
        auto* in_ptr  = input.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();
        if (input.is_cuda()) {
#ifdef USE_CUDA
            round_mantissa(in_ptr, out_ptr, input.numel(), to_mantissa_bits, ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len}, static_cast<scalar_t>(scale));
#else
            throw std::runtime_error("CUDA not available in this build");
#endif
        } else {
#ifndef USE_CUDA
            // CPU path: pre-multiply the input by scale (less perf-critical than GPU).
            auto scaled = (scale == 1.0) ? input : input.mul(scale);
            virtual_round(scaled.data_ptr<scalar_t>(), out_ptr, to_mantissa_bits, input.numel(), ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len});
#endif
        }
    }));
    #else
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_mantissa", ([&] {
        auto* in_ptr  = input.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();
        if (input.is_cuda()) {
#ifdef USE_CUDA
            round_mantissa(in_ptr, out_ptr, input.numel(), to_mantissa_bits, ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len}, static_cast<scalar_t>(scale));
#else
            throw std::runtime_error("CUDA not available in this build");
#endif
        } else {
#ifndef USE_CUDA
            auto scaled = (scale == 1.0) ? input : input.mul(scale);
            virtual_round(scaled.data_ptr<scalar_t>(), out_ptr, to_mantissa_bits, input.numel(), ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len});
#endif
        }
    }));
    #endif

    return output;
}

torch::Tensor virtual_round_params(const torch::Tensor &input, const FloatingPointParamsPy &params, lo_float::Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven, int stoch_len = 0, double scale = 1.0) {
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
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "virtual_round_params_cuda", ([&] {
            output = input.clone();
            round_fp_params(output.data_ptr<scalar_t>(), output.numel(), cpp_params, ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len}, static_cast<scalar_t>(scale));
        }));
        #else
        throw std::runtime_error("CUDA not available in this build");
        #endif
    } else {
        #ifndef USE_CUDA
        // CPU path: pre-multiply the input by scale (less perf-critical than GPU).
        auto scaled = (scale == 1.0) ? input : input.mul(scale);
        PyInfCheckerAdapter inf_adapter{params.IsInf};
        PyNaNCheckerAdapter nan_adapter{params.IsNaN};
        auto cpp_params = FloatingPointParams<PyInfCheckerAdapter, PyNaNCheckerAdapter>{
            params.bitwidth, params.mantissa_bits, params.bias,
            params.OV_behavior, params.NA_behavior, params.is_signed,
            inf_adapter, nan_adapter, Unsigned_behavior::NegtoZero
        };
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "virtual_round_params_cpu", ([&] {
            virtual_round(scaled.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.numel(), cpp_params, ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len});
        }));
        #endif
    }
    return output;
}

// Microscaling (MX) block fake-quantization. Float32 only. The tensor is
// partitioned into contiguous blocks of `block_size` elements (the caller keeps
// the MX/reduction axis last and contiguous, with block_size dividing it); each
// block gets one auto-computed shared scale = round(amax/priv_max_normal) in the
// scale_format, then each element is round(x/scale) in element_format, rescaled.
// Returns a new fp32 tensor in the original numeric domain.
torch::Tensor virtual_mx_round(
        const torch::Tensor &input, int block_size,
        const FloatingPointParamsPy &element_params,
        const FloatingPointParamsPy &scale_params,
        lo_float::Rounding_Mode round_mode = lo_float::Rounding_Mode::RoundToNearestEven,
        lo_float::Rounding_Mode scale_round_mode = lo_float::Rounding_Mode::RoundToNearestEven,
        int stoch_len = 0) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "virtual_mx_round: float32 tensors only");
    TORCH_CHECK(block_size > 0, "virtual_mx_round: block_size must be positive");

    auto out = input.contiguous().clone();
    const int64_t n = out.numel();

    // Inf/NaN checkers are unused by virtual_round; build params with the device
    // checker type in both paths (zero-filled on CPU) so results match exactly.
    auto build = [](const FloatingPointParamsPy &p) {
        return FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>{
            p.bitwidth, p.mantissa_bits, p.bias, p.OV_behavior, p.NA_behavior,
            p.is_signed, DeviceInfChecker{0,0,0,0}, DeviceNaNChecker{0,0,0,0},
            Unsigned_behavior::NegtoZero };
    };
    auto priv = build(element_params);
    auto pub  = build(scale_params);

    if (input.is_cuda()) {
        #ifdef USE_CUDA
        lo_float::run_virtual_mx_round(out.data_ptr<float>(), n, block_size,
                                       pub, priv, ProjSpec{scale_round_mode},
                                       ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len});
        #else
        throw std::runtime_error("CUDA not available in this build");
        #endif
    } else {
        // CPU scalar MX (works in both builds; mirrors the kernel exactly).
        const int me = priv.is_signed == Signedness::Signed
            ? (1 << (priv.bitwidth - priv.mantissa_bits - 1)) - 1 - priv.bias
            : (1 << (priv.bitwidth - priv.mantissa_bits))     - 1 - priv.bias;
        const float pmn = std::ldexp(1.0f, me) * (2.0f - std::ldexp(1.0f, -priv.mantissa_bits));
        float* p = out.data_ptr<float>();
        for (int64_t b = 0; b < n; b += block_size) {
            const int len = (int)std::min<int64_t>(block_size, n - b);
            float amax = 0.0f;
            for (int i = 0; i < len; ++i) amax = std::max(amax, std::fabs(p[b + i]));
            float scale = lo_float::virtual_round(amax / pmn, pub, ProjSpec{scale_round_mode});
            if (scale > 0.0f)
                for (int i = 0; i < len; ++i)
                    p[b + i] = lo_float::virtual_round(p[b + i] / scale, priv, ProjSpec{round_mode, Saturation_Mode::OvfInf, stoch_len}) * scale;
            else
                for (int i = 0; i < len; ++i) p[b + i] = 0.0f;
        }
    }
    return out.view_as(input);
}

// Maps a runtime length N to the matching compile-time kernel instantiation.
// Defined at file scope (not inside the AT_DISPATCH lambda) because preprocessor
// directives are not allowed within a macro argument.
#define LOF_PWL_SILU_CASE(NV) case NV: pwl_silu<scalar_t, NV>(ip, op, lp, n, Rf); break;

// Piecewise-linear SiLU approximation. `lut` is a 1-D tensor of N+1 uniform
// knots over [-R, R] (built by PWLSiLU.init_lut). On GPU it dispatches the
// runtime length N to the matching compile-time kernel instantiation; on CPU it
// runs an equivalent scalar loop. Output matches PWLSiLU.forward.
torch::Tensor pwl_silu_forward(const torch::Tensor &input, const torch::Tensor &lut, double R) {
    TORCH_CHECK(lut.dim() == 1 && lut.numel() >= 2, "pwl_silu: lut must be 1-D with >= 2 entries");
    TORCH_CHECK(R > 0.0, "pwl_silu: R must be positive");
    const int N = static_cast<int>(lut.numel()) - 1;   // segment count (power of 2)
    auto input_c = input.contiguous();
    auto output  = torch::empty_like(input_c);
    const float Rf = static_cast<float>(R);

    if (input.is_cuda()) {
        #ifdef USE_CUDA
        auto lut_f = lut.to(input.device(), at::kFloat).contiguous();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "pwl_silu_cuda", ([&] {
            auto* ip = input_c.data_ptr<scalar_t>();
            auto* op = output.data_ptr<scalar_t>();
            const float* lp = lut_f.data_ptr<float>();
            const int64_t n = input_c.numel();
            switch (N) {
                LOF_PWL_SILU_LENGTHS(LOF_PWL_SILU_CASE)
                default:
                    TORCH_CHECK(false, "pwl_silu: lut length ", N,
                                " is not a supported power of 2 in [2, 4096]");
            }
        }));
        #else
        throw std::runtime_error("CUDA not available in this build");
        #endif
    } else {
        auto lut_f = lut.to(at::kCPU, at::kFloat).contiguous();
        const float* lp = lut_f.data_ptr<float>();
        const float inv_step = static_cast<float>(N) / (2.0f * Rf);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "pwl_silu_cpu", ([&] {
            const scalar_t* ip = input_c.data_ptr<scalar_t>();
            scalar_t* op = output.data_ptr<scalar_t>();
            const int64_t n = input_c.numel();
            for (int64_t k = 0; k < n; ++k) {
                float x  = static_cast<float>(ip[k]);
                float xc = std::min(std::max(x, -Rf), Rf);
                float u  = (xc + Rf) * inv_step;
                int   i  = static_cast<int>(u);
                if (i > N - 1) i = N - 1;
                float frac = u - static_cast<float>(i);
                float y0 = lp[i], y1 = lp[i + 1];
                float y  = std::fma(frac, y1 - y0, y0);
                if (std::fabs(x) > Rf) y = x > 0.0f ? x : 0.0f;
                op[k] = static_cast<scalar_t>(y);
            }
        }));
    }
    return output;
}
#undef LOF_PWL_SILU_CASE


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
        .value("RoundToOdd", Rounding_Mode::RoundToOdd)
        .value("RoundTiesTowardsZero", Rounding_Mode::RoundTiesTowardsZero)
        .value("StochasticRoundingD", Rounding_Mode::StochasticRoundingD)
        .export_values();

    py::enum_<NaN_Behaviors>(m, "NaNBehavior")
        .value("_3109", NaN_Behaviors::_3109)
        .value("_754", NaN_Behaviors::_754)
        .export_values();

    py::class_<FloatingPointParamsPy>(m, "FloatFormatDescriptor")
    .def(py::init<int, int, int, Inf_Behaviors, NaN_Behaviors, Signedness, py::object, py::object>(),
    py::arg("total_bits"), py::arg("mantissa_bits"), py::arg("bias"),
    py::arg("inf_behavior"), py::arg("nan_behavior"), py::arg("signedness"),
    py::arg("is_inf_checker"), py::arg("is_nan_checker"))
        .def_readonly("total_bits",    &FloatingPointParamsPy::bitwidth)
        .def_readonly("mantissa_bits", &FloatingPointParamsPy::mantissa_bits)
        .def_readonly("bias",          &FloatingPointParamsPy::bias)
        .def_readonly("inf_behavior",  &FloatingPointParamsPy::OV_behavior)
        .def_readonly("nan_behavior",  &FloatingPointParamsPy::NA_behavior)
        .def_readonly("signedness",    &FloatingPointParamsPy::is_signed)
        .def_readonly("is_inf_checker",&FloatingPointParamsPy::IsInf)
        .def_readonly("is_nan_checker",&FloatingPointParamsPy::IsNaN);

    m.def("virtual_round", &virtual_round_mantissa,
          py::arg("input"), py::arg("to_mantissa_bits"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven, py::arg("stoch_len") = 0,
          py::arg("scale") = 1.0,
          "Round to a target mantissa bitwidth. If scale != 1, computes round(scale * input) (fused on CUDA).");

    m.def("virtual_round", &virtual_round_params,
          py::arg("input"), py::arg("params"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven, py::arg("stoch_len") = 0,
          py::arg("scale") = 1.0,
          "Round to a custom float format. If scale != 1, computes round(scale * input) (fused on CUDA).");

    m.def("virtual_mx_round", &virtual_mx_round,
          py::arg("input"), py::arg("block_size"),
          py::arg("element_format"), py::arg("scale_format"),
          py::arg("round_mode") = Rounding_Mode::RoundToNearestEven,
          py::arg("scale_round_mode") = Rounding_Mode::RoundToNearestEven,
          py::arg("stoch_len") = 0,
          "Microscaling (MX) block fake-quantization (float32). Splits the tensor into "
          "contiguous blocks of `block_size` (keep the reduction axis last & contiguous, "
          "block_size dividing it), computes one shared scale per block in `scale_format` "
          "(e.g. E8M0), rounds each element into `element_format`, and rescales. Returns a "
          "new fp32 tensor in the original domain.");

    m.def("pwl_silu", &pwl_silu_forward,
          py::arg("input"), py::arg("lut"), py::arg("R"),
          "Piecewise-linear SiLU approximation. `lut` holds N+1 uniform knots over "
          "[-R, R]; interpolates the table and applies the relu asymptote outside the "
          "range. Fused CUDA kernel on GPU tensors, scalar loop on CPU. Matches "
          "PWLSiLU.forward.");

    #ifdef USE_CUDA
    m.def("lof_gemm",
        &lo_float::LoF_gemm,
        py::arg("A"),
        py::arg("B"),
        py::arg("accum_mant_bits"),
        py::arg("round_mode")              = Rounding_Mode::RoundToNearestEven,
        py::arg("stochastic_rounding_bits") = 0,
        py::arg("scale_a")                 = 1.0,
        py::arg("scale_b")                 = 1.0,
        "Low-precision GEMM (float32 only). Output is divided by (scale_a * scale_b) "
        "to rescale back to the original (unscaled) domain when A, B were obtained by "
        "scale-then-quantize. Returns output tensor D.");

    m.def("lof_conv2d",
        &lo_float::LoF_conv2d,
        py::arg("input"),
        py::arg("weight"),
        py::arg("pad_h"),
        py::arg("pad_w"),
        py::arg("stride_h"),
        py::arg("stride_w"),
        py::arg("dilation_h"),
        py::arg("dilation_w"),
        py::arg("accum_mant_bits"),
        py::arg("round_mode")              = Rounding_Mode::RoundToNearestEven,
        py::arg("stochastic_rounding_bits") = 0,
        py::arg("weight_scale")            = 1.0,
        py::arg("input_scale")             = 1.0,
        "Low-precision Conv2d forward (float32 only, groups=1) via CUTLASS implicit "
        "GEMM. input: NCHW, weight: (C_out, C_in, kH, kW). Output is divided by "
        "(weight_scale * input_scale) to rescale back to the unscaled domain. "
        "Returns NCHW output.");
    #endif
}