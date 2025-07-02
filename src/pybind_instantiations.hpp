#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "lo_float.h"

using namespace lo_float;
namespace py = pybind11;

// --- Macros for Name Encoding ---
#define SIGN_ABBREV(A) ((A) == Signedness::Signed ? "s" : "u")
#define INF_ABBREV(A) \
    ((A) == Inf_Behaviors::Extended ? "e" : \
     (A) == Inf_Behaviors::Saturating ? "f" : "t")  // 't' for Saturating (trapping-style)

#define ROUND_MODE_ABBREV(A) \
    ((A) == Rounding_Mode::RoundToNearestEven ? "rne" : "x")

#define IEEE_NAME(K, E, S, I) \
    (std::string("binary") + std::to_string(K) + "p" + std::to_string(E) + \
     SIGN_ABBREV(S) + INF_ABBREV(I))

// --- Binding Macro ---
#define BIND_FLOAT(K, E, S, I) \
{ \
    using FloatType = P3109_float<K, E, S, I>; \
    std::string type_name = IEEE_NAME(K, E, S, I); \
    py::class_<FloatType>(m, type_name.c_str()) \
        .def(py::init<>()) \
        .def(py::init<float>()) \
        .def(py::self + py::self) \
        .def(py::self - py::self) \
        .def(py::self * py::self) \
        .def(py::self / py::self) \
        .def(py::self == py::self) \
        .def(py::self != py::self) \
        .def(py::self < py::self) \
        .def(py::self <= py::self) \
        .def(py::self > py::self) \
        .def(py::self >= py::self) \
        .def("add", [](const FloatType& a, const FloatType& b, const Rounding_Mode& rm, const int& stoch_len){ return add(a,b,rm,stoch_len); }) \
        .def("sub", [](const FloatType& a, const FloatType& b, const Rounding_Mode& rm, const int& stoch_len){ return sub(a,b,rm,stoch_len); }) \
        .def("mul", [](const FloatType& a, const FloatType& b, const Rounding_Mode& rm, const int& stoch_len){ return mul(a,b,rm,stoch_len); }) \
        .def("div", [](const FloatType& a, const FloatType& b, const Rounding_Mode& rm, const int& stoch_len){ return div(a,b,rm,stoch_len); }) \
        .def("__float__", [](const FloatType& f) { return static_cast<float>(f); }) \
        .def("__repr__", [type_name](const FloatType& f) { return type_name + "(" + std::to_string(static_cast<float>(f)) + ")"; }); \
}

// --- Enum Lists ---
constexpr Signedness SIGNEDNESS_LIST[] = {
    Signedness::Signed, Signedness::Unsigned
};

constexpr Inf_Behaviors INF_LIST[] = {
    Inf_Behaviors::Saturating, Inf_Behaviors::Extended, Inf_Behaviors::Saturating
};


