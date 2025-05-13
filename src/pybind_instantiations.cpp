#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "lo_float.h"

using namespace lo_float;

namespace py = pybind11;

#define SIGN_ABBREV(A) ((A) == Signedness::Signed ? "s" : "u")
#define INF_ABBREV(A) ((A) == Inf_Behaviors::Extended ? "e" : "f")
#define NAN_ABBREV(A) ((A) == NaN_Behaviors::QuietNaN ? "q" : (A) == NaN_Behaviors::NoNaN ? "n" : "s")
#define ROUND_MODE_ABBREV(A) \
    ((A) == Rounding_Mode::RoundToNearestEven ? "rne" : \
    (A) == Rounding_Mode::RoundTowardsZero ? "rtz" : \
    (A) == Rounding_Mode::RoundAwayFromZero ? "raw" : \
    (A) == Rounding_Mode::StochasticRoundingA ? "sra" : \
    (A) == Rounding_Mode::RoundToNearestOdd ? "rno" : \
    (A) == Rounding_Mode::RoundDown ? "rd" : \
    (A) == Rounding_Mode::RoundUp ? "ru" : \
    (A) == Rounding_Mode::RoundTiesToAway ? "rta" : \
    (A) == Rounding_Mode::StochasticRoundingB ? "srb" : \
    (A) == Rounding_Mode::StochasticRoundingC ? "src" : \
    (A) == Rounding_Mode::True_StochasticRounding ? "tsr" : \
    (A) == Rounding_Mode::ProbabilisticRounding ? "pr" : "")
#define U_BEHAVIOR_ABBREV(A) ((A) == Unsigned_behavior::NegtoZero ? "nz" : (A) == Unsigned_behavior::NegtoNaN ? "nn" : "")
#define NUM_STOCH(A) ((A) == 0 ? "" : #A)

#define IEEE_NAME(A, B, C, D, E) ( \
    std::string("binary") + std::to_string(A) + "p" + std::to_string(B) + \
    SIGN_ABBREV(C) + INF_ABBREV(D) + "_" + ROUND_MODE_ABBREV(E))

#define BIND_IEEE_8(A, B, C, D, E) \
PYBIND11_MODULE(example, m) { \
    using FloatType = float8_ieee_p<(B - 1)>; \
    std::string type_name = IEEE_NAME(A, B, C, D, E); \
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
        .def("__repr__", [type_name](const FloatType& f) { \
            return type_name + "(" + std::to_string(static_cast<float>(f)) + ")"; \
        }); \
}

BIND_IEEE_8(8, 4, Signedness::Signed, Inf_Behaviors::Saturating, Rounding_Mode::RoundToNearestEven);
