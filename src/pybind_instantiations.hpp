/// @author Sudhanva Kulkarni
/// @breif This file has instantiations of some defualt templates that are visible from Python through the use of pybind11

//instantiate all the 8-bit floats
#include <pybind11/pybind11.h>
#include "lo_float.h"
#include "lo_int.h"
#include "lo_float_sci.hpp"


using namespace lo_float;

namespace py = pybind11;

using namespace Lo_Gemm;

#define SIGN_ABBREV(A) \
    A == Signedness::Signed ? "s" : "u"

#define INF_ABBREV(A) \
    A == Inf_Behaviors::Extended ? "e" : "f"

#define NAN_ABBREV(A) \
    A == NaN_Behaviors::QuietNaN ? "q" : A == NaN_Behaviors::NoNaN ? "n" : "s"

#define ROUND_MODE_ABBREV(A) \
    A == Rounding_Mode::RoundToNearestEven ? "rne" : \
    A == Rounding_Mode::RoundTowardsZero ? "rtz" : \
    A == Rounding_Mode::RoundAwayFromZero ? "raw" : \
    A == Rounding_Mode::StochasticRoundingA ? "sra" : \
    A == Rounding_Mode::RoundToNearestOdd ? "rno" : \
    A == Rounding_Mode::RoundDown ? "rd" : \
    A == Rounding_Mode::RoundUp ? "ru" : \
    A == Rounding_Mode::RoundTiesToAway ? "rta" : \
    A == Rounding_Mode::StochasticRoundingB ? "srb" : \
    A == Rounding_Mode::StochasticRoundingC ? "src" : \
    A == Rounding_Mode::True_StochasticRounding ? "tsr" : \
    A == Rounding_Mode::ProbabilisticRounding ? "pr" : ""

#define U_BEHAVIOR_ABBREV(A) \
    A == Unsigned_behavior::NegtoZero ? "nz" : \
    A == Unsigned_behavior::NegtoNaN ? "nn" : ""

#define NUM_STOCH(A)    \
    A == 0 ? "" : A



/*  @params A is bitwidth, B is mantissa bits, C is signedness, D is finiteness, E is the rounding mode*/
#define IEEE_NAME(A, B, C, D, E) \
    "binary" #A "p" #B SIGN_ABBREV(#C) INF_ABBREV(#D) "_" ROUND_MODE_ABBREV(#E)

/*  @params first 5 are same as IEEE name, F is bias, G is the behaviour of NaN, H is number of bits used for stochastic rounding (ignored if 0), */
#define LOF_NAME(A, B, C, D, E, F, G, H, I) \
    "float" #A "_" #B SIGN_ABBREV(#C) INF_ABBREV(#D) "_" ROUND_MODE_ABBREV(#E) "_b" #F "_" NAN_ABBREV(#G) "_" NUM_STOCH(#H) "_" U_BEHAVIOR_ABBREV(#I)

#define BIND_ONE(A, B, C, D, E, F, G, H, I) \
    py::class_<Templated_Float<FloatingPointParams{Rounding_Mode::A, Signedness::B, Unsigned_behavior::C}> \
    (m, ("T_" #A "_" #B "_" #C).c_str()) \
        .def(py::init<>()) \
        .def("get_value", &T<Rounding_Mode::A, Signedness::B, Unsigned_behavior::C>::get_value);


PYBIND11_MODULE(your_module_name, m) {
   
}
