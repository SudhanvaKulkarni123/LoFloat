#include "LoPy_kernels.hpp"
using namespace lo_float::internal;

int main() {

    FloatFormatSpec e4m3(
        4, 3, FloatFormatSpec::Signedness::Signed, FloatFormatSpec::InfBehavior::Extended
    );
    int N = 1000000;

    double* arr = (double*) malloc(N*sizeof(double));
    double* out = (double*) malloc(N*sizeof(double));

    virtual_round_float_arr(
        arr,
        e4m3,
        RoundingMode::RoundToNearestEven,
        0
    );
}