#include "lo_float.h"
#include <cuda_runtime.h>

namespace lo_float {
template <typename From>
__global__ void round_mantissa_kernel(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = virtual_round(in[idx], mantissa_bits, round_mode, stoch_len);
}
template <typename From>
void round_mantissa(const From* in, From* out, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len) {
    round_mantissa_kernel<<<(n + 255) / 256, 256>>>(in, out, n, mantissa_bits, round_mode, stoch_len);
}
template <typename From>
__global__ void round_fp_params_kernel(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) inout[idx] = virtual_round(inout[idx], params, round_mode, stoch_len);
}
template <typename From>
void round_fp_params(From* inout, int64_t n, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker> params, Rounding_Mode round_mode, int stoch_len) {
    round_fp_params_kernel<<<(n + 255) / 256, 256>>>(inout, n, params, round_mode, stoch_len);
}

template void round_mantissa<float>(const float*, float*, int64_t, int, Rounding_Mode, int);
template void round_mantissa<double>(const double*, double*, int64_t, int, Rounding_Mode, int);
template void round_fp_params<float>(float*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
template void round_fp_params<double>(double*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
}