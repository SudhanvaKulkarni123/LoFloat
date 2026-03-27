#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include "lo_float.h"


namespace cg = cooperative_groups;
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

// template <typename From, int num_threads_per_cta = 256, int block_size = 32>
// __global__ void round_mantissa_kernel_mx(From* inout, int64_t n, int mantissa_bits, Rounding_Mode round_mode, int stoch_len) {
//     int64_t idx = blockIdx.x * num_threads_per_cta + threadIdx.x;
//     int warp_id = threadIdx.x / 32;
//     using mx_partition = cg::partition<block_size>;
//     if (idx < n) {
//         auto val = From[idx];
//         auto max_num = cg::reduce(mx_partition, val, cg::greater<From>());
//         auto max_exp = frexp(max_num);
//         cg::sync(max_partition);
//         val = virtual_round(From[idx]/max_exp, params, round_mode, stoch_len);
//         From[idx] = val
//     }

// }

template void round_mantissa<c10::Half>(const c10::Half*, c10::Half*, int64_t, int, Rounding_Mode, int);
template void round_mantissa<float>(const float*, float*, int64_t, int, Rounding_Mode, int);
template void round_mantissa<double>(const double*, double*, int64_t, int, Rounding_Mode, int);
template void round_fp_params<c10::Half>(c10::Half*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
template void round_fp_params<float>(float*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
template void round_fp_params<double>(double*, int64_t, FloatingPointParams<DeviceInfChecker, DeviceNaNChecker>, Rounding_Mode, int);
}