#include "LoPy_kernels.hpp"

namespace lo_float {
namespace lo_float_internal {

// Explicit template instantiations for RuntimeFloatConverter

// Float → uint8/uint16/uint32/uint64
template class RuntimeFloatConverter<float, uint8_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<float, uint16_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<float, uint32_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<float, uint64_t, NaN_Behaviors::QuietNaN, xs::default_arch>;

// Double → uint8/uint16/uint32/uint64
template class RuntimeFloatConverter<double, uint8_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<double, uint16_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<double, uint32_t, NaN_Behaviors::QuietNaN, xs::default_arch>;
template class RuntimeFloatConverter<double, uint64_t, NaN_Behaviors::QuietNaN, xs::default_arch>;

} // namespace lo_float_internal
} // namespace lo_float