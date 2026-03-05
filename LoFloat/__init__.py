from .LoFloat import virtual_round, RoundingMode, InfBehavior, NaNBehavior, Signedness, FloatFormatDescriptor  # compiled C++ extension
from .formats import create_p3109_params, create_half_params, create_single_params
from .layers import LoF_Quantize, LoF_Linear, LoF_Conv2d, _quantize, mantissa_quantize, exp_mant_quantize
from .utils import set_mantissa_fields, set_exponent_fields, set_exponentbias_fields, lofloatify, set_all_to_half, set_all_to_3109, print_exp_mant, record_formats