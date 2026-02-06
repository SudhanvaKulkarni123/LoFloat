import torch
import torch.nn as nn
from typing import Optional

from torch.utils.cpp_extension import load

fp_sim = load(name='lofloat', sources=['LoPy_bind.cpp'])


class LoF_quantize(nn.Module):
    def __init__(self, format, rounding_mode: str = "nearest", use_float64: bool = False):
        super().__init__()
        self.format = format
        self.rounding_mode = rounding_mode
        self.use_float64 = use_float64
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_float64:
            return fp_sim.quantize_float64(x.data, self.format, self.rounding_mode)
        else:
            return fp_sim.quantize_float32(x.data, self.format, self.rounding_mode)


class LoF_fake_quantize(nn.Module):

    def __init__(self, format, rounding_mode: str = "nearest", use_float64: bool = False):

        super().__init__()
        self.format = format
        self.rounding_mode = rounding_mode
        self.use_float64 = use_float64
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_float64:
            return fp_sim.fake_quantize_float64(x.data, self.format, self.rounding_mode)
        else:
            return fp_sim.fake_quantize_float32(x.data, self.format, self.rounding_mode)

class LoF_Linear(nn.Module):
    """
    Linear layer with fake quantization applied to both weights and activations.
    Accumulation is performed in float32, and output is float32.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        format_weight,
        format_activation,
        bias: bool = True,
        rounding_mode: str = "nearest",
        device=None,
        dtype=None
    ):
        """
        Args:
            in_features: Size of input features
            out_features: Size of output features
            format_weight: Quantization format for weights
            format_activation: Quantization format for activations
            bias: If True, adds a learnable bias (default: True)
            rounding_mode: Rounding mode for quantization (default: "nearest")
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_features = in_features
        self.out_features = out_features
        self.format_weight = format_weight
        self.format_activation = format_activation
        self.rounding_mode = rounding_mode
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Initialize fake quantization layers
        self.weight_fake_quant = LoF_fake_quantize(
            format_weight, 
            rounding_mode=rounding_mode,
            use_float64=False
        )
        self.activation_fake_quant = LoF_fake_quantize(
            format_activation,
            rounding_mode=rounding_mode,
            use_float64=False
        )
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_quant = self.activation_fake_quant(x)
        w_quant = self.weight_fake_quant(self.weight)
        x_quant = x_quant.float()
        w_quant = w_quant.float()
        output = torch.matmul(x_quant, w_quant.t())
        if self.bias is not None:
            output = output + self.bias
        
        return output 
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, rounding_mode={self.rounding_mode}')


