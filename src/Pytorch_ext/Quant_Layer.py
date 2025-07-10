import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.quantization_mappings import get_default_static_quant_module_mappings
# use this to get the mapping from high prec to low prec layers - mapping = get_default_static_quant_module_mappings()
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'Documents', 'LoFloat', 'src'))
sys.path.insert(0, module_path)

import LoFloat
from LoFloat import fake_quantize_tensor
from LoFloat import real_quantize_tensor



__all__ = ["Quantizer"]




@torch.fx.wrap
class LoFloatFakeQuant(nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
#k, p, signedness, saturating p3109<8, 4, ..>; <8, 5,...> <6, ...>
# PyTorch -> fake_quantize -> [fake_quantize<8, 4>, fake_quantize<8, 5>]
    def forward(self, x):
        x = fake_quantize_tensor(x, self.scale, self.zero_point)
        return x
    
    def backward(self, x):
        return x

@torch.fx.wrap
class LoFloatRealQuant(nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = real_quantize_tensor(x, self.scale, self.zero_point)
        return x
    
    def backward(self, x):
        return x

        
QAT_mapping = {
    LoFloatFakeQuant : LoFloatRealQuant
}

