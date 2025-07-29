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
from LoFloat import fake_mx_quantize_tensor
from LoFloat import real_mx_quantize_tensor


__all__ = ["Quantizer"]



def get_3109_string(**kwargs):
    return "binary" + str(kwargs['k']) + "p" + str(kwargs['p'])


LoFloat_supported_types = {
    "binary8p4se", "binary8p4sf"
}

LoFloat_supported_MX_types = {
    "binary8p1ue", "binary7p4uf"
}


@torch.fx.wrap
class LoFloatFakeQuant(nn.Module):
    def __init__(self, scale, zero_point, k, p, signedness, inf_behavior):
        
        super().__init__()

        type_string = get_3109_string(k=k, p=p, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")
        
        self.scale = scale
        self.zero_point = zero_point
        
        #nned some kind of global list to add to whenever I add a new type.... Two possible options here. One throw error message if type unavailable, 2 perform some kind of JIt. The former is lesser work - so I will use that

#k, p, signedness, saturating p3109<8, 4, ..>; <8, 5,...> <6, ...>
# PyTorch -> fake_quantize -> [fake_quantize<8, 4>, fake_quantize<8, 5>]
    def forward(self, x):
        x = fake_quantize_tensor(x, self.scale, self.zero_point, self.k, self.p, self.signedness, self.inf_behavior)
        return x
    
    def backward(self, x):
        return x
    
@torch.fx.wrap
class LoFloatMXFakeQuant(nn.Module):
    def __init__(self, scale, zero_point, k, p, signedness, inf_behavior, block_size, k_scale, p_scale, inf_behavior_scale):
        super().__init__()
        type_string = get_3109_string(k=k, p=p, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported private type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")
        type_string = get_3109_string(k=k_scale, p=p_scale, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported scaling type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")
    
        self.scale = scale
        self.zp = zero_point
        self.k = k
        self.p = p
        self.signedness = signedness
        self.inf_behavior = inf_behavior
        self.block_size = block_size
        self.k_scale = k_scale
        self.p_scale = p_scale
        self.inf_behavior_scale = inf_behavior_scale

    def forward(self, x):
        x = fake_mx_quantize_tensor(x, self.scale, self.zero_point, self.k, self.p, self.signedness, self.inf_behavior,
                                 self.block_size, self.k_scale, self.p_scale, self.inf_behavior_scale)
        return x
    def backward(self, x):
        return x

@torch.fx.wrap
class LoFloatRealQuant(nn.Module):
    def __init__(self, scale, zero_point, k, p, signedness, inf_behavior):
        super().__init__()
        type_string = get_3109_string(k=k, p=p, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")
        
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = real_quantize_tensor(x, self.scale, self.zero_point, self.k, self.p, self.signedness, self.inf_behavior)
        return x
    
    def backward(self, x):
        return x

@torch.fx.wrap
class LoFloatMXRealQuant(nn.Module):
    def __init__(self, scale, zero_point, k, p, signedness, inf_behavior, block_size, k_scale, p_scale, inf_behavior_scale):
        super().__init__()
        type_string = get_3109_string(k=k, p=p, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported private type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")
        type_string = get_3109_string(k=k_scale, p=p_scale, signedness = signedness, inf_behavior = inf_behavior)
        if type_string not in LoFloat_supported_types:
            raise SystemExit(f"err: unsupported scaling type '{type_string}', Add support by instantiating type in file PyTorch_extensions.cpp")

        self.scale = scale
        self.zp = zero_point
        self.k = k
        self.p = p
        self.signedness = signedness
        self.inf_behavior = inf_behavior
        self.block_size = block_size
        self.k_scale = k_scale
        self.p_scale = p_scale
        self.inf_behavior_scale = inf_behavior_scale
    def forward(self, x):
        x = real_mx_quantize_tensor(x, self.scale, self.zero_point, self.k, self.p, self.signedness, self.inf_behavior,
                                 self.block_size, self.k_scale, self.p_scale, self.inf_behavior_scale)
        return x
    def backward(self, x):
        return x
        
QAT_mapping = {
    LoFloatFakeQuant : LoFloatRealQuant,
    LoFloatMXFakeQuant : LoFloatMXRealQuant
}

