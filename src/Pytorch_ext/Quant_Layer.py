import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize
import numpy as np

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, module_path)

import LoFloat
from LoFloat import *

__all__ = ["Quantizer"]

#need a function to return data type based on input params
floatingpoint_list = {
    "binary8p4sf": binary8p4sf,
    
}

def get_dtype(bitwidth : int, mantissa_bits : int, signed : bool, extended : bool):
    #construct string for dtype
    dtpe_str = f"binary{bitwidth}p{mantissa_bits}"
    if signed :
        if extended:
            dtpe_str += "se"
        else:
            dtpe_str += "sf"
    else:
        if extended:
            dtpe_str += "ue"
        else:
            dtpe_str += "uf"

    return floatingpoint_list[dtpe_str]
        


def quantizer(bitwidth: int = 8, mantissa_bits: int = 4, bias: int = 0, signed: bool = True, extended: bool = False):
    

class LoFloatFakeQuant(FakeQuantize):
    def forward(self, x, **kwargs):
        self.scale = kwargs.get('scale', 1.0)
        self.zero_point = kwargs.get('zero_point', 0)
        self.quant_min = kwargs.get('quant_min', 0)
        self.quant_max = kwargs.get('quant_max', 255)
        x = (x / self.scale + self.zero_point).clamp(self.quant_min, self.quant_max).round()
        x = (x - self.zero_point) * self.scale
        x = torch.float32(kwargs.get('type_cast', torch.float32)(x))
        return x
    


class CustomLinear(nn.Module):
    def forward(self, x : Tensor, A: Tensor):
        

