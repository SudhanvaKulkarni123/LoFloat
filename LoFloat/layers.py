import torch
import torch.nn as nn
import torch.nn.functional as F
import LoFloat as lof

class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, params, rounding_mode=lof.RoundingMode.RoundToNearestEven):
        return lof.virtual_round(tensor, params, round_mode=rounding_mode, stoch_len=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

    
class LoF_Quantize(nn.Module):
    def __init__(self, params, rounding_mode=None):
        super().__init__()
        self.params = params
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven

    def __deepcopy__(self, memo):
        new = LoF_Quantize(self.params, self.rounding_mode)
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def forward(self, x):
        return STERound.apply(x, self.params, self.rounding_mode)

    def extra_repr(self):
        return f'params={self.params}, rounding_mode={self.rounding_mode}'

def _quantize(self, tensor, params):
    if params is None:
        return tensor
    return STERound.apply(tensor, params, self.rounding_mode)

def mantissa_quantize(tensor, mantissa_bits):
    if mantissa_bits is None:
        return tensor
    return STERound.apply(tensor, lof.create_p3109_params(8 + mantissa_bits + 1, mantissa_bits, True, True))

def exp_mant_quantize(tensor, exp_bits, mantissa_bits):
    if exp_bits is None or mantissa_bits is None:
        return tensor
    return STERound.apply(tensor, lof.create_p3109_params(exp_bits + mantissa_bits + 1, mantissa_bits, True, True))

class LoF_Linear(nn.Module):

    @staticmethod
    def _resolve_params(exp, mant, params):
        if params is not None:
            return params
        if exp is not None and mant is not None:
            return lof.create_p3109_params(exp + mant + 1, mant)
        raise ValueError("Must provide either (exp, mant) or params for each tensor")

    def __init__(self, in_features, out_features,
                 act_params=None, weight_params=None,
                 bias=True, bias_params=None,
                 act_exp=None, act_mant=None,
                 weight_exp=None, weight_mant=None,
                 bias_exp=None, bias_mant=None,
                 rounding_mode=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven

        if act_params is None:
            self.act_params = lof.create_p3109_params(act_exp + act_mant + 1, act_mant, True, True)
        else:
            self.act_params = act_params

        if weight_params is None:
            self.weight_params = lof.create_p3109_params(weight_exp + weight_mant + 1, weight_mant, True, True)
        else:
            self.weight_params = weight_params

        if bias_params is None:
            self.bias_params = lof.create_p3109_params(bias_exp + bias_mant + 1, bias_mant, True, True)
        else:
            self.bias_params = bias_params

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, act_params, weight_params, bias_params=None, rounding_mode=None):
        layer = cls(
            linear.in_features,
            linear.out_features,
            act_params=act_params,
            weight_params=weight_params,
            bias=linear.bias is not None,
            bias_params=bias_params,
            rounding_mode=rounding_mode,
        )
        layer.weight = nn.Parameter(self._quantize(linear.weight.clone(), layer.weight_params))
        if linear.bias is not None:
            layer.bias = nn.Parameter(self._quantize(linear.bias.clone(), layer.bias_params))
        return layer

    def __deepcopy__(self, memo):
        new = LoF_Linear(
            self.in_features, self.out_features,
            act_params=self.act_params,
            weight_params=self.weight_params,
            bias=self.bias is not None,
            bias_params=self.bias_params,
            rounding_mode=self.rounding_mode,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, x):
        x_q = self._quantize(x, self.act_params)
        w_q = self._quantize(self.weight, self.weight_params)
        out = torch.matmul(x_q, w_q.t())
        if self.bias is not None:
            b_q = self._quantize(self.bias, self.bias_params)
            out = out + b_q
        return out


    def set_mantissa(self, activ_mant, weight_mant, bias_mant):
        self.act_params = lof.create_p3109_params(self.act_params.total_bits, activ_mant, True, True)
        self.weight_params = lof.create_p3109_params(self.weight_params.total_bits, weight_mant, True, True)
        self.bias_params = lof.create_p3109_params(self.bias_params.total_bits, bias_mant, True, True)

    def set_exponent(self, activ_exp, weight_exp, bias_exp):
        act_total_bits = self.act_params.mantissa_bits + 1 + activ_exp
        weights_total_bits = self.weight_params.mantissa_bits + 1 + weight_exp
        bias_total_bits = self.bias_params.mantissa_bits + 1 + weight_exp
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True)
        self.act_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True)
    
    def set_exponentbias(self, activ_expbias, weight_expbias, bias_expbias):
        act_total_bits = self.act_params.total_bits
        weights_total_bits = self.weight_params.total_bits
        bias_total_bits = self.bias_params.total_bits
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True, activ_expbias)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True, weight_expbias)
        self.bias_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True, bias_expbias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'act_mantissa={self.act_params.mantissa_bits}, '
            f'act_exponent={self.act_params.total_bits - self.act_params.mantissa_bits - 1}, '
            f'weight_mantissa={self.weight_params.mantissa_bits}, '
            f'weight_exponent={self.weight_params.total_bits - self.weight_params.mantissa_bits - 1}, '
            f'bias_mantissa={self.bias_params.mantissa_bits}, '
            f'bias_exponent={self.bias_params.total_bits - self.bias_params.mantissa_bits - 1}'
        )


class LoF_Conv2d(nn.Module):

    @staticmethod
    def _resolve_params(exp, mant, params):
        if params is not None:
            return params
        if exp is not None and mant is not None:
            return lof.create_p3109_params(exp + mant + 1, mant)
        raise ValueError("Must provide either (exp, mant) or params for each tensor")

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 act_params=None, weight_params=None,
                 bias=True, bias_params=None,
                 act_exp=None, act_mant=None,
                 weight_exp=None, weight_mant=None,
                 bias_exp=None, bias_mant=None,
                 rounding_mode=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven

        if act_params is None:
            self.act_params = lof.create_p3109_params(act_exp + act_mant + 1, act_mant, True, True)
        else:
            self.act_params = act_params

        if weight_params is None:
            self.weight_params = lof.create_p3109_params(weight_exp + weight_mant + 1, weight_mant, True, True)
        else:
            self.weight_params = weight_params

        if bias_params is None:
            self.bias_params = lof.create_p3109_params(bias_exp + bias_mant + 1, bias_mant, True, True)
        else:
            self.bias_params = bias_params

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, act_params, weight_params, bias_params=None, rounding_mode=None):
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            act_params=act_params,
            weight_params=weight_params,
            bias=conv.bias is not None,
            bias_params=bias_params,
            rounding_mode=rounding_mode,
        )
        layer.weight = nn.Parameter(layer._quantize(conv.weight.clone(), layer.weight_params))
        if conv.bias is not None:
            layer.bias = nn.Parameter(layer._quantize(conv.bias.clone(), layer.bias_params))
        return layer

    def __deepcopy__(self, memo):
        new = LoF_Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            act_params=self.act_params,
            weight_params=self.weight_params,
            bias=self.bias is not None,
            bias_params=self.bias_params,
            rounding_mode=self.rounding_mode,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, x):
        x_q = self._quantize(x, self.act_params)
        w_q = self._quantize(self.weight, self.weight_params)

        bias_q = self._quantize(self.bias, self.bias_params) if self.bias is not None else None
        return F.conv2d(x_q, w_q, bias_q,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def set_mantissa(self, activ_mant, weight_mant, bias_mant):
        self.act_params = lof.create_p3109_params(self.act_params.total_bits, activ_mant, True, True)
        self.weight_params = lof.create_p3109_params(self.weight_params.total_bits, weight_mant, True, True)
        self.bias_params = lof.create_p3109_params(self.bias_params.total_bits, bias_mant, True, True)

    def set_exponent(self, activ_exp, weight_exp, bias_exp):
        act_total_bits = self.act_params.mantissa_bits + 1 + activ_exp
        weights_total_bits = self.weight_params.mantissa_bits + 1 + weight_exp
        bias_total_bits = self.bias_params.mantissa_bits + 1 + weight_exp
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True)
        self.act_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True)

    def set_exponentbias(self, activ_expbias, weight_expbias, bias_expbias):
        act_total_bits = self.act_params.total_bits
        weights_total_bits = self.weight_params.total_bits
        bias_total_bits = self.bias_params.total_bits
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True, activ_expbias)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True, weight_expbias)
        self.bias_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True, bias_expbias)

    def extra_repr(self):
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
            f'dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, '
            f'act_mantissa={self.act_params.mantissa_bits}, '
            f'act_exponent={self.act_params.total_bits - self.act_params.mantissa_bits - 1}, '
            f'weight_mantissa={self.weight_params.mantissa_bits}, '
            f'weight_exponent={self.weight_params.total_bits - self.weight_params.mantissa_bits - 1}, '
            f'bias_mantissa={self.bias_params.mantissa_bits}, '
            f'bias_exponent={self.bias_params.total_bits - self.bias_params.mantissa_bits - 1}'
        )