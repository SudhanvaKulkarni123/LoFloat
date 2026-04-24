import math
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
    return STERound.apply(tensor, lof.create_p3109_params(8 + mantissa_bits + 1, mantissa_bits + 1, True, True))

def exp_mant_quantize(tensor, exp_bits, mantissa_bits):
    if exp_bits is None or mantissa_bits is None:
        return tensor
    return STERound.apply(tensor, lof.create_p3109_params(exp_bits + mantissa_bits + 1, mantissa_bits + 1, True, True))

def _lof_gemm_2d(A, B, accum_mant_bits, gemm_round_mode, stochastic_rounding_bits):
    """Wrapper for (..., M, K) x (..., K, N) -> (..., M, N).

    Assumes the binary expects both A and B as contiguous RowMajor.
    """
    if A.dim() == 2 and B.dim() == 2:
        return lof.lof_gemm(
            A.contiguous(), B.contiguous(),
            accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
        )

    batch_shape = A.shape[:-2]
    M, K = A.shape[-2], A.shape[-1]
    N = B.shape[-1]
    A_flat = A.reshape(-1, M, K).contiguous()
    B_flat = B.reshape(-1, K, N).contiguous()

    out = torch.stack([
        lof.lof_gemm(
            A_flat[i], B_flat[i],         # already contiguous, indexing preserves that
            accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
        )
        for i in range(A_flat.shape[0])
    ])
    return out.reshape(*batch_shape, M, N)


def _lof_linear(x, weight, bias, accum_mant_bits, gemm_round_mode, stochastic_rounding_bits):
    """Drop-in for F.linear using lof_gemm.  x: (..., K), weight: (N, K) -> (..., N)."""
    leading = x.shape[:-1]
    K = x.shape[-1]
    x_2d = x.reshape(-1, K).contiguous()            # (B, K) RowMajor
    wt   = weight.t().contiguous()                   # (K, N) RowMajor (actual copy)
    out = lof.lof_gemm(x_2d, wt,
                       accum_mant_bits, gemm_round_mode, stochastic_rounding_bits)
    out = out.reshape(*leading, weight.shape[0])     # (..., N)
    if bias is not None:
        out = out + bias
    return out

# ═══════════════════════════════════════════════════════════════════════
#  LoF_Linear
# ═══════════════════════════════════════════════════════════════════════
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
                 rounding_mode=None,
                 act_scale_factor=1.0, w_scale_factor=1.0, b_scale_factor=1.0,
                 accum_mant_bits=23,
                 gemm_round_mode=None,
                 stochastic_rounding_bits=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.act_scale_factor = act_scale_factor
        self.w_scale_factor = w_scale_factor
        self.b_scale_factor = b_scale_factor

        # lof_gemm parameters
        self.accum_mant_bits = accum_mant_bits
        self.gemm_round_mode = gemm_round_mode if gemm_round_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.stochastic_rounding_bits = stochastic_rounding_bits

        if act_params is None:
            self.act_params = lof.create_p3109_params(act_exp + act_mant + 1, act_mant + 1, True, True)
        else:
            self.act_params = act_params

        if weight_params is None:
            self.weight_params = lof.create_p3109_params(weight_exp + weight_mant + 1, weight_mant + 1, True, True)
        else:
            self.weight_params = weight_params

        if bias_params is None:
            self.bias_params = lof.create_p3109_params(bias_exp + bias_mant + 1, bias_mant + 1, True, True)
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
    def from_linear(cls, linear: nn.Linear, act_params, weight_params, bias_params=None,
                    rounding_mode=None,
                    accum_mant_bits=23, gemm_round_mode=None, stochastic_rounding_bits=0):
        layer = cls(
            linear.in_features,
            linear.out_features,
            act_params=act_params,
            weight_params=weight_params,
            bias=linear.bias is not None,
            bias_params=bias_params,
            rounding_mode=rounding_mode,
            accum_mant_bits=accum_mant_bits,
            gemm_round_mode=gemm_round_mode,
            stochastic_rounding_bits=stochastic_rounding_bits,
        )
        layer.weight = nn.Parameter(linear.weight.clone())
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.clone())
        return layer

    def __deepcopy__(self, memo):
        new = LoF_Linear(
            self.in_features, self.out_features,
            act_params=self.act_params,
            weight_params=self.weight_params,
            bias=self.bias is not None,
            bias_params=self.bias_params,
            rounding_mode=self.rounding_mode,
            act_scale_factor=self.act_scale_factor,
            w_scale_factor=self.w_scale_factor,
            b_scale_factor=self.b_scale_factor,
            accum_mant_bits=self.accum_mant_bits,
            gemm_round_mode=self.gemm_round_mode,
            stochastic_rounding_bits=self.stochastic_rounding_bits,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, x):
        x_q = self._quantize(x * self.act_scale_factor, self.act_params)
        w_q = self._quantize(self.weight * self.w_scale_factor, self.weight_params)

        # ── lof_gemm replaces torch.matmul(x_q, w_q.t()) ──
        out = _lof_linear(x_q, w_q, None,
                          self.accum_mant_bits, self.gemm_round_mode,
                          self.stochastic_rounding_bits)

        if self.bias is not None:
            b_q = self._quantize(self.bias * self.b_scale_factor, self.bias_params)
            out = out + b_q
        return out

    # ── set_* and extra_repr unchanged ──────────────────────────────────
    def set_mantissa(self, activ_mant, weight_mant, bias_mant):
        act_exp = self.act_params.total_bits - self.act_params.mantissa_bits - 1
        weight_exp = self.weight_params.total_bits - self.weight_params.mantissa_bits - 1
        bias_exp = self.bias_params.total_bits - self.bias_params.mantissa_bits - 1
        self.act_params = lof.create_p3109_params(1 + act_exp + activ_mant, activ_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponent(self, activ_exp, weight_exp, bias_exp):
        act_mant = self.act_params.mantissa_bits
        weight_mant = self.weight_params.mantissa_bits
        bias_mant = self.bias_params.mantissa_bits
        self.act_params = lof.create_p3109_params(1 + activ_exp + act_mant, act_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponentbias(self, activ_expbias, weight_expbias, bias_expbias):
        act_total_bits = self.act_params.total_bits
        weights_total_bits = self.weight_params.total_bits
        bias_total_bits = self.bias_params.total_bits
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True, activ_expbias)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True, weight_expbias)
        self.bias_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True, bias_expbias)

    def set_saturation_mode(self, activ_sat, weight_sat, bias_sat):
        self.act_params = lof.create_p3109_params(self.act_params.total_bits, self.act_params.mantissa_bits, True, activ_sat)
        self.weight_params = lof.create_p3109_params(self.weight_params.total_bits, self.weight_params.mantissa_bits, True, weight_sat)
        self.bias_params = lof.create_p3109_params(self.bias_params.total_bits, self.bias_params.mantissa_bits, True, bias_sat)
    
    def set_accumulation_precision(self, accum_mant_bits):
        self.accum_mant_bits = accum_mant_bits

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'act_mantissa={self.act_params.mantissa_bits}, '
            f'act_exponent={self.act_params.total_bits - self.act_params.mantissa_bits - 1}, '
            f'weight_mantissa={self.weight_params.mantissa_bits}, '
            f'weight_exponent={self.weight_params.total_bits - self.weight_params.mantissa_bits - 1}, '
            f'bias_mantissa={self.bias_params.mantissa_bits}, '
            f'bias_exponent={self.bias_params.total_bits - self.bias_params.mantissa_bits - 1}, '
            f'accum_mant_bits={self.accum_mant_bits}'
        )


# ═══════════════════════════════════════════════════════════════════════
#  LoF_Conv2d  (im2col + lof_gemm)
# ═══════════════════════════════════════════════════════════════════════
class LoF_Conv2d(nn.Module):

    @staticmethod
    def _resolve_params(exp, mant, params):
        if params is not None:
            return params
        if exp is not None and mant is not None:
            return lof.create_p3109_params(exp + mant + 1, mant + 1)
        raise ValueError("Must provide either (exp, mant) or params for each tensor")

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 act_params=None, weight_params=None,
                 bias=True, bias_params=None,
                 act_exp=None, act_mant=None,
                 weight_exp=None, weight_mant=None,
                 bias_exp=None, bias_mant=None,
                 rounding_mode=None,
                 act_scale_factor=1.0, w_scale_factor=1.0, b_scale_factor=1.0,
                 accum_mant_bits=23,
                 gemm_round_mode=None,
                 stochastic_rounding_bits=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.act_scale_factor = act_scale_factor
        self.w_scale_factor = w_scale_factor
        self.b_scale_factor = b_scale_factor

        # lof_gemm parameters
        self.accum_mant_bits = accum_mant_bits
        self.gemm_round_mode = gemm_round_mode if gemm_round_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.stochastic_rounding_bits = stochastic_rounding_bits

        if act_params is None:
            self.act_params = lof.create_p3109_params(act_exp + act_mant + 1, act_mant + 1, True, True)
        else:
            self.act_params = act_params

        if weight_params is None:
            self.weight_params = lof.create_p3109_params(weight_exp + weight_mant + 1, weight_mant + 1, True, True)
        else:
            self.weight_params = weight_params

        if bias_params is None:
            self.bias_params = lof.create_p3109_params(bias_exp + bias_mant + 1, bias_mant + 1, True, True)
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
    def from_conv2d(cls, conv: nn.Conv2d, act_params, weight_params, bias_params=None,
                    rounding_mode=None,
                    accum_mant_bits=23, gemm_round_mode=None, stochastic_rounding_bits=0):
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
            accum_mant_bits=accum_mant_bits,
            gemm_round_mode=gemm_round_mode,
            stochastic_rounding_bits=stochastic_rounding_bits,
        )
        layer.weight = nn.Parameter(conv.weight.clone())
        if conv.bias is not None:
            layer.bias = nn.Parameter(conv.bias.clone())
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
            act_scale_factor=self.act_scale_factor,
            w_scale_factor=self.w_scale_factor,
            b_scale_factor=self.b_scale_factor,
            accum_mant_bits=self.accum_mant_bits,
            gemm_round_mode=self.gemm_round_mode,
            stochastic_rounding_bits=self.stochastic_rounding_bits,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, x):
        x_q = self._quantize(x * self.act_scale_factor, self.act_params)
        w_q = self._quantize(self.weight * self.w_scale_factor, self.weight_params)

        B, C_in, H_in, W_in = x_q.shape
        C_out = self.out_channels
        kH, kW = self.kernel_size

        # ── Compute output spatial dimensions ──
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (kH - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (kW - 1) - 1) // self.stride[1] + 1
        L = H_out * W_out  # number of output spatial positions

        if self.groups == 1:
            # ── im2col: unfold input → (B, C_in*kH*kW, L) ──
            x_col = F.unfold(x_q, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding,
                             stride=self.stride)                         # (B, C_in*kH*kW, L)

            # Weight as 2-D: (C_out, C_in*kH*kW)
            w_col = w_q.view(C_out, -1)                                  # (C_out, C_in*kH*kW)

            # Per-sample GEMM: w_col @ x_col[b] → (C_out, L)
            out = torch.stack([
                lof.lof_gemm(w_col, x_col[b].contiguous(),
                             self.accum_mant_bits, self.gemm_round_mode,
                             self.stochastic_rounding_bits)
                for b in range(B)
            ])                                                           # (B, C_out, L)

        else:
            # ── Grouped convolution via im2col ──
            # Split input channels per group
            c_per_group_in = C_in // self.groups
            c_per_group_out = C_out // self.groups

            x_col = F.unfold(x_q, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding,
                             stride=self.stride)                         # (B, C_in*kH*kW, L)

            # Reshape for groups: (B, groups, c_per_group_in*kH*kW, L)
            x_col = x_col.view(B, self.groups, c_per_group_in * kH * kW, L)

            # Weight per group: (groups, c_per_group_out, c_per_group_in*kH*kW)
            w_col = w_q.view(self.groups, c_per_group_out, -1)

            results = []
            for b in range(B):
                group_outs = []
                for g in range(self.groups):
                    # (c_per_group_out, c_per_group_in*kH*kW) @ (c_per_group_in*kH*kW, L)
                    group_outs.append(
                        lof.lof_gemm(w_col[g], x_col[b, g].contiguous(),
                                     self.accum_mant_bits, self.gemm_round_mode,
                                     self.stochastic_rounding_bits)
                    )
                results.append(torch.cat(group_outs, dim=0))             # (C_out, L)
            out = torch.stack(results)                                   # (B, C_out, L)

        # ── Fold back to spatial layout ──
        out = out.view(B, C_out, H_out, W_out)

        if self.bias is not None:
            b_q = self._quantize(self.bias * self.b_scale_factor, self.bias_params)
            out = out + b_q.view(1, -1, 1, 1)

        return out

    # ── set_* and extra_repr unchanged ──────────────────────────────────
    def set_mantissa(self, activ_mant, weight_mant, bias_mant):
        act_exp = self.act_params.total_bits - self.act_params.mantissa_bits - 1
        weight_exp = self.weight_params.total_bits - self.weight_params.mantissa_bits - 1
        bias_exp = self.bias_params.total_bits - self.bias_params.mantissa_bits - 1
        self.act_params = lof.create_p3109_params(1 + act_exp + activ_mant, activ_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponent(self, activ_exp, weight_exp, bias_exp):
        act_mant = self.act_params.mantissa_bits
        weight_mant = self.weight_params.mantissa_bits
        bias_mant = self.bias_params.mantissa_bits
        self.act_params = lof.create_p3109_params(1 + activ_exp + act_mant, act_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponentbias(self, activ_expbias, weight_expbias, bias_expbias):
        act_total_bits = self.act_params.total_bits
        weights_total_bits = self.weight_params.total_bits
        bias_total_bits = self.bias_params.total_bits
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True, activ_expbias)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True, weight_expbias)
        self.bias_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True, bias_expbias)

    def set_saturation_mode(self, activ_sat, weight_sat, bias_sat):
        self.act_params = lof.create_p3109_params(self.act_params.total_bits, self.act_params.mantissa_bits, True, activ_sat)
        self.weight_params = lof.create_p3109_params(self.weight_params.total_bits, self.weight_params.mantissa_bits, True, weight_sat)
        self.bias_params = lof.create_p3109_params(self.bias_params.total_bits, self.bias_params.mantissa_bits, True, bias_sat)
    
    def set_accumulation_precision(self, accum_mant_bits):
        self.accum_mant_bits = accum_mant_bits

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
            f'bias_exponent={self.bias_params.total_bits - self.bias_params.mantissa_bits - 1}, '
            f'accum_mant_bits={self.accum_mant_bits}'
        )



def _make_scale_buffer(scale, num_features):
    if isinstance(scale, torch.Tensor):
        s = scale.detach().float().reshape(-1)
        if s.numel() == 1:
            return torch.full((num_features,), s.item())
        if s.numel() != num_features:
            raise ValueError(
                f"scale tensor must have 1 or {num_features} elements; got {s.numel()}"
            )
        return s.clone()
    return torch.full((num_features,), float(scale))


class L1BatchNorm(nn.Module):
    """BatchNorm variant using mean absolute deviation (L1) instead of std dev (L2).

    scale is a per-channel buffer of shape [num_features]. Default sqrt(2/pi)
    broadcasts to all channels (the Gaussian-assumption constant). Passing a
    calibrated per-channel MAD/std ratio makes pre-affine output match BN2d's
    (x-mean)/std regardless of input distribution.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 scale=math.sqrt(2 / math.pi)):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_mad", torch.ones(num_features))
        self.register_buffer("scale", _make_scale_buffer(scale, num_features))

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.dim()))
        view_shape  = [1, -1] + [1] * (x.dim() - 2)

        if self.training:
            mean = x.mean(dim=reduce_dims)
            x_centered = x - mean.view(view_shape)
            mad = x_centered.abs().mean(dim=reduce_dims)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_mad.mul_(1 - self.momentum).add_(mad.detach(),  alpha=self.momentum)
        else:
            mean = self.running_mean
            mad  = self.running_mad
            x_centered = x - mean.view(view_shape)

        out = x_centered / (mad.view(view_shape) + self.eps)
        out = out * self.scale.view(view_shape)
        if self.affine:
            out = out * self.weight.view(view_shape) + self.bias.view(view_shape)
        return out


class LinfBatchNorm(nn.Module):
    """BatchNorm variant using max absolute deviation (L-infinity) instead of std.

    scale is a per-channel buffer of shape [num_features]. Default 1.0 recovers
    the previous behavior (no scale). Passing a calibrated maxdev/std ratio
    makes pre-affine output match BN2d's (x-mean)/std. There is no distribution-
    invariant constant default (E[max|X_i - mu|] grows with n), so when scale
    is left at 1.0 the affine weight absorbs the magnitude difference.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 scale=1.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_maxdev", torch.ones(num_features))
        self.register_buffer("scale", _make_scale_buffer(scale, num_features))

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.dim()))
        view_shape  = [1, -1] + [1] * (x.dim() - 2)

        if self.training:
            mean = x.mean(dim=reduce_dims)
            x_centered = x - mean.view(view_shape)
            maxdev = x_centered.abs().amax(dim=reduce_dims)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(),   alpha=self.momentum)
                self.running_maxdev.mul_(1 - self.momentum).add_(maxdev.detach(), alpha=self.momentum)
        else:
            mean   = self.running_mean
            maxdev = self.running_maxdev
            x_centered = x - mean.view(view_shape)

        out = x_centered / (maxdev.view(view_shape) + self.eps)
        out = out * self.scale.view(view_shape)
        if self.affine:
            out = out * self.weight.view(view_shape) + self.bias.view(view_shape)
        return out
# ═══════════════════════════════════════════════════════════════════════
#  LoF_MultiHeadAttention
# ═══════════════════════════════════════════════════════════════════════
class LoF_MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads,
                 act_params=None, weight_params=None,
                 bias=True, bias_params=None,
                 act_exp=None, act_mant=None,
                 weight_exp=None, weight_mant=None,
                 bias_exp=None, bias_mant=None,
                 rounding_mode=None, dropout=0.0,
                 accum_mant_bits=23,
                 gemm_round_mode=None,
                 stochastic_rounding_bits=0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven

        # lof_gemm parameters
        self.accum_mant_bits = accum_mant_bits
        self.gemm_round_mode = gemm_round_mode if gemm_round_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.stochastic_rounding_bits = stochastic_rounding_bits

        if act_params is None:
            self.act_params = lof.create_p3109_params(act_exp + act_mant + 1, act_mant + 1, True, True)
        else:
            self.act_params = act_params

        if weight_params is None:
            self.weight_params = lof.create_p3109_params(weight_exp + weight_mant + 1, weight_mant + 1, True, True)
        else:
            self.weight_params = weight_params

        if bias_params is None:
            self.bias_params = lof.create_p3109_params(bias_exp + bias_mant + 1, bias_mant + 1, True, True)
        else:
            self.bias_params = bias_params

        self.q_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dim))
            self.k_bias = nn.Parameter(torch.zeros(embed_dim))
            self.v_bias = nn.Parameter(torch.zeros(embed_dim))
            self.out_bias = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('k_bias', None)
            self.register_parameter('v_bias', None)
            self.register_parameter('out_bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.q_weight, self.k_weight, self.v_weight, self.out_weight]:
            nn.init.xavier_uniform_(w)

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, act_params, weight_params, bias_params=None,
                 rounding_mode=None,
                 accum_mant_bits=23, gemm_round_mode=None, stochastic_rounding_bits=0):
        layer = cls(
            mha.embed_dim,
            mha.num_heads,
            act_params=act_params,
            weight_params=weight_params,
            bias=mha.in_proj_bias is not None,
            bias_params=bias_params,
            rounding_mode=rounding_mode,
            dropout=mha.dropout,
            accum_mant_bits=accum_mant_bits,
            gemm_round_mode=gemm_round_mode,
            stochastic_rounding_bits=stochastic_rounding_bits,
        )

        E = mha.embed_dim
        in_w = mha.in_proj_weight.data
        layer.q_weight = nn.Parameter(layer._quantize(in_w[:E, :].clone(), layer.weight_params))
        layer.k_weight = nn.Parameter(layer._quantize(in_w[E:2*E, :].clone(), layer.weight_params))
        layer.v_weight = nn.Parameter(layer._quantize(in_w[2*E:, :].clone(), layer.weight_params))
        layer.out_weight = nn.Parameter(layer._quantize(mha.out_proj.weight.clone(), layer.weight_params))

        if mha.in_proj_bias is not None:
            in_b = mha.in_proj_bias.data
            layer.q_bias = nn.Parameter(layer._quantize(in_b[:E].clone(), layer.bias_params))
            layer.k_bias = nn.Parameter(layer._quantize(in_b[E:2*E].clone(), layer.bias_params))
            layer.v_bias = nn.Parameter(layer._quantize(in_b[2*E:].clone(), layer.bias_params))
            layer.out_bias = nn.Parameter(layer._quantize(mha.out_proj.bias.clone(), layer.bias_params))

        return layer

    def __deepcopy__(self, memo):
        new = LoF_MultiHeadAttention(
            self.embed_dim, self.num_heads,
            act_params=self.act_params,
            weight_params=self.weight_params,
            bias=self.q_bias is not None,
            bias_params=self.bias_params,
            rounding_mode=self.rounding_mode,
            dropout=self.dropout,
            accum_mant_bits=self.accum_mant_bits,
            gemm_round_mode=self.gemm_round_mode,
            stochastic_rounding_bits=self.stochastic_rounding_bits,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        B, T, E = query.shape
        S = key.shape[1]

        # Quantize layer inputs
        query = self._quantize(query, self.act_params)
        key = self._quantize(key, self.act_params)
        value = self._quantize(value, self.act_params)

        # Quantize stored weights and biases
        q_w = self._quantize(self.q_weight, self.weight_params)
        k_w = self._quantize(self.k_weight, self.weight_params)
        v_w = self._quantize(self.v_weight, self.weight_params)
        out_w = self._quantize(self.out_weight, self.weight_params)

        q_b = self._quantize(self.q_bias, self.bias_params) if self.q_bias is not None else None
        k_b = self._quantize(self.k_bias, self.bias_params) if self.k_bias is not None else None
        v_b = self._quantize(self.v_bias, self.bias_params) if self.v_bias is not None else None
        out_b = self._quantize(self.out_bias, self.bias_params) if self.out_bias is not None else None

        # ── Projections via lof_gemm (replaces F.linear) ──
        q = _lof_linear(query, q_w, q_b,
                        self.accum_mant_bits, self.gemm_round_mode, self.stochastic_rounding_bits)
        k = _lof_linear(key, k_w, k_b,
                        self.accum_mant_bits, self.gemm_round_mode, self.stochastic_rounding_bits)
        v = _lof_linear(value, v_w, v_b,
                        self.accum_mant_bits, self.gemm_round_mode, self.stochastic_rounding_bits)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # ── Attention scores via lof_gemm (replaces torch.matmul) ──
        # q: (B, H, T, D),  k^T: (B, H, D, S)  →  (B, H, T, S)
        attn = _lof_gemm_2d(q, k.transpose(-2, -1),
                            self.accum_mant_bits, self.gemm_round_mode,
                            self.stochastic_rounding_bits) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(attn, dim=-1)

        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)

        # ── Context via lof_gemm (replaces torch.matmul) ──
        # attn: (B, H, T, S),  v: (B, H, S, D)  →  (B, H, T, D)
        out = _lof_gemm_2d(attn, v,
                           self.accum_mant_bits, self.gemm_round_mode,
                           self.stochastic_rounding_bits)
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # ── Output projection via lof_gemm ──
        out = _lof_linear(out, out_w, out_b,
                          self.accum_mant_bits, self.gemm_round_mode,
                          self.stochastic_rounding_bits)

        if need_weights:
            return out, attn
        return out, None

    # ── set_* and extra_repr unchanged ──────────────────────────────────
    def set_mantissa(self, activ_mant, weight_mant, bias_mant):
        act_exp = self.act_params.total_bits - self.act_params.mantissa_bits - 1
        weight_exp = self.weight_params.total_bits - self.weight_params.mantissa_bits - 1
        bias_exp = self.bias_params.total_bits - self.bias_params.mantissa_bits - 1
        self.act_params = lof.create_p3109_params(1 + act_exp + activ_mant, activ_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponent(self, activ_exp, weight_exp, bias_exp):
        act_mant = self.act_params.mantissa_bits
        weight_mant = self.weight_params.mantissa_bits
        bias_mant = self.bias_params.mantissa_bits
        self.act_params = lof.create_p3109_params(1 + activ_exp + act_mant, act_mant + 1, True, True)
        self.weight_params = lof.create_p3109_params(1 + weight_exp + weight_mant, weight_mant + 1, True, True)
        self.bias_params = lof.create_p3109_params(1 + bias_exp + bias_mant, bias_mant + 1, True, True)

    def set_exponentbias(self, activ_expbias, weight_expbias, bias_expbias):
        act_total_bits = self.act_params.total_bits
        weights_total_bits = self.weight_params.total_bits
        bias_total_bits = self.bias_params.total_bits
        self.act_params = lof.create_p3109_params(act_total_bits, self.act_params.mantissa_bits, True, True, activ_expbias)
        self.weight_params = lof.create_p3109_params(weights_total_bits, self.weight_params.mantissa_bits, True, True, weight_expbias)
        self.bias_params = lof.create_p3109_params(bias_total_bits, self.bias_params.mantissa_bits, True, True, bias_expbias)

    def set_saturation_mode(self, activ_sat, weight_sat, bias_sat):
        self.act_params = lof.create_p3109_params(self.act_params.total_bits, self.act_params.mantissa_bits, True, activ_sat)
        self.weight_params = lof.create_p3109_params(self.weight_params.total_bits, self.weight_params.mantissa_bits, True, weight_sat)
        self.bias_params = lof.create_p3109_params(self.bias_params.total_bits, self.bias_params.mantissa_bits, True, bias_sat)
    
    def set_accumulation_precision(self, accum_mant_bits):
        self.accum_mant_bits = accum_mant_bits

    def extra_repr(self):
        return (
            f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
            f'head_dim={self.head_dim}, dropout={self.dropout}, '
            f'bias={self.q_bias is not None}, '
            f'act_mantissa={self.act_params.mantissa_bits}, '
            f'act_exponent={self.act_params.total_bits - self.act_params.mantissa_bits - 1}, '
            f'weight_mantissa={self.weight_params.mantissa_bits}, '
            f'weight_exponent={self.weight_params.total_bits - self.weight_params.mantissa_bits - 1}, '
            f'bias_mantissa={self.bias_params.mantissa_bits}, '
            f'bias_exponent={self.bias_params.total_bits - self.bias_params.mantissa_bits - 1}, '
            f'accum_mant_bits={self.accum_mant_bits}'
        )


# ═══════════════════════════════════════════════════════════════════════
#  ExplicitMultiheadAttention & replace_mha_with_explicit  (unchanged)
# ═══════════════════════════════════════════════════════════════════════
class ExplicitMultiheadAttention(torch.nn.Module):
    def __init__(self, mha: torch.nn.MultiheadAttention):
        super().__init__()
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.dropout = mha.dropout

        bias = mha.in_proj_bias is not None if mha.in_proj_weight is not None \
               else (hasattr(mha, 'q_proj_weight') and mha.q_proj_weight is not None)

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        if mha.in_proj_weight is not None:
            w = mha.in_proj_weight.data
            self.q_proj.weight.data = w[:embed_dim].clone()
            self.k_proj.weight.data = w[embed_dim:2*embed_dim].clone()
            self.v_proj.weight.data = w[2*embed_dim:].clone()
            if bias:
                b = mha.in_proj_bias.data
                self.q_proj.bias.data = b[:embed_dim].clone()
                self.k_proj.bias.data = b[embed_dim:2*embed_dim].clone()
                self.v_proj.bias.data = b[2*embed_dim:].clone()
        else:
            self.q_proj.weight.data = mha.q_proj_weight.data.clone()
            self.k_proj.weight.data = mha.k_proj_weight.data.clone()
            self.v_proj.weight.data = mha.v_proj_weight.data.clone()
            if bias:
                self.q_proj.bias.data = mha.bias_q.data.clone()
                self.k_proj.bias.data = mha.bias_k.data.clone()
                self.v_proj.bias.data = mha.bias_v.data.clone()

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim,
                                         bias=mha.out_proj.bias is not None)
        self.out_proj.weight.data = mha.out_proj.weight.data.clone()
        if mha.out_proj.bias is not None:
            self.out_proj.bias.data = mha.out_proj.bias.data.clone()

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = query

        B, T, C = query.shape
        S = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights
        return output, None


def replace_mha_with_explicit(model):
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ExplicitMultiheadAttention(module))
    return model