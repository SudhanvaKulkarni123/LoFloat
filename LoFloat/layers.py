import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import LoFloat as lof
from LoFloat._custom_ops import register_format

class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, params, rounding_mode=lof.RoundingMode.RoundToNearestEven, scale=1.0):
        # Fused scale-then-round: returns round(scale * tensor, params).
        # Output is in the *scaled* domain — callers that want the original
        # numeric range must divide by `scale` afterwards (or rely on the
        # GEMM's scale_a/scale_b output rescale).
        ctx.scale = float(scale)
        # Traceable path: pre-resolved int format-id + int rounding mode go
        # through the registered torch.ops.lofloat.virtual_round so Dynamo
        # can put a real node in the graph. The legacy eager path keeps the
        # pybind callable for callers that still pass a params object.
        if isinstance(params, int):
            return torch.ops.lofloat.virtual_round(
                tensor, params, int(rounding_mode), 0, float(scale)
            )
        return lof.virtual_round(tensor, params, round_mode=rounding_mode, stoch_len=0, scale=float(scale))

    @staticmethod
    def backward(ctx, grad_output):
        # STE: round(scale * x) ≈ scale * x in backward, so ∂/∂x = scale.
        if ctx.scale == 1.0:
            return grad_output, None, None, None
        return grad_output * ctx.scale, None, None, None


class STEMXRound(torch.autograd.Function):
    """Straight-through microscaling (MX) fake-quantize. Forward applies
    virtual_mx_round: the tensor is split into contiguous blocks of `block_size`
    along its LAST axis (keep the reduction/K axis last; block_size must divide
    it), each block gets one auto-computed shared scale in `scale_params`, and
    each element is rounded into `element_params` then rescaled — so the output
    is in the ORIGINAL numeric domain (no scale bookkeeping downstream).
    Backward is the identity because round(x/s)*s ≈ x (the scale cancels)."""
    @staticmethod
    def forward(ctx, tensor, element_params, scale_params, block_size,
                rounding_mode=lof.RoundingMode.RoundToNearestEven):
        # Traceable path: pre-resolved int format-ids go through the registered
        # op; eager path keeps the pybind callable with params objects.
        if isinstance(element_params, int):
            return torch.ops.lofloat.virtual_mx_round(
                tensor, int(block_size), element_params, scale_params,
                int(rounding_mode), int(lof.RoundingMode.RoundToNearestEven), 0)
        return lof.virtual_mx_round(
            tensor, int(block_size), element_params, scale_params,
            round_mode=rounding_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

    
class LoF_Quantize(nn.Module):
    """Quantize a tensor to a low-precision format.

    scaling:
      "per_tensor" (default) — fake-quant with no scaling (the legacy path; pass
                   a scalar scale via the layer that owns this if needed).
      "mx"         — microscaling: per-block auto scale along the LAST axis with
                     block size `mx_block_size`, scale stored in `scale_format`
                     (default E8M0). Requires the last dim divisible by the block
                     size. Output is in the original numeric domain.
    """
    def __init__(self, params, rounding_mode=None,
                 scaling="per_tensor", mx_block_size=32, scale_format=None):
        super().__init__()
        if scaling not in ("per_tensor", "mx"):
            raise ValueError(f"scaling must be 'per_tensor' or 'mx', got {scaling!r}")
        self.params = params
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.scaling = scaling
        self.mx_block_size = mx_block_size
        self.scale_format = scale_format if scale_format is not None else lof.create_e8m0_params()

    def __deepcopy__(self, memo):
        new = LoF_Quantize(self.params, self.rounding_mode,
                           scaling=self.scaling, mx_block_size=self.mx_block_size,
                           scale_format=self.scale_format)
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def forward(self, x):
        if self.scaling == "mx":
            if x.shape[-1] % self.mx_block_size != 0:
                raise ValueError(
                    f"mx scaling needs the last dim ({x.shape[-1]}) divisible by "
                    f"mx_block_size ({self.mx_block_size})")
            return STEMXRound.apply(x, self.params, self.scale_format,
                                    self.mx_block_size, self.rounding_mode)
        return STERound.apply(x, self.params, self.rounding_mode)

    def extra_repr(self):
        s = f'params={self.params}, rounding_mode={self.rounding_mode}, scaling={self.scaling}'
        if self.scaling == "mx":
            s += f', mx_block_size={self.mx_block_size}'
        return s

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

def _lof_gemm_2d(A, B, accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
                 scale_a=1.0, scale_b=1.0):
    """Wrapper for (..., M, K) x (..., K, N) -> (..., M, N).

    Assumes the binary expects both A and B as contiguous RowMajor.
    `scale_a`/`scale_b` rescale the output by 1/(scale_a*scale_b) — pass the same
    scales used for scale-then-quantize on A and B to recover the original domain.
    """
    if A.dim() == 2 and B.dim() == 2:
        return lof.lof_gemm(
            A.contiguous(), B.contiguous(),
            accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
            scale_a, scale_b,
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
            scale_a, scale_b,
        )
        for i in range(A_flat.shape[0])
    ])
    return out.reshape(*batch_shape, M, N)


def _lof_linear(x, weight, bias, accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
                scale_a=1.0, scale_b=1.0):
    """Drop-in for F.linear using lof_gemm.  x: (..., K), weight: (N, K) -> (..., N).
    `scale_a`/`scale_b` are forwarded to the GEMM for output rescale.
    """
    leading = x.shape[:-1]
    K = x.shape[-1]
    x_2d = x.reshape(-1, K).contiguous()            # (B, K) RowMajor
    wt   = weight.t().contiguous()                   # (K, N) RowMajor (actual copy)
    out = lof.lof_gemm(x_2d, wt,
                       accum_mant_bits, gemm_round_mode, stochastic_rounding_bits,
                       scale_a, scale_b)
    out = out.reshape(*leading, weight.shape[0])     # (..., N)
    if bias is not None:
        out = out + bias
    return out

# ═══════════════════════════════════════════════════════════════════════
#  LoF_Linear
# ═══════════════════════════════════════════════════════════════════════
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────
#  Fast Walsh-Hadamard Transform (orthonormal, O(n log n))
# ───────────────────────────────────────────────────────────────────────
def _fwht(x: torch.Tensor, block_size=None) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform along the last dim.

    If `block_size` is None, transforms the entire last dim (requires last
    dim to be a power of 2). Otherwise the last dim is reshaped into
    contiguous blocks of size `block_size` (must be a power of 2 dividing
    the last dim) and the transform is applied within each block. The
    block-diagonal rotation diag(H_B, H_B, ...) is still orthonormal and
    self-inverse.
    """
    n = x.shape[-1]
    if block_size is None:
        block_size = n
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    if n % block_size != 0:
        raise ValueError(f"block_size {block_size} must divide last dim {n}")

    orig_shape = x.shape
    nb = n // block_size                     # number of blocks
    x = x.reshape(-1, nb, block_size)        # (B, nb, block_size)
    flat = x.reshape(-1, block_size)         # (B*nb, block_size)
    M = flat.shape[0]

    h = 1
    while h < block_size:
        flat = flat.view(M, block_size // (2 * h), 2, h)
        a = flat[:, :, 0, :]
        b = flat[:, :, 1, :]
        flat = torch.stack([a + b, a - b], dim=2)
        flat = flat.reshape(M, block_size)
        h *= 2

    flat = flat * (1.0 / math.sqrt(block_size))
    return flat.view(*orig_shape[:-1], n)

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
                 stochastic_rounding_bits=0,
                 hadamard_transform=False, hadamard_block_size=None,
                 scaling="per_tensor", mx_block_size=32, scale_format=None):
        super().__init__()
        if scaling not in ("per_tensor", "mx"):
            raise ValueError(f"scaling must be 'per_tensor' or 'mx', got {scaling!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.act_scale_factor = act_scale_factor
        self.w_scale_factor = w_scale_factor
        self.b_scale_factor = b_scale_factor

        # Microscaling: activations and weights are MX-quantized along the
        # reduction (in_features = K) axis, which is the last axis of both x and
        # weight. The block scale is applied inside the round, so the GEMM runs
        # with scale_a = scale_b = 1.
        self.scaling = scaling
        self.mx_block_size = mx_block_size
        self.scale_format = scale_format if scale_format is not None else lof.create_e8m0_params()
        if scaling == "mx" and in_features % mx_block_size != 0:
            raise ValueError(
                f"mx scaling needs in_features ({in_features}) divisible by "
                f"mx_block_size ({mx_block_size})")

        self.accum_mant_bits = accum_mant_bits
        self.gemm_round_mode = gemm_round_mode if gemm_round_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.stochastic_rounding_bits = stochastic_rounding_bits

        # Hadamard rotation of activations before GEMM
        self.hadamard_transform = hadamard_transform
        self.hadamard_block_size = hadamard_block_size

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

        # Cache integer ids for the traceable path — Dynamo sees only ints
        # flowing into STERound / the registered ops, never the pybind objects.
        self._refresh_traced_ids()

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
                    accum_mant_bits=23, gemm_round_mode=None, stochastic_rounding_bits=0,
                    hadamard_transform=False, hadamard_block_size=None):
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
            hadamard_transform=hadamard_transform,
            hadamard_block_size=hadamard_block_size,
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
            hadamard_transform=self.hadamard_transform,
            hadamard_block_size=self.hadamard_block_size,
            scaling=self.scaling,
            mx_block_size=self.mx_block_size,
            scale_format=self.scale_format,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def _refresh_traced_ids(self):
        self._act_fid = register_format(self.act_params)
        self._weight_fid = register_format(self.weight_params)
        self._bias_fid = register_format(self.bias_params)
        self._round_mode_int = int(self.rounding_mode)
        self._gemm_round_mode_int = int(self.gemm_round_mode)

    def forward(self, x):
        # Optional Hadamard rotation of activations along the in_features axis.
        if self.hadamard_transform:
            x = _fwht(x, block_size=self.hadamard_block_size)

        if self.scaling == "mx":
            # Microscaling: per-block auto scale along K (the last axis of both x
            # and weight). Output is already in the original domain, so the GEMM
            # runs with scale_a = scale_b = 1.
            x_q = STEMXRound.apply(x, self.act_params, self.scale_format,
                                   self.mx_block_size, self.rounding_mode)
            w_q = STEMXRound.apply(self.weight, self.weight_params, self.scale_format,
                                   self.mx_block_size, self.rounding_mode)
            out = _lof_linear(x_q, w_q, None,
                              self.accum_mant_bits, self.gemm_round_mode,
                              self.stochastic_rounding_bits,
                              scale_a=1.0, scale_b=1.0)
        else:
            # Scale-then-quantize is fused inside the CUDA round kernel — no
            # intermediate scaled tensor materialized on GPU.
            x_q = STERound.apply(x, self._act_fid, self._round_mode_int, self.act_scale_factor)
            w_q = STERound.apply(self.weight, self._weight_fid, self._round_mode_int, self.w_scale_factor)

            # GEMM divides output by (act_scale * w_scale) to rescale back to the
            # original (unscaled) domain.
            out = _lof_linear(x_q, w_q, None,
                              self.accum_mant_bits, self._gemm_round_mode_int,
                              self.stochastic_rounding_bits,
                              scale_a=float(self.act_scale_factor),
                              scale_b=float(self.w_scale_factor))

        if self.bias is not None:
            # Quantize bias in its scaled domain, then descale to match the
            # rescaled GEMM output.
            b_q = STERound.apply(self.bias, self._bias_fid, self._round_mode_int, self.b_scale_factor)
            if self.b_scale_factor != 1.0:
                b_q = b_q / self.b_scale_factor
            out = out + b_q
        return out

    # ── set_* unchanged ────────────────────────────────────────────────
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
            f'accum_mant_bits={self.accum_mant_bits}, '
            f'hadamard_transform={self.hadamard_transform}, '
            f'hadamard_block_size={self.hadamard_block_size}'
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
                 stochastic_rounding_bits=0,
                 hadamard_transform=False, hadamard_block_size=None,
                 scaling="per_tensor", mx_block_size=32, scale_format=None,
                 backend="im2col"):
        super().__init__()
        if scaling not in ("per_tensor", "mx"):
            raise ValueError(f"scaling must be 'per_tensor' or 'mx', got {scaling!r}")
        if backend not in ("im2col", "cutlass"):
            raise ValueError(f"backend must be 'im2col' or 'cutlass', got {backend!r}")
        if backend == "cutlass":
            if scaling == "mx":
                raise NotImplementedError("backend='cutlass' does not yet support scaling='mx'")
            if groups != 1:
                raise NotImplementedError("backend='cutlass' currently supports groups=1 only")
        self.backend = backend
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

        self.accum_mant_bits = accum_mant_bits
        self.gemm_round_mode = gemm_round_mode if gemm_round_mode is not None else lof.RoundingMode.RoundToNearestEven
        self.stochastic_rounding_bits = stochastic_rounding_bits

        # Microscaling: block along the im2col reduction axis K = (C_in/groups)*kH*kW
        # (the last axis of both the unfolded activation and the reshaped weight).
        self.scaling = scaling
        self.mx_block_size = mx_block_size
        self.scale_format = scale_format if scale_format is not None else lof.create_e8m0_params()
        if scaling == "mx":
            kH, kW = self.kernel_size
            K = (in_channels // groups) * kH * kW
            if groups != 1:
                raise NotImplementedError("scaling='mx' for LoF_Conv2d currently supports groups=1 only")
            if K % mx_block_size != 0:
                raise ValueError(
                    f"mx scaling needs the im2col K=(C_in/groups)*kH*kW={K} divisible by "
                    f"mx_block_size ({mx_block_size})")

        # Hadamard rotation along the C_in axis (per spatial position) before im2col.
        self.hadamard_transform = hadamard_transform
        self.hadamard_block_size = hadamard_block_size
        if hadamard_transform:
            c_per_group_in = in_channels // groups
            eff = hadamard_block_size if hadamard_block_size is not None else c_per_group_in
            if eff <= 0 or (eff & (eff - 1)) != 0:
                raise ValueError(
                    f"hadamard_transform=True requires (hadamard_block_size or "
                    f"in_channels//groups) to be a power of 2, got {eff}"
                )
            if c_per_group_in % eff != 0:
                raise ValueError(
                    f"hadamard_block_size={eff} must divide in_channels//groups="
                    f"{c_per_group_in}"
                )

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
                    accum_mant_bits=23, gemm_round_mode=None, stochastic_rounding_bits=0,
                    hadamard_transform=False, hadamard_block_size=None):
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
            hadamard_transform=hadamard_transform,
            hadamard_block_size=hadamard_block_size,
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
            hadamard_transform=self.hadamard_transform,
            hadamard_block_size=self.hadamard_block_size,
            scaling=self.scaling,
            mx_block_size=self.mx_block_size,
            scale_format=self.scale_format,
            backend=self.backend,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        return STERound.apply(tensor, params, self.rounding_mode)

    def forward(self, x):
        # Optional Hadamard rotation along the channel axis (before im2col).
        # Move C to last dim, transform, then move back.
        if self.hadamard_transform:
            if self.groups == 1:
                x = x.movedim(1, -1).contiguous()       # (B, H, W, C_in)
                x = _fwht(x, block_size=self.hadamard_block_size)
                x = x.movedim(-1, 1).contiguous()       # (B, C_in, H, W)
            else:
                # Apply Hadamard within each group's channel slab
                B, C_in, H_in, W_in = x.shape
                x = x.view(B, self.groups, C_in // self.groups, H_in, W_in)
                x = x.movedim(2, -1).contiguous()       # (B, G, H, W, C/G)
                x = _fwht(x, block_size=self.hadamard_block_size)
                x = x.movedim(-1, 2).contiguous()       # (B, G, C/G, H, W)
                x = x.view(B, C_in, H_in, W_in)

        if self.backend == "cutlass":
            # Native CUTLASS implicit-GEMM Conv2d (no im2col materialization).
            # groups=1, scaling='per_tensor' only (enforced in __init__).
            x_q = STERound.apply(x, self.act_params, self.rounding_mode, self.act_scale_factor)
            w_q = STERound.apply(self.weight, self.weight_params, self.rounding_mode, self.w_scale_factor)

            out = lof.lof_conv2d(
                x_q, w_q,
                self.padding[0], self.padding[1],
                self.stride[0], self.stride[1],
                self.dilation[0], self.dilation[1],
                self.accum_mant_bits, self.gemm_round_mode,
                self.stochastic_rounding_bits,
                float(self.w_scale_factor), float(self.act_scale_factor))

            if self.bias is not None:
                b_q = STERound.apply(self.bias, self.bias_params, self.rounding_mode, self.b_scale_factor)
                if self.b_scale_factor != 1.0:
                    b_q = b_q / self.b_scale_factor
                out = out + b_q.view(1, -1, 1, 1)
            return out

        if self.scaling == "mx":
            # Microscaling: im2col first, then MX-quantize the activation and the
            # weight along the reduction axis K = (C_in)*kH*kW (the last axis of
            # the unfolded activation and the reshaped weight). GEMM scale = 1.
            B, C_in, H_in, W_in = x.shape
            C_out = self.out_channels
            kH, kW = self.kernel_size
            H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (kH - 1) - 1) // self.stride[0] + 1
            W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (kW - 1) - 1) // self.stride[1] + 1

            x_col = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation,
                             padding=self.padding, stride=self.stride)        # (B, K, L)
            x_col = STEMXRound.apply(
                x_col.transpose(1, 2).contiguous(), self.act_params,           # (B, L, K)
                self.scale_format, self.mx_block_size, self.rounding_mode
            ).transpose(1, 2).contiguous()                                     # (B, K, L)
            w_col = STEMXRound.apply(
                self.weight.view(C_out, -1), self.weight_params,               # (C_out, K)
                self.scale_format, self.mx_block_size, self.rounding_mode)

            out = torch.stack([
                lof.lof_gemm(w_col, x_col[b].contiguous(),
                             self.accum_mant_bits, self.gemm_round_mode,
                             self.stochastic_rounding_bits, 1.0, 1.0)
                for b in range(B)
            ]).view(B, C_out, H_out, W_out)

            if self.bias is not None:
                b_q = STERound.apply(self.bias, self.bias_params, self.rounding_mode)
                out = out + b_q.view(1, -1, 1, 1)
            return out

        # Fused scale-then-quantize on GPU; no intermediate scaled tensor.
        x_q = STERound.apply(x, self.act_params, self.rounding_mode, self.act_scale_factor)
        w_q = STERound.apply(self.weight, self.weight_params, self.rounding_mode, self.w_scale_factor)

        B, C_in, H_in, W_in = x_q.shape
        C_out = self.out_channels
        kH, kW = self.kernel_size

        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (kH - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (kW - 1) - 1) // self.stride[1] + 1
        L = H_out * W_out

        # GEMM divides output by (act_scale * w_scale) — note the conv im2col
        # GEMM is W @ X_col, so scale_a corresponds to the weight scale and
        # scale_b to the activation scale.
        s_w = float(self.w_scale_factor)
        s_a = float(self.act_scale_factor)

        if self.groups == 1:
            x_col = F.unfold(x_q, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding,
                             stride=self.stride)
            w_col = w_q.view(C_out, -1)

            out = torch.stack([
                lof.lof_gemm(w_col, x_col[b].contiguous(),
                             self.accum_mant_bits, self.gemm_round_mode,
                             self.stochastic_rounding_bits,
                             s_w, s_a)
                for b in range(B)
            ])
        else:
            c_per_group_in = C_in // self.groups
            c_per_group_out = C_out // self.groups

            x_col = F.unfold(x_q, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding,
                             stride=self.stride)
            x_col = x_col.view(B, self.groups, c_per_group_in * kH * kW, L)
            w_col = w_q.view(self.groups, c_per_group_out, -1)

            results = []
            for b in range(B):
                group_outs = []
                for g in range(self.groups):
                    group_outs.append(
                        lof.lof_gemm(w_col[g], x_col[b, g].contiguous(),
                                     self.accum_mant_bits, self.gemm_round_mode,
                                     self.stochastic_rounding_bits,
                                     s_w, s_a)
                    )
                results.append(torch.cat(group_outs, dim=0))
            out = torch.stack(results)

        out = out.view(B, C_out, H_out, W_out)

        if self.bias is not None:
            b_q = STERound.apply(self.bias, self.bias_params, self.rounding_mode, self.b_scale_factor)
            if self.b_scale_factor != 1.0:
                b_q = b_q / self.b_scale_factor
            out = out + b_q.view(1, -1, 1, 1)

        return out

    # ── set_* unchanged ────────────────────────────────────────────────
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
            f'accum_mant_bits={self.accum_mant_bits}, '
            f'hadamard_transform={self.hadamard_transform}, '
            f'hadamard_block_size={self.hadamard_block_size}'
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
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("scale", _make_scale_buffer(scale, num_features))

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.dim()))
        view_shape = [1, -1] + [1] * (x.dim() - 2)

        if self.training:
            mean = x.mean(dim=reduce_dims)
            x_centered = x - mean.view(view_shape)
            mad = x_centered.abs().mean(dim=reduce_dims)
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_mad.mul_(1 - self.momentum).add_(mad.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            mad = self.running_mad
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

class FISRBatchNorm(nn.Module):
    """BatchNorm variant using Quake III fast inverse square root + 1 Newton-Raphson step
    in place of 1/sqrt(var + eps). FP32 only (bit-hack assumes IEEE 754 single precision).
    """
    MAGIC = 0x5F3759DF

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
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
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _fast_rsqrt(x):
        x32 = x.to(torch.float32).contiguous()
        x_half = x32 * 0.5
        i = x32.view(torch.int32)
        i = FISRBatchNorm.MAGIC - (i >> 1)
        y = i.view(torch.float32).clone()
        y = y * (1.5 - x_half * y * y)   # one Newton-Raphson step
        y = y * (1.5 - x_half * y * y)   # second Newton-Raphson step for improved accuracy
        return y.to(x.dtype)

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.dim()))
        view_shape = [1, -1] + [1] * (x.dim() - 2)
        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        rsqrt = self._fast_rsqrt(var + self.eps)
        out = (x - mean.view(view_shape)) * rsqrt.view(view_shape)
        if self.affine:
            out = out * self.weight.view(view_shape) + self.bias.view(view_shape)
        return out


class PWLBatchNorm(nn.Module):
    """BatchNorm variant using a piecewise-linear LUT for 1/sqrt(var + eps).
    Range-reduces var+eps via frexp to mantissa m in [1, 2), looks up 1/sqrt(m)
    by linear interpolation over 2^lut_bits segments, then applies the exponent
    correction.

    lut_method: 'uniform' | 'midpoint' | 'minimax'.
    """
    INV_SQRT2 = 0.7071067811865476

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 lut_bits=4, lut_method="minimax"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.lut_bits = lut_bits
        self.lut_size = 1 << lut_bits
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("lut", torch.empty(self.lut_size + 1))
        self.init_lut(lut_method)

    def init_lut(self, method="minimax"):
        N = self.lut_size
        knots = torch.linspace(1.0, 2.0, N + 1, dtype=self.lut.dtype)
        f_at_knots = 1.0 / knots.sqrt()
        if method == "uniform":
            lut = f_at_knots.clone()
        elif method == "midpoint":
            mids = 0.5 * (knots[:-1] + knots[1:])
            f_at_mids = 1.0 / mids.sqrt()
            lut = f_at_knots.clone()
            lut[1:-1] = f_at_mids[:-1]
        elif method == "minimax":
            mids = 0.5 * (knots[:-1] + knots[1:])
            delta = 1.0 / mids.sqrt() - 0.5 * (f_at_knots[:-1] + f_at_knots[1:])
            c = torch.empty_like(f_at_knots)
            c[0] = delta[0] / 2
            c[1:-1] = (delta[:-1] + delta[1:]) / 4
            c[-1] = delta[-1] / 2
            lut = f_at_knots + c
        else:
            raise ValueError(f"Unknown lut_method {method!r}")
        self.lut.copy_(lut)
        self.lut_method = method

    def _rsqrt_pwl(self, x):
        m, e = torch.frexp(x)             # m in [0.5, 1)
        m = m * 2.0                        # m in [1, 2)
        e = e - 1
        u = (m - 1.0) * self.lut_size
        idx = u.floor().long().clamp(0, self.lut_size - 1)
        frac = u - idx.to(x.dtype)
        f = self.lut[idx]
        g = self.lut[idx + 1]
        rsqrt_m = f + frac * (g - f)
        e_half = e // 2
        odd = (e % 2 != 0)
        scale = torch.ldexp(torch.ones_like(x), -e_half)
        scale = torch.where(odd, scale * self.INV_SQRT2, scale)
        return rsqrt_m * scale

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.dim()))
        view_shape = [1, -1] + [1] * (x.dim() - 2)
        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        rsqrt = self._rsqrt_pwl(var + self.eps)
        out = (x - mean.view(view_shape)) * rsqrt.view(view_shape)
        if self.affine:
            out = out * self.weight.view(view_shape) + self.bias.view(view_shape)
        return out

class _PWLSiLUApprox(torch.autograd.Function):
    """Autograd wrapper around the fused C++/CUDA ``lof.pwl_silu`` forward.

    Forward runs the LUT kernel (one read + one write); backward uses the
    analytic SiLU derivative σ(x)·(1 + x·(1 − σ(x))) as a smooth surrogate. The
    PWL forward's exact gradient is piecewise-constant (the segment slope), which
    is a poor training signal, so we differentiate the function it approximates.
    """

    @staticmethod
    def forward(ctx, x, lut, R):
        ctx.save_for_backward(x)
        return lof.pwl_silu(x, lut, float(R))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        dsilu = s * (1 + x * (1 - s))
        return grad_output * dsilu, None, None


class PWLSiLU(nn.Module):
    """SiLU(x) = x * sigmoid(x) approximated via a piecewise-linear LUT on [-R, R].
    Outside the range, falls back to the asymptotes: 0 for x < -R, x for x > R.
    Uniform knot spacing; lut_size must be even so x=0 lands on a knot.
    lut_method: 'uniform' | 'midpoint' | 'minimax'.

    Forward dispatches to the fused ``lof.pwl_silu`` kernel when available
    (GPU or CPU); set ``use_ext=False`` to force the pure-torch reference path.
    """

    def __init__(self, R=8.0, lut_bits=4, lut_method="minimax", use_ext=True):
        super().__init__()
        self.R = float(R)
        self.lut_bits = lut_bits
        self.lut_size = 1 << lut_bits
        if self.lut_size % 2 != 0:
            raise ValueError("lut_size must be even so x=0 is a knot.")
        # The kernel is instantiated for power-of-2 lengths up to 4096 (= 2^12).
        self.use_ext = use_ext and hasattr(lof, "pwl_silu") and self.lut_size <= 4096
        self.register_buffer("lut", torch.empty(self.lut_size + 1))
        self.init_lut(lut_method)

    @staticmethod
    def _silu(x):
        return x * torch.sigmoid(x)

    def init_lut(self, method="minimax"):
        N = self.lut_size
        knots = torch.linspace(-self.R, self.R, N + 1, dtype=self.lut.dtype)
        f_at_knots = self._silu(knots)
        if method == "uniform":
            lut = f_at_knots.clone()
        elif method == "midpoint":
            mids = 0.5 * (knots[:-1] + knots[1:])
            f_at_mids = self._silu(mids)
            lut = f_at_knots.clone()
            lut[1:-1] = f_at_mids[:-1]
        elif method == "minimax":
            mids = 0.5 * (knots[:-1] + knots[1:])
            delta = self._silu(mids) - 0.5 * (f_at_knots[:-1] + f_at_knots[1:])
            c = torch.empty_like(f_at_knots)
            c[0] = delta[0] / 2
            c[1:-1] = (delta[:-1] + delta[1:]) / 4
            c[-1] = delta[-1] / 2
            lut = f_at_knots + c
        else:
            raise ValueError(f"Unknown lut_method {method!r}")
        self.lut.copy_(lut)
        self.lut_method = method

    
    def forward(self, x):
        if self.use_ext:
            # Fused LUT kernel: one read + one write, differentiable via the
            # analytic SiLU derivative. Falls back to the reference path below
            # only if disabled at construction.
            return _PWLSiLUApprox.apply(x, self.lut, self.R)
        return self._forward_ref(x)

    def _forward_ref(self, x):
        """Pure-torch reference (the op chain the fused kernel replaces)."""
        N = self.lut_size
        R = self.R

        # Clamp once so the LUT index is always in range — avoids wild extrapolation
        # for out-of-range x and removes the need for a separate "below -R" branch.
        u = (x.clamp(-R, R) + R) * (N / (2 * R))
        idx = u.long().clamp_(max=N - 1)          # in-place clamp
        frac = u - idx.to(x.dtype)

        # lerp = f + frac*(g - f), but as one fused op
        out = torch.lerp(self.lut[idx], self.lut[idx + 1], frac)

        # Single asymptote fix: F.relu(x) is x for x>R and 0 for x<-R,
        # which is exactly the asymptote you want. One where, no zeros_like.
        return torch.where(x.abs() > R, F.relu(x), out)

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
                 stochastic_rounding_bits=0,
                 scaling="per_tensor", mx_block_size=32, scale_format=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if scaling not in ("per_tensor", "mx"):
            raise ValueError(f"scaling must be 'per_tensor' or 'mx', got {scaling!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.rounding_mode = rounding_mode if rounding_mode is not None else lof.RoundingMode.RoundToNearestEven

        # Microscaling: every _quantize() call (q/k/v activations + projection
        # weights/biases) blocks along the LAST axis, which is the embedding dim
        # E — the contraction (K) axis of the projections. Needs E % block == 0.
        self.scaling = scaling
        self.mx_block_size = mx_block_size
        self.scale_format = scale_format if scale_format is not None else lof.create_e8m0_params()
        if scaling == "mx" and embed_dim % mx_block_size != 0:
            raise ValueError(
                f"mx scaling needs embed_dim ({embed_dim}) divisible by "
                f"mx_block_size ({mx_block_size})")

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
            scaling=self.scaling,
            mx_block_size=self.mx_block_size,
            scale_format=self.scale_format,
        )
        new.load_state_dict(self.state_dict())
        memo[id(self)] = new
        return new

    def _quantize(self, tensor, params):
        if params is None:
            return tensor
        if self.scaling == "mx":
            # MX along the last axis (= E, the projection contraction). Applies to
            # q/k/v activations and projection weights/biases alike.
            if tensor.shape[-1] % self.mx_block_size != 0:
                raise ValueError(
                    f"mx scaling needs the last dim ({tensor.shape[-1]}) divisible by "
                    f"mx_block_size ({self.mx_block_size})")
            return STEMXRound.apply(tensor, params, self.scale_format,
                                    self.mx_block_size, self.rounding_mode)
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