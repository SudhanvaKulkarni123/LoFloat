import torch
import torch.library

from .LoFloat import (
    virtual_round as _raw_virtual_round,
    virtual_mx_round as _raw_virtual_mx_round,
    lof_gemm as _raw_lof_gemm,
    RoundingMode,
    FloatFormatDescriptor,
)

# Format registry: lets us pass a FloatFormatDescriptor through a custom op
# (whose schema only accepts tensors + primitive scalars) by indirecting via
# an integer id. Identity-deduped so reusing the same params object in many
# layers doesn't grow the registry.
_FORMAT_REGISTRY: list[FloatFormatDescriptor] = []
_FORMAT_BY_PYID: dict[int, int] = {}


def register_format(params: FloatFormatDescriptor) -> int:
    key = id(params)
    fid = _FORMAT_BY_PYID.get(key)
    if fid is None:
        fid = len(_FORMAT_REGISTRY)
        _FORMAT_REGISTRY.append(params)
        _FORMAT_BY_PYID[key] = fid
    return fid


def get_format(fid: int) -> FloatFormatDescriptor:
    return _FORMAT_REGISTRY[fid]


@torch.library.custom_op("lofloat::virtual_round", mutates_args=())
def virtual_round_op(
    input: torch.Tensor,
    format_id: int,
    round_mode: int,
    stoch_len: int,
    scale: float,
) -> torch.Tensor:
    return _raw_virtual_round(
        input,
        _FORMAT_REGISTRY[format_id],
        round_mode=RoundingMode(round_mode),
        stoch_len=stoch_len,
        scale=scale,
    )


@virtual_round_op.register_fake
def _virtual_round_fake(input, format_id, round_mode, stoch_len, scale):
    return torch.empty_like(input)


@torch.library.custom_op("lofloat::virtual_mx_round", mutates_args=())
def virtual_mx_round_op(
    input: torch.Tensor,
    block_size: int,
    element_format_id: int,
    scale_format_id: int,
    round_mode: int,
    scale_round_mode: int,
    stoch_len: int,
) -> torch.Tensor:
    return _raw_virtual_mx_round(
        input,
        block_size,
        _FORMAT_REGISTRY[element_format_id],
        _FORMAT_REGISTRY[scale_format_id],
        round_mode=RoundingMode(round_mode),
        scale_round_mode=RoundingMode(scale_round_mode),
        stoch_len=stoch_len,
    )


@virtual_mx_round_op.register_fake
def _virtual_mx_round_fake(input, block_size, element_format_id, scale_format_id,
                           round_mode, scale_round_mode, stoch_len):
    return torch.empty_like(input)


@torch.library.custom_op("lofloat::lof_gemm", mutates_args=())
def lof_gemm_op(
    A: torch.Tensor,
    B: torch.Tensor,
    accum_mant_bits: int,
    round_mode: int,
    stochastic_rounding_bits: int,
    scale_a: float,
    scale_b: float,
) -> torch.Tensor:
    return _raw_lof_gemm(
        A,
        B,
        accum_mant_bits,
        round_mode=RoundingMode(round_mode),
        stochastic_rounding_bits=stochastic_rounding_bits,
        scale_a=scale_a,
        scale_b=scale_b,
    )


@lof_gemm_op.register_fake
def _lof_gemm_fake(A, B, accum_mant_bits, round_mode, stochastic_rounding_bits, scale_a, scale_b):
    M = A.shape[0]
    N = B.shape[1]
    return A.new_empty(M, N)
