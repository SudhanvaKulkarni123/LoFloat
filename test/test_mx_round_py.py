# Python smoke/regression test for the virtual_mx_round torch binding and the
# layers.py "mx" scaling path. Run: python test/test_mx_round_py.py
import sys
import torch
import LoFloat as lof
from LoFloat.layers import LoF_Quantize, LoF_Linear, LoF_Conv2d, LoF_MultiHeadAttention

errors = 0
def check(cond, msg):
    global errors
    print(("ok  " if cond else "FAIL") + " " + msg)
    if not cond:
        errors += 1

e4m3 = lof.create_p3109_params(8, 4, True, True)   # 8-bit, 3 mantissa bits
e8m0 = lof.create_e8m0_params()                    # unsigned power-of-two scale

torch.manual_seed(0)
x = (torch.rand(4, 256) - 0.5) * 10.0              # last dim 256, divisible by 32

# 1) binding callable, returns same shape, finite, and actually changes values.
y = lof.virtual_mx_round(x, 32, e4m3, e8m0)
check(y.shape == x.shape and torch.isfinite(y).all(), "binding: shape + finite")
check((y != x).any(), "binding: low-precision rounding actually applied")

# 2) CPU and CUDA paths must agree bit-for-bit (deterministic).
if torch.cuda.is_available():
    yg = lof.virtual_mx_round(x.cuda(), 32, e4m3, e8m0).cpu()
    check(torch.equal(yg, y), "CPU and CUDA virtual_mx_round agree bit-exact")
else:
    print("ok   (no CUDA: skipping CPU/GPU agreement)")

# 3) exactly-representable block round-trips (all elements e4m3-representable,
#    amax == priv_max_normal so the e8m0 scale is exactly 1).
maxn = 448.0
exact = torch.tensor([[maxn, maxn/2, -maxn/4, 1.0, 0.5, -2.0, 8.0, 0.0] * 4]) # len 32
ye = lof.virtual_mx_round(exact, 32, e4m3, e8m0)
check(torch.equal(ye, exact), "exact-representable block round-trips unchanged")

# 4) LoF_Quantize(scaling="mx"): forward runs and STE backward is identity.
q = LoF_Quantize(e4m3, scaling="mx", mx_block_size=32)
xr = x.clone().requires_grad_(True)
out = q(xr)
out.sum().backward()
check(out.shape == x.shape, "LoF_Quantize(mx): forward shape")
check(torch.allclose(xr.grad, torch.ones_like(xr.grad)), "LoF_Quantize(mx): STE grad is identity (ones)")

# 5) LoF_Linear(scaling="mx"): forward+backward. lof_gemm is GPU-only, so this
#    runs on CUDA when available.
if torch.cuda.is_available():
    lin = LoF_Linear(in_features=256, out_features=64, act_exp=4, act_mant=3,
                     weight_exp=4, weight_mant=3, bias_exp=4, bias_mant=3, bias=False,
                     scaling="mx", mx_block_size=32).cuda()
    xl = x.cuda()
    yl = lin(xl)
    check(yl.shape == (4, 64) and torch.isfinite(yl).all(), "LoF_Linear(mx): forward shape + finite")
    # (no backward check: lof_gemm is non-differentiable by design — inference
    #  path. The MX STE itself is covered by the LoF_Quantize grad check above.)
else:
    print("ok   (no CUDA: skipping LoF_Linear(mx) — lof_gemm is GPU-only)")

# 5b) LoF_Conv2d(scaling="mx") and LoF_MultiHeadAttention(scaling="mx") forward
#     (CUDA — lof_gemm is GPU-only). K_conv = C_in*kH*kW must be block-divisible.
if torch.cuda.is_available():
    conv = LoF_Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1,
                      act_exp=4, act_mant=3, weight_exp=4, weight_mant=3,
                      bias_exp=4, bias_mant=3, bias=True,
                      scaling="mx", mx_block_size=32).cuda()          # K = 32*9 = 288
    xc = torch.randn(2, 32, 8, 8, device="cuda")
    yc = conv(xc)
    check(yc.shape == (2, 16, 8, 8) and torch.isfinite(yc).all(), "LoF_Conv2d(mx): forward shape + finite")

    mha = LoF_MultiHeadAttention(embed_dim=64, num_heads=4,
                                 act_exp=4, act_mant=3, weight_exp=4, weight_mant=3,
                                 bias_exp=4, bias_mant=3, bias=True,
                                 scaling="mx", mx_block_size=32).cuda()
    t = torch.randn(2, 5, 64, device="cuda")
    yo, _ = mha(t, t, t)
    check(yo.shape == (2, 5, 64) and torch.isfinite(yo).all(), "LoF_MultiHeadAttention(mx): forward shape + finite")
else:
    print("ok   (no CUDA: skipping LoF_Conv2d(mx) / LoF_MultiHeadAttention(mx))")

# 6) divisibility guard fires.
try:
    LoF_Quantize(e4m3, scaling="mx", mx_block_size=32)(torch.rand(3, 30))
    check(False, "non-divisible last dim should raise")
except ValueError:
    check(True, "non-divisible last dim raises ValueError")

print(f"\n=== TOTAL ERRORS: {errors} ===")
sys.exit(1 if errors else 0)
