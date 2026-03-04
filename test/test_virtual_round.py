import torch
import LoFloat as lof
import numpy as np

def test_virtual_round():
    N = 4
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    arr = torch.rand(N, dtype=torch.float32, device=device)
    og_arr = arr.clone()
    print(f"Testing with {N} elements\n")

    class HalfPrecisionInfChecker:
        def __call__(self, bits):
            exp_mask = 0x7C00
            mant_mask = 0x03FF
            return (bits & exp_mask) == exp_mask and (bits & mant_mask) == 0
        def minNegInf(self):
            return 0xFC00
        def minPosInf(self):
            return 0x7C00

    class HalfPrecisionNaNChecker:
        def __call__(self, bits):
            exp_mask = 0x7C00
            mant_mask = 0x03FF
            return (bits & exp_mask) == exp_mask and (bits & mant_mask) != 0
        def qNanBitPattern(self):
            return 0x7E00
        def sNanBitPattern(self):
            return 0x7D00

    half_params = lof.FloatFormatDescriptor(
        16, 10, 15,
        inf_behavior=lof.InfBehavior.Extended,
        nan_behavior=lof.NaNBehavior.QuietNaN,
        signedness=lof.Signedness.Signed,
        is_inf_checker=HalfPrecisionInfChecker(),
        is_nan_checker=HalfPrecisionNaNChecker()
    )

    rounded = lof.virtual_round(
        arr,
        half_params,
        round_mode=lof.RoundingMode.RoundToNearestEven,
        stoch_len=0
    )

    if N < 5:
        for i in range(N):
            og_val = og_arr[i].item()
            rounded_val = rounded[i].item()
            og_bits = og_arr[i].view(torch.int32).item()
            rounded_bits = rounded[i].view(torch.int32).item()
            print(f"Original: {og_val:.10f}, Rounded: {rounded_val:.10f}")
            print(f"bit patterns in hex: Original: {og_bits:08x}, Rounded: {rounded_bits:08x}\n")

    test_passed = True
    og_arr_cpu = og_arr.cpu()
    rounded_cpu = rounded.cpu()
    for i in range(N):
        og_val = og_arr_cpu[i].item()
        rounded_val = rounded_cpu[i].item()
        err = abs(rounded_val - og_val)
        if og_val == 0:
            continue
        mantissa, exp = np.frexp(og_val)
        two_pow = np.ldexp(1.0, exp)
        threshold = (2.0 ** -11) / two_pow
        if err > threshold:
            print(f"Value: {rounded_val:.10f}, Original: {og_val:.10f}, Error: {err:.10f}")
            print(f"Threshold: {threshold:.10f}")
            print("Test failed!")
            test_passed = False
            break

    if test_passed:
        print("All tests passed!")
    return test_passed

if __name__ == "__main__":
    test_virtual_round()