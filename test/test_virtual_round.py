import torch
import LoFloat as lof
import numpy as np

def test_virtual_round():
    N = 400
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    arr = torch.rand(N, dtype=torch.float32)
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
    # Apply virtual rounding
    rounded = lof.virtual_round(
        arr, 
        half_params,
        round_mode=lof.RoundingMode.RoundToNearestEven,
        stoch_len=0
    )
    
    # Print results
    if N < 5:
        for i in range(N):
            og_val = og_arr[i].item()
            rounded_val = rounded[i].item()
            
            og_bits = og_arr[i].view(torch.int32).item()
            rounded_bits = rounded[i].view(torch.int32).item()
            
            print(f"Original: {og_val:.6f}, Rounded: {rounded_val:.6f}")
            print(f"bit patterns in hex: Original: {og_bits:08x}, Rounded: {rounded_bits:08x}\n")
    
    # Check error
    test_passed = True
    for i in range(N):
        og_val = og_arr[i].item()
        rounded_val = rounded[i].item()
        err = abs(rounded_val - og_val)
        
        if og_val == 0:
            continue
            
        mantissa, exp = np.frexp(og_val)
        two_pow = np.ldexp(1.0, exp)
        threshold = (2.0 ** -11) / two_pow
        
        if err > threshold:
            print(f"Value: {rounded_val:.6f}, Original: {og_val:.6f}, Error: {err:.10f}")
            print(f"Threshold: {threshold:.10f}")
            print("Test failed!")
            test_passed = False
            break
    
    if test_passed:
        print("All tests passed!")
    
    return test_passed


# def test_virtual_round_mantissa_only():
#     """Test the simpler mantissa-only interface"""
#     N = 4
#     torch.manual_seed(42)
    
#     arr = torch.rand(N, dtype=torch.float32)
#     og_arr = arr.clone()
    
#     print("\n\nTesting mantissa-only interface\n")
    
#     rounded = lof.virtual_round(
#         arr,
#         to_mantissa_bits=10,
#         round_mode=lof.RoundingMode.RoundToNearestEven
#     )
    
#     for i in range(N):
#         og_val = og_arr[i].item()
#         rounded_val = rounded[i].item()
#         print(f"Original: {og_val:.6f}, Rounded: {rounded_val:.6f}")
    
#     print("Mantissa-only test completed!")


if __name__ == "__main__":
    test_virtual_round()