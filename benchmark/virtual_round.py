import torch
import time
import csv
import os
from pathlib import Path

import LoFloat as lof


class P3109_NaNChecker:
    def __init__(self, k, is_signed=True):
        self.k = k
        self.is_signed = is_signed
    
    def __call__(self, bits):
        if self.is_signed:
            return bits == (1 << (self.k - 1))
        else:
            return bits == ((1 << self.k) - 1)
    
    def qNanBitPattern(self):
        if self.is_signed:
            return 1 << (self.k - 1)
        else:
            return (1 << self.k) - 1
    
    def sNanBitPattern(self):
        if self.is_signed:
            return 1 << (self.k - 1)
        else:
            return (1 << self.k) - 1


class P3109_InfChecker:
    def __init__(self, k, is_signed=True, has_inf_saturating=True):
        self.k = k
        self.is_signed = is_signed
        self.has_inf_saturating = has_inf_saturating
    
    def __call__(self, bits):
        if not self.has_inf_saturating:
            if self.is_signed:
                target = (1 << self.k) - 1
            else:
                target = (1 << self.k) - 2
            return (bits | (1 << (self.k - 1))) == target
        else:
            return False
    
    def minNegInf(self):
        if self.is_signed:
            return (1 << self.k) - 1
        else:
            return 0
    
    def minPosInf(self):
        if self.is_signed:
            return (1 << (self.k - 1)) - 1
        else:
            return (1 << self.k) - 2

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

def create_p3109_params(k, p, is_signed=True, saturating=True):
    mantissa_bits = p - 1
    bias = 1 << (k - p - 1)
    inf_behavior = lof.InfBehavior.Saturating if saturating else lof.InfBehavior.Extended
    
    return lof.FloatFormatDescriptor(
        k,
        mantissa_bits,
        bias,
        inf_behavior=inf_behavior,
        nan_behavior=lof.NaNBehavior.QuietNaN,
        signedness=lof.Signedness.Signed if is_signed else lof.Signedness.Unsigned,
        is_inf_checker=P3109_InfChecker(k, is_signed, saturating),
        is_nan_checker=P3109_NaNChecker(k, is_signed)
    )


def create_half_params():
    return lof.FloatFormatDescriptor(
        16,
        10,
        15,
        inf_behavior=lof.InfBehavior.Extended,
        nan_behavior=lof.NaNBehavior.QuietNaN,
        signedness=lof.Signedness.Signed,
        is_inf_checker=HalfPrecisionInfChecker(),
        is_nan_checker=HalfPrecisionNaNChecker()
    )

def sanitize(s):
    return ''.join(c if c.isalnum() else '_' for c in s)


def logspace_int(nmin, nmax, points):
    import math
    if points == 1:
        return [nmin]
    
    a = math.log10(nmin)
    b = math.log10(nmax)
    
    v = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        x = 10.0 ** (a + (b - a) * t)
        xi_int = round(x)
        xi_int = max(nmin, min(nmax, xi_int))
        if not v or xi_int > v[-1]:
            v.append(xi_int)
    
    if not v or v[-1] != nmax:
        v.append(nmax)
    
    return v


def bench_virtual_round(name, params, iters=50, warmup_iters=5, 
                        round_mode=None, csv_path="virtual_round_bench_py.csv"):
    if round_mode is None:
        round_mode = lof.RoundingMode.RoundToNearestEven
    
    NMIN = 10
    NMAX = 100_000_000
    POINTS = 20
    sizes = logspace_int(NMIN, NMAX, POINTS)
    
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if need_header:
        csv_writer.writerow(['type', 'n', 'avg_us', 'min_us', 'max_us', 'iters_used', 'warmup_iters'])
    
    type_name = sanitize(name)
    
    print(f"\nBenchmarking {name}:")
    print("-" * 80)
    
    for n in sizes:
        arr = torch.rand(n, dtype=torch.float32)
        
        for _ in range(0):
            _ = lof.virtual_round(arr, params, round_mode=round_mode, stoch_len=0)
        
        iters_used = iters
        if n >= 10_000_000:
            iters_used = min(iters_used, 5)
        if n >= 50_000_000:
            iters_used = min(iters_used, 3)
        iters_used = max(1, iters_used)
        
        samples_us = []
        a = 0
        for _ in range(iters_used):
            t0 = time.perf_counter()
            result = lof.virtual_round(arr, params, round_mode=round_mode, stoch_len=0)
            t1 = time.perf_counter()

            a = a +  result.sum().item()
            
            
            samples_us.append((t1 - t0) * 1e6)
        
        avg_us = sum(samples_us) / len(samples_us)
        min_us = min(samples_us)
        max_us = max(samples_us)
        print(a)
        
        print(f"{name:30s}  n={n:>10d}  avg={avg_us:>12.2f} us  "
              f"min={min_us:>12.2f} us  max={max_us:>12.2f} us  iters={iters_used}")
        
        csv_writer.writerow([type_name, n, avg_us, min_us, max_us, iters_used, warmup_iters])
        csv_file.flush()
    
    csv_file.close()
    return {'avg_us': avg_us, 'min_us': min_us, 'max_us': max_us}


def main():
    torch.manual_seed(12345)
    warmup = 5
    iters = 50
    
    print("=" * 80)
    print("Virtual Round Benchmark")
    print("=" * 80)
    
    binary8p4_params = create_p3109_params(k=8, p=4, is_signed=True, saturating=True)
    bench_virtual_round("binary8p4 (P3109<8,4>)", binary8p4_params, iters, warmup)
    
    binary8p3_params = create_p3109_params(k=8, p=3, is_signed=True, saturating=True)
    bench_virtual_round("binary8p3 (P3109<8,3>)", binary8p3_params, iters, warmup)
    
    half_params = create_half_params()
    bench_virtual_round("half (binary16)", half_params, iters, warmup)
    
    print("\n" + "=" * 80)
    print("Benchmark complete! Results saved to virtual_round_bench_py.csv")
    print("=" * 80)


if __name__ == "__main__":
    print("num threads = ")
    print(torch.get_num_threads())
    torch.set_num_threads(1)
    main()