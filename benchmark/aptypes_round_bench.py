#!/usr/bin/env python3
"""
Benchmark APyTypes floating-point rounding/casting performance.

APyTypes is a Python library with C++ backend for custom floating-point types.
Install with: pip install apytypes

Note: APyTypes is designed for hardware simulation and doesn't have a direct
"cast array from float32" operation like lo_float::Project(). Instead, it
creates APyFloat objects from Python floats. This benchmark measures that
conversion process.
"""

import numpy as np
import time
import csv
from pathlib import Path

try:
    import apytypes as apy
    APYTYPES_AVAILABLE = True
except ImportError:
    print("APyTypes not installed. Install with: pip install apytypes")
    APYTYPES_AVAILABLE = False
    exit(1)


def logspace_int(nmin, nmax, points):
    """Generate logarithmically spaced integer array sizes."""
    result = []
    a = np.log10(nmin)
    b = np.log10(nmax)
    
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        x = 10 ** (a + (b - a) * t)
        xi = int(round(x))
        xi = max(nmin, min(nmax, xi))
        if not result or xi > result[-1]:
            result.append(xi)
    
    if not result or result[-1] != nmax:
        result.append(nmax)
    
    return result


def bench_apytypes_cast(
    format_name,
    exp_bits,
    man_bits,  
    bias,
    iters=50,
    warmup_iters=5
):
    """
    Benchmark APyTypes casting from Python floats to APyFloat.
    
    Args:
        format_name: Name for the format (e.g., "binary8_E4M3")
        exp_bits: Number of exponent bits
        man_bits: Number of mantissa bits (excluding implicit bit)
        bias: Exponent bias
        iters: Number of timing iterations
        warmup_iters: Number of warmup iterations
    """
    NMIN = 10
    NMAX = 100_000_000
    POINTS = 20
    
    sizes = logspace_int(NMIN, NMAX, POINTS)
    csv_path = Path("apytypes_bench.csv")
    
    # Check if we need CSV header
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if need_header:
            writer.writerow([
                'format', 'exp_bits', 'man_bits', 'bias', 
                'n', 'avg_us', 'min_us', 'max_us', 
                'iters_used', 'warmup_iters'
            ])
        
        for n in sizes:
            # Generate random input data
            np.random.seed(12345)  # For reproducibility
            input_data = np.random.rand(n).astype(np.float32)
            
            # Warmup
            for _ in range(warmup_iters):
                # APyTypes conversion
                # Note: APyFloatArray can be created from numpy array
                result = apy.APyFloatArray.from_float(
                    input_data,
                    exp_bits=exp_bits,
                    man_bits=man_bits,
                    bias=bias
                )
                # Touch result to prevent optimization
                _ = result[0] if len(result) > 0 else None
            
            # Adapt iterations for very large arrays
            iters_used = iters
            if n >= 10_000_000:
                iters_used = min(iters_used, 5)
            if n >= 50_000_000:
                iters_used = min(iters_used, 3)
            iters_used = max(1, iters_used)
            
            # Benchmark
            times_us = []
            for i in range(iters_used):
                t0 = time.perf_counter()
                
                result = apy.APyFloatArray.from_float(
                    input_data,
                    exp_bits=exp_bits,
                    man_bits=man_bits,
                    bias=bias
                )
                
                t1 = time.perf_counter()
                
                # Touch result
                _ = result[(i * 17) % n] if len(result) > 0 else None
                
                elapsed_us = (t1 - t0) * 1e6
                times_us.append(elapsed_us)
            
            avg_us = np.mean(times_us)
            min_us = np.min(times_us)
            max_us = np.max(times_us)
            
            print(f"{format_name} (E{exp_bits}M{man_bits})  "
                  f"n={n:>9}  "
                  f"avg={avg_us:>10.2f} us  "
                  f"min={min_us:>10.2f} us  "
                  f"max={max_us:>10.2f} us  "
                  f"iters={iters_used}")
            
            writer.writerow([
                format_name, exp_bits, man_bits, bias,
                n, avg_us, min_us, max_us,
                iters_used, warmup_iters
            ])
        
        csvfile.flush()


def main():
    if not APYTYPES_AVAILABLE:
        return
    
    print("Benchmarking APyTypes floating-point casting\n")
    print("Note: APyTypes creates custom float types rather than casting")
    print("      arrays in-place, so performance characteristics differ")
    print("      from lo_float::Project() or CPFloat.\n")
    
    warmup = 5
    iters = 50
    
    # Binary8 formats similar to P3109
    # E4M3: 4-bit exp, 3-bit mantissa (+ 1 implicit)
    # Standard bias for E4M3 is 7 (2^(k-1) - 1 where k=4)
    bench_apytypes_cast("binary8_E4M3", exp_bits=4, man_bits=3, bias=7, 
                       iters=iters, warmup_iters=warmup)
    
    # E5M2: 5-bit exp, 2-bit mantissa (+ 1 implicit)
    # Standard bias for E5M2 is 15
    bench_apytypes_cast("binary8_E5M2", exp_bits=5, man_bits=2, bias=15,
                       iters=iters, warmup_iters=warmup)
    
    # IEEE 754 binary16 (half precision)
    # 5-bit exp, 10-bit mantissa, bias=15
    bench_apytypes_cast("binary16", exp_bits=5, man_bits=10, bias=15,
                       iters=iters, warmup_iters=warmup)
    
    print("\nBenchmark complete. Results saved to apytypes_bench.csv")


if __name__ == "__main__":
    main()
