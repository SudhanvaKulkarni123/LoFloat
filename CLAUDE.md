# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

LoFloat is a library for simulating variable-precision floating-point formats (custom bit widths, exponent biases, rounding modes, NaN/Inf behaviors). It has a header-only C++ core, CUDA kernels, and PyTorch Python bindings — the primary use case is quantization research for neural networks.

## Build and Install

**Full install (C++ headers + Python extension):**
```bash
./install.sh
```
This runs CMake then `pip install -e . --no-build-isolation`. Requires an active Python virtual environment.

**Python extension only (with CUDA):**
```bash
USE_CUDA=1 pip install -e . --no-build-isolation
```

**Python extension only (CPU only):**
```bash
pip install -e . --no-build-isolation
```

**OpenMP support:**
```bash
_LOFOPENMP=1 USE_CUDA=1 pip install -e . --no-build-isolation
```

## Running Tests

**Python tests:**
```bash
python test/pybind_tests.py
python test/test_virtual_round.py
```

**C++ tests:**
```bash
cd test && make all
./test_lo_float        # float arithmetic
./test_lo_int          # integer types
./test_gemm            # GEMM reference implementation
./test_gemm_cuda       # GPU GEMM (requires CUDA build)
./test_rounding_modes  # rounding behavior
./test_subnormal       # subnormal numbers
# ... see test/Makefile for all 18 targets
```

Run a single C++ test: `cd test && make test_lo_float && ./test_lo_float`

## Architecture

### C++ Layer (`src/`, header-only)
- `fp_tools.hpp` — `FloatingPointParams` struct: describes a float format (total bits, mantissa bits, exponent bias, rounding mode, NaN/Inf behavior). This is the central configuration object passed everywhere.
- `lo_float.h` — CRTP base `lo_float_base<Derived, UnderlyingType>` for custom float types with configurable precision.
- `lo_float_half.hpp`, `lo_float_bfloat.hpp`, `lo_float_sci.hpp` — Concrete specializations for FP16, BF16, and scientific formats.
- `Vector.h` / `Matrix.h` — `Vector<T>` and `MX_Vector` (shared-exponent vector) and matrix types built on custom floats.
- `gemms.hpp`, `Dot.hpp`, `Gemvs.hpp` — Reference GEMM, dot-product, and GEMV implementations in pure C++.
- `lo_int.h` — Custom integer type `i_n<len, Sign>` for fixed-width integers.

### CUDA Layer (`src/Lof_kernel.cu`)
- `round_mantissa` / `round_fp_params` kernels — GPU rounding to custom formats.
- CUTLASS-based GEMM kernels for accelerated matrix multiply with quantization.
- Dispatched from `LoPy_bind.cpp` when tensors are on CUDA.

### Python Bindings (`src/LoPy_bind.cpp`)
- Pybind11 bridge. Adapters (`PyInfCheckerAdapter`, `PyNaNCheckerAdapter`) allow Python callables as NaN/Inf checkers.
- `FloatingPointParamsPy` — Python-facing parameter descriptor.
- `virtual_round(tensor, params)` — quantize a tensor to a custom float format in-place/out-of-place.
- `lof_gemm(A, B, params_a, params_b, params_acc)` — GEMM with per-operand quantization.

### Python Package (`LoFloat/`)
- `__init__.py` — Imports C++ extension and re-exports the public API: `virtual_round`, `lof_gemm`, `RoundingMode`, `InfBehavior`, `NaNBehavior`, `Signedness`, `FloatFormatDescriptor`.
- `formats.py` — Predefined format helpers: `create_p3109_params()`, `create_half_params()`, `create_single_params()`.
- `layers.py` — PyTorch `nn.Module` subclasses for quantized inference/training:
  - `STERound` — straight-through estimator wrapper around `virtual_round`.
  - `LoF_Quantize` — generic quantization layer.
  - `LoF_Linear`, `LoF_Conv2d` — linear/conv with separate precision for activations, weights, biases.
  - `LoF_MultiHeadAttention` — attention using `lof_gemm` for the matrix multiplies.
  - BatchNorm variants: `L1BatchNorm`, `LinfBatchNorm`, `FISRBatchNorm`, `PWLBatchNorm`.
  - `PWLSiLU` — piecewise-linear LUT approximation of SiLU.
- `utils.py` — `lofloatify(model, ...)` converts `nn.Linear`/`nn.Conv2d` layers in an existing model to their LoFloat equivalents; `set_mantissa_fields()` / `set_exponent_fields()` for per-layer precision tuning.

## Key Concepts

- **FloatingPointParams** is the central config — it carries total bit width, mantissa bits, exponent bias, rounding mode (`RoundingMode`), and NaN/Inf behavior. Almost every operation takes one.
- **virtual_round** simulates quantization without actually changing the storage type — values stay in FP32/FP16 but are rounded to fit a custom format.
- **Stochastic rounding** is a first-class `RoundingMode` value; use it to reduce quantization bias during training.
- **MX (Microscaling) vectors** share a block exponent across a vector — see `MX_Vector` in `Vector.h`.
- The CUDA path is only compiled when `USE_CUDA=1`; the Python bindings dispatch at runtime based on tensor device.

## Dependencies

- C++20 compiler, CMake ≥ 3.16
- PyTorch (for Python bindings and `layers.py`)
- CUDA 13.2 (optional, for GPU kernels)
- OpenMP (optional, for CPU parallelism)
- xsimd (bundled in `third_party/`)
- CUTLASS (external, required for GPU GEMM kernels)
