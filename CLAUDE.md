# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LoFloat simulates arbitrary low-precision / non-standard floating-point (and integer) formats. It is a header-only C++20 library (`src/`) plus a PyTorch C++/CUDA extension and Python package (`LoFloat/`) that exposes the simulation as "fake quantization" ops and quantized NN layers. Funded by an STTR grant (UC Berkeley + Arith Inc).

## Build & run

### Submodules first
The library depends on git submodules that are not vendored:
```bash
git submodule update --init --recursive   # xsimd, cutlass, blis
```
- `third_party/xsimd` — CPU SIMD backend (always needed for header builds).
- `third_party/cutlass` — GPU GEMM/conv backend (only when `USE_CUDA=1`).
- `third_party/blis` — must be **compiled** (`./configure` + `make`) before the BLIS micro-kernel tests; they link `third_party/blis/lib/generic/libblis.a`.

### Header-only C++ library (CMake)
```bash
cmake -B build && cmake --build build      # installs/exports the lo_float INTERFACE target
```
`lo_float` is an INTERFACE (header-only) target requiring C++20; consumers just add `src/` and `third_party/xsimd/include` to their include path.

### Python / PyTorch extension
```bash
pip install -e . --no-build-isolation      # or: python setup.py build_ext --inplace
./install.sh                               # does CMake + pip install together
```
The extension compiles `src/LoPy_bind.cpp` (+ `src/Lof_kernel.cu` under CUDA) into `LoFloat/LoFloat.*.so`.

### Build-controlling environment variables
These are read by **both** `setup.py` and `CMakeLists.txt`:
- `USE_CUDA=1` — build CUDA device paths (requires `nvcc`; auto-detects `CUDA_HOME`, else set it). Pulls in Torch + CUTLASS.
- `_LOFOPENMP=1` — enable OpenMP (`-fopenmp`) for multithreaded CPU GEMM.
- `LOFLOAT_EXCEPTIONS=1` — enable IEEE-754 exception-flag signaling (`-DENABLE_EXCEPT`). **Silently ignored** if `USE_CUDA` or `_LOFOPENMP` is set: the global exception env (`f_env` in `f_exceptions.hpp`) is host-only and unsynchronized, so it is only valid for single-threaded CPU builds.
- `XSIMD_INCLUDE_PATH` — override the xsimd include dir for the non-CUDA build.

### Tests
Tests are standalone executables built by `test/Makefile` (one source → one binary), then run directly:
```bash
cd test
make all                 # build the CPU test suite (ALL_EXEC)
make test_lo_float       # build a single test, then run: ./test_lo_float
USE_CUDA=1 make test_gemm_cuda CUDA_ARCH=sm_90   # CUDA tests need nvcc + -arch
```
The Makefile auto-detects OpenMP and CUDA availability. CUDA test targets carry a `_cuda` suffix and require `USE_CUDA=1`. BLIS tests (`test_blis_lof*`) require the built `libblis.a`. Python binding tests: `python test/pybind_tests.py`, `python test/test_mx_round_py.py`. Example programs live in `examples/` with their own `Makefile`.

## Architecture

### Two simulation strategies (this is the core design)
1. **Native typed values** — `Templated_Float<Fp>` is a real C++ scalar type (stored in a `uint8/16/32_t` rep) that behaves like a narrow float. Arithmetic operators upcast to a wider `UnderlyingFloat` (`float` or `double`, chosen by mantissa width via `AOpType`), compute, then round back into the format. Predefined aliases in `lo_float.h`: `float8_e4m3_fn`, `float8_e5m2`, `float6_e3m2`, `float4_e2m1`, `half`, `bfloat16`, `tf32`, OCP MX types (`ocp_e4m3`…), P3109 (`P_3109_float`), Dojo cfloats, etc.
2. **`virtual_round`** — operates on ordinary `float`/`double` buffers and rounds each element's mantissa/exponent down to a target format *in place* (the type stays float32/64). This is the "fake quant" path the PyTorch layer uses. Overloads in `lo_float.h` take either a mantissa-bit count or a full `FloatingPointParams`.

### Format descriptor & class hierarchy
- `FloatingPointParams<InfChecker, NaNChecker>` (`fp_tools.hpp`) is the **compile-time** format descriptor: `bitwidth`, `mantissa_bits`, `bias`, inf behavior (Extended/Saturating), NaN behavior (`_3109`/`_754`), signedness, and pluggable inf/NaN checker functors. A `Templated_Float` is parameterized by a `constexpr` instance of this struct.
- CRTP chain: `lo_float_base<Derived, Rep>` (operators, conversions, comparison) → `Var_lo_float<Derived, Fp>` → `Templated_Float<Fp>`. Format properties are read through `get_mantissa_bits_v<T>`, `get_bitwidth_v<T>`, `get_signedness_v<T>`, `get_NaN_Behavior_v<T>`, etc.
- Enums (rounding modes, signedness, inf/nan/unsigned/saturation behaviors) are generated from **X-macros** at the top of `fp_tools.hpp` — add a new variant there and it propagates to the enum + name tables.
- `ProjSpec` bundles the runtime rounding choice: `{rounding_mode, saturation_mode, stoch_length}`. It is threaded through `virtual_round` → `RoundMantissa` → the individual rounding helpers.

### Rounding
`lo_float.h` (~4400 lines) holds many rounding implementations: nearest-even/odd, toward/away zero, up/down, ties-to-away, round-to-odd, and **five stochastic variants** (A/B/C/D + true stochastic + probabilistic). Each is a **single template** that uses `if constexpr (is_xsimd_batch<...>)` to branch between scalar and xsimd-batch paths, wrapped in `#ifndef USE_CUDA` for the SIMD branch. `RoundMantissa` dispatches on `ProjSpec.rounding_mode`.

### Host / device / SIMD portability
- `platform_macros.h` defines `LOFLOAT_HOST_DEVICE`, `LOFLOAT_DEVICE`, `LOFLOAT_FORCEINLINE`. Most functions are annotated to compile for both host and CUDA device.
- CUDA code is gated by `USE_CUDA` (build flag) and `__CUDA_ARCH__` (device-side codegen). Key gotcha encoded in the code: `std::bit_cast` silently yields 0 in nvcc device code — use the library's `bit_cast` helper which dispatches to `cuda::std::bit_cast` on device.
- Stochastic rounding on GPU cannot use `<random>`/`thread_local`; instead it derives decorrelated samples statelessly via a SplitMix64 hash of the element bits + per-call salt + global thread index (`lof_device_rand64`).

### GEMM / conv
- `Lo_Gemm::Gemm` (`gemms.hpp`) is a BLIS-style, multi-strategy CPU GEMM (naive → blocked+packed) that selects strategy by matrix size and accumulates in a configurable precision. `Matrix.h`/`Vector.h` provide layout-templated (`ColMajor`/`RowMajor`) views. `blis_lof_gemm.hpp` integrates the real BLIS micro-kernel.
- GPU GEMM/conv use CUTLASS (`cutlass_gemms.cuh`, `cutlass_conv.cuh`) and custom kernels (`Lof_kernel.cu`, `mx_round.cuh`).

### Python / PyTorch layer (`LoFloat/`)
- `LoFloat/__init__.py` re-exports the compiled extension symbols. Core ops bound in `src/LoPy_bind.cpp`: `virtual_round` (mantissa-bits or `FloatFormatDescriptor`), `virtual_mx_round` (microscaling with a block shared-scale, e.g. e8m0), `lof_gemm`, `lof_conv2d`, `pwl_silu`, and the `FloatFormatDescriptor` class (the Python face of `FloatingPointParams`).
- `formats.py` — Python factories for descriptors (`create_p3109_params`, `create_half_params`, `create_single_params`, `create_e8m0_params`) with matching inf/NaN checker classes.
- `layers.py` — quantization-aware NN modules: `LoF_Linear`, `LoF_Conv2d`, `LoF_Quantize`, batchnorm variants, `PWLSiLU`, explicit/quantized MHA. Forward passes round activations/weights/accumulators; backward uses straight-through estimators (`STERound`, `STEMXRound`).
- `_custom_ops.py` registers `torch.ops.lofloat.*` so the ops are traceable. Because op schemas only take tensors + scalars, `FloatFormatDescriptor`s are passed by integer id through a `_FORMAT_REGISTRY`.
- `utils.py` — helpers to walk a model and set per-layer mantissa/exponent/bias/accumulation formats (`lofloatify`, `set_all_to_half`, `set_all_to_3109`, …).

### Custom integers
`lo_int.h` defines `i_n<len, Signedness>` — arbitrary-width (1–128 bit) two's-complement integers with masking on every op, usable as the index type in `Matrix`/`Vector` and convertible to/from the float types.
