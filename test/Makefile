# Compiler / flags
CXX       ?= g++
CXXFLAGS  += -std=c++20 -O3
CDBG      += -w

OPENMP_FLAG := -fopenmp -funroll-loops -fprefetch-loop-arrays -frename-register 

USE_CUDA_VAR := $(if $(USE_CUDA),$(USE_CUDA),0)

OPENMP_AVAILABLE := $(shell \
  echo "int main(){return 0;}" | \
  $(CC) $(OPENMP_FLAG) -x c - -o /dev/null 2>/dev/null && echo yes)

ifeq ($(OPENMP_AVAILABLE),yes)
  CXXFLAGS += $(OPENMP_FLAG) -D_LOFOPENMP
endif

CUDA_AVAILABLE := $(shell \
  echo "int main(){return 0;}" | \
  nvcc -x cu - -o /dev/null 2>/dev/null && echo yes)



# LoFloat root: Makefile is in LoFloat/test, so root is one level up
MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LOFLOAT_ROOT ?= $(abspath $(MAKEFILE_DIR)/..)

# Include paths (relative to LOFLOAT_ROOT so it works anywhere)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/src
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/test

# BLIS (built submodule) — needed by the BLIS micro-kernel tests only.
BLIS_INC  := -I$(LOFLOAT_ROOT)/third_party/blis/include/generic
BLIS_LIB  := $(LOFLOAT_ROOT)/third_party/blis/lib/generic/libblis.a


# LLVM IR flags
LLVM_FLAGS = -S -emit-llvm

# Executables
# NOTE: test_ultra_low / test_compare_int were dropped: their source files
# (ultra_low_tests.cpp / compare_int_test.cpp) do not exist and have no equivalent.
ALL_EXEC = test_lo_float test_lo_float_arith test_expectation \
           test_probability test_prob_round test_exceptions test_unsigned \
           test_rounding_modes test_sat_mode test_small test_enum \
           test_compare test_recip test_float_int test_fixed_point test_lo_int \
           test_dot test_gemm \
           test_blis_lof test_blis_lof_accum \
           test_virtual_mx_round test_pack



# Torch include paths — queried from the active Python/torch install so the
# build works on any machine instead of a hardcoded site-packages path.
PYTHON      ?= python
TORCH_INCLUDE := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths()))" 2>/dev/null)

# ── CUDA settings ────────────────────────────────────────────────
NVCC      ?= nvcc
NVCCFLAGS += -std=c++20 -O3 -w -lineinfo
# Auto-detect the local GPU's compute capability (e.g. "9.0" -> sm_90).
# Falls back to sm_86 (RTX 3070) if nvidia-smi is unavailable.
# Override explicitly with: make ... CUDA_ARCH=sm_XX
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
CUDA_ARCH ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_86)

ifeq ($(USE_CUDA_VAR),1)
	ifeq ($(CUDA_AVAILABLE),yes)
		INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/cutlass/include
		INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/cutlass/tools/util/include
		INCLUDE_PATH += $(TORCH_INCLUDE)
		NVCCFLAGS += -DUSE_CUDA
	endif
else 
	INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/xsimd/include
endif

# Add to ALL_EXEC conditionally
ifeq ($(CUDA_AVAILABLE),yes)
  ALL_EXEC += test_gemm_cuda test_lo_int_cuda
endif

# Build rule
ifeq ($(CUDA_AVAILABLE),yes)
test_gemm_cuda: gemm_test.cu
	$(NVCC) $(NVCCFLAGS) -w -Xcompiler -w -v -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@ -lcublas > build_log.txt 2>&1

test_round_to_odd: round_to_odd.cu
	$(NVCC) $(NVCCFLAGS) -w -Xcompiler -w -v -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@ -lcublas > build_log.txt 2>&1

test_virtual_round_cuda: test_virtual_round.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

test_virtual_mx_round_cuda: test_virtual_mx_round_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

perf_mx_round_cuda: perf_mx_round_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

test_conv2d_cuda: test_conv2d_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

test_fixed_point_cuda: test_fixed_point_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

# i_n is not LOFLOAT_HOST_DEVICE-annotated; --expt-relaxed-constexpr lets its
# constexpr (host) operators run in device code.
test_lo_int_cuda: test_lo_int_cuda.cu
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@

test_pack_cuda: test_pack_cuda.cu
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr -DUSE_CUDA -w -Xcompiler -w -arch=$(CUDA_ARCH) $(INCLUDE_PATH) $< -o $@
endif
# Default / "all" option
all: $(ALL_EXEC)

# Build rules
test_lo_float: lo_float_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_lo_float_arith: test_lo_float_arith.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_lo_int: lo_int_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_expectation: stoch_expect_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_probability: stoch_prob_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_prob_round: test_prob_round.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_virtual_round: test_virtual_round.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_virtual_mx_round: test_virtual_mx_round.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

# Exception-flag signaling is gated behind ENABLE_EXCEPT; this test needs it on.
test_exceptions: test_exceptions.cpp
	$(CXX) $(CXXFLAGS) -DENABLE_EXCEPT $(INCLUDE_PATH) $< -o $@

test_unsigned: test_unsigned.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_rounding_modes: test_rounding_modes.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_sat_mode: test_sat_mode.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_float_int: test_float_to_int.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_fixed_point: test_fixed_point.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_small: small_rounding_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_enum: fp8_enum_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_compare: comparison_tests.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_recip: recip_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_dot: dot_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_gemm: gemm_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_pack: test_pack.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

# BLIS micro-kernel tests — link the built libblis.a.
test_blis_lof: test_blis_lof.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $(BLIS_INC) $< $(BLIS_LIB) -lpthread -lm -o $@

test_blis_lof_accum: test_blis_lof_accum.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $(BLIS_INC) $< $(BLIS_LIB) -lpthread -lm -o $@

# Perf harness (not part of ALL_EXEC) — stock BLIS vs the custom-accum kernel.
bench_blis_accum: bench_blis_accum.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $(BLIS_INC) $< $(BLIS_LIB) -lpthread -lm -o $@


# LLVM IR targets (optional)
lo_float_test.ll: lo_float_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

lo_int_test.ll: lo_int_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

stoch_expect_test.ll: stoch_expect_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

stoch_prob_test.ll: stoch_prob_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

exceptions_test.ll: exceptions_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

test_unsigned.ll: test_unsigned.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

ultra_low_tests.ll: ultra_low_tests.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

test_rounding_modes.ll: test_rounding_modes.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

small_rounding_test.ll: small_rounding_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

fp8_enum_test.ll: fp8_enum_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

subnormal_test.ll: subnormal_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

compare_test.ll: compare_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

compare_int_test.ll: compare_int_test.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) $< -o $@

clean:
	rm -rf *.ll *.dSYM
	rm -f $(ALL_EXEC)

.PHONY: all clean
