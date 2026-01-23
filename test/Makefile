# Compiler / flags
CXX       ?= g++
CXXFLAGS  += -std=c++20 -g -O3 -v
CDBG      += -w

OPENMP_FLAG := -fopenmp

OPENMP_AVAILABLE := $(shell \
  echo "int main(){return 0;}" | \
  $(CC) $(OPENMP_FLAG) -x c - -o /dev/null 2>/dev/null && echo yes)

ifeq ($(OPENMP_AVAILABLE),yes)
  CXXFLAGS += $(OPENMP_FLAG) -D_LOFOPENMP
endif

# LoFloat root: Makefile is in LoFloat/test, so root is one level up
MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LOFLOAT_ROOT ?= $(abspath $(MAKEFILE_DIR)/..)

# Include paths (relative to LOFLOAT_ROOT so it works anywhere)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/src
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/test
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/xsimd/include

# LLVM IR flags
LLVM_FLAGS = -S -emit-llvm

# Executables
ALL_EXEC = test_lo_float test_lo_int test_expectation \
           test_probability test_exceptions test_unsigned test_ultra_low \
           test_rounding_modes test_small test_enum test_subnormal \
           test_compare test_compare_int test_recip test_float_int \
           test_dot test_gemm

# Default / "all" option
all: $(ALL_EXEC)

# Build rules
test_lo_float: lo_float_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_lo_int: lo_int_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_expectation: stoch_expect_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_probability: stoch_prob_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_exceptions: exceptions_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_unsigned: test_unsigned.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_ultra_low: ultra_low_tests.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_rounding_modes: test_rounding_modes.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_float_int: test_float_to_int.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_small: small_rounding_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_enum: fp8_enum_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_subnormal: subnormal_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_compare: compare_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_compare_int: compare_int_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_recip: recip_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_dot: dot_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

test_gemm: gemm_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@

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
