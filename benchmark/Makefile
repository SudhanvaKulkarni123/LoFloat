MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LOFLOAT_ROOT ?= $(abspath $(MAKEFILE_DIR)/..)

# Include paths (relative to LOFLOAT_ROOT so it works anywhere)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/src
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/test
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/xsimd/include

# Toolchain
CXX ?= g++

# Base flags (keep these consistent with your tests)
CXXFLAGS ?= -O3 -march=native -std=c++20 -Wall -Wextra
LDFLAGS  ?=
LDLIBS   ?=

# OS detection
UNAME_S := $(shell uname -s)

# TORCH_INCLUDES := $(shell python3 -c "import torch.utils.cpp_extension; print(' '.join(['-I' + p for p in torch.utils.cpp_extension.include_paths()]))")
# TORCH_LDFLAGS  := $(shell python3 -c "import torch.utils.cpp_extension; print(' '.join(['-L' + p for p in torch.utils.cpp_extension.library_paths()]))")
# TORCH_LIBS     := -ltorch -ltorch_cpu -lc10

# CXXFLAGS += $(TORCH_INCLUDES)
# LDFLAGS  += $(TORCH_LDFLAGS) $(TORCH_LIBS) -Wl,-rpath,$(shell python3 -c "import torch.utils.cpp_extension; print(':'.join(torch.utils.cpp_extension.library_paths()))")

# Default (Linux GCC / Clang)
OPENMP_CXXFLAGS := -fopenmp
OPENMP_LDFLAGS  := -fopenmp

# macOS (Apple Clang + Homebrew libomp)
ifeq ($(UNAME_S),Darwin)
  OPENMP_CXXFLAGS := -Xpreprocessor -fopenmp

  HOMEBREW_OMP_INC := $(firstword \
    $(wildcard /opt/homebrew/opt/libomp/include) \
    $(wildcard /usr/local/opt/libomp/include))

  HOMEBREW_OMP_LIB := $(firstword \
    $(wildcard /opt/homebrew/opt/libomp/lib) \
    $(wildcard /usr/local/opt/libomp/lib))

  ifneq ($(HOMEBREW_OMP_INC),)
    OPENMP_CXXFLAGS += -I$(HOMEBREW_OMP_INC)
  endif
  ifneq ($(HOMEBREW_OMP_LIB),)
    OPENMP_LDFLAGS  += -L$(HOMEBREW_OMP_LIB) -lomp
  endif
endif

# Test whether OpenMP actually works
OPENMP_AVAILABLE := $(shell \
  printf "int main(){return 0;}\n" | \
  $(CXX) $(OPENMP_CXXFLAGS) -x c++ - -o /dev/null $(OPENMP_LDFLAGS) 2>/dev/null && echo yes)

ifeq ($(OPENMP_AVAILABLE),yes)
  CXXFLAGS += $(OPENMP_CXXFLAGS) -D_LOFOPENMP
  LDFLAGS  += $(OPENMP_LDFLAGS)
  $(info OpenMP: enabled)
else
  $(info OpenMP: not available)
endif
# ---- Targets ----
TARGETS := vec_round axpy_bench

SRC_ROUND := vec_round.cpp
SRC_AXPY  := axpy_bench.cpp
SRC_VIRTUAL := virtual_round.cpp
SRC_ADD := vec_add.cpp
.PHONY: all clean

all: $(TARGETS)

# Build each target in current directory
vec_round: $(SRC_ROUND)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

axpy_bench: $(SRC_AXPY)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

virtual_round: $(SRC_VIRTUAL)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

vec_add: $(SRC_ADD)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	@rm -f $(TARGETS)

