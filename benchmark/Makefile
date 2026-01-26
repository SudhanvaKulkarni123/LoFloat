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
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -Wall -Wextra
LDFLAGS  ?=
LDLIBS   ?=

# OS detection
UNAME_S := $(shell uname -s)

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

.PHONY: all clean

all: $(TARGETS)

# Build each target in current directory
vec_round: $(SRC_ROUND)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

axpy_bench: $(SRC_AXPY)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	@rm -f $(TARGETS)

