CXXFLAGS = -std=c++20
CXX = g++

CDBG += -w
CXXFLAGS += $(CDBG) -v
CXXFLAGS += -g -O3


# LoFloat root: Makefile is in LoFloat/examples, so root is one level up
MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LOFLOAT_ROOT ?= $(abspath $(MAKEFILE_DIR)/..)

# Include paths (relative to LOFLOAT_ROOT so it works anywhere)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/src
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/test
INCLUDE_PATH += -I$(LOFLOAT_ROOT)/third_party/xsimd/include


OSCILLATOR:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) Oscillator.cpp -o oscillator

NEWTON:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) newtons_method.cpp -o newtons_method

clean:
	rm -f oscillator
	rm -f *.o
	rm -rf *.dSYM
	rm -f *.exe
	rm -f *.out
	rm -f *.log
	rm -f *.txt
