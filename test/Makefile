CXXFLAGS = -std=c++20
CXX = clang++

CDBG += -w
CXXFLAGS += $(CDBG) -v
CXXFLAGS += -g -O3


INCLUDE_PATH = -I$(HOME)/Documents/tlapack
INCLUDE_PATH += -I$(HOME)
INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/src

INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/test

VAR_FLOAT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) var_float_test.cpp -o test_var_float

LO_FLOAT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) lo_float_test.cpp -o test_lo_float

LO_INT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) lo_int_test.cpp -o test_lo_int

EXPECTATION:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) stoch_expect_test.cpp -o test_expectation 

PROBABILITY:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) stoch_prob_test.cpp -o test_probability

EXCEPTIONS:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) exceptions_test.cpp -o test_exceptions