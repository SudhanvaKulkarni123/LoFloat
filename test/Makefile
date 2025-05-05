CXXFLAGS = -std=c++20
CXX = g++

CDBG += -w
CXXFLAGS += $(CDBG) -v
CXXFLAGS += -g -O3


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

UNSIGNED:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) test_unsigned.cpp -o test_unsigned

ULTRA_LOW:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) ultra_low_tests.cpp -o test_ultra_low

ROUNDING_MODES:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) test_rounding_modes.cpp -o test_rounding_modes

SMALL:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) small_rounding_test.cpp -o test_small

ENUM:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) fp8_enum_test.cpp -o test_enum

SUBNORMAL:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) subnormal_test.cpp -o test_subnormal

ALL: LO_FLOAT LO_INT EXPECTATION PROBABILITY EXCEPTIONS UNSIGNED ULTRA_LOW ROUNDING_MODES