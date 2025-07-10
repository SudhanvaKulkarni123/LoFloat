CXXFLAGS = -std=c++20
CXX = g++

CDBG += -w
CXXFLAGS += $(CDBG) -v
CXXFLAGS += -g -O0


INCLUDE_PATH += -I$(HOME)
INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/src

INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/test

LLVM_FLAGS = -S -emit-llvm 

ALL_EXEC = test_var_float test_lo_float test_lo_int test_expectation \
		   test_probability test_exceptions test_unsigned test_ultra_low \
		   test_rounding_modes test_small test_enum test_subnormal \
		   test_compare test_compare_int test_recip



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

COMPARE_FLOAT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) compare_test.cpp -o test_compare

COMPARE_INT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) compare_int_test.cpp -o test_compare_int

RECIP:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) recip_test.cpp -o test_recip

DOT:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) dot_test.cpp -o test_dot

ALL: LO_FLOAT LO_INT EXPECTATION PROBABILITY EXCEPTIONS UNSIGNED ULTRA_LOW ROUNDING_MODES

LO_FLOAT_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) lo_float_test.cpp -o lo_float_test.ll

VAR_FLOAT_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) var_float_test.cpp -o var_float_test.ll

# Add more as needed
LO_INT_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) lo_int_test.cpp -o lo_int_test.ll

EXPECTATION_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) stoch_expect_test.cpp -o stoch_expect_test.ll

PROBABILITY_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) stoch_prob_test.cpp -o stoch_prob_test.ll

EXCEPTIONS_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) exceptions_test.cpp -o exceptions_test.ll

UNSIGNED_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) test_unsigned.cpp -o test_unsigned.ll

ULTRA_LOW_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) ultra_low_tests.cpp -o ultra_low_tests.ll

ROUNDING_MODES_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) test_rounding_modes.cpp -o test_rounding_modes.ll

SMALL_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) small_rounding_test.cpp -o small_rounding_test.ll

ENUM_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) fp8_enum_test.cpp -o fp8_enum_test.ll


SUBNORMAL_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) subnormal_test.cpp -o subnormal_test.ll

COMPARE_FLOAT_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) compare_test.cpp -o compare_test.ll

COMPARE_INT_LLVM:
	$(CXX) $(CXXFLAGS) $(LLVM_FLAGS) $(INCLUDE_PATH) compare_int_test.cpp -o compare_int_test.ll

clean:
	rm -rf *.ll *.dSYM 
	rm -f $(ALL_EXEC)

.PHONY: all clean
