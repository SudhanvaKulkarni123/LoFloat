///@author Sudhanva Kulkarni, UC Berkeley
/*
    * @file
    * @brief This file contains the definition of the FloatingPointException class, which is used to handle exceptions related to floating-point operations.
    * 
    * @note This file is part of the lo_float library, which provides a set of tools for working with low to medium floating-point numbers in C++.
*/

#pragma once

#include <cstdint>
#include <concepts>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>

#include "fp_tools.hpp"  // for lo_float::NaN_Behaviors (the _754 gate)


//need some kind of functors that tell us how to deal with traps. Also need some global flags that are visible to users

namespace lo_float {

    namespace lo_float_internal {

        //need to set DivisionByZero in the parent class along with arith ops. Might need to set it in lo_float_sci as well
        //Set Overflow, underflow in the convert function. I count going from negative number to 0 in the case of unsigned floats as an underflow
        //Set InvalidOperation according to 754 doc and also provide some way to allow users to define operations/wrappers that let users define their own invalid ops
        //Set inexact in the conversion function when round the mantissa. If the rounded bits are the exact same as the original bits, then we don't set inexact


        //all flags are sticky
        enum LF_exception_flags : uint8_t {
            // No exception
            NoException = 0,

            // Floating-point exceptions
            DivisionByZero = 1 << 0,
            Overflow = 1 << 1,
            Underflow = 1 << 2,
            InvalidOperation = 1 << 3,
            Inexact = 1 << 4,

            // All exceptions
            AllExceptions = DivisionByZero | Overflow | Underflow | InvalidOperation | Inexact
        };

        inline std::string exception_names[] = {
            "NoException",
            "DivisionByZero",
            "Overflow",
            "Underflow",
            "InvalidOperation",
            "Inexact"
        };



        inline void DefaultTrapHandler(LF_exception_flags f) noexcept {
                std::cerr << "[Default Trap] Exception (s) : ";
                for(int i = 0; i < 5; i++) {
                    if ((uint8_t)f & (1 << i)) {
                        std::cerr << exception_names[i] << " ";
                    }
                }
                std::cerr << "\n";
                std::abort();
        }
        
        struct TrapHandlerRegistry {
            public:
                using TrapHandler = void(*)(LF_exception_flags);
    
                TrapHandler custom_handler = nullptr;
                TrapHandler default_handler = nullptr;
                bool use_custom_handler = false;
    
                TrapHandlerRegistry()
                    : default_handler(&DefaultTrapHandler) {}
    
                void set_custom_handler(TrapHandler handler) {
                    custom_handler = handler;
                    use_custom_handler = true;
                }
    
                void reset_custom_handler() {
                    custom_handler = nullptr;
                    use_custom_handler = false;
                }
    
                void raise(LF_exception_flags f) const {
                    if (use_custom_handler) {
                        custom_handler(f);
                    } else {
                        default_handler(f);
                    }
                }
            };
    
            struct Environment {
                uint8_t exception_flags;
                uint8_t trapping_flags;
                TrapHandlerRegistry trap_handler_registry;
    
                Environment()
                    : exception_flags(static_cast<uint8_t>(NoException)),
                      trapping_flags(static_cast<uint8_t>(NoException)) {}

                void check_and_trap(uint8_t flags) {
                    uint8_t trap_mask = static_cast<uint8_t>(trapping_flags);
                    uint8_t active_flags = static_cast<uint8_t>(flags);
        
                    if (trap_mask & active_flags) {
                        trap_handler_registry.raise(static_cast<LF_exception_flags>(trap_mask & active_flags));
                    }
                }
    
                void set_exception_flag(LF_exception_flags flags) {
                    exception_flags |= static_cast<uint8_t>(flags);
                    check_and_trap(exception_flags);
                }
    
                void set_trapping_flag(LF_exception_flags flags) {
                    trapping_flags |= static_cast<uint8_t>(flags);
                }
    
                void clear_exception_flag(LF_exception_flags flags) {
                    exception_flags &= ~static_cast<uint8_t>(flags);
                }
    
                void clear_trapping_flag(LF_exception_flags flags) {
                    trapping_flags &= ~static_cast<uint8_t>(flags);
                }
    
                void reset_exception_flags() {
                    exception_flags = static_cast<uint8_t>(NoException);
                }
    
                void reset_trapping_flags() {
                    trapping_flags = static_cast<uint8_t>(NoException);
                }
    
                void print_flags() const {
                    std::cout << "Exception Flags: ";
                    for (int i = 0; i < 5; ++i) {
                        if (exception_flags & (1 << i)) {
                            std::cout << exception_names[i] << " ";
                        }
                    }
                    std::cout << "\n";
                }
    

    
                void raise_if_enabled(LF_exception_flags f) {
                    if (trapping_flags & static_cast<uint8_t>(f)) {
                        trap_handler_registry.raise(f);
                    }
                }
    
                uint8_t get_exception_flags() const {
                    return exception_flags;
                }
    
                uint8_t get_trapping_flags() const {
                    return trapping_flags;
                }
                

                //specialize for
            };


        // Single, header-safe global environment shared by every translation unit.
        // (inline => one definition even when ENABLE_EXCEPT is on across TUs.)
        inline Environment f_env{};

        // Gate helper: raise `f` on the global environment, but only for formats that
        // opted into IEEE-754 exception semantics (NA_behavior == _754). The behavior
        // is a compile-time constant, so the non-_754 case folds away to nothing.
        template <lo_float::NaN_Behaviors NAB>
        inline void signal_if_754(LF_exception_flags f) {
            if constexpr (NAB == lo_float::NaN_Behaviors::_754) {
                f_env.set_exception_flag(f);
            }
        }


    } // namespace lo_float_internal
}
