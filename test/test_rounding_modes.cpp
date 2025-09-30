// -------------------------------------------------------------
// rounding-modes-test.cpp   ― rewritten for call-site rounding
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>
#include <limits>

#include "lo_float.h"      
#include "lo_float_sci.hpp"


using namespace lo_float;

double get_denom(double d) {
    // treat zero / non-finite as the fallback case
    if (d == 0.0 || !std::isfinite(d)) return 1.0;

    int exp = 0;
    std::frexp(d, &exp);         // d == mantissa * 2^exp
    return std::ldexp(1.0, exp); // returns 1.0 * 2^exp
}

template<Float F>
bool is_normal(F f) {
    return std::numeric_limits<F>::min() < f;
}

template<int l, int p> 
int test_n2n_3109() {
    

    int num_errors = 0;

    using P3109_type = P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    double mach_eps = std::pow(2.0, -p + 1);
    auto rnd32 = []() -> float {
        return static_cast<float>(std::rand()) / RAND_MAX;
    };
    auto is_even = [](uint32_t bits){ return (bits & 1u) == 0; };
    auto is_odd  = [](uint32_t bits){ return (bits & 1u) == 1; };

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    // ----------------------------------------------------------------------
    // 1.  RoundUp / RoundDown
    // ----------------------------------------------------------------------
    
    for (int i = 0; i < 2000; ++i) {
        float d  = rnd32();

        double UNT = (double)std::numeric_limits<P3109_type>::denorm_min();

        P3109_type fd   = Round<float,P3109_type, Rounding_Mode::RoundDown>(d);
        P3109_type fu   = Round<float,P3109_type, Rounding_Mode::RoundUp>(d);
        P3109_type frne = Round<float,P3109_type, Rounding_Mode::RoundToNearestEven>(d);
        P3109_type frno = Round<float,P3109_type, Rounding_Mode::RoundToNearestOdd>(d);
        P3109_type frta = Round<float,P3109_type, Rounding_Mode::RoundTiesToAway>(d);
        P3109_type frtz = Round<float,P3109_type, Rounding_Mode::RoundTowardsZero>(d);
        P3109_type fraw = Round<float,P3109_type, Rounding_Mode::RoundAwayFromZero>(d);

        double denom = get_denom(d);

        double abs_down = std::fabs(static_cast<double>(fd)   - d);
        double abs_up   = std::fabs(static_cast<double>(fu)   - d);
        double abs_rne  = std::fabs(static_cast<double>(frne) - d);
        double abs_rno  = std::fabs(static_cast<double>(frno) - d);
        double abs_rta  = std::fabs(static_cast<double>(frta) - d);
        double abs_rtz  = std::fabs(static_cast<double>(frtz) - d);
        double abs_raw  = std::fabs(static_cast<double>(fraw) - d);

        double rel_down = std::fabs(static_cast<double>(fd)   - d) / denom;
        double rel_up   = std::fabs(static_cast<double>(fu)   - d) / denom;
        double rel_rne  = std::fabs(static_cast<double>(frne) - d) / denom;
        double rel_rno  = std::fabs(static_cast<double>(frno) - d) / denom;
        double rel_rta  = std::fabs(static_cast<double>(frta) - d) / denom;
        double rel_rtz  = std::fabs(static_cast<double>(frtz) - d) / denom;
        double rel_raw  = std::fabs(static_cast<double>(fraw) - d) / denom;

       if (!isnan(d)) {
    if (is_normal(fd)) {
        if ((double)fd > d || rel_down > mach_eps) {
            std::cout << "RoundDown failed (x=" << d << " fd=" << (double)fd
                      << " rel_err=" << rel_down << ")\n";
            num_errors++;
        }
    } else {
        if ((double)fd > d || abs_down > UNT) {
            std::cout << "RoundDown failed (x=" << d << " fd=" << (double)fd
                      << " abs_err=" << abs_down << ")\n";
            num_errors++;
        }
    }

    if (is_normal(fu)) {
        if ((double)fu < d || rel_up > mach_eps) {
            std::cout << "RoundUp failed (x=" << d << " fu=" << (double)fu
                      << " rel_err=" << rel_up << ")\n";
            num_errors++;
        }
    } else {
        if ((double)fu < d || abs_up > UNT) {
            std::cout << "RoundUp failed (x=" << d << " fu=" << (double)fu
                      << " abs_err=" << abs_up << ")\n";
            num_errors++;
        }
    }

    if (is_normal(frtz)) {
        if (std::fabs((double)frtz) > std::fabs(d) || rel_rtz > mach_eps) {
            std::cout << "RoundTowardsZero failed (x=" << d << " frtz=" << (double)frtz
                      << " rel_err=" << rel_rtz << ")\n";
            num_errors++;
        }
    } else {
        if (std::fabs((double)frtz) > std::fabs(d) || abs_rtz > UNT) {
            std::cout << "RoundTowardsZero failed (x=" << d << " frtz=" << (double)frtz
                      << " abs_err=" << abs_rtz << ")\n";
            num_errors++;
        }
    }

    if (is_normal(fraw)) {
        if (std::fabs((double)fraw) < std::fabs(d) || rel_raw > mach_eps) {
            std::cout << "RoundAwayFromZero failed (x=" << d << " fraw=" << (double)fraw
                      << " rel_err=" << rel_raw << ")\n";
            num_errors++;
        }
    } else {
        if (std::fabs((double)fraw) < std::fabs(d) || abs_raw > UNT) {
            std::cout << "RoundAwayFromZero failed (x=" << d << " fraw=" << (double)fraw
                      << " abs_err=" << abs_raw << ")\n";
            num_errors++;
        }
    }

    if (is_normal(frne)) {
        if (rel_rne > mach_eps) {
            std::cout << "RoundTiesToEven failed (x=" << d << " frne=" << (double)frne
                      << " rel_err=" << rel_rne << ")\n";
            num_errors++;
        }
    } else {
        if (abs_rne > UNT) {
            std::cout << "RoundTiesToEven failed (x=" << d << " frne=" << (double)frne
                      << " abs_err=" << abs_rne << ")\n";
            num_errors++;
        }
    }

    if (is_normal(frno)) {
        if (rel_rno > mach_eps) {
            std::cout << "RoundTiesToOdd failed (x=" << d << " frno=" << (double)frno
                      << " rel_err=" << rel_rno << ")\n";
            num_errors++;
        }
    } else {
        if (abs_rno > UNT) {
            std::cout << "RoundTiesToOdd failed (x=" << d << " frno=" << (double)frno
                      << " abs_err=" << abs_rno << ")\n";
            num_errors++;
        }
    }

    if (is_normal(frta)) {
        if (rel_rta > mach_eps) {
            std::cout << "RoundTiesToAway failed (x=" << d << " frta=" << (double)frta
                      << " rel_err=" << rel_rta << ")\n";
            num_errors++;
        }
    } else {
        if (abs_rta > UNT) {
            std::cout << "RoundTiesToAway failed (x=" << d << " frta=" << (double)frta
                      << " abs_err=" << abs_rta << ")\n";
            num_errors++;
        }
    }
}



    }
     //generate special "in between cases for the  tie breaking modes" -- these cases don't make sense when there are no explicit mantissa bits since all numbers are even
     if constexpr (p > 1) {
        for(uint32_t rep = 1; rep < (1u << l) - 2; rep++) {
            P3109_type a = P3109_type::FromRep(rep);
            double a_d = (double)a;
            double b_d = (double)P3109_type::FromRep(rep+1);
            double tie = (a_d + b_d) / 2.0;

            P3109_type rne = Round<double,P3109_type, Rounding_Mode::RoundToNearestEven>(tie);
            P3109_type rno = Round<double,P3109_type, Rounding_Mode::RoundToNearestOdd>(tie);
            P3109_type rta = Round<double,P3109_type, Rounding_Mode::RoundTiesToAway>(tie);

            if(isnan(tie)) continue;
            if (!is_even(rne.rep())) {
                std::cout << "RNE tie-round not even  (x=" << tie << ")\n";
                std::cout << "rne: " << (float)rne << "\n";
                num_errors++;
            }
            if (!is_odd(rno.rep())) {
                std::cout << "RNO tie-round not odd   (x=" << tie << ")\n";
                std::cout << "rno.rep(): " <<  (int)rno.rep() << "\n";
                num_errors++;
            }

            double ref_next = std::nextafter(tie, tie > 0 ? 1e300 : -1e300);
            double rta_d    = static_cast<double>(rta);
            if (std::fabs(rta_d) < std::fabs(ref_next) - 1e-20) {
                std::cout << "RTiesToAway tie not away (x=" << tie
                          << " res=" << rta_d << ")\n";
                num_errors++;
            }
        }
    }

std::cout << "P3109<" << l << "," << p << "> : " 
                  << (num_errors == 0 ? "pass" : "FAIL") << "\n";

    return num_errors;


}

template<int l, int... Ps>
void instantiate_for_l(std::integer_sequence<int, Ps...>) {
    (test_n2n_3109<l, Ps+1>(), ...);
}
template<int... Ls>
void instantiate_all_l(std::integer_sequence<int, Ls...>) {
    (instantiate_for_l<Ls>(std::make_integer_sequence<int, Ls-1 >{}), ...);
}

// Offset sequence helper: converts [0,1,2,...,N-1] to [Offset, Offset+1, ..., Offset+N-1]
template<int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

void instantiate_all() {
    // For l from 2 to 8
    instantiate_all_l(offset_sequence<2>(std::make_integer_sequence<int, 7>{}));
}



// template<int l, int p>
// int test_s2n_P3109() {

// }



// ------------------------------------------------------------------
int main() {
    instantiate_all();

    using fp8 = P3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>;
    double a = 0.99;
    fp8 a_fp8 = fp8(a);
    std::cout << "a_fp8: " << (double)a_fp8 << "\n";
        return 0;
}
