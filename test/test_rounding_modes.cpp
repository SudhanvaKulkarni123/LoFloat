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
int test_n2sn_3109() {
    

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

        P3109_type fd   = Round<P3109_type>(d, Rounding_Mode::RoundDown);
        P3109_type fu   = Round<P3109_type>(d, Rounding_Mode::RoundUp);
        P3109_type frne = Round<P3109_type>(d, Rounding_Mode::RoundToNearestEven);
        P3109_type frno = Round<P3109_type>(d, Rounding_Mode::RoundToNearestOdd);
        P3109_type frta = Round<P3109_type>(d, Rounding_Mode::RoundTiesToAway);
        P3109_type frtz = Round<P3109_type>(d, Rounding_Mode::RoundTowardsZero);
        P3109_type fraw = Round<P3109_type>(d, Rounding_Mode::RoundAwayFromZero);

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

            P3109_type rne = Round<P3109_type>(tie, Rounding_Mode::RoundToNearestEven);
            P3109_type rno = Round<P3109_type>(tie, Rounding_Mode::RoundToNearestOdd);
            P3109_type rta = Round<P3109_type>(tie, Rounding_Mode::RoundTiesToAway);

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

//add test cases for fp16. bf16 and tf32
int test_n2n_old()
{
    auto is_even = [](uint32_t bits){ return (bits & 1u) == 0; };
    auto is_odd  = [](uint32_t bits){ return (bits & 1u) == 1; };
    int num_errors = 0;

    for(int i = 0; i < 10000; i++) {
    float d = static_cast<float>(std::rand()) / RAND_MAX * 1000.0f;

        using bf16 = Templated_Float<bfloatPrecisionParams>;
        using tf32 = Templated_Float<tf32PrecisionParams>;
        using binary16 = Templated_Float<halfPrecisionParams>;

        double denom = get_denom(d);

        // bf16 rounding and error calculations
        bf16 fb_d   = Round<bf16>(d, Rounding_Mode::RoundDown);
        bf16 fb_u   = Round<bf16>(d, Rounding_Mode::RoundUp);
        bf16 fb_rne = Round<bf16>(d, Rounding_Mode::RoundToNearestEven);
        bf16 fb_rno = Round<bf16>(d, Rounding_Mode::RoundToNearestOdd);
        bf16 fb_rta = Round<bf16>(d, Rounding_Mode::RoundTiesToAway);
        bf16 fb_rtz = Round<bf16>(d, Rounding_Mode::RoundTowardsZero);
        bf16 fb_raw = Round<bf16>(d, Rounding_Mode::RoundAwayFromZero);

        double abs_b_d = std::fabs(static_cast<double>(fb_d) - d);
        double abs_b_u = std::fabs(static_cast<double>(fb_u) - d);
        double abs_b_rne = std::fabs(static_cast<double>(fb_rne) - d);
        double abs_b_rno = std::fabs(static_cast<double>(fb_rno) - d);
        double abs_b_rta = std::fabs(static_cast<double>(fb_rta) - d);
        double abs_b_rtz = std::fabs(static_cast<double>(fb_rtz) - d);
        double abs_b_raw = std::fabs(static_cast<double>(fb_raw) - d);

        double rel_b_d = abs_b_d / denom;
        double rel_b_u = abs_b_u / denom;
        double rel_b_rne = abs_b_rne / denom;
        double rel_b_rno = abs_b_rno / denom;
        double rel_b_rta = abs_b_rta / denom;
        double rel_b_rtz = abs_b_rtz / denom;
        double rel_b_raw = abs_b_raw / denom;

        // tf32 rounding and error calculations
        tf32 ft_d   = Round<tf32>(d, Rounding_Mode::RoundDown);
        tf32 ft_u   = Round<tf32>(d, Rounding_Mode::RoundUp);
        tf32 ft_rne = Round<tf32>(d, Rounding_Mode::RoundToNearestEven);
        tf32 ft_rno = Round<tf32>(d, Rounding_Mode::RoundToNearestOdd);
        tf32 ft_rta = Round<tf32>(d, Rounding_Mode::RoundTiesToAway);
        tf32 ft_rtz = Round<tf32>(d, Rounding_Mode::RoundTowardsZero);
        tf32 ft_raw = Round<tf32>(d, Rounding_Mode::RoundAwayFromZero);

        double abs_t_d = std::fabs(static_cast<double>(ft_d) - d);
        double abs_t_u = std::fabs(static_cast<double>(ft_u) - d);
        double abs_t_rne = std::fabs(static_cast<double>(ft_rne) - d);
        double abs_t_rno = std::fabs(static_cast<double>(ft_rno) - d);
        double abs_t_rta = std::fabs(static_cast<double>(ft_rta) - d);
        double abs_t_rtz = std::fabs(static_cast<double>(ft_rtz) - d);
        double abs_t_raw = std::fabs(static_cast<double>(ft_raw) - d);

        double rel_t_d = abs_t_d / denom;
        double rel_t_u = abs_t_u / denom;
        double rel_t_rne = abs_t_rne / denom;
        double rel_t_rno = abs_t_rno / denom;
        double rel_t_rta = abs_t_rta / denom;
        double rel_t_rtz = abs_t_rtz / denom;
        double rel_t_raw = abs_t_raw / denom;

        // binary16 rounding and error calculations
        binary16 fh_d   = Round<binary16>(d, Rounding_Mode::RoundDown);
        binary16 fh_u   = Round<binary16>(d, Rounding_Mode::RoundUp);
        binary16 fh_rne = Round<binary16>(d, Rounding_Mode::RoundToNearestEven);
        binary16 fh_rno = Round<binary16>(d, Rounding_Mode::RoundToNearestOdd);
        binary16 fh_rta = Round<binary16>(d, Rounding_Mode::RoundTiesToAway);
        binary16 fh_rtz = Round<binary16>(d, Rounding_Mode::RoundTowardsZero);
        binary16 fh_raw = Round<binary16>(d, Rounding_Mode::RoundAwayFromZero);

        double abs_h_d = std::fabs(static_cast<double>(fh_d) - d);
        double abs_h_u = std::fabs(static_cast<double>(fh_u) - d);
        double abs_h_rne = std::fabs(static_cast<double>(fh_rne) - d);
        double abs_h_rno = std::fabs(static_cast<double>(fh_rno) - d);
        double abs_h_rta = std::fabs(static_cast<double>(fh_rta) - d);
        double abs_h_rtz = std::fabs(static_cast<double>(fh_rtz) - d);
        double abs_h_raw = std::fabs(static_cast<double>(fh_raw) - d);

        double rel_h_d = abs_h_d / denom;
        double rel_h_u = abs_h_u / denom;
        double rel_h_rne = abs_h_rne / denom;
        double rel_h_rno = abs_h_rno / denom;
        double rel_h_rta = abs_h_rta / denom;
        double rel_h_rtz = abs_h_rtz / denom;
        double rel_h_raw = abs_h_raw / denom;

        if (!isnan(d)) {
            double mach_eps_b = std::pow(2.0, -7); //bf16 epsilon
            double mach_eps_t = std::pow(2.0, -10); //tf32 epsilon
            double mach_eps_h = std::pow(2.0, -10); //half precision epsilon

            double UNT_b = (double)std::numeric_limits<bf16>::denorm_min();
            double UNT_t = (double)std::numeric_limits<tf32>::denorm_min();
            double UNT_h = (double)std::numeric_limits<binary16>::denorm_min();

            // --- bf16 Tests ---
            if (is_normal(fb_d)) { if ((double)fb_d > d || rel_b_d > mach_eps_b) { std::cout << "bf16 RoundDown failed (x=" << d << " res=" << (double)fb_d << " rel_err=" << rel_b_d << ")\n"; num_errors++; } } else { if ((double)fb_d > d || abs_b_d > UNT_b) { std::cout << "bf16 RoundDown failed (x=" << d << " res=" << (double)fb_d << " abs_err=" << abs_b_d << ")\n"; num_errors++; } }
            if (is_normal(fb_u)) { if ((double)fb_u < d || rel_b_u > mach_eps_b) { std::cout << "bf16 RoundUp failed (x=" << d << " res=" << (double)fb_u << " rel_err=" << rel_b_u << ")\n"; num_errors++; } } else { if ((double)fb_u < d || abs_b_u > UNT_b) { std::cout << "bf16 RoundUp failed (x=" << d << " res=" << (double)fb_u << " abs_err=" << abs_b_u << ")\n"; num_errors++; } }
            if (is_normal(fb_rtz)) { if (std::fabs((double)fb_rtz) > std::fabs(d) || rel_b_rtz > mach_eps_b) { std::cout << "bf16 RoundTowardsZero failed (x=" << d << " res=" << (double)fb_rtz << " rel_err=" << rel_b_rtz << ")\n"; num_errors++; } } else { if (std::fabs((double)fb_rtz) > std::fabs(d) || abs_b_rtz > UNT_b) { std::cout << "bf16 RoundTowardsZero failed (x=" << d << " res=" << (double)fb_rtz << " abs_err=" << abs_b_rtz << ")\n"; num_errors++; } }
            if (is_normal(fb_raw)) { if (std::fabs((double)fb_raw) < std::fabs(d) || rel_b_raw > mach_eps_b) { std::cout << "bf16 RoundAwayFromZero failed (x=" << d << " res=" << (double)fb_raw << " rel_err=" << rel_b_raw << ")\n"; num_errors++; } } else { if (std::fabs((double)fb_raw) < std::fabs(d) || abs_b_raw > UNT_b) { std::cout << "bf16 RoundAwayFromZero failed (x=" << d << " res=" << (double)fb_raw << " abs_err=" << abs_b_raw << ")\n"; num_errors++; } }
            if (is_normal(fb_rne)) { if (rel_b_rne > mach_eps_b) { std::cout << "bf16 RoundTiesToEven failed (x=" << d << " res=" << (double)fb_rne << " rel_err=" << rel_b_rne << ")\n"; num_errors++; } } else { if (abs_b_rne > UNT_b) { std::cout << "bf16 RoundTiesToEven failed (x=" << d << " res=" << (double)fb_rne << " abs_err=" << abs_b_rne << ")\n"; num_errors++; } }
            if (is_normal(fb_rno)) { if (rel_b_rno > mach_eps_b) { std::cout << "bf16 RoundTiesToOdd failed (x=" << d << " res=" << (double)fb_rno << " rel_err=" << rel_b_rno << ")\n"; num_errors++; } } else { if (abs_b_rno > UNT_b) { std::cout << "bf16 RoundTiesToOdd failed (x=" << d << " res=" << (double)fb_rno << " abs_err=" << abs_b_rno << ")\n"; num_errors++; } }
            if (is_normal(fb_rta)) { if (rel_b_rta > mach_eps_b) { std::cout << "bf16 RoundTiesToAway failed (x=" << d << " res=" << (double)fb_rta << " rel_err=" << rel_b_rta << ")\n"; num_errors++; } } else { if (abs_b_rta > UNT_b) { std::cout << "bf16 RoundTiesToAway failed (x=" << d << " res=" << (double)fb_rta << " abs_err=" << abs_b_rta << ")\n"; num_errors++; } }

            // --- tf32 Tests ---
            if (is_normal(ft_d)) { if ((double)ft_d > d || rel_t_d > mach_eps_t) { std::cout << "tf32 RoundDown failed (x=" << d << " res=" << (double)ft_d << " rel_err=" << rel_t_d << ")\n"; num_errors++; } } else { if ((double)ft_d > d || abs_t_d > UNT_t) { std::cout << "tf32 RoundDown failed (x=" << d << " res=" << (double)ft_d << " abs_err=" << abs_t_d << ")\n"; num_errors++; } }
            if (is_normal(ft_u)) { if ((double)ft_u < d || rel_t_u > mach_eps_t) { std::cout << "tf32 RoundUp failed (x=" << d << " res=" << (double)ft_u << " rel_err=" << rel_t_u << ")\n"; num_errors++; } } else { if ((double)ft_u < d || abs_t_u > UNT_t) { std::cout << "tf32 RoundUp failed (x=" << d << " res=" << (double)ft_u << " abs_err=" << abs_t_u << ")\n"; num_errors++; } }
            if (is_normal(ft_rtz)) { if (std::fabs((double)ft_rtz) > std::fabs(d) || rel_t_rtz > mach_eps_t) { std::cout << "tf32 RoundTowardsZero failed (x=" << d << " res=" << (double)ft_rtz << " rel_err=" << rel_t_rtz << ")\n"; num_errors++; } } else { if (std::fabs((double)ft_rtz) > std::fabs(d) || abs_t_rtz > UNT_t) { std::cout << "tf32 RoundTowardsZero failed (x=" << d << " res=" << (double)ft_rtz << " abs_err=" << abs_t_rtz << ")\n"; num_errors++; } }
            if (is_normal(ft_raw)) { if (std::fabs((double)ft_raw) < std::fabs(d) || rel_t_raw > mach_eps_t) { std::cout << "tf32 RoundAwayFromZero failed (x=" << d << " res=" << (double)ft_raw << " rel_err=" << rel_t_raw << ")\n"; num_errors++; } } else { if (std::fabs((double)ft_raw) < std::fabs(d) || abs_t_raw > UNT_t) { std::cout << "tf32 RoundAwayFromZero failed (x=" << d << " res=" << (double)ft_raw << " abs_err=" << abs_t_raw << ")\n"; num_errors++; } }
            if (is_normal(ft_rne)) { if (rel_t_rne > mach_eps_t) { std::cout << "tf32 RoundTiesToEven failed (x=" << d << " res=" << (double)ft_rne << " rel_err=" << rel_t_rne << ")\n"; num_errors++; } } else { if (abs_t_rne > UNT_t) { std::cout << "tf32 RoundTiesToEven failed (x=" << d << " res=" << (double)ft_rne << " abs_err=" << abs_t_rne << ")\n"; num_errors++; } }
            if (is_normal(ft_rno)) { if (rel_t_rno > mach_eps_t) { std::cout << "tf32 RoundTiesToOdd failed (x=" << d << " res=" << (double)ft_rno << " rel_err=" << rel_t_rno << ")\n"; num_errors++; } } else { if (abs_t_rno > UNT_t) { std::cout << "tf32 RoundTiesToOdd failed (x=" << d << " res=" << (double)ft_rno << " abs_err=" << abs_t_rno << ")\n"; num_errors++; } }
            if (is_normal(ft_rta)) { if (rel_t_rta > mach_eps_t) { std::cout << "tf32 RoundTiesToAway failed (x=" << d << " res=" << (double)ft_rta << " rel_err=" << rel_t_rta << ")\n"; num_errors++; } } else { if (abs_t_rta > UNT_t) { std::cout << "tf32 RoundTiesToAway failed (x=" << d << " res=" << (double)ft_rta << " abs_err=" << abs_t_rta << ")\n"; num_errors++; } }

            // --- binary16 Tests ---
            if (is_normal(fh_d)) { if ((double)fh_d > d || rel_h_d > mach_eps_h) { std::cout << "binary16 RoundDown failed (x=" << d << " res=" << (double)fh_d << " rel_err=" << rel_h_d << ")\n"; num_errors++; } } else { if ((double)fh_d > d || abs_h_d > UNT_h) { std::cout << "binary16 RoundDown failed (x=" << d << " res=" << (double)fh_d << " abs_err=" << abs_h_d << ")\n"; num_errors++; } }
            if (is_normal(fh_u)) { if ((double)fh_u < d || rel_h_u > mach_eps_h) { std::cout << "binary16 RoundUp failed (x=" << d << " res=" << (double)fh_u << " rel_err=" << rel_h_u << ")\n"; num_errors++; } } else { if ((double)fh_u < d || abs_h_u > UNT_h) { std::cout << "binary16 RoundUp failed (x=" << d << " res=" << (double)fh_u << " abs_err=" << abs_h_u << ")\n"; num_errors++; } }
            if (is_normal(fh_rtz)) { if (std::fabs((double)fh_rtz) > std::fabs(d) || rel_h_rtz > mach_eps_h) { std::cout << "binary16 RoundTowardsZero failed (x=" << d << " res=" << (double)fh_rtz << " rel_err=" << rel_h_rtz << ")\n"; num_errors++; } } else { if (std::fabs((double)fh_rtz) > std::fabs(d) || abs_h_rtz > UNT_h) { std::cout << "binary16 RoundTowardsZero failed (x=" << d << " res=" << (double)fh_rtz << " abs_err=" << abs_h_rtz << ")\n"; num_errors++; } }
            if (is_normal(fh_raw)) { if (std::fabs((double)fh_raw) < std::fabs(d) || rel_h_raw > mach_eps_h) { std::cout << "binary16 RoundAwayFromZero failed (x=" << d << " res=" << (double)fh_raw << " rel_err=" << rel_h_raw << ")\n"; num_errors++; } } else { if (std::fabs((double)fh_raw) < std::fabs(d) || abs_h_raw > UNT_h) { std::cout << "binary16 RoundAwayFromZero failed (x=" << d << " res=" << (double)fh_raw << " abs_err=" << abs_h_raw << ")\n"; num_errors++; } }
            if (is_normal(fh_rne)) { if (rel_h_rne > mach_eps_h) { std::cout << "binary16 RoundTiesToEven failed (x=" << d << " res=" << (double)fh_rne << " rel_err=" << rel_h_rne << ")\n"; num_errors++; } } else { if (abs_h_rne > UNT_h) { std::cout << "binary16 RoundTiesToEven failed (x=" << d << " res=" << (double)fh_rne << " abs_err=" << abs_h_rne << ")\n"; num_errors++; } }
            if (is_normal(fh_rno)) { if (rel_h_rno > mach_eps_h) { std::cout << "binary16 RoundTiesToOdd failed (x=" << d << " res=" << (double)fh_rno << " rel_err=" << rel_h_rno << ")\n"; num_errors++; } } else { if (abs_h_rno > UNT_h) { std::cout << "binary16 RoundTiesToOdd failed (x=" << d << " res=" << (double)fh_rno << " abs_err=" << abs_h_rno << ")\n"; num_errors++; } }
            if (is_normal(fh_rta)) { if (rel_h_rta > mach_eps_h) { std::cout << "binary16 RoundTiesToAway failed (x=" << d << " res=" << (double)fh_rta << " rel_err=" << rel_h_rta << ")\n"; num_errors++; } } else { if (abs_h_rta > UNT_h) { std::cout << "binary16 RoundTiesToAway failed (x=" << d << " res=" << (double)fh_rta << " abs_err=" << abs_h_rta << ")\n"; num_errors++; } }
        }

    
}
if (num_errors == 0) std::cout << "bf16, fp16, tf32, pass\n";
    else                std::cout << "bf16, fp16, tf32, FAIL\n";


}


template<int l, int... Ps>
void instantiate_for_l(std::integer_sequence<int, Ps...>) {
    (test_n2sn_3109<l, Ps+1>(), ...);
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

template<int l, int p>
int test_array_round() {
    
    int n = 400;
    float* float_arr = (float*) malloc(n*sizeof(float));
    P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>* arr = (P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>*) malloc(sizeof(P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>)*n);
    for (int i = 0; i < n; i++) {
        float_arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    lo_float::Project(float_arr, arr, n, Rounding_Mode::RoundToNearestEven);

    int num_errors = 0;
    for (int i = 0; i < n; i++) {
        P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating> single_val = arr[i];
        float UNT = (float)std::numeric_limits<P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>>::denorm_min();
        float x  = float_arr[i];
        float xr = static_cast<float>(arr[i]);           // [x] as float
        float diff = std::fabs(xr - x);
        float eps = std::ldexp(1.0f, -(p - 1));          // machine eps = 2^-(p-1)
        int e = (x == 0.0f) ? 0 : std::ilogb(x);         // exponent of x
        float tol = is_normal(single_val) ? 0.5f * eps * std::ldexp(1.0f, e) : UNT ;    // (eps/2) * 2^e
        
        if (!(diff < tol)) {
            if constexpr ( l == 8 && (p == 7 || p == 6)) std::cout << "x = " << x << " xr = " << xr << " tol = " << tol; 
            num_errors++;
        }
    }

    if (num_errors == 0) {
        std::cout << "Array rounding test passed for P3109<" << l << "," << p << ">\n";
    } else {
        std::cout << "Array rounding test failed for P3109<" << l << "," << p << "> with " << num_errors << " errors\n";
    }

    free(float_arr);
    free(arr);

    return num_errors;
}

template<int l, int... Ps>
void test_array_for_l(std::integer_sequence<int, Ps...>) {
    (test_array_round<l, Ps+1>(), ...);
}

template<int... Ls>
void instantiate_array_test_all_l(std::integer_sequence<int, Ls...>) {
    (test_array_for_l<Ls>(std::make_integer_sequence<int, Ls-1 >{}), ...);
}

void instantiate_array_all() {
    // For l from 2 to 8
    instantiate_array_test_all_l(offset_sequence<2>(std::make_integer_sequence<int, 7>{}));
}





// ------------------------------------------------------------------
int main() {
   instantiate_all();
    //test_n2n_old();
   instantiate_array_all();

//    float float_arr[4];
//    for (int i = 0; i < 4; i++) {
//     float_arr[i] = 0.0067845568992197514f;
//    }

//    P3109_float<8, 7, Signedness::Signed, Inf_Behaviors::Saturating> arr[4];
//    lo_float::Project(float_arr, arr, 4, Rounding_Mode::RoundToNearestEven);

//    std::cout << (float)arr[0] << "\n";
//    std::cout << (float)P3109_float<8, 7, Signedness::Signed, Inf_Behaviors::Saturating>(float_arr[0]) << "\n";

   




        return 0;
}
