#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "lo_float.h"

using namespace lo_float;




template<int l, int p>
int test_comparisons_exhaustive()
{
    using binary_lp = P_3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    using unsigned_binary_lp = P_3109_float<l, p, Signedness::Unsigned, Inf_Behaviors::Saturating>;
    const int total_values = 1 << l;

    int case1_bound = total_values/2;
    

    int errors = 0;

    //check all cases where 0 < a < b for signed and unsigned
    for(unsigned int rep1 = 0; rep1 < total_values/2; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values/2; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
   
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
        }
    }

    for(unsigned int rep1 = 0; rep1 < total_values-1; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values-1; rep2++) {
            unsigned_binary_lp a = unsigned_binary_lp::FromRep(rep1);
            unsigned_binary_lp b = unsigned_binary_lp::FromRep(rep2);
   
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
        }
    }



    //check cases where a < 0 < b (signed)
    for(unsigned int rep1 = total_values/2 + 1; rep1 < total_values; rep1++) {
        for(unsigned int rep2 = 0; rep2 < total_values/2; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
            if(!(a <= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a > b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b >= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b < a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
            
        }
    }

    //check cases where b < a  < 0
    for(unsigned int rep1 = total_values/2 + 1; rep1 < total_values; rep1++) {
        for(unsigned int rep2 = rep1; rep2 < total_values; rep2++) {
            binary_lp a = binary_lp::FromRep(rep1);
            binary_lp b = binary_lp::FromRep(rep2);
            if(!(a >= b)) {
                std::cout << "Error: " << (float)a << " <= " << (float)b << " failed\n";
                errors++;
            }
            if(a < b) {
                std::cout << "Error: " << (float)a << " > " << (float)b << " failed\n";
                errors++;
            }
            if(!(b <= a)) {
                std::cout << "Error: " << (float)b << " >= " << (float)a << " failed\n";
                errors++;
            }
            if(b > a) {
                std::cout << "Error: " << (float)b << " < " << (float)a << " failed\n";
                errors++;
            }
            
        }
    }


    printf("Total errors for P_3109<%d,%d>: %d\n", l, p, errors);
    return errors;
    
}

template<int l, int... Ps>
int instantiate_for_l(std::integer_sequence<int, Ps...>) {
    return (test_comparisons_exhaustive<l, Ps+1>() + ...);
}
template<int... Ls>
int instantiate_all_l(std::integer_sequence<int, Ls...>) {
    return (instantiate_for_l<Ls>(std::make_integer_sequence<int, Ls-1 >{}) + ...);
}

// Offset sequence helper: converts [0,1,2,...,N-1] to [Offset, Offset+1, ..., Offset+N-1]
template<int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

int instantiate_all() {
    // For l from 2 to 8
    return instantiate_all_l(offset_sequence<2>(std::make_integer_sequence<int, 6>{}));
}

//used to test comparisons for binary16, bf16 and tf32
int test_comparisons_random(int num_tests)
{

    // F4 format-preset aliases (exercises the new public aliases too)
    using bf16     = bfloat16;
    using binary16 = half;
    // tf32 resolves to the global lo_float::tf32 alias

    for(int i = 0; i < num_tests; i++) {
        float a = static_cast<float>(((double) rand()) / (double) RAND_MAX * 1000.0);
        float b = static_cast<float>(((double) rand()) / (double) RAND_MAX * 1000.0);

        int sign_a = rand() % 2;
        int sign_b = rand() % 2;

        if(sign_a) a = -a;
        if(sign_b) b = -b;

        bf16 a_bf16 = bf16(a);
        bf16 b_bf16 = bf16(b);

        binary16 a_binary16 = binary16(a);
        binary16 b_binary16 = binary16(b);

        tf32 a_tf32 = tf32(a);
        tf32 b_tf32 = tf32(b);

        float a_ref_bf16 = (float)a_bf16;
        float b_ref_bf16 = (float)b_bf16;
        float a_ref_half = (float)a_binary16;
        float b_ref_half = (float)b_binary16;
        float a_ref_tf32 = (float)a_tf32;
        float b_ref_tf32 = (float)b_tf32;

        if((a_ref_bf16 <= b_ref_bf16) != (a_bf16 <= b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " <= " << (float)b_bf16 << " failed\n";

            std::cout << "a.rep(): " << std::hex << a_bf16.rep() << "\n";
            std::cout << "b.rep(): " << std::hex << b_bf16.rep() << "\n";
            return 1;
        }
        if((a_ref_bf16 > b_ref_bf16) != (a_bf16 > b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " > " << (float)b_bf16 << " failed\n";
            return 1;
        }
        if((a_ref_bf16 >= b_ref_bf16) != (a_bf16 >= b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " >= " << (float)b_bf16 << " failed\n";
            return 1;
        }
        if((a_ref_bf16 < b_ref_bf16) != (a_bf16 < b_bf16)) {
            std::cout << "Error: bf16 " << (float)a_bf16 << " < " << (float)b_bf16 << " failed\n";
            return 1;
        }

        if((a_ref_half <= b_ref_half) != (a_binary16 <= b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " <= " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half > b_ref_half) != (a_binary16 > b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " > " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half >= b_ref_half) != (a_binary16 >= b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " >= " << (float)b_binary16 << " failed\n";
            return 1;
        }
         if((a_ref_half < b_ref_half) != (a_binary16 < b_binary16)) {
            std::cout << "Error: binary16 " << (float)a_binary16 << " < " << (float)b_binary16 << " failed\n";
            return 1;
        }

         if ((a_ref_tf32 <= b_ref_tf32) != (a_tf32 <= b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " <= " << (float)b_tf32 << " failed\n";
                std::cout << "a : " << a << ", b: " << b << "\n";
                std::cout << "a_ref_tf32: " << a_ref_tf32 << ", b_ref_tf32: " << b_ref_tf32 << "\n";
                std::cout << "a.rep(): " << std::hex << a_tf32.rep() << "\n";
                std::cout << "b.rep(): " << std::hex << b_tf32.rep() << "\n";
                std::cout << (float) abs(a_tf32) << ", " << (float) abs(b_tf32) << "\n";
                return 1;
            }
            if ((a_ref_tf32 > b_ref_tf32) != (a_tf32 > b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " > " << (float)b_tf32 << " failed\n";
                return 1;
            }
            if ((a_ref_tf32 >= b_ref_tf32) != (a_tf32 >= b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " >= " << (float)b_tf32 << " failed\n";
                return 1;
            }
            if ((a_ref_tf32 < b_ref_tf32) != (a_tf32 < b_tf32)) {
                std::cout << "Error: tf32 " << (float)a_tf32 << " < " << (float)b_tf32 << " failed\n";
                return 1;
            }

    }

    return 0;



}


// ---------------------------------------------------------------------------
// T3 — special-value comparison edge cases (NaN / ±Inf / ±0).
// Reference-vs-lo idiom: float already encodes correct IEEE semantics
// (NaN unordered, Inf ordering, ±0 equality), and Compare is built the same
// way, so we compare each of the six operators on (float)a,(float)b against
// a,b and assert agreement. Only Inf/NaN-capable formats belong here.
// ---------------------------------------------------------------------------
// Reference-vs-lo all-pairs sweep over the special-value set. This is robust
// for ANY format: it never hard-codes a truth table — it asserts the six
// operators agree between the float reference and the lo_float value. It is
// self-consistent even when a format lacks a given special value (e.g. P3109's
// signbit pattern is NaN, tf32's infinity() encodes a finite value): the same
// (float)v is compared against the same lo v, so Compare's actual path for that
// encoding is validated against float either way.
template <typename T>
int edge_pairs_sweep(const char* name)
{
    int errors = 0;

    T plus0    = T::FromRep(0);
    T minus0   = T::FromRep(1u << (T::bitwidth - 1));
    T plusInf  = std::numeric_limits<T>::infinity();
    T minusInf = T::FromRep(T::IsInfFunctor.minNegInf());
    T nanv     = std::numeric_limits<T>::quiet_NaN();
    T one      = T(1.0f);
    T negtwo   = T(-2.0f);

    T vals[]            = { plus0, minus0, plusInf, minusInf, nanv, one, negtwo };
    const char* lbls[]  = { "+0", "-0", "+Inf", "-Inf", "NaN", "1.0", "-2.0" };
    const int N = 7;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float fa = (float)vals[i];
            float fb = (float)vals[j];
            #define CHECK_OP(op, opname) \
                if ((fa op fb) != (vals[i] op vals[j])) { \
                    std::cout << "Error: " << name << " " << lbls[i] << " " opname " " << lbls[j] \
                              << "  ref=" << (fa op fb) << " lo=" << (vals[i] op vals[j]) << "\n"; \
                    errors++; }
            CHECK_OP(==, "==")
            CHECK_OP(!=, "!=")
            CHECK_OP(<,  "<")
            CHECK_OP(<=, "<=")
            CHECK_OP(>,  ">")
            CHECK_OP(>=, ">=")
            #undef CHECK_OP
        }
    }

    if (errors) std::cout << name << ": " << errors << " edge-pair errors\n";
    return errors;
}

// Explicit headline invariants (self-describing). Only valid for formats with
// full IEEE special-value encodings (real -0, real ±Inf, real NaN). See the
// probe in PROGRESS/NOTES: half + bfloat16 qualify; tf32 has a broken inf
// encoding and P3109 reserves the signbit pattern for NaN (no -0).
template <typename T>
int ieee_invariants(const char* name)
{
    int errors = 0;

    T plus0    = T::FromRep(0);
    T minus0   = T::FromRep(1u << (T::bitwidth - 1));
    T plusInf  = std::numeric_limits<T>::infinity();
    T minusInf = T::FromRep(T::IsInfFunctor.minNegInf());
    T nanv     = std::numeric_limits<T>::quiet_NaN();
    T one      = T(1.0f);

    // sanity: the special values really are special in this format
    if (!isinf(plusInf))  { std::cout << "Error: " << name << " +Inf not isinf\n"; errors++; }
    if (!isinf(minusInf)) { std::cout << "Error: " << name << " -Inf not isinf\n"; errors++; }
    if (!isnan(nanv))     { std::cout << "Error: " << name << " NaN not isnan\n";  errors++; }

    if (!(plus0 == minus0))                    { std::cout << "Error: " << name << " +0==-0 failed\n";          errors++; }
    if (!(plus0 <= minus0 && plus0 >= minus0)) { std::cout << "Error: " << name << " +0<=>=-0 failed\n";        errors++; }
    if (plus0 < minus0 || plus0 > minus0)      { std::cout << "Error: " << name << " +0 strict vs -0 failed\n"; errors++; }
    if (nanv == nanv)                          { std::cout << "Error: " << name << " NaN==NaN true\n";          errors++; }
    if (!(nanv != nanv))                       { std::cout << "Error: " << name << " NaN!=NaN false\n";         errors++; }
    if (nanv < one || one < nanv || nanv <= one || nanv > one || nanv >= one) {
        std::cout << "Error: " << name << " NaN ordered vs finite\n"; errors++; }
    if (!(minusInf < plusInf))                 { std::cout << "Error: " << name << " -Inf<+Inf failed\n";       errors++; }
    if (!(plusInf == plusInf))                 { std::cout << "Error: " << name << " +Inf==+Inf failed\n";      errors++; }
    if (!(plusInf > one))                      { std::cout << "Error: " << name << " +Inf>finite failed\n";     errors++; }
    if (!(minusInf < one))                     { std::cout << "Error: " << name << " -Inf<finite failed\n";     errors++; }

    if (errors) std::cout << name << ": " << errors << " ieee-invariant errors\n";
    return errors;
}

int test_comparisons_edge_cases()
{
    int errors = 0;
    // Reference-vs-lo all-pairs sweep on every Inf/NaN-capable format — this is
    // the actual coverage of Compare's special-value paths.
    errors += edge_pairs_sweep<half>("half");
    errors += edge_pairs_sweep<bfloat16>("bfloat16");
    errors += edge_pairs_sweep<tf32>("tf32");
    errors += edge_pairs_sweep<
        P_3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>>("P3109<8,4,Ext>");
    // Explicit headline invariants on the full-IEEE formats (extra teeth).
    errors += ieee_invariants<half>("half");
    errors += ieee_invariants<bfloat16>("bfloat16");
    return errors;
}

// ---------------------------------------------------------------------------
// F4 — format-preset alias smoke check.
// ---------------------------------------------------------------------------
int test_format_aliases()
{
    int errors = 0;
    static_assert(std::is_same_v<float16, half>,   "float16 must alias half");
    static_assert(std::is_same_v<float32, single>, "float32 must alias single");

    auto chk = [&](bool cond, const char* msg) {
        if (!cond) { std::cout << "Error: alias " << msg << "\n"; errors++; }
    };

    // IEEE / ML + OCP element presets: a couple of finite floats round-trip
    chk((float)ocp_e4m3(1.5f)  == 1.5f, "ocp_e4m3(1.5)");
    chk((float)ocp_e5m2(1.5f)  == 1.5f, "ocp_e5m2(1.5)");
    chk((float)ocp_e3m2(1.5f)  == 1.5f, "ocp_e3m2(1.5)");
    chk((float)ocp_e2m1(1.5f)  == 1.5f, "ocp_e2m1(1.5)");
    chk((float)bfloat16(2.0f)  == 2.0f, "bfloat16(2.0)");
    chk((float)half(2.0f)      == 2.0f, "half(2.0)");
    chk((float)tf32(2.0f)      == 2.0f, "tf32(2.0)");
    chk((float)single(3.25f)   == 3.25f,"single(3.25)");

    // Dojo: default + non-default template bias compile and round-trip
    chk((float)dojo_cfloat8_1_4_3<>(1.5f)   == 1.5f, "dojo_cfloat8_1_4_3(1.5)");
    chk((float)dojo_cfloat8_1_5_2<>(1.5f)   == 1.5f, "dojo_cfloat8_1_5_2(1.5)");
    chk((float)dojo_cfloat16_1_8_7<>(1.5f)  == 1.5f, "dojo_cfloat16_1_8_7(1.5)");
    chk((float)dojo_cfloat16_1_6_9<>(1.5f)  == 1.5f, "dojo_cfloat16_1_6_9(1.5)");
    chk((float)dojo_cfloat8_1_4_3<5>(1.5f)  == 1.5f, "dojo_cfloat8_1_4_3<5>(1.5)"); // non-default bias

    // Dojo no-Inf / no-NaN invariant — every encoding is finite, incl. all-ones
    chk(!isinf(dojo_cfloat8_1_4_3<>(1.5f)) && !isnan(dojo_cfloat8_1_4_3<>(1.5f)),
        "dojo value finite");
    chk(!isinf(dojo_cfloat8_1_4_3<>::FromRep(0xFF)) && !isnan(dojo_cfloat8_1_4_3<>::FromRep(0xFF)),
        "dojo 0xFF not special");

    // E8M0 is primarily an MX scale format: power-of-two round-trip + NaN sanity
    chk((float)ocp_e8m0(4.0f) == 4.0f, "ocp_e8m0(4.0)");
    chk(isnan(std::numeric_limits<ocp_e8m0>::quiet_NaN()), "ocp_e8m0 qNaN");

    if (errors) std::cout << "format-alias smoke: " << errors << " errors\n";
    return errors;
}


int main()
{
    //run exhaustive test on P_3109 formats for length <= 8

    int total_errors = 0;

    total_errors += instantiate_all();

    int rnd = test_comparisons_random(10000);
    if (rnd) {
        printf("Randomized comparison tests failed\n");
        total_errors += rnd;
    }

    total_errors += test_comparisons_edge_cases();
    total_errors += test_format_aliases();

    if (total_errors == 0)
        std::cout << "comparison tests passed\n";
    else
        std::cout << "comparison tests FAILED: " << total_errors << " errors\n";

    return total_errors ? 1 : 0;
}