#include "Vector.h"
#include "Dot.hpp"
#include <iostream>
#include <chrono>
#include <cassert>
#include <limits>

using namespace lo_float;
using namespace Lo_Gemm;

/*broadly, there are 2 kinds of tests we should include - the first is the "normal test" where I only use number in the range (-1,1).
The total number of mantissa bits I can fill using this approach is num_mantissa1 + num_mantissa_2 + min_exp1 + min_exp2. So, this test can. maybe work with fp8 input fp16 accum
The problem arises when the number of mantissa bits in the accumulator is greater than num_mantissa1 + num_mantissa_2 + min_exp1 + min_exp2. Here, we can stretch it thin a bit more by using inputs in the range (-max, max)
Then, our maximum mantissa bit quota becomes  num_mantissa1 + num_mantissa_2 + min_exp1 + min_exp2 + max_exp1 + max_exp2, which should be high enough for most reasonable test cases.
If num_mantissa1 + num_mantissa_2 + min_exp1 + min_exp2 + max_exp1 + max_exp2 is still not enough , we can try a stratgey of first displacing the exponent of the accumulator upward by a large amount.
This can be done by simply adding the maximum possible product a sufficient number of times. If we have to add it to much (say more than 1024 times), then the accumulation will be exact in nearly all cases anyway, so we ignore these cases.
*/
template<typename T>
void print_vec(T* x, int n) {
    for(int i = 0; i < n; i++) {
        std::cout << (double)x[i] << " ";
    }
    std::cout << "\n";
}
template<Float Fp1, Float Fp2>
double ref_dot(const Fp1* x, const Fp2* y, int n)
{
    double s = 0.0;
    for(int i = 0; i < n; i++)
    {
        s += (double)x[i]* (double)y[i];
                if(isnan(s)) {
            std::cout << "nan encountered in one_norm\n";
            std::cout << "x[" << i << "] = " << (double)x[i] << ", y[" << i << "] = " << (double)y[i] << "\n";
            return 0.0;
        }
    }
    return s;
}
template<Float Fp>
double one_norm(const Fp* x, int n)
{
    double s = 0.0;
    for(int i = 0; i < n; i++)
    {
        s += std::abs((double)x[i]);

    }
    return s;
}

template<Float Fp1, Float Fp2>
double ref_abs_dot(const Fp1* x, const Fp2* y, int n)
{
    double s = 0.0;
    for(int i = 0; i < n; i++)
    {
        s += abs((double)x[i]* (double)y[i]);
    }
    return s;
}

template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, Int idx>
bool dot_strat_first(Vector<Fp1, idx>& x, Vector<Fp2, idx>& y)
{
    int k = (ceil(log2((double)std::numeric_limits<Fp_out>::epsilon())/log2((double)std::numeric_limits<Fp_out>::epsilon())));
    int n = x.len();
    y[0] = (Fp2) (((float) rand())/(float) RAND_MAX);
    Fp_accum s = x[0]*y[0];

}

template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, Int idx>
bool dot_case_A(Vector<Fp1, idx>& x, Vector<Fp2, idx>& y)
{

}

template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, Int idx>
bool dot_case_B()
{

}

template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, Int idx>
bool dot_case_C()
{

}


//helpeer func to generate random exp from min_exp to max_exp
int rand_int_gen(int min_exp, int max_exp)
{
    return min_exp + (rand() % (max_exp - min_exp + 1));
}

template<Float Fp1, Float Fp2, Float Fp_accum, Float Fp_out, Int idx>
bool test_mixed_dot(Vector<Fp1, idx>& x, Vector<Fp2, idx>& y)
{
    const int k = (ceil(log2((double)std::numeric_limits<double>::epsilon())/-3.0)); 
    std::cout << "k: " << k << "\n";
    const int n = x.len();
    // const int max_x_exp = std::numeric_limits<Fp1>::max_exponent;
    // const int min_x_exp = std::numeric_limits<Fp1>::min_exponent;
    // const int max_y_exp = std::numeric_limits<Fp2>::max_exponent;
    // const int min_y_exp = std::numeric_limits<Fp2>::min_exponent;
    y[0] = (Fp2) (((float) rand())/(float) RAND_MAX);

    //start by adding the max exp to x
    double s = (double)x[0]*(double)y[0];

    for(int j = 1; j < n - k + 2; j++) {
        s = s * std::pow(2.0, -1);
        if((double)x[j] == 0.0) std::cout << "some x is 0\n";
        y[j] = (Fp2) (s/(double)x[j]);
    }

    for(int j = n - k + 2; j < n; j++)
    {
        s = ref_dot(x.data, y.data, j);
        if((double)x[j] == 0.0) std::cout << "some x is 0\n";
        y[j] = (Fp2) (-s/(double)x[j]);
    }
    std::cout << "n = " << n << "\n";
    s = ref_dot(x.data, y.data, n);
    Fp_out r = (Fp_out) -s;

    std::cout << "constructred inputs, running test\n";

    auto result = dot<Fp1, Fp2, idx, Fp_out, Fp_accum, Fp_accum, 4>(x, y);
    std::cout << "result: " << (double)result << "\n";
    std::cout << "expected: " << s << "\n";

    double abs_s = abs(s);
    double S = ref_abs_dot(x.data, y.data, n);
    double U = std::max( double(2*n + 3), std::max(one_norm(y.data, n) + 2*n + 1, one_norm(x.data, n) + 2*n + 1));
    U *= (double)std::numeric_limits<Fp_accum>::denorm_min();
    U += (double)std::numeric_limits<Fp_accum>::denorm_min();
    std::cout << "U: " << U << "\n";
    std::cout << "abs_s: " << abs_s << "\n";
    std::cout << "accum epsilon: " << (double)std::numeric_limits<Fp_accum>::epsilon() << "\n";
    std::cout << "out epsilon: " << 0.25 << "\n";
    const double bound = (n + 2)*((double)std::numeric_limits<Fp_accum>::epsilon())*S + U + (double)(0.25)*abs_s;
    std::cout << "expected bound: " << bound << "\n";
    std::cout << "abs error: " << abs((double)result - s) << "\n";
    return abs((double)result - s) <= bound;

}



int main() {
    int N = 32768; // Size of the vectors, can be adjusted as needed

    // FP8 types
    using binary8p4 = P3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>;
    using binary8p3 = P3109_float<8, 3, Signedness::Signed, Inf_Behaviors::Extended>; 

    // FP16 metadata and type
    struct IsInf_f16 {
        bool operator()(uint32_t bits) const {
            uint32_t exponent = (bits >> 10) & 0x1F;
            uint32_t fraction = bits & 0x3FF;
            return (exponent == 0x1F && fraction == 0);
        }
        uint32_t infBitPattern() const { return 0x7F800000; }
        uint32_t minNegInf()     const { return 0xFF800000; }
        uint32_t minPosInf()     const { return 0x7F800000; }
    };

    struct IsNaN_f16 {
        bool operator()(uint32_t bits) const {
            uint32_t exponent = (bits >> 23) & 0xFF;
            uint32_t fraction = bits & 0x7FFFFF;
            return (exponent == 0xFF && fraction != 0);
        }
        uint32_t qNanBitPattern() const { return 0x7FC00000; }
        uint32_t sNanBitPattern() const { return 0x7FA00000; }
    };

    constexpr FloatingPointParams param_fp16(
        16, 10, 15,
        Inf_Behaviors::Extended,
        NaN_Behaviors::QuietNaN,
        Signedness::Signed,
        IsInf_f16(),
        IsNaN_f16()
    );

    using half = lo_float::Templated_Float<param_fp16>;
    using t1 = binary8p3;
    using t2 = binary8p4;

    // Allocate and initialize vectors
    t1* x_buf = (t1*) malloc(N * sizeof(t1));
    t2* y_buf = (t2*) malloc(N * sizeof(t2));
    Vector<t1, int> X(x_buf, N);
    Vector<t2, int> Y(y_buf, N);

    // Fill vectors with random data
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float a = ((float) rand()) / RAND_MAX;
        float b = (float)t1::FromRep(1);
        X[i] = (t1)(a + b);
        Y[i] = (t2)(a + b);
    }

    // ---- Test with t3 = half (FP16 accumulator) ----
    using Accumulator1 = half;
    auto start1 = std::chrono::high_resolution_clock::now();
    bool ret1 = test_mixed_dot<t1, t2, Accumulator1, t2, int>(X, Y);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

    // ---- Test with t3 = float (FP32 accumulator) ----
    using Accumulator2 = float;
    auto start2 = std::chrono::high_resolution_clock::now();
    bool ret2 = test_mixed_dot<t1, t2, Accumulator2, t2, int>(X, Y);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

    // ---- Output Results ----
    std::cout << "[FP16 Accumulator] " << (ret1 ? "Passed" : "Failed") << " in " << duration1 << " us\n";
    std::cout << "[FP32 Accumulator] " << (ret2 ? "Passed" : "Failed") << " in " << duration2 << " us\n";

    if (ret1) std::cout << "fp16 test passed!\n";
    if (ret2) std::cout << "fp32 test passed!\n";
    else std::cout << "Some tests failed!\n";

    // Cleanup
    free(x_buf);
    free(y_buf);
    return 0;
}