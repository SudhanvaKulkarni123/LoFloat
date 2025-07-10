#include "Vector.h"
#include "Dot.hpp"
#include <iostream>
#include <cassert>

using namespace Lo_Gemm;

int main() {
    /* --------------------------------------------------
     * Plain Vector – dot + copy
     * ------------------------------------------------*/
    using e8m0 = lo_float::float8_ieee_p<1>;
    double a_data[] = {1.0, 2.0, 3.0};
    double b_data[] = {4.0, 5.0, 6.0};
    Vector<double, int> va(a_data, 3);
    Vector<double, int> vb(b_data, 3);

    assert(dot(va, vb) == 32.0);

    double c_data[3]{};
    Vector<double, int> vc(c_data, 3);
    copy(va, vc);
    for (int i = 0; i < 3; ++i)
        assert(vc[i] == a_data[i]);

    /* --------------------------------------------------
     * MX‑Vector – dot, copy, slice
     * ------------------------------------------------*/
    float mant_a[] = {1.0f, 2.0f, 4.0f, 8.0f};
    float mant_b[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int   exp_a[]  = {0, 1};           // two exponent slots (r = 2)
    int   exp_b[]  = {0, 0};

    constexpr int len = 4;
    constexpr int r   = 2;

    MX_Vector<float, int, e8m0> mx_a(mant_a, exp_a, len, 2, 1, r);
    MX_Vector<float, int, e8m0> mx_b(mant_b, exp_b, len, 2, 1, r);

    mx_a.set_exp(0, 0);   // first group (indices 0,1)
    mx_a.set_exp(2, 1);   // second group (indices 2,3)
    mx_b.set_exp(0, 0);
    mx_b.set_exp(2, 0);

    // mx_a: {1,2,8,16}, mx_b: {1,1,1,1}  ==> dot = 28
    double mx_dot = dot<double>(mx_a, mx_b);
    assert(mx_dot == 28.0);

    /* copy MX_Vector -> Vector */
    float plain[len]{};
    Vector<float, int> vplain(plain, len);
    copy(mx_a, vplain);
    assert(vplain[0] == 1.0f && vplain[3] == 16.0f);

    /* slice */
    auto sub = slice(va, range<int>{1, 3}); // picks {2,3}
    assert(sub.m == 2 && sub[0] == 2.0 && sub[1] == 3.0);

    std::cout << "All tests passed!\n";
    return 0;
}
