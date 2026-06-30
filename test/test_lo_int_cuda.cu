///@author loft loop — GPU test for i_n<len, Signedness> (T5, saturating)
//
// Mirrors test_lo_int.cpp on the device: i_n arithmetic is pure integer logic,
// so device results must match the host saturating oracle bit-exact (and hence
// the CPU i_n already validated against that same oracle). Compiled with
// --expt-relaxed-constexpr so i_n's constexpr (host) operators are callable in
// device code (i_n is not LOFLOAT_HOST_DEVICE-annotated).
//
// 0 = pass; nonzero prints the first mismatches and exits nonzero.
#include <cstdint>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "lo_int.h"

using namespace lo_float;

static int g_errors = 0;
static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(e) << "\n";
        std::exit(2);
    }
}

template <int len, Signedness S>
struct ref_range {
    static constexpr long long LO = (S == Signedness::Signed) ? -(1LL << (len - 1)) : 0LL;
    static constexpr long long HI = (S == Signedness::Signed) ? ((1LL << (len - 1)) - 1)
                                                              : ((1LL << len) - 1);
    static constexpr long long sat(long long x) { return x < LO ? LO : (x > HI ? HI : x); }
};

// Pairwise arithmetic over every (a,b) in the format's range.
template <int len, Signedness S>
__global__ void k_arith(long long lo, int m,
                        long long* add, long long* sub, long long* mul, long long* dv) {
    using T = i_n<len, S>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * m) {
        long long a = lo + idx / m, b = lo + idx % m;
        T A(a), B(b);
        add[idx] = (long long)(int)(A + B);
        sub[idx] = (long long)(int)(A - B);
        mul[idx] = (long long)(int)(A * B);
        dv[idx]  = (b != 0) ? (long long)(int)(A / B) : 0;
    }
}

template <int len, Signedness S>
void test_format(const char* name) {
    using R = ref_range<len, S>;
    const long long LO = R::LO, HI = R::HI;
    const int m = int(HI - LO + 1);
    const int mm = m * m;

    long long *d_add, *d_sub, *d_mul, *d_dv;
    cudaCheck(cudaMalloc(&d_add, mm * sizeof(long long)), "malloc add");
    cudaCheck(cudaMalloc(&d_sub, mm * sizeof(long long)), "malloc sub");
    cudaCheck(cudaMalloc(&d_mul, mm * sizeof(long long)), "malloc mul");
    cudaCheck(cudaMalloc(&d_dv,  mm * sizeof(long long)), "malloc dv");

    k_arith<len, S><<<(mm + 255) / 256, 256>>>(LO, m, d_add, d_sub, d_mul, d_dv);
    cudaCheck(cudaGetLastError(), "launch");
    cudaCheck(cudaDeviceSynchronize(), "sync");

    std::vector<long long> add(mm), sub(mm), mul(mm), dv(mm);
    cudaCheck(cudaMemcpy(add.data(), d_add, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy add");
    cudaCheck(cudaMemcpy(sub.data(), d_sub, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy sub");
    cudaCheck(cudaMemcpy(mul.data(), d_mul, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy mul");
    cudaCheck(cudaMemcpy(dv.data(),  d_dv,  mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy dv");

    int reported = 0, local = 0;
    for (int ia = 0; ia < m; ++ia)
        for (int ib = 0; ib < m; ++ib) {
            long long a = LO + ia, b = LO + ib;
            int idx = ia * m + ib;
            auto bad = [&](const char* op, long long got, long long exp) {
                ++local;
                if (reported++ < 5)
                    std::cerr << "FAIL " << name << " " << a << op << b
                              << " gpu=" << got << " exp=" << exp << "\n";
            };
            if (add[idx] != R::sat(a + b)) bad("+", add[idx], R::sat(a + b));
            if (sub[idx] != R::sat(a - b)) bad("-", sub[idx], R::sat(a - b));
            if (mul[idx] != R::sat(a * b)) bad("*", mul[idx], R::sat(a * b));
            if (b != 0 && dv[idx] != R::sat(a / b)) bad("/", dv[idx], R::sat(a / b));
        }

    cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_dv);
    g_errors += local;
    std::cout << name << " : " << (local == 0 ? "pass" : "FAIL") << "\n";
}

int main() {
    test_format<4, Signedness::Signed>("INT4  signed ");
    test_format<4, Signedness::Unsigned>("INT4  unsigned");
    test_format<6, Signedness::Signed>("INT6  signed ");
    test_format<6, Signedness::Unsigned>("INT6  unsigned");
    test_format<8, Signedness::Signed>("INT8  signed ");
    test_format<8, Signedness::Unsigned>("INT8  unsigned");
    std::cout << "\nlo_int GPU (T5) saturating test: " << (g_errors == 0 ? "PASS" : "FAIL") << "\n";
    return g_errors == 0 ? 0 : 1;
}
