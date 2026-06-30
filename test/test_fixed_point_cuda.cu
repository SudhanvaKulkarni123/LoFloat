///@author loft loop — GPU test for fixed_point<IntBits,FracBits,Sign>
//
// Per loop/NUMERICAL_TESTING.md §8: a deterministic op must produce the SAME
// result on CPU and GPU.  fixed_point's arithmetic is pure integer + IEEE double,
// so device and host must agree BIT-EXACT.  The host side here is the same
// fixed_point already validated against an independent int64/llround oracle in
// test_fixed_point.cpp, so device==host bit-exact closes the chain.
//
//  - from_double over a dense sweep incl. half-ULP ties and overflow;
//  - exhaustive pairwise add/sub/mul/div over every raw pattern (W<=8 formats);
//  - field decomposition int_part()/frac_part().
// 0 = pass (quiet); nonzero prints the first mismatches and exits nonzero.

#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include "lo_int.h"

using namespace lo_float;

static int g_errors = 0;
#define CK(cond, ...) do { if(!(cond)){ ++g_errors; std::cerr << "FAIL " << __VA_ARGS__ << "\n"; } } while(0)

static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(e) << "\n";
        std::exit(2);
    }
}

// ---- device kernels (templated on the fixed_point type) ----
template <class FP>
__global__ void k_from_double(const double* x, long long* out_raw, double* out_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        FP f(x[i]);
        out_raw[i] = (long long)f.raw();
        out_val[i] = f.to_double();
    }
}

template <class FP>
__global__ void k_binary(const long long* raws, int m,
                         long long* add, long long* sub, long long* mul, long long* dv,
                         long long* ipart, long long* fpart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * m) {
        int ia = idx / m, ib = idx % m;
        using St = decltype(FP{}.raw());
        FP A = FP::FromRaw((St)raws[ia]);
        FP B = FP::FromRaw((St)raws[ib]);
        add[idx] = (long long)(A + B).raw();
        sub[idx] = (long long)(A - B).raw();
        mul[idx] = (long long)(A * B).raw();
        dv[idx]  = (long long)(A / B).raw();
        if (ib == 0) { ipart[ia] = (long long)A.int_part(); fpart[ia] = (long long)A.frac_part(); }
    }
}

template <int I, int F, Signedness S>
void test_format(const char* name) {
    using FP = fixed_point<I, F, S>;
    constexpr int  W  = I + F;
    constexpr bool sg = (S == Signedness::Signed);
    const long long HI = sg ? ((1LL << (W - 1)) - 1) : ((1LL << W) - 1);
    const long long LO = sg ? -(1LL << (W - 1)) : 0LL;
    const long long ONE = 1LL << F;

    // ---------- from_double sweep ----------
    std::vector<double> xs;
    double step = 1.0 / double(ONE);
    for (double x = double(LO) / ONE - 2.0; x <= double(HI) / ONE + 2.0; x += step * 0.25) xs.push_back(x);
    for (long long k = LO; k < HI; ++k) xs.push_back((double(k) + 0.5) / double(ONE)); // ties
    xs.push_back(1e30); xs.push_back(-1e30);                                            // overflow
    int n = (int)xs.size();

    double*    d_x;   long long* d_raw; double* d_val;
    cudaCheck(cudaMalloc(&d_x,   n * sizeof(double)),    "malloc x");
    cudaCheck(cudaMalloc(&d_raw, n * sizeof(long long)), "malloc raw");
    cudaCheck(cudaMalloc(&d_val, n * sizeof(double)),    "malloc val");
    cudaCheck(cudaMemcpy(d_x, xs.data(), n * sizeof(double), cudaMemcpyHostToDevice), "cpy x");
    k_from_double<FP><<<(n + 255) / 256, 256>>>(d_x, d_raw, d_val, n);
    cudaCheck(cudaGetLastError(), "launch from_double");
    cudaCheck(cudaDeviceSynchronize(), "sync from_double");
    std::vector<long long> h_raw(n); std::vector<double> h_val(n);
    cudaCheck(cudaMemcpy(h_raw.data(), d_raw, n * sizeof(long long), cudaMemcpyDeviceToHost), "cpy raw");
    cudaCheck(cudaMemcpy(h_val.data(), d_val, n * sizeof(double),    cudaMemcpyDeviceToHost), "cpy val");

    int reported = 0;
    for (int i = 0; i < n; ++i) {
        FP host(xs[i]);
        bool ok = ((long long)host.raw() == h_raw[i]) && (host.to_double() == h_val[i]);
        if (!ok && reported++ < 5)
            CK(false, name << " from_double x=" << xs[i] << " gpu_raw=" << h_raw[i]
                           << " cpu_raw=" << (long long)host.raw());
        else if (!ok) ++g_errors;
    }
    cudaFree(d_x); cudaFree(d_raw); cudaFree(d_val);

    // ---------- exhaustive pairwise arithmetic + decomposition ----------
    std::vector<long long> raws;
    for (long long r = LO; r <= HI; ++r) raws.push_back(r);
    int m = (int)raws.size();
    int mm = m * m;

    long long *d_raws, *d_add, *d_sub, *d_mul, *d_div, *d_ip, *d_fp;
    cudaCheck(cudaMalloc(&d_raws, m  * sizeof(long long)), "malloc raws");
    cudaCheck(cudaMalloc(&d_add,  mm * sizeof(long long)), "malloc add");
    cudaCheck(cudaMalloc(&d_sub,  mm * sizeof(long long)), "malloc sub");
    cudaCheck(cudaMalloc(&d_mul,  mm * sizeof(long long)), "malloc mul");
    cudaCheck(cudaMalloc(&d_div,  mm * sizeof(long long)), "malloc div");
    cudaCheck(cudaMalloc(&d_ip,   m  * sizeof(long long)), "malloc ip");
    cudaCheck(cudaMalloc(&d_fp,   m  * sizeof(long long)), "malloc fp");
    cudaCheck(cudaMemcpy(d_raws, raws.data(), m * sizeof(long long), cudaMemcpyHostToDevice), "cpy raws");
    k_binary<FP><<<(mm + 255) / 256, 256>>>(d_raws, m, d_add, d_sub, d_mul, d_div, d_ip, d_fp);
    cudaCheck(cudaGetLastError(), "launch binary");
    cudaCheck(cudaDeviceSynchronize(), "sync binary");

    std::vector<long long> add(mm), sub(mm), mul(mm), dv(mm), ip(m), fp(m);
    cudaCheck(cudaMemcpy(add.data(), d_add, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy add");
    cudaCheck(cudaMemcpy(sub.data(), d_sub, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy sub");
    cudaCheck(cudaMemcpy(mul.data(), d_mul, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy mul");
    cudaCheck(cudaMemcpy(dv.data(),  d_div, mm * sizeof(long long), cudaMemcpyDeviceToHost), "cpy div");
    cudaCheck(cudaMemcpy(ip.data(),  d_ip,  m  * sizeof(long long), cudaMemcpyDeviceToHost), "cpy ip");
    cudaCheck(cudaMemcpy(fp.data(),  d_fp,  m  * sizeof(long long), cudaMemcpyDeviceToHost), "cpy fp");

    reported = 0;
    for (int ia = 0; ia < m; ++ia) {
        using St = decltype(FP{}.raw());
        FP A = FP::FromRaw((St)raws[ia]);
        if ((long long)A.int_part()  != ip[ia]) { if (reported++ < 5) CK(false, name << " int_part raw=" << raws[ia]); else ++g_errors; }
        if ((long long)A.frac_part() != fp[ia]) { if (reported++ < 5) CK(false, name << " frac_part raw=" << raws[ia]); else ++g_errors; }
        for (int ib = 0; ib < m; ++ib) {
            FP B = FP::FromRaw((St)raws[ib]);
            int idx = ia * m + ib;
            long long e_add = (long long)(A + B).raw();
            long long e_sub = (long long)(A - B).raw();
            long long e_mul = (long long)(A * B).raw();
            long long e_div = (long long)(A / B).raw();
            if (e_add != add[idx] || e_sub != sub[idx] || e_mul != mul[idx] || e_div != dv[idx]) {
                if (reported++ < 5)
                    CK(false, name << " arith a=" << raws[ia] << " b=" << raws[ib]
                                   << " gpu(add,sub,mul,div)=" << add[idx] << "," << sub[idx] << "," << mul[idx] << "," << dv[idx]
                                   << " cpu=" << e_add << "," << e_sub << "," << e_mul << "," << e_div);
                else ++g_errors;
            }
        }
    }
    cudaFree(d_raws); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div); cudaFree(d_ip); cudaFree(d_fp);
}

int main() {
    test_format<4, 4, Signedness::Signed>("Q4.4s");
    test_format<4, 4, Signedness::Unsigned>("Q4.4u");
    test_format<2, 2, Signedness::Signed>("Q2.2s");
    test_format<5, 3, Signedness::Unsigned>("Q5.3u");
    test_format<1, 7, Signedness::Signed>("Q1.7s");

    if (g_errors == 0) { std::cout << "test_fixed_point_cuda: ALL PASSED (CPU==GPU bit-exact)\n"; return 0; }
    std::cerr << "test_fixed_point_cuda: " << g_errors << " FAILURES\n";
    return 1;
}
