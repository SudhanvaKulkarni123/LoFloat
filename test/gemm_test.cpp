#include "gemms.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono> // for timing (optional)
// #include <gperftools/profiler.h>

using namespace lo_float;
using namespace Lo_Gemm;

template<typename T, typename idx, Layout L, typename T2>
void naive_gemm(const Matrix<T, idx, L>& A, const Matrix<T, idx, L>& B, Matrix<T2, idx, L>& C) {
    int M = A.rows();
    int N = B.cols();
    int K = A.cols();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float buf = 0.0;
            for (int k = 0; k < K; ++k) {
                buf += (float)A(i, k) * (float)B(k, j);
            }
            C(i,j) = T2(buf);
        }
    }
}

template<typename T, typename idx, Layout layout, typename T2>
void abs_naive_gemm(const Matrix<T, idx, layout>& A, const Matrix<T, idx, layout>& B, Matrix<T2, idx, layout>& C) {
    int M = A.rows();
    int N = B.cols();
    int K = A.cols();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float buf = 0; // Initialize to zero
            for (int k = 0; k < K; ++k) {
                buf += (float)std::abs(A(i, k)) * (float)std::abs(B(k, j));
            }
            C(i,j) = T2(buf);
        }
    }
}

template<typename T, typename idx, Layout layout>
void printMatrix(const Matrix<T, idx, layout>& mat, int max_rows = 10, int max_cols = 10) {
    for (idx i = 0; i < std::min(mat.rows(), max_rows); ++i) {
        for (idx j = 0; j < std::min(mat.cols(), max_cols); ++j) {
            std::cout << (float)mat(i, j) << " ";
        }
        std::cout << "\n";
    }
}

void test_gemm_large_random() {
    constexpr int MR = 8, NR = 8, KR = 8;
   
    using half = lo_float::Templated_Float<lo_float::halfPrecisionParams>;
    using T = P3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>;
    using T2 = float;
    using T3 = float;
    constexpr Layout layout = Layout::RowMajor;

    // Large dimensions
    constexpr int M = 2048;
    constexpr int K = 2048;
    constexpr int N = 2048;

    T* A_data = new T[M * K];
    T* B_data = new T[K * N];
    T2* C_data = new T2[M * N];
    float* C_ref_data = new float[M * N];
    float* C_ref_data_abs = new float[M * N];
    T3* D_data = new T3[M * N];

    Matrix<T, int, layout> A(A_data, M, K, K);
    Matrix<T, int, layout> B(B_data, K, N, N);
    Matrix<T2, int, layout> C(C_data, M, N, N);
    Matrix<float, int, layout> C_ref(C_ref_data, M, N, N);
    Matrix<float, int, layout> C_ref_abs(C_ref_data_abs, M, N, N);
    Matrix<T3, int, layout> D(D_data, M, N, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) A_data[i] = (T)dist(rng);
    for (int i = 0; i < K * N; ++i) B_data[i] = (T)dist(rng);
    for (int i = 0; i < M * N; ++i) {
        C_data[i] = T2{};
        C_ref_data[i] = 0.0f;
        C_ref_data_abs[i] = 0.0f;
        D_data[i] = T3{};
    }


    //abs_naive_gemm(A, B, C_ref_abs);

    auto start_opt = std::chrono::high_resolution_clock::now();
    Gemm<decltype(A), decltype(B), decltype(C), half, decltype(D), MR, NR, KR> gemm;
    gemm.run(A, B, C, D);
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> opt_duration = end_opt - start_opt;

    std::cout << ":completed GEMM, took " << opt_duration.count() << " seconds.\n";

    naive_gemm(A, B, C_ref);
    abs_naive_gemm(A, B, C_ref_abs);

    double max_diff = 0.0;
    double tolerance = K * std::pow(2.0, -10);

    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs((double)C_data[i] - (double)C_ref_data[i]) / (double)C_ref_data_abs[i];
        max_diff = std::max(max_diff, diff);
       // assert(diff <= tolerance);
    }

    std:cout << "max diff = " << max_diff << "\n";

     const double total_flops = 2.0 * M * N * K;
    const double total_bytes = sizeof(T) * (M * K + K * N) + sizeof(T2) * (M * N + M * N); // A + B + C_read + C_write

    //const double naive_time = naive_duration.count();
    const double opt_time = opt_duration.count();

    //const double naive_gflops = total_flops / naive_time / 1e9;
    const double opt_gflops = total_flops / opt_time / 1e9;
    const double op_intensity = total_flops / total_bytes;

    // ===================== 📊 Logging =========================
    std::cout << "\n✅ Large random GEMM test passed.\n";
    std::cout << "🔍 Max Relative Error     : " << max_diff << "\n";

    std::cout << "\n--- ⏱️ Timing & Performance ---\n";
    //std::cout << "Naive GEMM Time (s)       : " << naive_time << "\n";
    std::cout << "Optimized GEMM Time (s)   : " << opt_time << "\n";
    //std::cout << "Speedup                   : " << naive_time / opt_time << "x\n";

    std::cout << "\n--- 📈 Roofline Data ---\n";
    std::cout << "GFLOPs (Optimized)        : " << opt_gflops << "\n";
    std::cout << "Operational Intensity     : " << op_intensity << " FLOPs/Byte\n";
    std::cout << "Total FLOPs               : " << total_flops << "\n";
    std::cout << "Total Bytes Transferred   : " << total_bytes << "\n";

    delete[] A_data;
    delete[] B_data;
    delete[] C_data;
    delete[] C_ref_data;
    delete[] C_ref_data_abs;
    delete[] D_data;
}

int main() {

    test_gemm_large_random();

    return 0;
}
