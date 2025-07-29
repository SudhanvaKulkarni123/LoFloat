/// @author Sudhanva Kulkarni
/// This file tests Gemm with the updated Matrix implementation.

#include "Gemms.hpp"
#include <iostream>
#include <cassert>
#include <cmath> // for std::abs

using namespace lo_float;
using namespace Lo_Gemm;

template<typename T, typename idx, Layout layout>
void printMatrix(const Matrix<T, idx, layout>& mat) {
    for (idx i = 0; i < mat.rows(); ++i) {
        for (idx j = 0; j < mat.cols(); ++j) {
            std::cout << mat(i, j) << " ";
        }
        std::cout << "\n";
    }
}

void test_gemm_basic() {
    constexpr int MR = 3, NR = 3;
    using T = float;
    constexpr Layout layout = Layout::RowMajor;

    // Dimensions
    constexpr int M = 2;
    constexpr int K = 3;
    constexpr int N = 2;

    // Allocate memory manually as required by Matrix
    T A_data[M * K];
    T B_data[K * N];
    T C_data[M * N];

    // Initialize matrices
    Matrix<T, int, layout> A(A_data, M, K, K);
    Matrix<T, int, layout> B(B_data, K, N, N);
    Matrix<T, int, layout> C(C_data, M, N, N);

    using MatrixType = Matrix<T, int, layout>;

    // Fill A (2x3): [1 2 3; 4 5 6]
    A(0, 0) = 1; A(1, 0) = 4;
    A(0, 1) = 2; A(1, 1) = 5;
    A(0, 2) = 3; A(1, 2) = 6;

    // Fill B (3x2): [7 8; 9 10; 11 12]
    B(0, 0) = 7;  B(1, 0) = 9;  B(2, 0) = 11;
    B(0, 1) = 8;  B(1, 1) = 10; B(2, 1) = 12;

    // Clear C
    for (int i = 0; i < M * N; ++i) C_data[i] = 0;

    std::cout << "initialization complee! Print matrices\n";
    printMatrix(A);
    printMatrix(B);
    std::cout << "starting gemm\n";

    // Compute C = A * B
    Gemm<MatrixType, MatrixType, MatrixType, float, float, MR, NR> gemm;
    gemm.run(C, A, B);

    std::cout << "C =\n";
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            std::cout << C(i, j) << " ";
        }
        std::cout << "\n";
    }

    // Expected:
    // C(0,0) = 1*7 + 2*9 + 3*11 = 58
    // C(0,1) = 1*8 + 2*10 + 3*12 = 64
    // C(1,0) = 4*7 + 5*9 + 6*11 = 139
    // C(1,1) = 4*8 + 5*10 + 6*12 = 154
    assert(std::abs(C(0, 0) - 58) < 1e-4);
    assert(std::abs(C(0, 1) - 64) < 1e-4);
    assert(std::abs(C(1, 0) - 139) < 1e-4);
    assert(std::abs(C(1, 1) - 154) < 1e-4);

    std::cout << "✅ Gemm test passed.\n";
}

int main() {
    test_gemm_basic();
    return 0;
}
