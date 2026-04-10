// ═══════════════════════════════════════════════════════════════════════════
//  test_lof_gemm_gpu.cu
//
//  GPU test for lo_float::Gemm (CUTLASS + LoFMma accumulation rounding).
//  Compares against cuBLAS SGEMM as full-precision reference.
//  Reports: kernel time, GFLOP/s, max element-wise backward error.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── Your project headers ─────────────────────────────────────────────────
#include "lo_float.h"
#include "layouts.h"

// ── The lo_float::Gemm wrapper ───────────────────────────────────────────
#include "cutlass_gemms.cuh"

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(x) do {                                              \
  cudaError_t e = (x);                                                  \
  if (e != cudaSuccess) {                                               \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
            __FILE__, __LINE__, cudaGetErrorString(e));                  \
    exit(1);                                                            \
  }                                                                     \
} while (0)

#define CUBLAS_CHECK(x) do {                                            \
  cublasStatus_t s = (x);                                               \
  if (s != CUBLAS_STATUS_SUCCESS) {                                     \
    fprintf(stderr, "cuBLAS error %s:%d: status %d\n",                  \
            __FILE__, __LINE__, (int)s);                                 \
    exit(1);                                                            \
  }                                                                     \
} while (0)

// ─────────────────────────────────────────────────────────────────────────
//  cuBLAS SGEMM wrapper for: D = alpha * A * B + beta * C
//
//  A : RowMajor  M × K, ld = K
//  B : ColMajor  K × N, ld = K
//  C/D : RowMajor  M × N, ld = N
//
//  cuBLAS is column-major native, so we compute:
//    D^T = B^T * A^T    (all viewed as column-major)
//
//  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//              N, M, K, &alpha, B, K, A, K, &beta, D, N)
// ─────────────────────────────────────────────────────────────────────────
void cublas_sgemm_rm_cm(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha, float beta,
    const float* d_A,     // RowMajor M×K, ld=K
    const float* d_B,     // ColMajor K×N, ld=K
    const float* d_C,     // RowMajor M×N, ld=N  (source for beta)
    float*       d_D)     // RowMajor M×N, ld=N  (output)
{
  if (d_D != d_C && beta != 0.0f) {
    CUDA_CHECK(cudaMemcpy(d_D, d_C, sizeof(float) * M * N,
                          cudaMemcpyDeviceToDevice));
  }

  CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_T,    // op on first matrix (B): transpose
      CUBLAS_OP_N,    // op on second matrix (A as col-major K×M): no transpose
      N, M, K,        // m, n, k for the column-major computation
      &alpha,
      d_B, K,         // B col-major K×N
      d_A, K,         // A row-major M×K = col-major K×M
      &beta,
      d_D, N));       // D row-major M×N = col-major N×M
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test
// ═══════════════════════════════════════════════════════════════════════════

void test_lof_gemm(int M, int N, int K,
                   int accum_mant_bits,
                   lo_float::Rounding_Mode rounding_mode,
                   int stochastic_rounding_bits)
{
  std::cout << "\n══════════════════════════════════════════════════\n";
  std::cout << "  lo_float::Gemm GPU Test\n";
  std::cout << "  M=" << M << "  N=" << N << "  K=" << K << "\n";
  std::cout << "  accum_mant_bits=" << accum_mant_bits
            << "  stochastic_bits=" << stochastic_rounding_bits << "\n";
  std::cout << "══════════════════════════════════════════════════\n\n";

  const size_t size_A = (size_t)M * K;
  const size_t size_B = (size_t)K * N;
  const size_t size_C = (size_t)M * N;

  // ── Host allocations ──────────────────────────────────────────────────

  float* h_A       = new float[size_A];
  float* h_B       = new float[size_B];
  float* h_C       = new float[size_C];
  float* h_D       = new float[size_C];
  float* h_D_ref   = new float[size_C];
  float* h_absA    = new float[size_A];
  float* h_absB    = new float[size_B];
  float* h_abs_ref = new float[size_C];

  // ── Fill with random data ─────────────────────────────────────────────

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < size_A; ++i) {
    h_A[i] = dist(rng);
    h_absA[i] = std::abs(h_A[i]);
  }

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      float val = dist(rng);
      h_B[k + n * K] = val;
      h_absB[k + n * K] = std::abs(val);
    }
  }

  std::memset(h_C,       0, size_C * sizeof(float));
  std::memset(h_D,       0, size_C * sizeof(float));
  std::memset(h_D_ref,   0, size_C * sizeof(float));
  std::memset(h_abs_ref, 0, size_C * sizeof(float));

  // ── Device allocations ────────────────────────────────────────────────

  float *d_A, *d_B, *d_C, *d_D, *d_D_ref;
  float *d_absA, *d_absB, *d_abs_ref;

  CUDA_CHECK(cudaMalloc(&d_A,       size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B,       size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C,       size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D,       size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D_ref,   size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_absA,    size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_absB,    size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_abs_ref, size_C * sizeof(float)));

  // ── Copy H → D ────────────────────────────────────────────────────────

  CUDA_CHECK(cudaMemcpy(d_A,       h_A,    size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B,       h_B,    size_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C,       h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D,       h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_ref,   h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absA,    h_absA, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absB,    h_absB, size_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_abs_ref, h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));

  // ── CUDA events for all timing ────────────────────────────────────────

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // ═════════════════════════════════════════════════════════════════════
  //  1. cuBLAS reference: full-precision SGEMM
  // ═════════════════════════════════════════════════════════════════════

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Warmup
  cublas_sgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_A, d_B, d_C, d_D_ref);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed cuBLAS
  const int cublas_reps = 20;
  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < cublas_reps; ++i)
    cublas_sgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_A, d_B, d_C, d_D_ref);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float cublas_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, ev_start, ev_stop));
  cublas_ms /= cublas_reps;

  // |A| * |B| for backward error normalization
  cublas_sgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_absA, d_absB, d_abs_ref, d_abs_ref);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ═════════════════════════════════════════════════════════════════════
  //  2. lo_float::Gemm  (CUTLASS + LoFMma)
  // ═════════════════════════════════════════════════════════════════════

  using MatA = lo_float::Matrix<float, int, lo_float::RowMajor>;
  using MatB = lo_float::Matrix<float, int, lo_float::ColMajor>;
  using MatC = lo_float::Matrix<float, int, lo_float::RowMajor>;
  using MatD = lo_float::Matrix<float, int, lo_float::RowMajor>;

  MatA A(d_A, M, K, K);
  MatB B(d_B, K, N, K);
  MatC C(d_C, M, N, N);
  MatD D(d_D, M, N, N);

  lo_float::Gemm<MatA, MatB, MatC, MatD> gemm(
      accum_mant_bits,
      rounding_mode,
      stochastic_rounding_bits);

  // ── Warmup + error check ──────────────────────────────────────────────

  {
    auto status = gemm(1.0f, 0.0f, A, B, C, D);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      std::cerr << "ERROR: Kernel launch failed: "
                << cudaGetErrorString(launch_err) << "\n";
      goto cleanup;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t exec_err = cudaGetLastError();
    if (exec_err != cudaSuccess) {
      std::cerr << "ERROR: Kernel execution failed: "
                << cudaGetErrorString(exec_err) << "\n";
      goto cleanup;
    }

    if (status != decltype(gemm)::Status::kSuccess) {
      std::cerr << "ERROR: lo_float::Gemm returned status "
                << static_cast<int>(status) << "\n";
      goto cleanup;
    }

    std::cout << "  Warmup OK — kernel launched successfully.\n\n";
  }

  // ── Timed LoF GEMM runs ──────────────────────────────────────────────

  {
    const int lof_reps = 20;

    // Reset D
    CUDA_CHECK(cudaMemset(d_D, 0, size_C * sizeof(float)));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < lof_reps; ++i)
      gemm(1.0f, 0.0f, A, B, C, D);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float lof_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&lof_ms, ev_start, ev_stop));
    lof_ms /= lof_reps;

    // ═════════════════════════════════════════════════════════════════════
    //  3. Copy results D → H
    // ═════════════════════════════════════════════════════════════════════

    CUDA_CHECK(cudaMemcpy(h_D,       d_D,       size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D_ref,   d_D_ref,   size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_abs_ref, d_abs_ref, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // ═════════════════════════════════════════════════════════════════════
    //  4. Compute max element-wise backward error
    //
    //     err_i = |D_lof[i] - D_ref[i]| / (|A|*|B|)[i]
    // ═════════════════════════════════════════════════════════════════════

    double max_err   = 0.0;
    double sum_err   = 0.0;
    int    err_count = 0;

    for (size_t i = 0; i < size_C; ++i) {
      double diff   = std::abs((double)h_D[i] - (double)h_D_ref[i]);
      double absval = (double)h_abs_ref[i];

      if (absval > 0.0) {
        double rel = diff / absval;
        max_err = std::max(max_err, rel);
        sum_err += rel;
        err_count++;
      }
    }
    double avg_err = (err_count > 0) ? sum_err / err_count : 0.0;

    // ═════════════════════════════════════════════════════════════════════
    //  5. Report
    // ═════════════════════════════════════════════════════════════════════

    double total_flops = 2.0 * M * N * K;
    double cublas_gflops = total_flops / (cublas_ms * 1e-3) / 1e9;
    double lof_gflops    = total_flops / (lof_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "--- Performance ---\n";
    std::cout << "  cuBLAS (fp32 ref)     : "
              << cublas_ms << " ms   "
              << cublas_gflops << " GFLOP/s\n";
    std::cout << "  lo_float::Gemm (LoF)  : "
              << lof_ms << " ms   "
              << lof_gflops << " GFLOP/s\n";
    std::cout << "  Ratio (cuBLAS/LoF)    : "
              << cublas_ms / lof_ms << "x\n";

    std::cout << "\n--- Accuracy (vs. cuBLAS fp32) ---\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Max backward error    : " << max_err << "\n";
    std::cout << "  Avg backward error    : " << avg_err << "\n";

    if (accum_mant_bits > 0) {
      double expected_eps = std::pow(2.0, -accum_mant_bits);
      std::cout << "  eps(accum_mant_bits=" << accum_mant_bits << ") : "
                << expected_eps << "\n";
      std::cout << "  Max err / (K * eps)   : "
                << max_err / (K * expected_eps) << "\n";
    }

    // ═════════════════════════════════════════════════════════════════════
    //  6. Print a small submatrix from top-left corner
    // ═════════════════════════════════════════════════════════════════════

    {
      const int print_rows = std::min(M, 6);
      const int print_cols = std::min(N, 6);

      std::cout << std::fixed << std::setprecision(4);

      std::cout << "\n--- D_ref (cuBLAS fp32) [" << print_rows << "x" << print_cols << "] ---\n";
      for (int i = 0; i < print_rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < print_cols; ++j) {
          std::cout << std::setw(10) << h_D_ref[i * N + j] << " ";
        }
        std::cout << "\n";
      }

      std::cout << "\n--- D_lof (CUTLASS LoF) [" << print_rows << "x" << print_cols << "] ---\n";
      for (int i = 0; i < print_rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < print_cols; ++j) {
          std::cout << std::setw(10) << h_D[i * N + j] << " ";
        }
        std::cout << "\n";
      }

      std::cout << "\n--- |D_lof - D_ref| [" << print_rows << "x" << print_cols << "] ---\n";
      std::cout << std::scientific << std::setprecision(2);
      for (int i = 0; i < print_rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < print_cols; ++j) {
          double diff = std::abs((double)h_D[i * N + j] - (double)h_D_ref[i * N + j]);
          std::cout << std::setw(10) << diff << " ";
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";
  }

cleanup:
  // ── Free ────────────────────────────────────────────────────────────

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  cudaFree(d_A);       cudaFree(d_B);       cudaFree(d_C);
  cudaFree(d_D);       cudaFree(d_D_ref);
  cudaFree(d_absA);    cudaFree(d_absB);    cudaFree(d_abs_ref);

  delete[] h_A;        delete[] h_B;        delete[] h_C;
  delete[] h_D;        delete[] h_D_ref;
  delete[] h_absA;     delete[] h_absB;     delete[] h_abs_ref;
}

// ═══════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════

int main() {

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name
            << " (SM " << prop.major << prop.minor
            << ", " << prop.sharedMemPerMultiprocessor / 1024 << " KB smem/SM)\n";

  // ── Test 2: Reduced precision accumulation (10-bit mantissa) ─────────
  test_lof_gemm(4096,4096,4096,
                /*accum_mant_bits=*/10,
                lo_float::Rounding_Mode::RoundToNearestEven,
                /*stochastic_rounding_bits=*/0);

  // // ── Test 3: Stochastic rounding ─────────────────────────────────────
  // test_lof_gemm(2048, 2048, 2048,
  //               /*accum_mant_bits=*/10,
  //               lo_float::Rounding_Mode::Stochastic,
  //               /*stochastic_rounding_bits=*/8);

  return 0;
}