// ═══════════════════════════════════════════════════════════════════════════
//  test_lof_gemm_gpu.cu
//
//  GPU test for lo_float::Gemm (CUTLASS + LoFMma accumulation rounding).
//  Compares against cuBLAS SGEMM (fp32) and HGEMM (fp16) as references.
//  Reports: kernel time, GFLOP/s, max element-wise backward error.
// ═══════════════════════════════════════════════════════════════════════════
#define USE_CUDA 1
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
//  Float ↔ Half conversion kernels
// ─────────────────────────────────────────────────────────────────────────

__global__ void float2half_kernel(const float* __restrict__ src,
                                  __half* __restrict__ dst, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = __float2half(src[idx]);
}

__global__ void half2float_kernel(const __half* __restrict__ src,
                                  float* __restrict__ dst, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = __half2float(src[idx]);
}

void float2half_gpu(const float* d_src, __half* d_dst, size_t n) {
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  float2half_kernel<<<blocks, threads>>>(d_src, d_dst, n);
  CUDA_CHECK(cudaGetLastError());
}

void half2float_gpu(const __half* d_src, float* d_dst, size_t n) {
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  half2float_kernel<<<blocks, threads>>>(d_src, d_dst, n);
  CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────
//  cuBLAS SGEMM wrapper for: D = alpha * A * B + beta * C
//
//  A : RowMajor  M × K, ld = K
//  B : ColMajor  K × N, ld = K
//  C/D : RowMajor  M × N, ld = N
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

  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
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

// ─────────────────────────────────────────────────────────────────────────
//  cuBLAS HGEMM wrapper for: D = alpha * A * B + beta * C
//
//  Same layout convention as SGEMM above, but with fp16 inputs/outputs
//  and fp16 accumulation.
//
//  A_h : RowMajor  M × K, ld = K  (half)
//  B_h : ColMajor  K × N, ld = K  (half)
//  D_h : RowMajor  M × N, ld = N  (half, output)
// ─────────────────────────────────────────────────────────────────────────
void cublas_hgemm_rm_cm(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha_f, float beta_f,
    const __half* d_A_h,    // RowMajor M×K, ld=K
    const __half* d_B_h,    // ColMajor K×N, ld=K
    __half*       d_D_h)    // RowMajor M×N, ld=N  (output)
{
  __half alpha_h = __float2half(alpha_f);
  __half beta_h  = __float2half(beta_f);

  // Same transpose trick: D^T = B^T * A^T viewed as column-major
  CUBLAS_CHECK(cublasHgemm(
      handle,
      CUBLAS_OP_T,    // B transpose
      CUBLAS_OP_N,    // A no-transpose (row-major = col-major transposed)
      N, M, K,
      &alpha_h,
      d_B_h, K,
      d_A_h, K,
      &beta_h,
      d_D_h, N));
}

// ─────────────────────────────────────────────────────────────────────────
//  Helper: print a submatrix from a float host array (RowMajor, ld = N)
// ─────────────────────────────────────────────────────────────────────────
void print_submatrix(const char* label, const float* h, int M, int N,
                     int rows, int cols, bool scientific = false) {
  rows = std::min(rows, M);
  cols = std::min(cols, N);
  std::cout << "\n--- " << label << " [" << rows << "x" << cols << "] ---\n";
  if (scientific)
    std::cout << std::scientific << std::setprecision(2);
  else
    std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < rows; ++i) {
    std::cout << "  ";
    for (int j = 0; j < cols; ++j)
      std::cout << std::setw(10) << h[i * N + j] << " ";
    std::cout << "\n";
  }
}

// ─────────────────────────────────────────────────────────────────────────
//  Helper: compute backward error stats between test and reference,
//          normalized by |A|*|B|.
// ─────────────────────────────────────────────────────────────────────────
struct ErrorStats {
  double max_err;
  double avg_err;
  int    count;
};

ErrorStats compute_backward_error(const float* h_test, const float* h_ref,
                                  const float* h_abs_ref, size_t n) {
  ErrorStats s{0.0, 0.0, 0};
  for (size_t i = 0; i < n; ++i) {
    double diff   = std::abs((double)h_test[i] - (double)h_ref[i]);
    double absval = (double)h_abs_ref[i];
    if (absval > 0.0) {
      double rel = diff / absval;
      s.max_err = std::max(s.max_err, rel);
      s.avg_err += rel;
      s.count++;
    }
  }
  if (s.count > 0) s.avg_err /= s.count;
  return s;
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
  float* h_D_lof   = new float[size_C];   // LoF result
  float* h_D_ref   = new float[size_C];   // SGEMM fp32 reference
  float* h_D_half  = new float[size_C];   // HGEMM fp16 reference (as float)
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
      h_B[k + n * K] = val;           // ColMajor: ld = K
      h_absB[k + n * K] = std::abs(val);
    }
  }

  std::memset(h_C,       0, size_C * sizeof(float));
  std::memset(h_D_lof,   0, size_C * sizeof(float));
  std::memset(h_D_ref,   0, size_C * sizeof(float));
  std::memset(h_D_half,  0, size_C * sizeof(float));
  std::memset(h_abs_ref, 0, size_C * sizeof(float));

  // ── Device allocations (fp32) ─────────────────────────────────────────

  float *d_A, *d_B, *d_C, *d_D, *d_D_ref, *d_D_half_f;
  float *d_absA, *d_absB, *d_abs_ref;

  CUDA_CHECK(cudaMalloc(&d_A,         size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B,         size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C,         size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D,         size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D_ref,     size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_D_half_f,  size_C * sizeof(float)));  // HGEMM result in float
  CUDA_CHECK(cudaMalloc(&d_absA,      size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_absB,      size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_abs_ref,   size_C * sizeof(float)));

  // ── Device allocations (fp16) ─────────────────────────────────────────

  __half *d_A_h, *d_B_h, *d_D_h;

  CUDA_CHECK(cudaMalloc(&d_A_h, size_A * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_B_h, size_B * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_D_h, size_C * sizeof(__half)));

  // ── Copy H → D ────────────────────────────────────────────────────────

  CUDA_CHECK(cudaMemcpy(d_A,       h_A,    size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B,       h_B,    size_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C,       h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D,       h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_ref,   h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absA,    h_absA, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absB,    h_absB, size_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_abs_ref, h_C,    size_C * sizeof(float), cudaMemcpyHostToDevice));

  // ── Convert A, B to half on device ────────────────────────────────────

  float2half_gpu(d_A, d_A_h, size_A);
  float2half_gpu(d_B, d_B_h, size_B);
  CUDA_CHECK(cudaMemset(d_D_h, 0, size_C * sizeof(__half)));
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── CUDA events for timing ────────────────────────────────────────────

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // ═════════════════════════════════════════════════════════════════════
  //  1. cuBLAS SGEMM reference (fp32)
  // ═════════════════════════════════════════════════════════════════════

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Warmup
  cublas_sgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_A, d_B, d_C, d_D_ref);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed
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
  //  2. cuBLAS HGEMM reference (fp16 inputs, fp16 accumulation)
  // ═════════════════════════════════════════════════════════════════════

  // Warmup
  CUDA_CHECK(cudaMemset(d_D_h, 0, size_C * sizeof(__half)));
  cublas_hgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_A_h, d_B_h, d_D_h);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed
  const int hgemm_reps = 20;
  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < hgemm_reps; ++i) {
    CUDA_CHECK(cudaMemset(d_D_h, 0, size_C * sizeof(__half)));
    cublas_hgemm_rm_cm(handle, M, N, K, 1.0f, 0.0f, d_A_h, d_B_h, d_D_h);
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float hgemm_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&hgemm_ms, ev_start, ev_stop));
  hgemm_ms /= hgemm_reps;

  // Convert HGEMM result back to float for comparison
  half2float_gpu(d_D_h, d_D_half_f, size_C);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ═════════════════════════════════════════════════════════════════════
  //  3. lo_float::Gemm  (CUTLASS + LoFMma)
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
    //  4. Copy results D → H
    // ═════════════════════════════════════════════════════════════════════

    CUDA_CHECK(cudaMemcpy(h_D_lof,   d_D,         size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D_ref,   d_D_ref,     size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D_half,  d_D_half_f,  size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_abs_ref, d_abs_ref,   size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // ═════════════════════════════════════════════════════════════════════
    //  5. Backward errors (all vs. fp32 SGEMM reference)
    // ═════════════════════════════════════════════════════════════════════

    ErrorStats lof_err  = compute_backward_error(h_D_lof,  h_D_ref, h_abs_ref, size_C);
    ErrorStats half_err = compute_backward_error(h_D_half, h_D_ref, h_abs_ref, size_C);

    // ═════════════════════════════════════════════════════════════════════
    //  6. Report
    // ═════════════════════════════════════════════════════════════════════

    double total_flops = 2.0 * M * N * K;
    double cublas_gflops = total_flops / (cublas_ms * 1e-3) / 1e9;
    double hgemm_gflops  = total_flops / (hgemm_ms * 1e-3) / 1e9;
    double lof_gflops    = total_flops / (lof_ms   * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "--- Performance ---\n";
    std::cout << "  cuBLAS SGEMM (fp32)   : "
              << cublas_ms << " ms   "
              << cublas_gflops << " GFLOP/s\n";
    std::cout << "  cuBLAS HGEMM (fp16)   : "
              << hgemm_ms << " ms   "
              << hgemm_gflops << " GFLOP/s\n";
    std::cout << "  lo_float::Gemm (LoF)  : "
              << lof_ms << " ms   "
              << lof_gflops << " GFLOP/s\n";
    std::cout << "  Ratio SGEMM/LoF       : "
              << cublas_ms / lof_ms << "x\n";
    std::cout << "  Ratio HGEMM/LoF       : "
              << hgemm_ms / lof_ms << "x\n";

    std::cout << "\n--- Accuracy (backward error vs. cuBLAS fp32) ---\n";
    std::cout << std::scientific << std::setprecision(6);

    std::cout << "  HGEMM (fp16):\n";
    std::cout << "    Max backward error  : " << half_err.max_err << "\n";
    std::cout << "    Avg backward error  : " << half_err.avg_err << "\n";

    std::cout << "  lo_float::Gemm (LoF, accum_mant_bits=" << accum_mant_bits << "):\n";
    std::cout << "    Max backward error  : " << lof_err.max_err << "\n";
    std::cout << "    Avg backward error  : " << lof_err.avg_err << "\n";

    if (accum_mant_bits > 0) {
      double eps_lof  = std::pow(2.0, -accum_mant_bits);
      double eps_fp16 = std::pow(2.0, -10.0);  // fp16 has 10 mantissa bits
      std::cout << "\n  eps(accum=" << accum_mant_bits << ")  : " << eps_lof << "\n";
      std::cout << "  eps(fp16=10)          : " << eps_fp16 << "\n";
      std::cout << "  LoF  max_err/(K*eps)  : "
                << lof_err.max_err / (K * eps_lof) << "\n";
      std::cout << "  fp16 max_err/(K*eps)  : "
                << half_err.max_err / (K * eps_fp16) << "\n";
    }

    // ═════════════════════════════════════════════════════════════════════
    //  7. Print submatrices
    // ═════════════════════════════════════════════════════════════════════

    const int pr = std::min(M, 6);
    const int pc = std::min(N, 6);

    print_submatrix("D_ref (cuBLAS fp32)", h_D_ref, M, N, pr, pc);
    print_submatrix("D_half (cuBLAS fp16)", h_D_half, M, N, pr, pc);
    print_submatrix("D_lof (CUTLASS LoF)", h_D_lof, M, N, pr, pc);

    // Diff tables
    {
      // |D_half - D_ref|
      float* h_diff_half = new float[size_C];
      for (size_t i = 0; i < size_C; ++i)
        h_diff_half[i] = (float)std::abs((double)h_D_half[i] - (double)h_D_ref[i]);
      print_submatrix("|D_half - D_ref|", h_diff_half, M, N, pr, pc, /*scientific=*/true);
      delete[] h_diff_half;

      // |D_lof - D_ref|
      float* h_diff_lof = new float[size_C];
      for (size_t i = 0; i < size_C; ++i)
        h_diff_lof[i] = (float)std::abs((double)h_D_lof[i] - (double)h_D_ref[i]);
      print_submatrix("|D_lof - D_ref|", h_diff_lof, M, N, pr, pc, /*scientific=*/true);
      delete[] h_diff_lof;
    }

    std::cout << "\n";
  }

cleanup:
  // ── Free ────────────────────────────────────────────────────────────

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  cudaFree(d_A);       cudaFree(d_B);       cudaFree(d_C);
  cudaFree(d_D);       cudaFree(d_D_ref);   cudaFree(d_D_half_f);
  cudaFree(d_absA);    cudaFree(d_absB);    cudaFree(d_abs_ref);
  cudaFree(d_A_h);     cudaFree(d_B_h);     cudaFree(d_D_h);

  delete[] h_A;        delete[] h_B;        delete[] h_C;
  delete[] h_D_lof;    delete[] h_D_ref;    delete[] h_D_half;
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

  // ── Test: Reduced precision accumulation (10-bit mantissa) ───────────
  test_lof_gemm(4096, 4096, 4096,
                /*accum_mant_bits=*/10,
                lo_float::Rounding_Mode::RoundTowardsZero,
                /*stochastic_rounding_bits=*/0);

  return 0;
}