// ═══════════════════════════════════════════════════════════════════════════
//  test_conv2d_cuda.cu
//
//  GPU correctness test for lo_float::Conv2d (CUTLASS implicit-GEMM Conv2d
//  fprop + LoFMma accumulation rounding). NHWC activation, KRSC filter,
//  NHWC output, cross-correlation mode (matches PyTorch's Conv2d).
//
//  Oracle: a host double-precision direct convolution (loop/NUMERICAL_TESTING.md
//  §1 — the reference is double, not the DUT). Error bound follows the same
//  family as test/dot_test.cpp / test/gemm_test.cu: for a reduction length
//  K = C*R*S and accumulation mantissa bits `accum_mant_bits`,
//
//      |D_lof - D_ref| <= SLACK * K * eps_lof * conv(|A|, |B|)
//
//  where conv(|A|,|B|) is the same convolution applied to the elementwise
//  absolute values of A and B (the GEMM analogue is K*eps*|A|*|B|).
//
//  Returns 0 on success (all configs/modes pass), nonzero error count
//  otherwise, per NUMERICAL_TESTING.md §11.
// ═══════════════════════════════════════════════════════════════════════════
#define USE_CUDA 1
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

#include "lo_float.h"
#include "cutlass_conv.cuh"

#define CUDA_CHECK(x) do {                                                   \
  cudaError_t e = (x);                                                       \
  if (e != cudaSuccess) {                                                    \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                                \
            __FILE__, __LINE__, cudaGetErrorString(e));                      \
    exit(1);                                                                 \
  }                                                                          \
} while (0)

// ─────────────────────────────────────────────────────────────────────────
//  Problem description (NHWC activation, KRSC filter, NHWC output).
// ─────────────────────────────────────────────────────────────────────────
struct ConvCfg {
  int N, H, W, C, K, R, S;
  int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;

  int P() const {
    return (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
  }
  int Q() const {
    return (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
  }
};

// Host double-precision direct convolution (cross-correlation), with an
// explicit zero-padding mask. Used both for the signed reference D_ref and
// for the |A|*|B|-convolution error-bound denominator (call with abs inputs).
static void host_conv2d_ref(
    const ConvCfg& cfg,
    const std::vector<double>& A,  // NHWC: A[((n*H+h)*W+w)*C+c]
    const std::vector<double>& B,  // KRSC: B[((k*R+r)*S+s)*C+c]
    std::vector<double>& D)        // NPQK: D[((n*P+p)*Q+q)*K+k]
{
  const int P = cfg.P(), Q = cfg.Q();
  D.assign((size_t)cfg.N * P * Q * cfg.K, 0.0);

  for (int n = 0; n < cfg.N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        for (int k = 0; k < cfg.K; ++k) {
          double acc = 0.0;
          for (int r = 0; r < cfg.R; ++r) {
            int h = p * cfg.stride_h - cfg.pad_h + r * cfg.dilation_h;
            if (h < 0 || h >= cfg.H) continue;
            for (int s = 0; s < cfg.S; ++s) {
              int w = q * cfg.stride_w - cfg.pad_w + s * cfg.dilation_w;
              if (w < 0 || w >= cfg.W) continue;
              for (int c = 0; c < cfg.C; ++c) {
                double a = A[((size_t)(n * cfg.H + h) * cfg.W + w) * cfg.C + c];
                double b = B[((size_t)(k * cfg.R + r) * cfg.S + s) * cfg.C + c];
                acc += a * b;
              }
            }
          }
          D[((size_t)(n * P + p) * Q + q) * cfg.K + k] = acc;
        }
      }
    }
  }
}

struct TestResult {
  int errors = 0;
  double max_ratio = 0.0;  // max_i |D_lof - D_ref| / (SLACK * K * eps * |conv|(i))
};

// Runs one (cfg, accum_mant_bits, rounding_mode) case on the GPU and checks
// the K*eps*conv(|A|,|B|) backward-error bound elementwise.
static TestResult run_case(
    const ConvCfg& cfg,
    int accum_mant_bits,
    lo_float::Rounding_Mode rounding_mode,
    int stochastic_rounding_bits,
    unsigned seed)
{
  TestResult result;
  const int P = cfg.P(), Q = cfg.Q();
  const size_t size_A = (size_t)cfg.N * cfg.H * cfg.W * cfg.C;
  const size_t size_B = (size_t)cfg.K * cfg.R * cfg.S * cfg.C;
  const size_t size_D = (size_t)cfg.N * P * Q * cfg.K;
  const int K_reduce = cfg.C * cfg.R * cfg.S;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> hA(size_A), hB(size_B);
  std::vector<double> hA_d(size_A), hB_d(size_A ? size_A : 0), hA_abs(size_A), hB_abs(size_B);
  hB_d.resize(size_B);
  for (size_t i = 0; i < size_A; ++i) { hA[i] = dist(rng); hA_d[i] = (double)hA[i]; hA_abs[i] = std::abs(hA_d[i]); }
  for (size_t i = 0; i < size_B; ++i) { hB[i] = dist(rng); hB_d[i] = (double)hB[i]; hB_abs[i] = std::abs(hB_d[i]); }

  // ── Host references (double precision) ──────────────────────────────
  std::vector<double> hD_ref, hD_absref;
  host_conv2d_ref(cfg, hA_d, hB_d, hD_ref);
  host_conv2d_ref(cfg, hA_abs, hB_abs, hD_absref);

  // ── Device buffers ────────────────────────────────────────────────────
  float *dA, *dB, *dC, *dD;
  CUDA_CHECK(cudaMalloc(&dA, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, size_D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, size_D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, size_D * sizeof(float)));
  CUDA_CHECK(cudaMemset(dD, 0, size_D * sizeof(float)));

  cutlass::conv::Conv2dProblemSize problem_size(
      cfg.N, cfg.H, cfg.W, cfg.C,
      cfg.K, cfg.R, cfg.S, P, Q,
      cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w, cfg.dilation_h, cfg.dilation_w,
      cutlass::conv::Mode::kCrossCorrelation);

  lo_float::Conv2d<> conv(accum_mant_bits, lo_float::ProjSpec{rounding_mode, lo_float::Saturation_Mode::OvfInf, stochastic_rounding_bits});

  auto status = conv(problem_size, dA, dB, dC, dD, 1.0f, 0.0f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (status != decltype(conv)::Status::kSuccess) {
    std::cerr << "ERROR: lo_float::Conv2d returned status "
              << static_cast<int>(status) << " for N=" << cfg.N << " H=" << cfg.H
              << " W=" << cfg.W << " C=" << cfg.C << " K=" << cfg.K
              << " R=" << cfg.R << " S=" << cfg.S << "\n";
    result.errors++;
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    return result;
  }

  std::vector<float> hD_lof(size_D);
  CUDA_CHECK(cudaMemcpy(hD_lof.data(), dD, size_D * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);

  // ── Check the K*eps*conv(|A|,|B|) backward-error bound ───────────────
  const double eps_lof = (accum_mant_bits > 0) ? std::pow(2.0, -accum_mant_bits) : 0.0;
  // Small slack for: (a) the epilogue's own fp32 alpha*acc+beta*c rounding,
  // (b) accum_mant_bits==0 (rounds to power-of-two -> needs a multiplicative,
  // not additive, allowance), (c) stochastic rounding's extra +-1ulp draw.
  const double SLACK = 4.0;

  for (size_t i = 0; i < size_D; ++i) {
    double diff = std::abs((double)hD_lof[i] - hD_ref[i]);
    double bound = SLACK * std::max(1, K_reduce) * std::max(eps_lof, 1e-7) * hD_absref[i]
                 + 1e-6;  // tiny absolute floor for near-zero outputs
    if (diff > bound) {
      if (result.errors < 5) {
        std::cerr << "ERROR: element " << i << " |diff|=" << diff
                  << " > bound=" << bound << " (D_lof=" << hD_lof[i]
                  << " D_ref=" << hD_ref[i] << ")\n";
      }
      result.errors++;
    }
    if (bound > 0) result.max_ratio = std::max(result.max_ratio, diff / bound);
  }

  return result;
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name << " (SM " << prop.major << prop.minor << ")\n";

  int total_errors = 0;

  struct Case {
    const char* name;
    ConvCfg cfg;
    int accum_mant_bits;
    lo_float::Rounding_Mode mode;
  };

  std::vector<Case> cases = {
    {"3x3 stride1 pad1 RNE",
     {2, 10, 10, 8, 12, 3, 3, /*pad*/1, 1, /*stride*/1, 1, /*dilation*/1, 1},
     10, lo_float::Rounding_Mode::RoundToNearestEven},
    {"3x3 stride2 pad1 RNE (downsample, sub-tile M/N)",
     {1, 9, 11, 6, 10, 3, 3, 1, 1, 2, 2, 1, 1},
     10, lo_float::Rounding_Mode::RoundToNearestEven},
    {"3x3 dilation2 pad2 stride1 RoundTowardsZero",
     {1, 12, 12, 4, 8, 3, 3, 2, 2, 1, 1, 2, 2},
     8, lo_float::Rounding_Mode::RoundTowardsZero},
    {"1x1 stride1 pad0 RoundToNearestOdd (pointwise conv == GEMM)",
     {2, 7, 7, 16, 20, 1, 1, 0, 0, 1, 1, 1, 1},
     12, lo_float::Rounding_Mode::RoundToNearestOdd},
    {"full precision (accum_mant_bits=23) tight bound",
     {1, 8, 8, 4, 6, 3, 3, 1, 1, 1, 1, 1, 1},
     23, lo_float::Rounding_Mode::RoundToNearestEven},
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& tc = cases[i];
    TestResult r = run_case(tc.cfg, tc.accum_mant_bits, tc.mode,
                            /*stochastic_rounding_bits=*/0, /*seed=*/1000u + (unsigned)i);
    std::cout << "[" << (r.errors == 0 ? "PASS" : "FAIL") << "] " << tc.name
              << "  K=" << (tc.cfg.C * tc.cfg.R * tc.cfg.S)
              << "  accum_mant_bits=" << tc.accum_mant_bits
              << "  max_ratio=" << std::fixed << std::setprecision(4) << r.max_ratio
              << "  errors=" << r.errors << "\n";
    total_errors += r.errors;
  }

  // ── groups != 1 must be rejected, not silently mis-computed ─────────
  {
    cutlass::conv::Conv2dProblemSize gp(
        1, 4, 4, 4, 4, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1,
        cutlass::conv::Mode::kCrossCorrelation, /*split_k_slices=*/1, /*groups=*/2);
    lo_float::Conv2d<> conv(23, lo_float::ProjSpec{lo_float::Rounding_Mode::RoundToNearestEven, lo_float::Saturation_Mode::OvfInf, 0});
    float *dA, *dB, *dC, *dD;
    CUDA_CHECK(cudaMalloc(&dA, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, 64 * sizeof(float)));
    auto status = conv(gp, dA, dB, dC, dD);
    bool ok = (status == decltype(conv)::Status::kErrorNotSupported);
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] groups=2 correctly rejected\n";
    if (!ok) total_errors++;
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  }

  std::cout << "\nTotal errors: " << total_errors << "\n";
  return total_errors == 0 ? 0 : 1;
}
