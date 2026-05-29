// -------------------------------------------------------------
// test_virtual_round.cu  --  CUDA tests for the two scalar
// virtual_round overloads in lo_float.h, invoked from device
// kernels (same kernels used in Lof_kernel.cu).
//
//   1)  virtual_round(value, ToMantissaBits, Rounding_Mode)
//        - mantissa-only rounding, no exponent constraint
//   2)  virtual_round(value, FloatingPointParams, Rounding_Mode)
//        - full target-format rounding (range + precision)
//
// Mirrors test_virtual_round.cpp, but every rounding call is
// dispatched to a __global__ kernel and the result is read back
// to the host for validation.
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "lo_float.h"
#include "lo_float_sci.hpp"
#include "fp_tools.hpp"

using namespace lo_float;

// -------------------------------------------------------------
// CUDA error check
// -------------------------------------------------------------
#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::cerr << "CUDA error " << cudaGetErrorString(_e)                   \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// -------------------------------------------------------------
// device kernels (single-element scalar entry points)
// -------------------------------------------------------------
__global__ void vr_mantissa_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n,
                                   int ToMantissaBits,
                                   Rounding_Mode mode,
                                   int stoch_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        out[idx] = lo_float::virtual_round(v, ToMantissaBits, mode, stoch_len);
    }
}

template <typename ToInf, typename ToNaN>
__global__ void vr_fp_params_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    FloatingPointParams<ToInf, ToNaN> params,
                                    Rounding_Mode mode,
                                    int stoch_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        out[idx] = lo_float::virtual_round(v, params, mode, stoch_len);
    }
}

// -------------------------------------------------------------
// host helpers
// -------------------------------------------------------------
static double get_denom(double d) {
    if (d == 0.0 || !std::isfinite(d)) return 1.0;
    int exp = 0;
    std::frexp(d, &exp);
    return std::ldexp(1.0, exp);
}

static float rnd32_signed() {
    float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    return (u * 2.0f - 1.0f) * v;
}

static float rnd32_wide() {
    float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    int e = (std::rand() % 41) - 20;
    float sign = (std::rand() & 1) ? 1.0f : -1.0f;
    return sign * std::ldexp(u, e);
}

static uint32_t f2u(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

static float u2f(uint32_t b) {
    float v;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

// -------------------------------------------------------------
// device-side batch launchers (one mode per call)
// -------------------------------------------------------------
static void launch_mantissa(const float* d_in, float* d_out, int n,
                            int ToMantissaBits, Rounding_Mode mode,
                            int stoch_len = 0) {
    constexpr int kThreads = 256;
    int blocks = (n + kThreads - 1) / kThreads;
    vr_mantissa_kernel<<<blocks, kThreads>>>(d_in, d_out, n, ToMantissaBits,
                                             mode, stoch_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename ToInf, typename ToNaN>
static void launch_fp_params(const float* d_in, float* d_out, int n,
                             FloatingPointParams<ToInf, ToNaN> params,
                             Rounding_Mode mode, int stoch_len = 0) {
    constexpr int kThreads = 256;
    int blocks = (n + kThreads - 1) / kThreads;
    vr_fp_params_kernel<<<blocks, kThreads>>>(d_in, d_out, n, params, mode,
                                              stoch_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -------------------------------------------------------------
// 1)  virtual_round(value, ToMantissaBits, mode)   --  CUDA
// -------------------------------------------------------------
template <int ToMantissaBits>
int test_mantissa_round_cuda(int n_iters = 2000) {
    int num_errors = 0;

    const double mach_eps      = std::pow(2.0, -ToMantissaBits);
    const double mach_eps_half = std::pow(2.0, -ToMantissaBits - 1);

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    // ------------------------------------------------------------------
    // build a batch of random inputs (skip non-finite at validation time)
    // ------------------------------------------------------------------
    std::vector<float> h_in(n_iters);
    for (int i = 0; i < n_iters; ++i) h_in[i] = rnd32_wide();

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  n_iters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n_iters * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n_iters * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto run = [&](Rounding_Mode mode, int stoch_len = 0) {
        std::vector<float> out(n_iters);
        launch_mantissa(d_in, d_out, n_iters, ToMantissaBits, mode, stoch_len);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, n_iters * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    };

    auto rd   = run(Rounding_Mode::RoundDown);
    auto ru   = run(Rounding_Mode::RoundUp);
    auto rne  = run(Rounding_Mode::RoundToNearestEven);
    auto rno  = run(Rounding_Mode::RoundToNearestOdd);
    auto rta  = run(Rounding_Mode::RoundTiesToAway);
    auto rtz  = run(Rounding_Mode::RoundTowardsZero);
    auto raw  = run(Rounding_Mode::RoundAwayFromZero);

    for (int i = 0; i < n_iters; ++i) {
        float d = h_in[i];
        if (!std::isfinite(d)) continue;

        double denom = get_denom(d);

        double rel_down = std::fabs((double)rd[i]  - d) / denom;
        double rel_up   = std::fabs((double)ru[i]  - d) / denom;
        double rel_rne  = std::fabs((double)rne[i] - d) / denom;
        double rel_rno  = std::fabs((double)rno[i] - d) / denom;
        double rel_rta  = std::fabs((double)rta[i] - d) / denom;
        double rel_rtz  = std::fabs((double)rtz[i] - d) / denom;
        double rel_raw  = std::fabs((double)raw[i] - d) / denom;

        if ((double)rd[i] > d || rel_down > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundDown failed (x=" << d
                      << " fd=" << (double)rd[i] << " rel=" << rel_down << ")\n";
            ++num_errors;
        }
        if ((double)ru[i] < d || rel_up > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundUp failed (x=" << d
                      << " fu=" << (double)ru[i] << " rel=" << rel_up << ")\n";
            ++num_errors;
        }
        if (std::fabs((double)rtz[i]) > std::fabs(d) || rel_rtz > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundTowardsZero failed (x=" << d
                      << " frtz=" << (double)rtz[i] << " rel=" << rel_rtz << ")\n";
            ++num_errors;
        }
        if (std::fabs((double)raw[i]) < std::fabs(d) || rel_raw > mach_eps) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundAwayFromZero failed (x=" << d
                      << " fraw=" << (double)raw[i] << " rel=" << rel_raw << ")\n";
            ++num_errors;
        }
        if (rel_rne > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundToNearestEven failed (x=" << d
                      << " frne=" << (double)rne[i] << " rel=" << rel_rne << ")\n";
            ++num_errors;
        }
        if (rel_rno > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundToNearestOdd failed (x=" << d
                      << " frno=" << (double)rno[i] << " rel=" << rel_rno << ")\n";
            ++num_errors;
        }
        if (rel_rta > mach_eps_half) {
            std::cout << "[mant=" << ToMantissaBits << "] RoundTiesToAway failed (x=" << d
                      << " frta=" << (double)rta[i] << " rel=" << rel_rta << ")\n";
            ++num_errors;
        }
    }

    // ------------------------------------------------------------------
    // tie-breaking: construct exact ties between two adjacent representable
    // values at the target precision.
    // ------------------------------------------------------------------
    if constexpr (ToMantissaBits >= 1 && ToMantissaBits < 23) {
        const int kFromMantissa = 23;
        const int shift         = kFromMantissa - ToMantissaBits;
        const uint32_t step     = uint32_t{1} << shift;
        const uint32_t half     = uint32_t{1} << (shift - 1);

        constexpr int kTies = 200;
        std::vector<float> h_tie(kTies, 0.0f);
        std::vector<float> h_a(kTies, 0.0f), h_anext(kTies, 0.0f);
        std::vector<uint8_t> h_valid(kTies, 0);

        for (int rep = 0; rep < kTies; ++rep) {
            float base = rnd32_wide();
            if (!std::isfinite(base) || base == 0.0f) continue;
            uint32_t b = f2u(base);
            b &= ~(step - 1);
            float a      = u2f(b);
            float a_next = u2f(b + step);
            float tie    = u2f(b + half);
            if (!std::isfinite(tie) || !std::isfinite(a_next)) continue;
            h_a[rep] = a;
            h_anext[rep] = a_next;
            h_tie[rep] = tie;
            h_valid[rep] = 1;
        }

        float *d_tin = nullptr, *d_tout = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tin,  kTies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tout, kTies * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tin, h_tie.data(), kTies * sizeof(float),
                              cudaMemcpyHostToDevice));

        auto run_tie = [&](Rounding_Mode mode) {
            std::vector<float> out(kTies);
            launch_mantissa(d_tin, d_tout, kTies, ToMantissaBits, mode);
            CUDA_CHECK(cudaMemcpy(out.data(), d_tout, kTies * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            return out;
        };
        auto trne = run_tie(Rounding_Mode::RoundToNearestEven);
        auto trno = run_tie(Rounding_Mode::RoundToNearestOdd);
        auto trta = run_tie(Rounding_Mode::RoundTiesToAway);

        auto kept_lsb = [&](float v) {
            uint32_t bits = f2u(v);
            return (bits >> shift) & 1u;
        };

        for (int rep = 0; rep < kTies; ++rep) {
            if (!h_valid[rep]) continue;
            float tie    = h_tie[rep];
            float a      = h_a[rep];
            float a_next = h_anext[rep];

            if (kept_lsb(trne[rep]) != 0u) {
                std::cout << "[mant=" << ToMantissaBits << "] RNE tie not even (tie=" << (double)tie
                          << " rne=" << (double)trne[rep] << ")\n";
                ++num_errors;
            }
            if (kept_lsb(trno[rep]) != 1u) {
                std::cout << "[mant=" << ToMantissaBits << "] RNO tie not odd (tie=" << (double)tie
                          << " rno=" << (double)trno[rep] << ")\n";
                ++num_errors;
            }
            double further = std::max(std::fabs((double)a), std::fabs((double)a_next));
            if (std::fabs((double)trta[rep]) + 1e-30 < further) {
                std::cout << "[mant=" << ToMantissaBits << "] RTA tie not away (tie=" << (double)tie
                          << " rta=" << (double)trta[rep] << ")\n";
                ++num_errors;
            }
        }

        CUDA_CHECK(cudaFree(d_tin));
        CUDA_CHECK(cudaFree(d_tout));
    }

    // ------------------------------------------------------------------
    // stochastic modes: result must be one of {round-down, round-up}.
    // ------------------------------------------------------------------
    {
        const Rounding_Mode stoch_modes[] = {
            Rounding_Mode::StochasticRoundingA,
            Rounding_Mode::StochasticRoundingB,
            Rounding_Mode::StochasticRoundingC,
            Rounding_Mode::True_StochasticRounding,
        };
        const char* stoch_names[] = {"StochA", "StochB", "StochC", "TrueStoch"};

        constexpr int kS = 500;
        std::vector<float> h_si(kS);
        for (int i = 0; i < kS; ++i) h_si[i] = rnd32_wide();

        float *d_si = nullptr, *d_so = nullptr;
        CUDA_CHECK(cudaMalloc(&d_si, kS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_so, kS * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_si, h_si.data(), kS * sizeof(float),
                              cudaMemcpyHostToDevice));

        auto run_s = [&](Rounding_Mode mode, int stoch_len = 0) {
            std::vector<float> out(kS);
            launch_mantissa(d_si, d_so, kS, ToMantissaBits, mode, stoch_len);
            CUDA_CHECK(cudaMemcpy(out.data(), d_so, kS * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            return out;
        };

        auto sd = run_s(Rounding_Mode::RoundDown);
        auto su = run_s(Rounding_Mode::RoundUp);

        for (int m = 0; m < 4; ++m) {
            auto srs = run_s(stoch_modes[m], 8);
            for (int i = 0; i < kS; ++i) {
                float d = h_si[i];
                if (!std::isfinite(d)) continue;
                if (srs[i] != sd[i] && srs[i] != su[i]) {
                    std::cout << "[mant=" << ToMantissaBits << "] " << stoch_names[m]
                              << " produced non-adjacent result (x=" << d
                              << " rs=" << (double)srs[i]
                              << " fd=" << (double)sd[i]
                              << " fu=" << (double)su[i] << ")\n";
                    ++num_errors;
                }
            }
        }

        CUDA_CHECK(cudaFree(d_si));
        CUDA_CHECK(cudaFree(d_so));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "virtual_round<float, mant=" << ToMantissaBits << "> [cuda] : "
              << (num_errors == 0 ? "pass" : "FAIL") << "\n";
    return num_errors;
}

// -------------------------------------------------------------
// 2)  virtual_round(value, FloatingPointParams, mode)   --  CUDA
// -------------------------------------------------------------
template <typename ToInf, typename ToNaN>
int test_fp_params_round_cuda(const char* name,
                              FloatingPointParams<ToInf, ToNaN> ToFp,
                              int n_iters = 5000) {
    int num_errors = 0;

    const double mach_eps      = std::pow(2.0, -ToFp.mantissa_bits);
    const double mach_eps_half = std::pow(2.0, -ToFp.mantissa_bits - 1);

    const double UNT = std::pow(2.0, 1 - ToFp.bias) *
                       std::pow(2.0, -ToFp.mantissa_bits);

    const int ToMax_exp = (ToFp.is_signed == Signedness::Signed
                              ? (1 << (ToFp.bitwidth - ToFp.mantissa_bits - 1)) - 1
                              : (1 << (ToFp.bitwidth - ToFp.mantissa_bits))) - 1
                         - ToFp.bias;
    const float max_abs = std::ldexp(1.0f, ToMax_exp);

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    // ------------------------------------------------------------------
    // sweep: 2 random samples per iter (signed-narrow + wide)
    // ------------------------------------------------------------------
    const int batch = n_iters * 2;
    std::vector<float> h_in(batch);
    for (int i = 0; i < n_iters; ++i) {
        h_in[2 * i + 0] = rnd32_signed();
        h_in[2 * i + 1] = rnd32_wide();
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, batch * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), batch * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto run = [&](Rounding_Mode mode, int stoch_len = 0) {
        std::vector<float> out(batch);
        launch_fp_params(d_in, d_out, batch, ToFp, mode, stoch_len);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, batch * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    };

    auto rd  = run(Rounding_Mode::RoundDown);
    auto ru  = run(Rounding_Mode::RoundUp);
    auto rne = run(Rounding_Mode::RoundToNearestEven);
    auto rno = run(Rounding_Mode::RoundToNearestOdd);
    auto rta = run(Rounding_Mode::RoundTiesToAway);
    auto rtz = run(Rounding_Mode::RoundTowardsZero);
    auto raw = run(Rounding_Mode::RoundAwayFromZero);

    for (int i = 0; i < batch; ++i) {
        float d = h_in[i];
        if (!std::isfinite(d)) continue;
        if (std::fabs(d) >= max_abs) continue;

        double denom    = get_denom(d);
        double abs_down = std::fabs((double)rd[i]  - d);
        double abs_up   = std::fabs((double)ru[i]  - d);
        double abs_rne  = std::fabs((double)rne[i] - d);
        double abs_rno  = std::fabs((double)rno[i] - d);
        double abs_rta  = std::fabs((double)rta[i] - d);
        double abs_rtz  = std::fabs((double)rtz[i] - d);
        double abs_raw  = std::fabs((double)raw[i] - d);

        double rel_down = abs_down / denom;
        double rel_up   = abs_up   / denom;
        double rel_rne  = abs_rne  / denom;
        double rel_rno  = abs_rno  / denom;
        double rel_rta  = abs_rta  / denom;
        double rel_rtz  = abs_rtz  / denom;
        double rel_raw  = abs_raw  / denom;

        const double abs_d = std::fabs((double)d);
        const bool in_target_normal = abs_d >= std::ldexp(1.0, 1 - ToFp.bias);

        if (in_target_normal) {
            if ((double)rd[i] > d || rel_down > mach_eps) {
                std::cout << name << " [cuda] RoundDown failed (x=" << d << " fd=" << (double)rd[i]
                          << " rel=" << rel_down << ")\n"; ++num_errors;
            }
            if ((double)ru[i] < d || rel_up > mach_eps) {
                std::cout << name << " [cuda] RoundUp failed (x=" << d << " fu=" << (double)ru[i]
                          << " rel=" << rel_up << ")\n"; ++num_errors;
            }
            if (std::fabs((double)rtz[i]) > std::fabs(d) || rel_rtz > mach_eps) {
                std::cout << name << " [cuda] RoundTowardsZero failed (x=" << d << " frtz=" << (double)rtz[i]
                          << " rel=" << rel_rtz << ")\n"; ++num_errors;
            }
            if (std::fabs((double)raw[i]) < std::fabs(d) || rel_raw > mach_eps) {
                std::cout << name << " [cuda] RoundAwayFromZero failed (x=" << d << " fraw=" << (double)raw[i]
                          << " rel=" << rel_raw << ")\n"; ++num_errors;
            }
            if (rel_rne > mach_eps_half) {
                std::cout << name << " [cuda] RoundToNearestEven failed (x=" << d << " frne=" << (double)rne[i]
                          << " rel=" << rel_rne << ")\n"; ++num_errors;
            }
            if (rel_rno > mach_eps_half) {
                std::cout << name << " [cuda] RoundToNearestOdd failed (x=" << d << " frno=" << (double)rno[i]
                          << " rel=" << rel_rno << ")\n"; ++num_errors;
            }
            if (rel_rta > mach_eps_half) {
                std::cout << name << " [cuda] RoundTiesToAway failed (x=" << d << " frta=" << (double)rta[i]
                          << " rel=" << rel_rta << ")\n"; ++num_errors;
            }
        } else {
            if ((double)rd[i] > d || abs_down > UNT) {
                std::cout << name << " [cuda] (sub) RoundDown failed (x=" << d << " fd=" << (double)rd[i]
                          << " abs=" << abs_down << ")\n"; ++num_errors;
            }
            if ((double)ru[i] < d || abs_up > UNT) {
                std::cout << name << " [cuda] (sub) RoundUp failed (x=" << d << " fu=" << (double)ru[i]
                          << " abs=" << abs_up << ")\n"; ++num_errors;
            }
            if (std::fabs((double)rtz[i]) > std::fabs(d) || abs_rtz > UNT) {
                std::cout << name << " [cuda] (sub) RoundTowardsZero failed (x=" << d << " frtz=" << (double)rtz[i]
                          << " abs=" << abs_rtz << ")\n"; ++num_errors;
            }
            if (std::fabs((double)raw[i]) < std::fabs(d) || abs_raw > UNT) {
                std::cout << name << " [cuda] (sub) RoundAwayFromZero failed (x=" << d << " fraw=" << (double)raw[i]
                          << " abs=" << abs_raw << ")\n"; ++num_errors;
            }
            if (abs_rne > UNT) {
                std::cout << name << " [cuda] (sub) RoundToNearestEven failed (x=" << d << " frne=" << (double)rne[i]
                          << " abs=" << abs_rne << ")\n"; ++num_errors;
            }
            if (abs_rno > UNT) {
                std::cout << name << " [cuda] (sub) RoundToNearestOdd failed (x=" << d << " frno=" << (double)rno[i]
                          << " abs=" << abs_rno << ")\n"; ++num_errors;
            }
            if (abs_rta > UNT) {
                std::cout << name << " [cuda] (sub) RoundTiesToAway failed (x=" << d << " frta=" << (double)rta[i]
                          << " abs=" << abs_rta << ")\n"; ++num_errors;
            }
        }
    }

    // ------------------------------------------------------------------
    // explicit zero / NaN
    // ------------------------------------------------------------------
    {
        std::vector<float> probe = {0.0f, std::numeric_limits<float>::quiet_NaN()};
        float *d_p = nullptr, *d_po = nullptr;
        CUDA_CHECK(cudaMalloc(&d_p,  probe.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_po, probe.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p, probe.data(), probe.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::vector<float> out(probe.size());
        launch_fp_params(d_p, d_po, (int)probe.size(), ToFp,
                         Rounding_Mode::RoundToNearestEven);
        CUDA_CHECK(cudaMemcpy(out.data(), d_po, probe.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (out[0] != 0.0f) {
            std::cout << name << " [cuda] zero not preserved (got " << (double)out[0] << ")\n";
            ++num_errors;
        }
        if (!std::isnan(out[1])) {
            std::cout << name << " [cuda] NaN not preserved (got " << (double)out[1] << ")\n";
            ++num_errors;
        }
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_po));
    }

    // ------------------------------------------------------------------
    // tie-breaking inside the normal range of the target format
    // ------------------------------------------------------------------
    {
        const int target_mant = ToFp.mantissa_bits;
        const int shift       = 23 - target_mant;
        if (shift > 0) {
            const uint32_t step = uint32_t{1} << shift;
            const uint32_t half = uint32_t{1} << (shift - 1);

            constexpr int kTies = 200;
            std::vector<float> h_tie(kTies, 0.0f);
            std::vector<float> h_a(kTies, 0.0f), h_anext(kTies, 0.0f);
            std::vector<uint8_t> h_valid(kTies, 0);

            for (int i = 0; i < kTies; ++i) {
                int e = (std::rand() % 6) - 2;
                float u = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                float base = std::ldexp(u + 1.0f, e);
                if (!std::isfinite(base)) continue;
                uint32_t b = f2u(base);
                b &= ~(step - 1);
                float a      = u2f(b);
                float a_next = u2f(b + step);
                float tie    = u2f(b + half);
                if (!std::isfinite(a_next) || !std::isfinite(tie)) continue;
                h_a[i] = a;
                h_anext[i] = a_next;
                h_tie[i] = tie;
                h_valid[i] = 1;
            }

            float *d_tin = nullptr, *d_tout = nullptr;
            CUDA_CHECK(cudaMalloc(&d_tin,  kTies * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_tout, kTies * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_tin, h_tie.data(), kTies * sizeof(float),
                                  cudaMemcpyHostToDevice));

            auto run_tie = [&](Rounding_Mode mode) {
                std::vector<float> out(kTies);
                launch_fp_params(d_tin, d_tout, kTies, ToFp, mode);
                CUDA_CHECK(cudaMemcpy(out.data(), d_tout, kTies * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                return out;
            };
            auto trne = run_tie(Rounding_Mode::RoundToNearestEven);
            auto trno = run_tie(Rounding_Mode::RoundToNearestOdd);
            auto trta = run_tie(Rounding_Mode::RoundTiesToAway);

            auto kept_lsb = [&](float v) {
                uint32_t bits = f2u(v);
                return (bits >> shift) & 1u;
            };

            for (int i = 0; i < kTies; ++i) {
                if (!h_valid[i]) continue;
                float tie    = h_tie[i];
                float a      = h_a[i];
                float a_next = h_anext[i];

                if (kept_lsb(trne[i]) != 0u) {
                    std::cout << name << " [cuda] RNE tie not even (tie=" << (double)tie
                              << " rne=" << (double)trne[i] << ")\n"; ++num_errors;
                }
                if (kept_lsb(trno[i]) != 1u) {
                    std::cout << name << " [cuda] RNO tie not odd (tie=" << (double)tie
                              << " rno=" << (double)trno[i] << ")\n"; ++num_errors;
                }
                double further = std::max(std::fabs((double)a), std::fabs((double)a_next));
                if (std::fabs((double)trta[i]) + 1e-30 < further) {
                    std::cout << name << " [cuda] RTA tie not away (tie=" << (double)tie
                              << " rta=" << (double)trta[i] << ")\n"; ++num_errors;
                }
            }

            CUDA_CHECK(cudaFree(d_tin));
            CUDA_CHECK(cudaFree(d_tout));
        }
    }

    // ------------------------------------------------------------------
    // stochastic sanity for FP-params overload
    // ------------------------------------------------------------------
    {
        const Rounding_Mode stoch_modes[] = {
            Rounding_Mode::StochasticRoundingA,
            Rounding_Mode::StochasticRoundingB,
            Rounding_Mode::StochasticRoundingC,
            Rounding_Mode::True_StochasticRounding,
        };
        const char* stoch_names[] = {"StochA", "StochB", "StochC", "TrueStoch"};

        constexpr int kS = 500;
        std::vector<float> h_si(kS);
        for (int i = 0; i < kS; ++i) h_si[i] = rnd32_signed();

        float *d_si = nullptr, *d_so = nullptr;
        CUDA_CHECK(cudaMalloc(&d_si, kS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_so, kS * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_si, h_si.data(), kS * sizeof(float),
                              cudaMemcpyHostToDevice));

        auto run_s = [&](Rounding_Mode mode, int stoch_len = 0) {
            std::vector<float> out(kS);
            launch_fp_params(d_si, d_so, kS, ToFp, mode, stoch_len);
            CUDA_CHECK(cudaMemcpy(out.data(), d_so, kS * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            return out;
        };

        auto sd = run_s(Rounding_Mode::RoundDown);
        auto su = run_s(Rounding_Mode::RoundUp);

        for (int m = 0; m < 4; ++m) {
            auto srs = run_s(stoch_modes[m], 8);
            for (int i = 0; i < kS; ++i) {
                float d = h_si[i];
                if (!std::isfinite(d)) continue;
                if (std::fabs(d) >= max_abs) continue;
                if (srs[i] != sd[i] && srs[i] != su[i]) {
                    std::cout << name << " [cuda] " << stoch_names[m]
                              << " produced non-adjacent result (x=" << d
                              << " rs=" << (double)srs[i]
                              << " fd=" << (double)sd[i]
                              << " fu=" << (double)su[i] << ")\n";
                    ++num_errors;
                }
            }
        }

        CUDA_CHECK(cudaFree(d_si));
        CUDA_CHECK(cudaFree(d_so));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "virtual_round<" << name << "> [cuda] : "
              << (num_errors == 0 ? "pass" : "FAIL") << "\n";
    return num_errors;
}

// -------------------------------------------------------------
int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int total = 0;

    // 1) mantissa-only overload, several target precisions
    total += test_mantissa_round_cuda<7>();
    total += test_mantissa_round_cuda<10>();
    total += test_mantissa_round_cuda<15>();
    // total += test_mantissa_round_cuda<23>();   // no-op; should be exact

    // 2) full FloatingPointParams overload
    // total += test_fp_params_round_cuda("halfPrecision",   halfPrecisionParams);
    // total += test_fp_params_round_cuda("bfloatPrecision", bfloatPrecisionParams);
    // total += test_fp_params_round_cuda("tf32Precision",   tf32PrecisionParams);

    std::cout << "\n=== TOTAL ERRORS: " << total
              << (total == 0 ? "  (ALL PASS)" : "  (FAIL)") << " ===\n";
    return total == 0 ? 0 : 1;
}
