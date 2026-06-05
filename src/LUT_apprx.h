#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "lo_float_sci.hpp"
namespace lo_float {

struct SiLURangeReduce {
    static constexpr float X_MAX = 8.0f;

    struct Ctx {
        float x_orig;
        bool  saturated;
    };

    template <FloatingPointParams Fp>
    __host__ __device__
    std::pair<Templated_Float<Fp>, Ctx> operator()(Templated_Float<Fp> x) const {
        Ctx ctx;
        ctx.x_orig    = static_cast<float>(x);
        float abs_x   = fabsf(ctx.x_orig);
        ctx.saturated = (abs_x >= X_MAX);
        Templated_Float<Fp> reduced = ctx.saturated
            ? Templated_Float<Fp>{X_MAX}
            : Templated_Float<Fp>{abs_x};
        return {reduced, ctx};
    }
};

template <std::size_t N>
struct SiLULUT {
    static constexpr float X_MAX    = SiLURangeReduce::X_MAX;
    static constexpr float INV_STEP = static_cast<float>(N - 1) / X_MAX;

    struct Node { float x, y; };
    std::array<Node, N> table{};

    template <class XGen>
    constexpr SiLULUT(XGen gen_x) {
        for (std::size_t i = 0; i < N; ++i) {
            float xi = gen_x(i, N);
            table[i] = { xi, xi / (1.0f + std::exp(-xi)) };
        }
    }

    constexpr SiLULUT()
        : SiLULUT([](std::size_t i, std::size_t n) {
              return static_cast<float>(i) * (X_MAX / static_cast<float>(n - 1));
          }) {}

    template <FloatingPointParams Fp>
    __host__ __device__
    Templated_Float<Fp> operator()(Templated_Float<Fp> reduced,
                                    const SiLURangeReduce::Ctx& ctx) const {
        float y_pos;

        if (ctx.saturated) {
            y_pos = fabsf(ctx.x_orig);
        } else {
            float xr    = static_cast<float>(reduced);
            float idx_f = xr * INV_STEP;
            std::size_t idx = static_cast<std::size_t>(idx_f);
            if (idx >= N - 1) idx = N - 2;
            float frac  = idx_f - static_cast<float>(idx);
            float y0    = table[idx].y;
            float y1    = table[idx + 1].y;
            y_pos       = fmaf(frac, y1 - y0, y0);
        }

        float result = (ctx.x_orig < 0.0f) ? (y_pos + ctx.x_orig) : y_pos;
        return Templated_Float<Fp>{result};
    }
};

template <typename RangeReducer, typename Approx_func>
class FuncApprox {
    RangeReducer reducer;
    Approx_func  approx_func;
public:
    __host__ __device__
    FuncApprox(RangeReducer r, Approx_func f) : reducer(r), approx_func(f) {}

    template <FloatingPointParams Fp>
    __host__ __device__
    Templated_Float<Fp> operator()(Templated_Float<Fp> x) const {
        auto [reduced, ctx] = reducer(x);
        return approx_func(reduced, ctx);
    }
};

inline const FuncApprox<SiLURangeReduce, SiLULUT<257>> g_silu{
    SiLURangeReduce{}, SiLULUT<257>{}
};

template <FloatingPointParams Fp, class Reducer, class ApproxFn>
__device__ __forceinline__
Templated_Float<Fp> silu_approx(const FuncApprox<Reducer, ApproxFn>& approx,
                                 Templated_Float<Fp> x) {
    return approx(x);
}

}