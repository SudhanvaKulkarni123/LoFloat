// examples/oscillator_8bit.cpp
// ---------------------------------------------------------------------------
// 8-bit LoFloat demo:
//   u'(t)=v(t), v'(t)=−u(t),  u(0)=1, v(0)=0
//   • Heun (explicit midpoint, 2-nd order)
//   • Implicit midpoint (2-nd order, A-stable)
//   • “Mixed precision” implicit midpoint:
//       state stored in float, stage equations solved in rne8 / sr8
//
//   h = 0.1,  T = 15
//   CSV columns: t,
//                u_rne_2nd,v_rne_2nd,u_sr_2nd,v_sr_2nd,
//                u_rne_imp,v_rne_imp,u_sr_imp,v_sr_imp,
//                u_rne_mix,v_rne_mix,u_sr_mix,v_sr_mix,
//                u_exact,v_exact
// ---------------------------------------------------------------------------

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include "lo_float.h"

using namespace lo_float;

// ────────────────────────────────────────────────────────────────────────────
// 8-bit parameter packs
// ────────────────────────────────────────────────────────────────────────────
static constexpr FloatingPointParams param_fp8(
    8, /*mant*/6, /*bias*/1,
    Rounding_Mode::StochasticRoundingA,
    Inf_Behaviors::Saturating, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    lo_float_internal::IEEE_F8_InfChecker(),
    lo_float_internal::IEEE_F8_NaNChecker(),
    0
);
static constexpr FloatingPointParams param2_fp8(
    8, 6, 1,
    Rounding_Mode::StochasticRoundingA,
    Inf_Behaviors::Saturating, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    lo_float_internal::IEEE_F8_InfChecker(),
    lo_float_internal::IEEE_F8_NaNChecker(),
    0
);

using rne8 = Templated_Float<param2_fp8>;   // round-to-nearest-even
using sr8  = Templated_Float<param_fp8>;    // stochastic rounding

// ────────────────────────────────────────────────────────────────────────────
// Heun (explicit midpoint) step
// ────────────────────────────────────────────────────────────────────────────
template<class F>
inline void heun_step(F& u, F& v, float h)
{
    const F du0 = v,   dv0 = -u;

    F u_pred = F(float(u) + h * float(du0));
    F v_pred = F(float(v) + h * float(dv0));

    const F du1 = v_pred, dv1 = -u_pred;

    u = F(float(u) + 0.5f * h * (float(du0) + float(du1)));
    v = F(float(v) + 0.5f * h * (float(dv0) + float(dv1)));
}

// ────────────────────────────────────────────────────────────────────────────
// Implicit midpoint step (generic: state type vs stage precision type)
// ────────────────────────────────────────────────────────────────────────────
template<class StageT, class StateT>
inline void implicit_midpoint_step(StateT& u, StateT& v, float h)
{
    const float half_h = 0.5f * h;                 // h/2
    const float denom  = 1.0f + half_h * half_h;   // 1 + (h/2)^2

    // Solve k = f(z + h/2 k)   with stage variables in StageT
    const StageT k1 = StageT( (static_cast<float>(v) - half_h * static_cast<float>(u)) / denom );
    const StageT k2 = StageT( static_cast<float>(-u) - half_h * float(k1) );

    u = StateT((float)u + h * float(k1));
    v = StateT((float)v + h * float(k2));
}

// ────────────────────────────────────────────────────────────────────────────
int main()
{
    constexpr float T = 1.0f, h = 0.00001f;
    const int N = static_cast<int>(T / h);

    lo_float::set_seed(static_cast<unsigned int>(std::time(nullptr)));

    // ── storage ────────────────────────────────────────────────────────────
    std::vector<float> t;
    std::vector<float> u_rne_2nd, v_rne_2nd, u_sr_2nd, v_sr_2nd;      // Heun
    std::vector<float> u_rne_imp, v_rne_imp, u_sr_imp, v_sr_imp;      // ImpMid
    std::vector<float> u_rne_mix, v_rne_mix, u_sr_mix, v_sr_mix;      // mixed
    std::vector<float> u_exact,   v_exact;
    t.reserve(N + 1);

    // ── initial conditions ────────────────────────────────────────────────
    rne8 u_r2 = (rne8)1.0f , v_r2 = (rne8)0.0f;      // Heun  (RNE)
    sr8  u_s2 = (sr8)1.0f , v_s2 = (sr8)0.0f;      // Heun  (SR)

    rne8 u_r_imp = (rne8)1.0f , v_r_imp = (rne8)0.0f;   // ImpMid (RNE)
    sr8  u_s_imp = (sr8)1.0f , v_s_imp = (sr8)0.0f;   // ImpMid (SR)

    float u_r_mix = 1.0f, v_r_mix = 0.0f;   // state=float, stage=rne8
    float u_s_mix = 1.0f, v_s_mix = 0.0f;   // state=float, stage=sr8

    // ── time-march loop ───────────────────────────────────────────────────
    for (int k = 0; k <= N; ++k)
    {
        float tk = k * h;
        t.push_back(tk);

        // store
        u_rne_2nd.push_back(float(u_r2));   v_rne_2nd.push_back(float(v_r2));
        u_sr_2nd .push_back(float(u_s2));   v_sr_2nd .push_back(float(v_s2));

        u_rne_imp.push_back(float(u_r_imp)); v_rne_imp.push_back(float(v_r_imp));
        u_sr_imp .push_back(float(u_s_imp)); v_sr_imp .push_back(float(v_s_imp));

        u_rne_mix.push_back(u_r_mix);       v_rne_mix.push_back(v_r_mix);
        u_sr_mix .push_back(u_s_mix);       v_sr_mix .push_back(v_s_mix);

        u_exact.push_back(std::cos(tk));
        v_exact.push_back(-std::sin(tk));

        // advance one step
        heun_step(u_r2, v_r2, h);
        heun_step(u_s2, v_s2, h);

        implicit_midpoint_step<rne8>(u_r_imp, v_r_imp, h);
        implicit_midpoint_step<sr8 >(u_s_imp, v_s_imp, h);

        implicit_midpoint_step<rne8>(u_r_mix, v_r_mix, h);
        implicit_midpoint_step<sr8 >(u_s_mix, v_s_mix, h);
    }

    // ── write CSV ──────────────────────────────────────────────────────────
    std::ofstream out("oscillator_8bit.csv");
    out << "t,"
        << "u_rne_2nd,v_rne_2nd,u_sr_2nd,v_sr_2nd,"
        << "u_rne_imp,v_rne_imp,u_sr_imp,v_sr_imp,"
        << "u_rne_mix,v_rne_mix,u_sr_mix,v_sr_mix,"
        << "u_exact,v_exact\n";

    for (std::size_t i = 0; i < t.size(); ++i)
        out << t[i] << ","
            << u_rne_2nd[i] << "," << v_rne_2nd[i] << ","
            << u_sr_2nd[i]  << "," << v_sr_2nd[i]  << ","
            << u_rne_imp[i] << "," << v_rne_imp[i] << ","
            << u_sr_imp[i]  << "," << v_sr_imp[i]  << ","
            << u_rne_mix[i] << "," << v_rne_mix[i] << ","
            << u_sr_mix[i]  << "," << v_sr_mix[i]  << ","
            << u_exact[i]   << "," << v_exact[i]   << "\n";

    // ── L2 error helper ───────────────────────────────────────────────────
    auto l2 = [&](const std::vector<float>& num)
    {
        double acc = 0.0;
        for (std::size_t i = 0; i < num.size(); ++i)
            acc += std::pow(u_exact[i] - num[i], 2);
        return std::sqrt(acc / num.size());
    };

    std::cout << "L2 error Heun   (RNE)      = " << l2(u_rne_2nd)  << '\n';
    std::cout << "L2 error Heun   (SR )      = " << l2(u_sr_2nd)   << '\n';
    std::cout << "L2 error ImpMid (RNE)      = " << l2(u_rne_imp)  << '\n';
    std::cout << "L2 error ImpMid (SR )      = " << l2(u_sr_imp)   << '\n';
    std::cout << "L2 error ImpMid (mix-RNE)  = " << l2(u_rne_mix)  << '\n';
    std::cout << "L2 error ImpMid (mix-SR )  = " << l2(u_sr_mix)   << '\n';

    std::cout << "✅  Wrote oscillator_8bit.csv\n";
    return 0;
}
