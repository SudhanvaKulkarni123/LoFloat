// examples/oscillator_8bit.cpp
// -----------------------------------------------------------
// 8‑bit LoFloat demo (no plotting):
//   u'(t) = v(t),  v'(t) = −u(t),  u(0)=1, v(0)=0
//   Forward Euler, h = 0.05, T = 20
//   Dumps CSV with columns: t, u_rne, u_sr, u_exact
// -----------------------------------------------------------
#include <vector>
#include <fstream>
#include <cmath>
#include "lo_float.h"
#include <chrono>   

using namespace lo_float;

// ---------- 8‑bit parameter packs ------------------------------------------


//define FloatingPointParams for 8-bit floats
static constexpr FloatingPointParams param_fp8(
    8, /*mant*/5, /*bias*/3,
    Rounding_Mode::StochasticRoundingC,
    Inf_Behaviors::Saturating, NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    lo_float_internal::IEEE_F8_InfChecker(),
    lo_float_internal::IEEE_F8_NaNChecker(),
    6
);

using rne8 = float8_ieee_p<6, RoundToNearestEven>;
using sr8  = Templated_Float<param_fp8>;

// ---------- one Euler step --------------------------------------------------
template<typename F>
inline void euler_step(F& u, F& v, float h)
{
    const F du = v;
    const F dv = -u;
    u = static_cast<F>((float)u + h * (float)du);
    v = static_cast<F>((float)v + h * (float)dv);
}

// Full Leapfrog Integrator Step
template<typename F>
inline void leapfrog_step(F& u, F& v, const F& u_old, const F& v_old, float h)
{
    // Half-step position update
    F u_half = static_cast<F>((float)u_old + 0.5f * h * (float)v_old);
    
    // Full-step velocity update (using u_half)
    F v_new = static_cast<F>((float)v_old - h * (float)u_half);
    
    // Another half-step position update using v_new
    F u_new = static_cast<F>((float)u_half + 0.5f * h * (float)v_new);

    // Assign updated values back
    u = u_new;
    v = v_new;
}


int main()
{

    constexpr float T = 10.0, h = 0.001;
    int N = static_cast<int>(T / h);
    std::cout << "N = " << N << "\n";

    //set seed with time
    lo_float::set_seed(static_cast<unsigned int>(std::time(nullptr)));

    // printr max vals of the formats
    std::cout << "max rne8 = " << static_cast<float>(std::numeric_limits<rne8>::max()) << "\n";
    std::cout << "max sr8  = " << static_cast<float>(std::numeric_limits<sr8>::max()) << "\n";
    std::cout << "min rne8 = " << static_cast<float>(std::numeric_limits<rne8>::min()) << "\n";
    std::cout << "min sr8  = " << static_cast<float>(std::numeric_limits<sr8>::min()) << "\n";
    std::cout << "denorm rne8 = " << static_cast<float>(std::numeric_limits<rne8>::denorm_min()) << "\n";
    std::cout << "denorm sr8  = " << static_cast<float>(std::numeric_limits<sr8>::denorm_min()) << "\n";


    std::vector<float> t, u_rne, v_rne, u_sr, v_sr, u_exact, v_exact;
    t.reserve(N + 1);

    rne8 u_r = rne8(1.0), v_r = rne8(0.0);
    sr8  u_s = sr8(1.0), v_s = sr8(-0.0);

    std::cout << "u_s = " << static_cast<float>(u_s) << "\n";
    std::cout << "v_s = " << static_cast<float>(v_s) << "\n";

    for (int k = 0; k <= N; ++k) {
        float tk = k * h;
        t.push_back(tk);
        u_rne.push_back(static_cast<float>(u_r));
        v_rne.push_back(static_cast<float>(v_r));
        u_sr .push_back(static_cast<float>(u_s));
        v_sr .push_back(static_cast<float>(v_s));
        u_exact.push_back(std::cos(tk));
        v_exact.push_back(-std::sin(tk));

        euler_step(u_r, v_r, h);
        euler_step(u_s, v_s, h);

    }

    // Write CSV with u and v for both modes
    std::ofstream out("oscillator_8bit.csv");
    out << "t,u_rne,v_rne,u_sr,v_sr,u_exact,v_exact\n";
    for (std::size_t i = 0; i < t.size(); ++i)
        out << t[i] << ","
            << u_rne[i] << "," << v_rne[i] << ","
            << u_sr[i]  << "," << v_sr[i]  << ","
            << u_exact[i] << "," << v_exact[i] << "\n";

    std::cout << "✅ Wrote oscillator_8bit.csv with u(t) and v(t) for both rounding modes\n";
    return 0;
}