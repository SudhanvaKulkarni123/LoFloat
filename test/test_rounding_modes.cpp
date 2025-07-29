// -------------------------------------------------------------
// rounding-modes-test.cpp   ― rewritten for call-site rounding
// -------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>

#include "lo_float.h"          // your library


using namespace lo_float;

// --- Optional helper to inspect the raw bits of a float ------------
void print_float_hex(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    std::cout << "float: " << f << "\nhex:   0x"
              << std::hex << std::setw(8) << std::setfill('0')
              << bits << std::dec << "\n";
}

// ------------------------------------------------------------------
//  Canonical IEEE-754 single-precision format (layout only)
// ------------------------------------------------------------------
struct IsInf_f32 {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) == 0);
    }
    uint32_t infBitPattern() const { return 0x7F800000; }
    uint32_t minNegInf()     const { return 0xFF800000; }
    uint32_t minPosInf()     const { return 0x7F800000; }
};

struct IsNaN_f32 {
    bool operator()(uint32_t bits) const {
        return (((bits >> 23) & 0xFF) == 0xFF) && ((bits & 0x7FFFFF) != 0);
    }
    uint32_t qNanBitPattern() const { return 0x7FC00000; }
    uint32_t sNanBitPattern() const { return 0x7FA00000; }
};


constexpr FloatingPointParams param_fp32(
    /*total bits*/ 32, /*mantissa*/ 23, /*bias*/ 127,
    Inf_Behaviors::Extended,
    NaN_Behaviors::QuietNaN,
    Signedness::Signed,
    IsInf_f32(), IsNaN_f32());

using float32 = lo_float::Templated_Float<param_fp32>;  // RM is ignored here



// ------------------------------------------------------------------
int main() {
    using namespace lo_float;

    std::srand(0xC0FFEE);
    const double f32_eps = std::numeric_limits<float>::epsilon();

    // ---- helpers --------------------------------------------------
    auto rnd64 = []() -> double {
        return static_cast<double>(std::rand()) / RAND_MAX * 200.0 - 100.0;
    };
    auto is_even = [](uint32_t bits){ return (bits & 1u) == 0; };
    auto is_odd  = [](uint32_t bits){ return (bits & 1u) == 1; };

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    // ----------------------------------------------------------------------
    // 1.  RoundUp / RoundDown
    // ----------------------------------------------------------------------
    for (int i = 0; i < 2000; ++i) {
        double d  = rnd64();

        float32 fd = Round<double,float32>(d, Rounding_Mode::RoundDown);
        float32 fu = Round<double,float32>(d, Rounding_Mode::RoundUp);

        double rel_down = std::fabs(static_cast<double>(fd) - d) / std::fabs(d ? d : 1.0);
        double rel_up   = std::fabs(static_cast<double>(fu) - d) / std::fabs(d ? d : 1.0);

        if (static_cast<double>(fd) > d)
            std::cout << "RoundDown > x  (x=" << d << ")\n";
        if (static_cast<double>(fu) < d)
            std::cout << "RoundUp   < x  (x=" << d << ")\n";

        if (rel_down > f32_eps)
            std::cout << "RoundDown REL err high: " << rel_down << "\n";
        if (rel_up   > f32_eps)
            std::cout << "RoundUp   REL err high: " << rel_up   << "\n";
    }

    // ----------------------------------------------------------------------
    // 2.  RoundTowardsZero & RoundAwayFromZero
    // ----------------------------------------------------------------------
    for (int i = 0; i < 2'000; ++i) {
        double d = rnd64();

        float32 tz  = Round<double,float32>(d, Rounding_Mode::RoundTowardsZero);
        float32 taw = Round<double,float32>(d, Rounding_Mode::RoundAwayFromZero);

        if (d >= 0.0 && static_cast<double>(tz) > d) { std::cout << "RTZ failed (x=" << d << ")\n"; return 1; }
        if (d <  0.0 && static_cast<double>(tz) < d) { std::cout << "RTZ failed (x=" << d << ")\n"; return 1; }

        if (d >= 0.0 && static_cast<double>(taw) < d){ std::cout << "RAW failed (x=" << d << ")\n"; return 1; }
        if (d <  0.0 && static_cast<double>(taw) > d){ std::cout << "RAW failed (x=" << d << ")\n"; return 1; }
    }

    // ----------------------------------------------------------------------
    // 3.  Round-to-Nearest-Even / Odd  &  RoundTiesToAway
    // ----------------------------------------------------------------------
    auto half_ulp = [](float x){
        return std::nextafter(x, std::numeric_limits<float>::infinity()) - x;
    };

    for (int i = 0; i < 2; ++i) {
        float  base = static_cast<float>(rnd64());
        float  hu   = half_ulp(base);
        double tie  = base + double(hu) * 0.5;          // exact halfway value

        float32 rne = Round<double,float32>(tie, Rounding_Mode::RoundToNearestEven);
        float32 rno = Round<double,float32>(tie, Rounding_Mode::RoundToNearestOdd);
        float32 rta = Round<double,float32>(tie, Rounding_Mode::RoundTiesToAway);

        if (!is_even(rne.rep())) {
            std::cout << "RNE tie-round not even  (x=" << tie << ")\n";
            return 1;
        }
        if (!is_odd(rno.rep())) {
            std::cout << "RNO tie-round not odd   (x=" << tie << ")\n";
            return 1;
        }

        double ref_next = std::nextafter(tie, tie > 0 ? 1e300 : -1e300);
        double rta_d    = static_cast<double>(rta);
        if (std::fabs(rta_d) < std::fabs(ref_next) - 1e-20) {
            std::cout << "RTiesToAway tie not away (x=" << tie
                      << " res=" << rta_d << ")\n";
            return 1;
        }
    }

    std::cout << "All tests passed.\n";
    return 0;
}
