#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cctype>
// Include your library headers here
#include "lo_float.h"  // example
// #include "p3109_float.hpp"
// #include "templated_float.hpp"

static volatile float g_sink = 0.0f; // prevents over-optimization
using namespace lo_float;
template <typename OutFloat>
struct BenchResult {
    double avg_us;
    double min_us;
    double max_us;
};

// Bench ONLY lo_float::Project(...) time
template <typename OutFloat>
BenchResult<OutFloat> bench_project(
    const char* name,
    int /*n_unused*/,
    int iters,
    int warmup_iters,
    Rounding_Mode mode = Rounding_Mode::RoundToNearestEven
) {
    // ---- helpers (local so you don't change anything else) ----
    auto sanitize = [](const char* s) {
        std::string out;
        for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
            if (std::isalnum(*p)) out.push_back((char)*p);
            else out.push_back('_');
        }
        return out;
    };

    auto logspace_int = [](int nmin, int nmax, int points) {
        std::vector<int> v;
        v.reserve(points);
        const double a = std::log10((double)nmin);
        const double b = std::log10((double)nmax);
        for (int i = 0; i < points; ++i) {
            double t = (points == 1) ? 0.0 : (double)i / (double)(points - 1);
            double x = std::pow(10.0, a + (b - a) * t);
            int xi = (int)std::llround(x);
            if (xi < nmin) xi = nmin;
            if (xi > nmax) xi = nmax;
            if (v.empty() || xi > v.back()) v.push_back(xi);
        }
        // ensure last is exactly nmax
        if (v.empty() || v.back() != nmax) v.push_back(nmax);
        return v;
    };

    // ---- experiment plan ----
    const int NMIN = 10;
    const int NMAX = 100000000; // 1e8
    const int POINTS = 20;
    const std::vector<int> sizes = logspace_int(NMIN, NMAX, POINTS);

    // ---- CSV output (append) ----
    const std::string csv_path = "vec_round_bench.csv";
    const bool need_header = [&]() {
        std::ifstream in(csv_path);
        return !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: could not open " << csv_path << " for writing\n";
    } else if (need_header) {
        csv << "type,n,avg_us,min_us,max_us,iters_used,warmup_iters\n";
    }

    // ---- run sweep ----
    BenchResult<OutFloat> last{0, 0, 0};
    const std::string type_name = sanitize(name);

    for (int n : sizes) {
        std::vector<float> in(n);
        for (int i = 0; i < n; i++) {
            in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        std::vector<OutFloat> out(n);

        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            lo_float::Project(in.data(), out.data(), n, mode);
            g_sink += static_cast<float>(out[i % n]);
        }

        // Optional: adapt iterations so the 1e8 case doesn't take forever.
        int iters_used = iters;
        if (n >= 10000000) iters_used = std::min(iters_used, 5);
        if (n >= 50000000) iters_used = std::min(iters_used, 3);
        if (iters_used < 1) iters_used = 1;

        std::vector<double> samples_us;
        samples_us.reserve(iters_used);

        for (int i = 0; i < iters_used; i++) {
            auto t0 = std::chrono::steady_clock::now();
            lo_float::Project(in.data(), out.data(), n, mode);
            auto t1 = std::chrono::steady_clock::now();

            g_sink += static_cast<float>(out[(i * 17) % n]);

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            samples_us.push_back(us);
        }

        auto [min_it, max_it] = std::minmax_element(samples_us.begin(), samples_us.end());
        double avg = 0.0;
        for (double x : samples_us) avg += x;
        avg /= (double)samples_us.size();

        last = BenchResult<OutFloat>{avg, *min_it, *max_it};

        std::cout << name
                  << "  n=" << n
                  << "  avg=" << avg << " us"
                  << "  min=" << *min_it << " us"
                  << "  max=" << *max_it << " us"
                  << "  iters=" << iters_used
                  << "\n";

        if (csv) {
            csv << type_name << ","
                << n << ","
                << avg << ","
                << *min_it << ","
                << *max_it << ","
                << iters_used << ","
                << warmup_iters
                << "\n";
        }
    }

    if (csv) csv.flush();
    return last;
}

 
// Convenient aliases matching your types
template<int l, int p>
using p3109_s_sat = P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;

using binary16 = Templated_Float<halfPrecisionParams>;

int main() {
    // Keep benchmark consistent
    srand(12345);

    // Choose sizes / iteration counts
    const int n = 1 << 20;        // ~1,048,576 elements (change as desired)
    const int warmup = 5;
    const int iters  = 50;

    // Benchmark binary8p4
    bench_project<p3109_s_sat<8,4>>("binary8p4 (P3109<8,4>)", n, iters, warmup);

    // Benchmark binary8p3
    bench_project<p3109_s_sat<8,3>>("binary8p3 (P3109<8,3>)", n, iters, warmup);

    // Benchmark half
    bench_project<binary16>("half (binary16)", n, iters, warmup);

    // Print sink so compiler can't assume it's unused
    std::cerr << "sink=" << g_sink << "\n";
    return 0;
}

