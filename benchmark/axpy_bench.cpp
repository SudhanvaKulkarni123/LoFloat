#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cctype>

#include "lo_float.h"

static volatile float g_sink = 0.0f;
using namespace lo_float;

template <typename TY>
struct BenchResult {
    double avg_us;
    double min_us;
    double max_us;
};

// Bench ONLY lo_float::axpy(...) time
template <typename TA, typename TX, typename TY, class arch = xsimd::default_arch>
BenchResult<TY> bench_axpy(
    const char* name,
    int /*n_unused*/,
    int iters,
    int warmup_iters,
    Rounding_Mode mode = Rounding_Mode::RoundToNearestEven
) {
    // ---- helpers ----
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
        if (v.empty() || v.back() != nmax) v.push_back(nmax);
        return v;
    };

    // ---- experiment plan ----
    const int NMIN = 10;
    const int NMAX = 100000000;
    const int POINTS = 20;
    const std::vector<int> sizes = logspace_int(NMIN, NMAX, POINTS);

    // ---- CSV output ----
    const std::string csv_path = "vec_axpy_bench.csv";
    const bool need_header = [&]() {
        std::ifstream in(csv_path);
        return !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: could not open " << csv_path << "\n";
    } else if (need_header) {
        csv << "type,n,avg_us,min_us,max_us,iters_used,warmup_iters\n";
    }

    BenchResult<TY> last{0,0,0};
    const std::string type_name = sanitize(name);

    for (int n : sizes) {
        std::vector<TX> x(n);
        std::vector<TY> y(n);

        for (int i = 0; i < n; ++i) {
            x[i] = static_cast<TX>(static_cast<float>(rand()) / RAND_MAX);
            y[i] = static_cast<TY>(static_cast<float>(rand()) / RAND_MAX);
        }

        TA a_val = static_cast<TA>(0.25f);   // constant scalar
        const TA* a = &a_val;

        // Warmup
        for (int w = 0; w < warmup_iters; ++w) {
            lo_float::axpy<TA, TX, TY, arch>(n, a, x.data(), 1, y.data(), 1, mode);
            g_sink += static_cast<float>(y[w % n]);
        }

        int iters_used = iters;
        if (n >= 10000000) iters_used = std::min(iters_used, 5);
        if (n >= 50000000) iters_used = std::min(iters_used, 3);
        if (iters_used < 1) iters_used = 1;

        std::vector<double> samples_us;
        samples_us.reserve(iters_used);

        for (int i = 0; i < iters_used; ++i) {
            auto t0 = std::chrono::steady_clock::now();

            lo_float::axpy<TA, TX, TY, arch>(n, a, x.data(), 1, y.data(), 1, mode);

            auto t1 = std::chrono::steady_clock::now();

            g_sink += static_cast<float>(y[(i * 17) % n]);

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            samples_us.push_back(us);
        }

        auto [min_it, max_it] = std::minmax_element(samples_us.begin(), samples_us.end());
        double avg = 0.0;
        for (double x : samples_us) avg += x;
        avg /= samples_us.size();

        last = BenchResult<TY>{avg, *min_it, *max_it};

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

// ---- Type aliases ----
template<int l, int p>
using p3109_s_sat = P3109_float<l, p, Signedness::Signed, Inf_Behaviors::Saturating>;

using binary16 = Templated_Float<halfPrecisionParams>;

int main() {
    srand(12345);

    const int n = 1 << 20;
    const int warmup = 5;
    const int iters  = 50;

    bench_axpy<p3109_s_sat<8,4>, float, p3109_s_sat<8,4>>(
        "AXPY binary8p4", n, iters, warmup);

    bench_axpy<p3109_s_sat<8,3>, float, p3109_s_sat<8,3>>(
        "AXPY binary8p3", n, iters, warmup);

    bench_axpy<binary16, float, binary16>(
        "AXPY half (binary16)", n, iters, warmup);

    std::cerr << "sink=" << g_sink << "\n";
    return 0;
}
