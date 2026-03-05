/*
 * Benchmark lo_float::add_vec for FP16 (binary16).
 * Compile:
 *   g++ -O2 -march=native -fopenmp -std=c++17 -o vec_add_bench vec_add_bench.cpp -I/path/to/lo_float
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cctype>
#include "lo_float.h"

static volatile float g_sink = 0.0f;
using namespace lo_float;
using binary16 = Templated_Float<halfPrecisionParams>;

static std::vector<int> logspace_int(int nmin, int nmax, int points) {
    std::vector<int> v;
    double a = std::log10((double)nmin), b = std::log10((double)nmax);
    for (int i = 0; i < points; ++i) {
        double t = (points == 1) ? 0.0 : (double)i / (points - 1);
        int s = (int)std::llround(std::pow(10.0, a + (b - a) * t));
        if (s < nmin) s = nmin;
        if (s > nmax) s = nmax;
        if (v.empty() || s > v.back()) v.push_back(s);
    }
    if (v.empty() || v.back() != nmax) v.push_back(nmax);
    return v;
}

void bench_add_vec(int iters, int warmup_iters) {
    const auto sizes = logspace_int(10, 100000000, 20);

    const std::string csv_path = "vec_add_bench.csv";
    bool need_header = [&]() {
        std::ifstream in(csv_path);
        return !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) { std::cerr << "ERROR: could not open " << csv_path << "\n"; return; }
    if (need_header)
        csv << "type,n,avg_us,min_us,max_us,iters_used,warmup_iters\n";

    for (int n : sizes) {
        std::vector<float> a_fp32(n), b_fp32(n);
        for (int i = 0; i < n; i++) {
            float scale = 0.001f + (float)rand() / RAND_MAX * 999.999f;
            a_fp32[i] = ((float)rand() / RAND_MAX) * scale;
            b_fp32[i] = ((float)rand() / RAND_MAX) * scale;
        }

        // Pre-round inputs to FP16
        std::vector<binary16> a(n), b(n), out(n);
        lo_float::Project(a_fp32.data(), a.data(), n);
        lo_float::Project(b_fp32.data(), b.data(), n);

        // Warmup
        for (int w = 0; w < warmup_iters; w++) {
            lo_float::add_vec(a.data(), b.data(), out.data(), n);
            g_sink += static_cast<float>(out[w % n]);
        }

        int iters_used = iters;
        if (n >= 10000000) iters_used = std::min(iters_used, 5);
        if (n >= 50000000) iters_used = std::min(iters_used, 3);
        if (iters_used < 1) iters_used = 1;

        std::vector<double> samples;
        samples.reserve(iters_used);

        for (int i = 0; i < iters_used; i++) {
            auto t0 = std::chrono::steady_clock::now();
            lo_float::add_vec(a.data(), b.data(), out.data(), n);
            auto t1 = std::chrono::steady_clock::now();
            g_sink += static_cast<float>(out[(i * 17) % n]);
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        auto [mn_it, mx_it] = std::minmax_element(samples.begin(), samples.end());
        double avg = 0;
        for (double x : samples) avg += x;
        avg /= samples.size();

        std::cout << "half (binary16) add  n=" << n
                  << "  avg=" << avg << " us"
                  << "  min=" << *mn_it << " us"
                  << "  max=" << *mx_it << " us"
                  << "  iters=" << iters_used << "\n";

        csv << "half__binary16_," << n << "," << avg << "," << *mn_it << ","
            << *mx_it << "," << iters_used << "," << warmup_iters << "\n";
    }

    csv.flush();
}

int main() {
    srand(12345);
    std::cout << "lo_float::add_vec FP16 Benchmark\n";
    std::cout << "======================================\n";
    bench_add_vec(50, 5);
    std::cerr << "sink=" << g_sink << "\n";
    return 0;
}