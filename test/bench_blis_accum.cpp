// ============================================================================
//  bench_blis_accum.cpp
//
//  Perf comparison for the BLIS custom-accumulation feature:
//    (1) stock BLIS sgemm            — the unmodified BLIS kernel (timed BEFORE
//                                       our micro-kernel override is installed),
//    (2) lof_sgemm_ukr, accum off    — our scalar fp32 micro-kernel,
//    (3) lof_sgemm_ukr, fp16 accum   — our kernel with round-to-odd + fp16
//                                       virtual_round custom accumulation.
//
//  Reports wall-clock ms and GFLOP/s (2*M*N*K flops) for each. The custom
//  kernel is a plain scalar triple-loop (the CPU analog of the CUTLASS path),
//  so it is expected to be much slower than stock BLIS — this just records the
//  cost of the accumulation hook so regressions are visible.
//
//  Build: see test/Makefile target `bench_blis_accum`.
//  Usage: ./bench_blis_accum [M N K reps]
// ============================================================================

#include "blis_lof_gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

using clk = std::chrono::steady_clock;

static double run_timed( int M, int N, int K, int reps,
                         const float* A, const float* B, std::vector<float>& C )
{
    const float alpha = 1.0f, beta = 0.0f;
    double best_ms = 1e300;
    for ( int r = 0; r < reps; ++r )
    {
        for ( auto& x : C ) x = 0.0f;
        auto t0 = clk::now();
        bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
                   M, N, K, &alpha,
                   const_cast<float*>( A ), 1, M,
                   const_cast<float*>( B ), 1, K,
                   &beta, C.data(), 1, M );
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
        if ( ms < best_ms ) best_ms = ms;
    }
    return best_ms;
}

static void report( const char* tag, double ms, long long flops )
{
    double gflops = ( ms > 0.0 ) ? ( (double)flops / ( ms * 1e6 ) ) : 0.0;
    printf( "  %-28s  %9.3f ms   %8.3f GFLOP/s\n", tag, ms, gflops );
}

int main( int argc, char** argv )
{
    int M = 256, N = 256, K = 256, reps = 5;
    if ( argc >= 4 ) { M = atoi( argv[1] ); N = atoi( argv[2] ); K = atoi( argv[3] ); }
    if ( argc >= 5 ) reps = atoi( argv[4] );

    const long long flops = 2LL * M * N * K;

    std::vector<float> A( (size_t)M * K ), B( (size_t)K * N ), C( (size_t)M * N );
    srand( 7 );
    auto rnd = [] { return (float)rand() / RAND_MAX - 0.5f; };
    for ( auto& x : A ) x = rnd();
    for ( auto& x : B ) x = rnd();

    printf( "BLIS custom-accumulation perf  (M=%d N=%d K=%d, best of %d)\n", M, N, K, reps );

    // (1) Stock BLIS — must be timed BEFORE the global micro-kernel override.
    double ms_stock = run_timed( M, N, K, reps, A.data(), B.data(), C );
    report( "stock BLIS sgemm", ms_stock, flops );

    // Install our scalar micro-kernel (process-global override).
    lof_blis_install_kernels();

    // (2) Our kernel, custom accumulation disabled (plain fp32).
    lof_blis_set_accum( -1 );
    double ms_fp32 = run_timed( M, N, K, reps, A.data(), B.data(), C );
    report( "lof_ukr fp32 (accum off)", ms_fp32, flops );

    // (3) Our kernel, fp16 custom accumulation (round-to-odd + virtual_round).
    lof_blis_set_accum( 10, lo_float::Rounding_Mode::RoundToNearestEven );
    double ms_fp16 = run_timed( M, N, K, reps, A.data(), B.data(), C );
    report( "lof_ukr fp16 accum (RNE)", ms_fp16, flops );

    printf( "\nslowdown vs stock BLIS:  fp32 x%.1f   fp16-accum x%.1f\n",
            ms_fp32 / ms_stock, ms_fp16 / ms_stock );
    return 0;
}
