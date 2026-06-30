// ============================================================================
//  test_blis_lof.cpp
//
//  Smoke test for the scalar BLIS micro-kernel override (Step 1).
//  Installs lof_sgemm_ukr into the BLIS context, runs C = alpha*A*B + beta*C
//  through the typed BLIS front-end (which reads the global context), and
//  compares against a DOUBLE-PRECISION reference.
//
//  Why a double reference (not an fp32 one): two fp32 implementations that
//  happen to sum in the same order match bit-for-bit, giving a meaningless
//  "0 error". Comparing fp32 against fp64 produces a genuine, informative
//  rounding error (~1e-7 here), and exactly-0 then becomes a RED FLAG that
//  something is wrong (kernel not run, output copied, NaNs masking the score).
//
//  We also explicitly screen for the pathologies that can fake a good score:
//    - NaN/Inf in the output            (std::max would otherwise hide them)
//    - all-zero / trivial output        (uninitialized or kernel not run)
//
//  Build (after BLIS is built):
//    g++ -std=c++20 -O2 -I src -I third_party/blis/include/generic \
//        test/test_blis_lof.cpp third_party/blis/lib/generic/libblis.a \
//        -lpthread -lm -o test_blis_lof
//  Self-check the override is live:  add -DLOF_BLIS_POISON  (test must FAIL).
// ============================================================================

#include "blis_lof_gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Double-precision column-major reference: C = alpha*A*B + beta*C.
// Each output entry is accumulated in double, then stored as fp32 — i.e. the
// near-best fp32 answer, so the fp32 kernel should land ~1e-7 away from it.
static void ref_gemm_f64( int M, int N, int K,
                          float alpha, float beta,
                          const float* A, const float* B, float* C )
{
    for ( int j = 0; j < N; ++j )
        for ( int i = 0; i < M; ++i )
        {
            double acc = 0.0;
            for ( int p = 0; p < K; ++p )
                acc += (double)A[ i + (size_t)p * M ] * (double)B[ p + (size_t)j * K ];
            C[ i + (size_t)j * M ] =
                (float)( (double)alpha * acc + (double)beta * (double)C[ i + (size_t)j * M ] );
        }
}

int main()
{
    lof_blis_install_kernels();

    const int M = 67, N = 53, K = 40;   // deliberately non-multiples of MR/NR
    const float alpha = 1.25f, beta = 0.5f;

    // Column-major storage: rs = 1, cs = leading dim.
    std::vector<float> A( (size_t)M * K ), B( (size_t)K * N );
    std::vector<float> C( (size_t)M * N ), C_ref( (size_t)M * N );

    srand( 7 );
    auto rnd = [] { return (float)rand() / RAND_MAX - 0.5f; };
    for ( auto& x : A ) x = rnd();
    for ( auto& x : B ) x = rnd();
    for ( size_t i = 0; i < C.size(); ++i ) { C[i] = rnd(); C_ref[i] = C[i]; }

    // Reference (double accumulation).
    ref_gemm_f64( M, N, K, alpha, beta, A.data(), B.data(), C_ref.data() );

    // BLIS (routes through lof_sgemm_ukr). Column-major: rs=1, cs=ld.
    bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
               M, N, K,
               &alpha,
               A.data(), 1, M,
               B.data(), 1, K,
               &beta,
               C.data(), 1, M );

    // -- Screen for pathologies, then measure error (all NaN-safe) -----------
    int    n_nan = 0, n_inf = 0;
    double max_abs_c   = 0.0;   // magnitude of the output (must be nontrivial)
    double err_fro2    = 0.0;   // ||C - C_ref||_F^2
    double ref_fro2    = 0.0;   // ||C_ref||_F^2
    double max_abs_err = 0.0;

    for ( size_t i = 0; i < C.size(); ++i )
    {
        const double c  = (double)C[i];
        const double cr = (double)C_ref[i];

        if ( std::isnan( c ) ) { ++n_nan; continue; }
        if ( std::isinf( c ) ) { ++n_inf; continue; }

        const double ac = std::fabs( c );
        if ( ac > max_abs_c ) max_abs_c = ac;

        const double d = std::fabs( c - cr );
        if ( d > max_abs_err ) max_abs_err = d;

        err_fro2 += d * d;
        ref_fro2 += cr * cr;
    }

    const double rel_fro = ( ref_fro2 > 0.0 )
                         ? std::sqrt( err_fro2 / ref_fro2 ) : 0.0;

    printf( "M=%d N=%d K=%d\n", M, N, K );
    printf( "  NaNs=%d  Infs=%d\n", n_nan, n_inf );
    printf( "  max|C|=%.6f  (nontrivial output check)\n", max_abs_c );
    printf( "  max_abs_err=%.3e  rel_fro_err=%.3e\n", max_abs_err, rel_fro );

    // -- Pass criteria -------------------------------------------------------
    bool pass = true;
    if ( n_nan || n_inf ) {
        printf( "  -> FAIL: output contains NaN/Inf\n" );
        pass = false;
    }
    if ( max_abs_c < 1e-6 ) {
        printf( "  -> FAIL: output is ~all-zero (kernel likely not run / uninitialized)\n" );
        pass = false;
    }
    if ( rel_fro == 0.0 ) {
        // With an fp64 reference and random data this is statistically
        // impossible if the fp32 kernel really ran — treat as suspicious.
        printf( "  -> FAIL: rel error is EXACTLY 0 (suspicious — fp32 vs fp64 must differ)\n" );
        pass = false;
    }
    if ( rel_fro > 1e-5 ) {
        printf( "  -> FAIL: rel error %.3e exceeds fp32 bound 1e-5\n", rel_fro );
        pass = false;
    }

    printf( "%s\n", pass ? "PASS" : "FAIL" );
    return pass ? 0 : 1;
}