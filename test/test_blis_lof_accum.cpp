// ============================================================================
//  test_blis_lof_accum.cpp
//
//  Tests the custom-accumulation + rounding-mode support added to the BLIS
//  micro-kernel (lof_sgemm_ukr via lof_blis_set_accum). The kernel accumulates
//  each rank-1 update in a low-precision accumulator using
//      acc = virtual_round( lof_round_to_odd_fma(a, b, acc), bits, mode );
//  i.e. a single effective rounding per step (round-to-odd intermediate avoids
//  double-rounding), the CPU analog of thread::Mma<OpLoFMultiplyAdd> in
//  cutlass_gemms.cuh.
//
//  Acceptance (from BACKLOG.md): fp16 accumulation with different rounding
//  modes must satisfy the classical inner-product error bound
//      |C_fp16 - C_exact|_ij  <=  gamma_K * (|A|*|B|)_ij
//  where the oracle C_exact and |A|*|B| are computed in DOUBLE, and
//      gamma_K = K*eps / (1 - K*eps),   eps = 2^-10  (fp16 machine epsilon).
//
//  Using eps = 2^-10 (= ULP at 1, twice the unit roundoff u = 2^-11) makes the
//  bound mode-agnostic: a single rounding step has error <= 0.5*ULP for nearest
//  modes and < 1*ULP for directed / stochastic modes, so every mode that does
//  ONE rounding per step is covered. (See NUMERICAL_TESTING.md: oracle is
//  double, error scaled by ULP, bound not tuned to pass.)
//
//  Build (after BLIS is built):
//    g++ -std=c++20 -O3 -I src -I third_party/xsimd/include \
//        -I third_party/blis/include/generic \
//        test/test_blis_lof_accum.cpp third_party/blis/lib/generic/libblis.a \
//        -lpthread -lm -o test_blis_lof_accum
// ============================================================================

#include "blis_lof_gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

using lo_float::Rounding_Mode;

// Column-major (rs=1, cs=ld) double reference: C_exact = A*B  and  absAB = |A|*|B|.
static void ref_exact_f64( int M, int N, int K,
                           const float* A, const float* B,
                           double* C_exact, double* absAB )
{
    for ( int j = 0; j < N; ++j )
        for ( int i = 0; i < M; ++i )
        {
            double acc = 0.0, aacc = 0.0;
            for ( int p = 0; p < K; ++p )
            {
                const double a = (double)A[ i + (size_t)p * M ];
                const double b = (double)B[ p + (size_t)j * K ];
                acc  += a * b;
                aacc += std::fabs( a ) * std::fabs( b );
            }
            C_exact[ i + (size_t)j * M ] = acc;
            absAB  [ i + (size_t)j * M ] = aacc;
        }
}

// Run BLIS sgemm (alpha=1,beta=0) with the currently-installed accumulation
// settings; A:MxK ld=M, B:KxN ld=K, C:MxN ld=M (all column-major).
static void run_blis( int M, int N, int K,
                      const float* A, const float* B, float* C )
{
    float alpha = 1.0f, beta = 0.0f;
    std::memset( C, 0, (size_t)M * N * sizeof( float ) );
    bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
               M, N, K,
               &alpha,
               const_cast<float*>( A ), 1, M,
               const_cast<float*>( B ), 1, K,
               &beta,
               C, 1, M );
}

// Max absolute error vs the double oracle, and a pathology screen.
static bool check_bound( const char* tag,
                         int M, int N, int K,
                         const float* C, const double* C_exact, const double* absAB,
                         double gammaK, double& max_abs_err_out )
{
    int    n_nan = 0, n_inf = 0, n_viol = 0;
    double max_abs_err = 0.0, max_abs_c = 0.0, worst_ratio = 0.0;

    for ( size_t idx = 0; idx < (size_t)M * N; ++idx )
    {
        const double c = (double)C[ idx ];
        if ( std::isnan( c ) ) { ++n_nan; continue; }
        if ( std::isinf( c ) ) { ++n_inf; continue; }

        max_abs_c = std::max( max_abs_c, std::fabs( c ) );

        const double err   = std::fabs( c - C_exact[ idx ] );
        max_abs_err        = std::max( max_abs_err, err );

        // Bound = gammaK * |A||B|, with a tiny absolute cushion for the entries
        // whose |A||B| is ~0 (there fp32 storage of C and the double oracle
        // still differ by an fp32 ULP of the magnitude involved).
        const double bound = gammaK * absAB[ idx ] + 1e-6;
        if ( err > bound ) {
            ++n_viol;
            worst_ratio = std::max( worst_ratio, err / ( bound + 1e-300 ) );
        }
    }

    bool ok = true;
    printf( "  [%-22s] max_abs_err=%.3e  max|C|=%.4f", tag, max_abs_err, max_abs_c );
    if ( n_nan || n_inf ) { printf( "  NaN=%d Inf=%d", n_nan, n_inf ); ok = false; }
    if ( max_abs_c < 1e-6 ) { printf( "  <ALL-ZERO>" ); ok = false; }
    if ( n_viol )           { printf( "  BOUND-VIOL=%d (worst x%.2f)", n_viol, worst_ratio ); ok = false; }
    printf( "  %s\n", ok ? "ok" : "FAIL" );

    max_abs_err_out = max_abs_err;
    return ok;
}

int main()
{
    lof_blis_install_kernels();

    bool pass = true;

    // ── Part 1: fp16 accumulation, all deterministic modes, error bound ─────
    const int M = 32, N = 32, K = 64;
    const int fp16_mant = 10;
    const double eps    = std::ldexp( 1.0, -10 );          // fp16 ULP at 1
    const double Keps   = (double)K * eps;
    const double gammaK = Keps / ( 1.0 - Keps );           // classical gamma_K

    std::vector<float>  A( (size_t)M * K ), B( (size_t)K * N ), C( (size_t)M * N );
    std::vector<double> C_exact( (size_t)M * N ), absAB( (size_t)M * N );

    srand( 13 );
    auto rnd = [] { return (float)rand() / RAND_MAX - 0.5f; };
    for ( auto& x : A ) x = rnd();
    for ( auto& x : B ) x = rnd();

    ref_exact_f64( M, N, K, A.data(), B.data(), C_exact.data(), absAB.data() );

    printf( "== Part 1: fp16 accumulation (K=%d), bound = gamma_K*|A||B|, "
            "gamma_K=%.4f ==\n", K, gammaK );

    struct ModeCase { Rounding_Mode mode; const char* name; };
    const ModeCase modes[] = {
        { Rounding_Mode::RoundToNearestEven,  "RoundToNearestEven"  },
        { Rounding_Mode::RoundTowardsZero,    "RoundTowardsZero"    },
        { Rounding_Mode::RoundAwayFromZero,   "RoundAwayFromZero"   },
        { Rounding_Mode::RoundToNearestOdd,   "RoundToNearestOdd"   },
        { Rounding_Mode::RoundDown,           "RoundDown"           },
        { Rounding_Mode::RoundUp,             "RoundUp"             },
        { Rounding_Mode::RoundTiesToAway,     "RoundTiesToAway"     },
    };

    double err_fp16_rne = 0.0;
    for ( const auto& mc : modes )
    {
        lof_blis_set_accum( fp16_mant, lo_float::ProjSpec{ mc.mode } );
        run_blis( M, N, K, A.data(), B.data(), C.data() );
        double maxerr = 0.0;
        pass &= check_bound( mc.name, M, N, K,
                             C.data(), C_exact.data(), absAB.data(), gammaK, maxerr );
        if ( mc.mode == Rounding_Mode::RoundToNearestEven ) err_fp16_rne = maxerr;
    }

    // Stochastic rounding (worst-case |error| < 1 ULP per step, so the same
    // mode-agnostic bound applies). 8 random bits, fixed seed for reproducibility.
    {
        std::srand( 12345 );
        lof_blis_set_accum( fp16_mant, lo_float::ProjSpec{ Rounding_Mode::StochasticRoundingA, lo_float::Saturation_Mode::OvfInf, 8 } );
        run_blis( M, N, K, A.data(), B.data(), C.data() );
        double maxerr = 0.0;
        pass &= check_bound( "StochasticRoundingA", M, N, K,
                             C.data(), C_exact.data(), absAB.data(), gammaK, maxerr );
    }

    // ── Part 2: confirm low precision actually engaged ──────────────────────
    // fp32 accumulation (disabled custom path) must be ORDERS more accurate
    // than the fp16 path; otherwise the rounding hook is being bypassed.
    {
        lof_blis_set_accum( -1 );                          // plain fp32
        run_blis( M, N, K, A.data(), B.data(), C.data() );
        double err32 = 0.0;
        check_bound( "fp32 (disabled)", M, N, K,
                     C.data(), C_exact.data(), absAB.data(), gammaK, err32 );

        printf( "== Part 2: engaged check  err_fp16=%.3e  err_fp32=%.3e ==\n",
                err_fp16_rne, err32 );
        if ( !( err_fp16_rne > 50.0 * err32 && err_fp16_rne > 1e-4 ) ) {
            printf( "  -> FAIL: fp16 accumulation not measurably lower precision "
                    "(hook bypassed or accum_bits ignored)\n" );
            pass = false;
        } else {
            printf( "  -> ok: fp16 error >> fp32 error (low-precision accumulation active)\n" );
        }
    }

    // ── Part 3: exactly-representable case -> every mode bit-identical ──────
    // Entries are halves; products are multiples of 0.25 and partial sums stay
    // small, so the whole inner product is fp16-exact. Round-to-odd returns the
    // exact value and virtual_round is a no-op, so all modes must agree
    // bit-for-bit and equal the exact double result.
    {
        const int m = 2, n = 2, k = 4;
        // A (mxk, ld=m) and B (kxn, ld=k), column-major, values in {-0.5,0,0.5,1}.
        const float Ae[ m * k ] = { 0.5f, -0.5f,  1.0f, 0.0f,  0.5f, 0.5f,  -0.5f, 1.0f };
        const float Be[ k * n ] = { 0.5f, 0.0f, 0.5f, 1.0f,   1.0f, -0.5f, 0.5f, 0.5f };

        double Cx[ m * n ], aAB[ m * n ];
        ref_exact_f64( m, n, k, Ae, Be, Cx, aAB );

        std::vector<float> ref( m * n );
        bool exact_ok = true;
        for ( size_t i = 0; i < (size_t)m * n; ++i )
            if ( (double)(float)Cx[ i ] != Cx[ i ] ) exact_ok = false;   // sanity: oracle is fp16/fp32-exact

        std::vector<uint32_t> first_bits;
        int mi = 0;
        for ( const auto& mc : modes )
        {
            std::vector<float> Cm( m * n );
            lof_blis_set_accum( fp16_mant, lo_float::ProjSpec{ mc.mode } );
            run_blis( m, n, k, Ae, Be, Cm.data() );

            if ( mi == 0 ) {
                for ( float v : Cm ) {
                    uint32_t b; std::memcpy( &b, &v, 4 ); first_bits.push_back( b );
                }
                // also must equal the exact double result
                for ( size_t i = 0; i < (size_t)m * n; ++i )
                    if ( (double)Cm[ i ] != Cx[ i ] ) exact_ok = false;
            } else {
                for ( size_t i = 0; i < (size_t)m * n; ++i ) {
                    uint32_t b; std::memcpy( &b, &Cm[ i ], 4 );
                    if ( b != first_bits[ i ] ) exact_ok = false;
                }
            }
            ++mi;
        }
        printf( "== Part 3: exact-representable case, all modes bit-identical & exact: %s ==\n",
                exact_ok ? "ok" : "FAIL" );
        pass &= exact_ok;
    }

    lof_blis_set_accum( -1 );      // restore default
    printf( "%s\n", pass ? "PASS" : "FAIL" );
    return pass ? 0 : 1;
}
