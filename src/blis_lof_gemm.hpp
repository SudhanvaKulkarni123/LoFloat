#pragma once
// ============================================================================
//  blis_lof_gemm.hpp
//
//  Step 1 of the BLIS + xsimd custom-accumulation GEMM (the CPU analog of
//  cutlass_gemms.cuh).  This file provides:
//
//    1. lof_sgemm_ukr     — a *straightforward scalar* float GEMM micro-kernel
//                           written to the BLIS 2.0 micro-kernel contract.
//    2. lof_blis_install_kernels() — installs that kernel into the BLIS
//                           context so ordinary BLIS GEMM calls route through
//                           it.
//
//  This first cut is INTENTIONALLY plain fp32 (no LoFloat rounding yet): the
//  goal is to validate that the context override + packed-panel plumbing is
//  correct.  The single accumulation line flagged below ("LOFLOAT HOOK") is
//  where the round-to-odd FMA + virtual_round custom accumulation will go in
//  the next step — the CPU counterpart of thread::Mma<OpLoFMultiplyAdd> in
//  cutlass_gemms.cuh.
//
//  Requires a built BLIS (libblis + the generated umbrella header blis.h).
//  Include path after building the submodule:
//      third_party/blis/include/<config>/blis.h
// ============================================================================

#include "blis.h"   // BLIS umbrella header (handles extern "C" for C++)

#include "lo_float.h"   // lo_float::virtual_round, Rounding_Mode

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

// ----------------------------------------------------------------------------
//  Host round-to-odd FMA  (CPU counterpart of round_to_odd_fma in
//  cutlass_gemms.cuh, which uses the device-only __fmaf_rd/__fmaf_ru intrinsics)
//
//  Returns a*b + c rounded into float with round-to-odd (sticky-bit) semantics:
//  the round-toward-zero result, with its mantissa LSB forced to 1 whenever the
//  exact a*b + c is not representable. Feeding this into a subsequent re-round
//  to a lower-precision accumulator reproduces the result of rounding the EXACT
//  value once — no double-rounding error, for every rounding mode.
//
//  We avoid <cfenv> directed rounding (which needs -frounding-math / FENV_ACCESS
//  to be safe and is slow). Instead, exactly:
//    * rn  = std::fmaf(a,b,c)  is the correctly-rounded (RN) float of a*b+c.
//    * res = (a*b + c) - rn    is computed EXACTLY in double:
//        - p = a*b           is exact (24+24 <= 53 mantissa bits),
//        - two_sum(p, c)     captures p+c = s+e exactly,
//        - (s - rn) is Sterbenz-exact (rn is the nearest float, ratio ~1),
//        - res = (s - rn) + e is exact (the two terms are ~29 bits apart).
//    * res == 0  => a*b+c is float-exact, return rn.
//    * otherwise => round-to-odd is the neighbor of the exact value with an odd
//      mantissa LSB; the two neighbors are rn and nextafterf(rn, toward res),
//      and consecutive floats differ by exactly one ULP in bit pattern, so one
//      is odd and one is even. Pick the odd one.
//
//  Correct for all finite inputs; inf/nan inputs propagate through (handled by
//  the subsequent virtual_round / the format's saturation policy).
// ----------------------------------------------------------------------------
inline float lof_round_to_odd_fma( float a, float b, float c )
{
    const float rn = std::fmaf( a, b, c );           // correctly-rounded RN float
    if ( !std::isfinite( rn ) ) return rn;           // overflow/NaN: nothing to round

    const double p  = (double)a * (double)b;         // exact
    const double cd = (double)c;

    // two_sum(p, cd) -> s + e == p + cd exactly
    const double s  = p + cd;
    const double bb = s - p;
    const double e  = ( p - ( s - bb ) ) + ( cd - bb );

    const double res = ( s - (double)rn ) + e;       // exact residual (a*b+c) - rn
    if ( res == 0.0 ) return rn;                      // exactly representable

    uint32_t brn;
    std::memcpy( &brn, &rn, sizeof brn );
    if ( brn & 1u ) return rn;                        // rn is already the odd neighbor

    // Otherwise the round-to-odd result is the OTHER neighbour — one ULP from rn
    // toward the exact value. Done with an integer bit +/-1 (the well-known
    // nextafter trick) to avoid a per-element libm call. Bit pattern is
    // sign|magnitude, so magnitude grows with ++ and shrinks with -- (binade
    // boundaries handled automatically).
    if ( rn == 0.0f ) {
        brn = ( res > 0.0 ) ? 0x00000001u : 0x80000001u;   // smallest subnormal, odd LSB
    } else {
        const bool neg        = brn >> 31;
        const bool toward_pos = res > 0.0;
        if ( toward_pos == neg ) --brn;              // magnitude decreases
        else                     ++brn;              // magnitude increases
    }
    float other;
    std::memcpy( &other, &brn, sizeof other );
    return other;
}

// ----------------------------------------------------------------------------
//  Custom-accumulation parameters for the micro-kernel.
//
//  The BLIS gemm_ukr_ft signature is fixed, so these are smuggled in as
//  process-global state (the same deliberate global-override model the kernel
//  install already uses; see PROTOTYPE NOTE below). Set them ONCE before a GEMM
//  call. BLIS shares the global context read-only across worker threads and the
//  kernel only reads these, so this is safe for BLIS's threading model.
//
//  accum_mant_bits <  0  => DISABLED: plain fp32 FMA (original Step-1 behavior,
//                           so the existing smoke test is unaffected).
//  accum_mant_bits >= 0  => each rank-1 update accumulates in the target
//                           precision via:
//      acc = virtual_round( lof_round_to_odd_fma(a, b, acc),
//                           accum_mant_bits, ps );
//  (e.g. accum_mant_bits = 10 simulates an fp16 accumulator.)
// ----------------------------------------------------------------------------
inline int&  lof_blis_accum_mant_bits()
{ static int v = -1; return v; }
inline lo_float::ProjSpec& lof_blis_projspec()
{ static lo_float::ProjSpec v = lo_float::ProjSpec{}; return v; }

//  Convenience setter. accum_mant_bits < 0 restores the plain-fp32 path.
inline void lof_blis_set_accum(
    int accum_mant_bits,
    lo_float::ProjSpec ps = lo_float::ProjSpec{} )
{
    lof_blis_accum_mant_bits() = accum_mant_bits;
    lof_blis_projspec()        = ps;
}

// ----------------------------------------------------------------------------
//  Scalar float GEMM micro-kernel  (BLIS 2.0 gemm_ukr_ft signature)
//
//  Computes the micro-tile:   C := beta * C + alpha * A_panel * B_panel
//
//  A_panel : packed MR x k micro-panel  (column-major within the panel)
//  B_panel : packed  k x NR micro-panel (row-major within the panel)
//  m, n    : the *live* tile dimensions (m <= MR, n <= NR at array edges)
//  k       : the current KC-panel length
//
//  Strides are queried from the active context rather than hardcoded, so the
//  kernel is correct for whatever MR/NR/broadcast factors the selected
//  sub-config declares (the packing has already been done with those values
//  before we are called).
// ----------------------------------------------------------------------------
inline void lof_sgemm_ukr(
          dim_t            m,
          dim_t            n,
          dim_t            k,
    const void*            alpha0,
    const void*            a0,
    const void*            b0,
    const void*            beta0,
          void*            c0,
          inc_t            rs_c,
          inc_t            cs_c,
    const auxinfo_t*       data,
    const cntx_t*          cntx )
{
    (void)data;

    const float* alpha = static_cast<const float*>( alpha0 );
    const float* a     = static_cast<const float*>( a0 );
    const float* b     = static_cast<const float*>( b0 );
    const float* beta  = static_cast<const float*>( beta0 );
          float* c     = static_cast<float*>( c0 );

    // -- Packed micro-panel strides (mirror ref_kernels/3/bli_gemm_ref.c) ----
    const inc_t packmr = bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_MR, cntx );
    const inc_t packnr = bli_cntx_get_blksz_max_dt( BLIS_FLOAT, BLIS_NR, cntx );

    const inc_t rs_a = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_BBM, cntx ); // broadcast factor (1 for real float)
    const inc_t cs_a = packmr;
    const inc_t rs_b = packnr;
    const inc_t cs_b = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_BBN, cntx );

    // -- Local MR x NR accumulator, column-major (rs_ab = 1, cs_ab = m) ------
    //    Same fixed stack buffer the reference kernel uses; it is sized for the
    //    largest possible micro-tile, so m*n always fits.
    float ab[ BLIS_STACK_BUF_MAX_SIZE / sizeof( float ) ]
        __attribute__(( aligned( BLIS_STACK_BUF_ALIGN_SIZE ) ));
    const inc_t rs_ab = 1;
    const inc_t cs_ab = m;

    for ( dim_t i = 0; i < m * n; ++i )
        ab[ i ] = 0.0f;

    // -- Custom-accumulation parameters (read once; see lof_blis_set_accum) ---
    const int                accum_bits = lof_blis_accum_mant_bits();
    const lo_float::ProjSpec ps         = lof_blis_projspec();
    const bool               custom     = ( accum_bits >= 0 );

    // -- k rank-1 updates ----------------------------------------------------
    //  The custom-vs-plain decision is hoisted OUT of the hot loop into two
    //  separate nests: the plain-fp32 inner loop is then a branch-free SAXPY the
    //  compiler can vectorize, and the (heavy) custom path doesn't perturb its
    //  register allocation.
#ifdef LOF_BLIS_POISON
    // Self-check: build with -DLOF_BLIS_POISON to confirm BLIS is actually
    // routing through this kernel. Results should then be wrong (test FAILs);
    // without the macro they must be correct.
    for ( dim_t l = 0; l < k; ++l )
        for ( dim_t j = 0; j < n; ++j )
        {
            const float b_lj = b[ l * rs_b + j * cs_b ];
            for ( dim_t i = 0; i < m; ++i )
                ab[ i * rs_ab + j * cs_ab ] += 2.0f * a[ i * rs_a + l * cs_a ] * b_lj;
        }
#else
    if ( custom )
    {
        // Round-to-odd FMA into fp32, then re-round the accumulator into the
        // target precision with the chosen mode — a single effective rounding
        // per step (no double-rounding). CPU counterpart of
        // thread::Mma<OpLoFMultiplyAdd>.
        for ( dim_t l = 0; l < k; ++l )
            for ( dim_t j = 0; j < n; ++j )
            {
                const float b_lj = b[ l * rs_b + j * cs_b ];
                for ( dim_t i = 0; i < m; ++i )
                {
                    float& acc = ab[ i * rs_ab + j * cs_ab ];
                    acc = (float)lo_float::virtual_round(
                              lof_round_to_odd_fma( a[ i * rs_a + l * cs_a ], b_lj, acc ),
                              accum_bits, ps );
                }
            }
    }
    else
    {
        // Plain fp32 (accumulation disabled) — original Step-1 behavior.
        for ( dim_t l = 0; l < k; ++l )
            for ( dim_t j = 0; j < n; ++j )
            {
                const float b_lj = b[ l * rs_b + j * cs_b ];
                for ( dim_t i = 0; i < m; ++i )
                    ab[ i * rs_ab + j * cs_ab ] += a[ i * rs_a + l * cs_a ] * b_lj;
            }
    }
#endif

    // -- Scale by alpha ------------------------------------------------------
    for ( dim_t i = 0; i < m * n; ++i )
        ab[ i ] *= *alpha;

    // -- Write back to C -----------------------------------------------------
    if ( *beta == 0.0f )
    {
        for ( dim_t j = 0; j < n; ++j )
            for ( dim_t i = 0; i < m; ++i )
                c[ i * rs_c + j * cs_c ] = ab[ i * rs_ab + j * cs_ab ];
    }
    else
    {
        for ( dim_t j = 0; j < n; ++j )
            for ( dim_t i = 0; i < m; ++i )
                c[ i * rs_c + j * cs_c ] =
                    (*beta) * c[ i * rs_c + j * cs_c ] +
                    ab[ i * rs_ab + j * cs_ab ];
    }
}

// ----------------------------------------------------------------------------
//  Install lof_sgemm_ukr as the float GEMM micro-kernel.
//
//  PROTOTYPE NOTE: this overrides the *process-global* native float gemm
//  micro-kernel in place.  BLIS 2.0 stores the context's kernel tables in
//  heap-backed stacks (stck_t holds `void** blocks` + a mutex), and there is
//  no bli_cntx_dup(), so a shallow `cntx_t` copy would alias and clobber the
//  global anyway.  A deliberate global override is the simplest correct option
//  for a single-purpose extension where every GEMM should use this kernel.
//  Call it once, before any GEMM, and before spawning worker threads (BLIS
//  shares the global context read-only across threads).
//
//  We change ONLY the ukr function pointer — not BLIS_MR/BLIS_NR/BBM/BBN or the
//  ukr row preference — so packing and the macro-kernel call convention are
//  unchanged, and this scalar kernel reads exactly what BLIS packed.
// ----------------------------------------------------------------------------
inline void lof_blis_install_kernels()
{
    bli_init();

    // The gks returns a const* to the live global context; we intentionally
    // mutate it (see PROTOTYPE NOTE above).
    cntx_t* cntx = const_cast<cntx_t*>( bli_gks_query_cntx() );

    bli_cntx_set_ukrs(
        cntx,
        BLIS_GEMM_UKR, BLIS_FLOAT, reinterpret_cast<void_fp>( lof_sgemm_ukr ),
        BLIS_VA_END );

    // Force the conventional (packed) GEMM path. The typed front-ends copy the
    // global runtime on each call (bli_rntm_init_from_global), so disabling the
    // small/unpacked "SUP" path here guarantees BLIS routes through our packed
    // BLIS_GEMM_UKR rather than the separate gemmsup kernels — otherwise small
    // or skinny problems would silently bypass this kernel.
    bli_rntm_disable_l3_sup( bli_global_rntm() );
}
