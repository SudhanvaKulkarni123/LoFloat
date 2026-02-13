#pragma once

#include "platform_macros.h"

#if LOFLOAT_HAS_HALF && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace lo_float {

#if LOFLOAT_HAS_HALF && defined(__ARM_FEATURE_SVE)
inline void half_fma_sve(__fp16* LOFLOAT_RESTRICT out,
                         const __fp16 a,
                         const __fp16* LOFLOAT_RESTRICT x,
                         const __fp16* LOFLOAT_RESTRICT y,
                         int count) noexcept
{
    svbool_t pg = svptrue_b16();
    svfloat16_t va = svdup_f16(a);
    int i = 0;
    while (i < count) {
        svfloat16_t vx = svld1_f16(pg, &x[i]);
        svfloat16_t vy = svld1_f16(pg, &y[i]);
        svfloat16_t vr = svmla_f16_x(pg, vy, va, vx);
        svst1_f16(pg, &out[i], vr);
        i += svcnth();
    }
}
#endif

}