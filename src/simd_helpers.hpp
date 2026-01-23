#include <xsimd/xsimd.hpp>
#include <type_traits>
#include <cstdint>
#include <algorithm>

using namespace xsimd;
namespace xs = xsimd;

template <class T, class Arch>
T first_lane(const xs::batch<T, Arch>& v)
{
    T buf[xs::batch<T, Arch>::size];
    xs::store_unaligned(buf, v);
    return buf[0];
}

template <class DstScalar, class SrcScalar, class arch = xsimd::default_arch>
xsimd::batch<DstScalar, arch>
widen_low_lanes(xsimd::batch<SrcScalar, arch> const& src,
                DstScalar fill = DstScalar{0})
{
    static_assert(std::is_arithmetic_v<SrcScalar>, "SrcScalar must be arithmetic");
    static_assert(std::is_arithmetic_v<DstScalar>, "DstScalar must be arithmetic");

    using src_batch = xsimd::batch<SrcScalar, arch>;
    using dst_batch = xsimd::batch<DstScalar, arch>;

    constexpr int src_lanes = int(src_batch::size);
    constexpr int dst_lanes = int(dst_batch::size);
    constexpr int step      = (src_lanes < dst_lanes) ? src_lanes : dst_lanes;

    // Use a conservative alignment that is safe for both batches.
    // (xsimd batches generally support unaligned too, but aligned is fine with alignas(64).)
    alignas(64) SrcScalar src_buf[src_lanes];
    alignas(64) DstScalar dst_buf[dst_lanes];

    // Fill destination with `fill`
    for (int i = 0; i < dst_lanes; ++i)
        dst_buf[i] = fill;

    // Store src and convert low lanes
    src.store_aligned(src_buf);

    #if defined(__clang__) || defined(__GNUC__)
    #pragma unroll
    #endif
    for (int i = 0; i < step; ++i)
        dst_buf[i] = static_cast<DstScalar>(src_buf[i]);

    return dst_batch::load_aligned(dst_buf);
}
