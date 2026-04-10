#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>


#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── CUTLASS ─────────────────────────────────────────────────────────────────
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/functional.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"


// We need these headers BEFORE our specializations
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"

#include "lo_float.h"
#include "layouts.h"
#include "Matrix.h"

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){             \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
  exit(1);}}while(0)
#define CUBLAS_CHECK(x) do{cublasStatus_t s=(x);if(s!=CUBLAS_STATUS_SUCCESS){\
  fprintf(stderr,"cuBLAS %s:%d err %d\n",__FILE__,__LINE__,(int)s);         \
  exit(1);}}while(0)

// ═══════════════════════════════════════════════════════════════════════════
//  Step 1: Custom op tag
// ═══════════════════════════════════════════════════════════════════════════
namespace cutlass { namespace arch {
struct OpLoFMultiplyAdd {};
}} // namespace cutlass::arch



namespace cutlass {
namespace gemm {
namespace thread {

template <typename Shape_, typename LayoutA_, typename LayoutB_, typename Enable>
struct Mma<Shape_, float, LayoutA_, float, LayoutB_,
           float, layout::RowMajor,
           arch::OpLoFMultiplyAdd, Enable> {

    using Shape = Shape_;
    using Operator = arch::OpLoFMultiplyAdd;
    using ElementC = float;


    // Required nested type — warp-level code queries ThreadMma::ArchMmaOperator
    using ArchMmaOperator = arch::Mma<
        gemm::GemmShape<1,1,1>, 1,
        float, LayoutA_, float, LayoutB_, float, layout::RowMajor,
        arch::OpMultiplyAdd>;

    using FragmentA = Array<float, Shape::kMK>;
    using FragmentB = Array<float, Shape::kKN>;
    using FragmentC = Array<float, Shape::kMN>;

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A,
                    FragmentB const& B, FragmentC const& C, int accum_mant_bits, lo_float::Rounding_Mode rounding_mode, int stochastic_bits) const {
        D = C;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Shape::kK; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; n++) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Shape::kM; m++) {
            float p = A[m * Shape::kK + k] * B[n * Shape::kK + k];
            D[m * Shape::kN + n] +=  lo_float::virtual_round(p, accum_mant_bits, rounding_mode, stochastic_bits);
            }
        }
        }
    }
    };


}  // namespace thread
}  // namespace gemm
}  // namespace cutlass

// ═══════════════════════════════════════════════════════════════════════════
//  Step 3: LoFMma — warp-level MMA that uses OpLoFMultiplyAdd
//
//  warp::MmaSimt hardcodes OpMultiplyAdd in its ThreadMma typedef.
//  We create a near-copy that differs only in the operator tag passed
//  to thread::Mma.
// ═══════════════════════════════════════════════════════════════════════════
namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape_,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename Policy_,
  int PartitionsK = 1,
  ComplexTransform TransformA = ComplexTransform::kNone,
  ComplexTransform TransformB = ComplexTransform::kNone,
  typename Enable = bool
>
class LoFMma {
public:
  using Shape = Shape_;
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<4, 8>,   // WarpShape in threads (M, N)
    cutlass::layout::RowMajorInterleaved<2>,
    cutlass::gemm::GemmShape<4, 1, 1>  // LaneMmaShape
>;
  using OperatorClass = arch::OpClassSimt;
  using ArchTag = arch::Sm80;

  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;

  using ThreadLayoutA = typename platform::conditional<
    platform::is_same< layout::ColumnMajorInterleaved<4>, LayoutA >::value,
    layout::ColumnMajor,
    typename platform::conditional<
      platform::is_same< layout::RowMajorInterleaved<4>, LayoutA >::value,
      layout::RowMajor,
      LayoutA>::type
  >::type;

  using ThreadLayoutB = typename platform::conditional<
    platform::is_same< layout::ColumnMajorInterleaved<4>, LayoutB >::value,
    layout::ColumnMajor,
    typename platform::conditional<
      platform::is_same< layout::RowMajorInterleaved<4>, LayoutB >::value,
      layout::RowMajor,
      LayoutB>::type
  >::type;

  static constexpr bool use_dp4a = false;
  using dp4a_type = bool;

  using ThreadMma = thread::Mma<
    GemmShape<
      Shape::kM / Policy::WarpShape::kRow,
      Shape::kN / Policy::WarpShape::kColumn,
      Policy::LaneMmaShape::kK>,
    ElementA,
    ThreadLayoutA,
    ElementB,
    ThreadLayoutB,
    ElementC,
    LayoutC,
    arch::OpLoFMultiplyAdd
  >;

  using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  using InstructionShape = GemmShape<1,1,1>;

  using IteratorA = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    Operand::kA, ElementA, LayoutA, Policy, PartitionsK, Shape::kK
  >;
  using FragmentA = typename IteratorA::Fragment;
  using TransformedFragmentA = FragmentA;

  using IteratorB = MmaSimtTileIterator<
    MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
    Operand::kB, ElementB, LayoutB, Policy, PartitionsK, Shape::kK
  >;
  using FragmentB = typename IteratorB::Fragment;
  using TransformedFragmentB = FragmentB;

  using IteratorC = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    Operand::kC, ElementC, LayoutC, Policy
  >;
  using FragmentC = typename ThreadMma::FragmentC;

  int accum_mant_bits;
  lo_float::Rounding_Mode rounding_mode;
  int stochastic_bits;


  CUTLASS_DEVICE
  LoFMma(int accum_mant_bits, lo_float::Rounding_Mode rounding_mode, int stochastic_bits) : accum_mant_bits(accum_mant_bits), rounding_mode(rounding_mode), stochastic_bits(stochastic_bits) {}

  CUTLASS_HOST_DEVICE void set_accum_bits(int bits) { accum_mant_bits = bits; }
  CUTLASS_HOST_DEVICE void set_rounding_mode(lo_float::Rounding_Mode mode) { rounding_mode = mode; }
  CUTLASS_HOST_DEVICE void set_stochastic_bits(int bits) { stochastic_bits = bits; }
  CUTLASS_HOST_DEVICE int get_accum_bits() const { return accum_mant_bits; }
  CUTLASS_HOST_DEVICE lo_float::Rounding_Mode get_rounding_mode() const { return rounding_mode; }
  CUTLASS_HOST_DEVICE int get_stochastic_bits() const { return stochastic_bits; }
  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, FragmentA a, FragmentB b,
    FragmentC const &c, int group_idx = 0) const {
    ThreadMma mma;
    mma(d, a, b, c, accum_mant_bits, rounding_mode, stochastic_bits);
  }

  CUTLASS_DEVICE
  TransformedFragmentA transform_A(FragmentA const &a) const { return a; }

  CUTLASS_DEVICE
  TransformedFragmentB transform_B(FragmentB const &b) const { return b; }

  /// Combined transform (required by MmaMultistage)
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    dst_A = A;
    dst_B = B;
  }
};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

namespace cutlass {
namespace gemm {
namespace threadblock {

/// 2-stage specialization
template <
  typename Shape_,
  typename WarpShape_,
  typename LayoutC_>
struct DefaultMmaCore<
    Shape_, WarpShape_, gemm::GemmShape<1,1,1>,
    float, layout::RowMajor,
    float, layout::ColumnMajor,
    float, LayoutC_,
    arch::OpClassSimt, 2,
    arch::OpLoFMultiplyAdd,
    false,
    cutlass::arch::CacheOperation::Global,
    cutlass::arch::CacheOperation::Global,
    ComplexTransform::kNone,
    ComplexTransform::kNone,
    false> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = gemm::GemmShape<1,1,1>;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const kStages = 2;
  using Operator = arch::OpLoFMultiplyAdd;

  using Base = DefaultMmaCore<
    Shape_, WarpShape_, gemm::GemmShape<1,1,1>,
    float, layout::RowMajor,
    float, layout::ColumnMajor,
    float, LayoutC_,
    arch::OpClassSimt, 2,
    arch::OpLoFMultiplyAdd>;

  using WarpCount = typename Base::WarpCount;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;
  using SmemIteratorA = typename Base::SmemIteratorA;
  using SmemIteratorB = typename Base::SmemIteratorB;
  static int const kThreads = Base::kThreads;

  using MmaWarpSimt = warp::LoFMma<
    WarpShape, ElementA, SmemLayoutA,
    ElementB, SmemLayoutB, ElementC, LayoutC,
    typename Base::MmaWarpSimt::Policy
  >;

  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass


template <
  typename Shape_, typename WarpShape_, typename InstructionShape_,
  typename ElementA_, typename LayoutA_,
  typename ElementB_, typename LayoutB_,
  typename ElementC_, typename LayoutC_,
  int Stages>
struct LoFMmaCore {
  using Base = cutlass::gemm::threadblock::DefaultMmaCore<
    Shape_, WarpShape_, InstructionShape_,
    ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_,
    cutlass::arch::OpClassSimt, Stages, cutlass::arch::OpMultiplyAdd>;

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using WarpCount = typename Base::WarpCount;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;
  using SmemIteratorA = typename Base::SmemIteratorA;
  using SmemIteratorB = typename Base::SmemIteratorB;
  static int const kThreads = Base::kThreads;

  using MmaWarpSimt = cutlass::gemm::warp::LoFMma<
    WarpShape, ElementA_, SmemLayoutA,
    ElementB_, SmemLayoutB, ElementC_, LayoutC_,
    typename Base::MmaWarpSimt::Policy>;

  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK>;

};


// ═══════════════════════════════════════════════════════════════════════════
//  GEMM runners
// ═══════════════════════════════════════════════════════════════════════════

// Stages == 2: device::Gemm works (our DefaultMmaCore spec is picked up)
template<int TbM,int TbN,int TbK, int WpM,int WpN,int WpK>
float run_lof_gemm_2stg(int M,int N,int K,float alpha,float beta,
    const float* dA,const float* dB,const float* dC,float* dD,int reps=20)
{
  using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TbM,TbN,TbK>,
    cutlass::gemm::GemmShape<WpM,WpN,WpK>,
    cutlass::gemm::GemmShape<1,1,1>,
    cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 1, 1, false,
    cutlass::arch::OpLoFMultiplyAdd>;

  typename Gemm::Arguments args({M,N,K},{dA,K},{dB,K},{dC,N},{dD,N},{alpha,beta});
  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.f;
  size_t ws = Gemm::get_workspace_size(args);
  void* dw = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&dw, ws));
  if (op.initialize(args, dw) != cutlass::Status::kSuccess){ if(dw)cudaFree(dw); return -1.f; }
  op(); CUDA_CHECK(cudaDeviceSynchronize());
  // Timer t; t.start();
  // for(int i=0;i<reps;i++) op();
  // float ms = t.stop() / reps;
  // if(dw) cudaFree(dw);
  return 0.0f;
}




namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    typename Shape_,
    typename WarpShape_,
    typename LayoutC_,
    int Stages,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<
    Shape_, WarpShape_, gemm::GemmShape<1,1,1>,
    float, layout::RowMajor,
    float, layout::ColumnMajor,
    float, LayoutC_,
    arch::OpClassSimt, Stages,
    arch::OpLoFMultiplyAdd,
    false, CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = gemm::GemmShape<1,1,1>;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const kStages = Stages;
  using Operator = arch::OpLoFMultiplyAdd;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA =
      cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB =
      cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<
      Shape::kM / WarpShape::kM,
      Shape::kN / WarpShape::kN,
      Shape::kK / WarpShape::kK>;

  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts (same as upstream SM80 SIMT RowMajor×ColumnMajor)
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>,
      kThreads,
      kElementsPerAccess>;

  using SmemThreadMapA =
      transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>,
      kThreads,
      kElementsPerAccess>;

  using SmemThreadMapB =
      transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator — LoFMma instead of MmaSimt
  //

  static const int WarpNumThreadsM = 4;
  static const int WarpNumThreadsN = 8;
  static_assert(
      !(WarpShape::kM % WarpNumThreadsM) &&
      !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout =
      ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(
      !((Shape::kK / 32) % LaneM) && !((Shape::kK / 32) % LaneN),
      "Padding must be divisible by Lane");

  using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,
      cutlass::layout::RowMajorInterleaved<LaneLayout>,
      LaneMmaShape>;

  // ** This is the key difference: LoFMma instead of MmaSimt **
  using MmaWarpSimt = warp::LoFMma<
      WarpShape,
      ElementA, SmemLayoutA,
      ElementB, SmemLayoutB,
      ElementC, LayoutC,
      Policy>;

  using MmaPolicy = MmaPolicy<
      MmaWarpSimt,
      MatrixShape<Shape::kK / 32, 0>,
      MatrixShape<0, Shape::kK / 32>,
      WarpCount::kK>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass


// ═══════════════════════════════════════════════════════════════════════════
//  Multistage GEMM runner
// ═══════════════════════════════════════════════════════════════════════════

template<int TbM, int TbN, int TbK,
         int WpM, int WpN, int WpK,
         int Stages>
float run_lof_gemm_multistage(
    int M, int N, int K,
    float alpha, float beta,
    const float* dA, const float* dB, const float* dC, float* dD,
    int accum_mant_bits = 0,
    lo_float::Rounding_Mode rounding_mode = lo_float::Rounding_Mode::RoundToNearestEven,
    int stochastic_rounding_bits = 0,
    int reps = 20)
{
  using Gemm = cutlass::gemm::device::Gemm<
      float, cutlass::layout::RowMajor,       // A
      float, cutlass::layout::ColumnMajor,     // B
      float, cutlass::layout::RowMajor,        // C/D
      float,                                    // Accumulator
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TbM, TbN, TbK>,
      cutlass::gemm::GemmShape<WpM, WpN, WpK>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages,                                   // <-- multistage!
      1, 1, false,
      cutlass::arch::OpLoFMultiplyAdd>;

  typename Gemm::Arguments args(
      {M, N, K},
      {dA, K}, {dB, K}, {dC, N}, {dD, N},
      {alpha, beta},
      1,             // split_k_slices
      nullptr,       // gather_A_indices
      nullptr,       // gather_B_indices
      nullptr,       // scatter_D_indices
      accum_mant_bits,
      rounding_mode,
      stochastic_rounding_bits);

  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.f;

  size_t ws = Gemm::get_workspace_size(args);
  void* dw = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&dw, ws));

  if (op.initialize(args, dw) != cutlass::Status::kSuccess) {
    if (dw) cudaFree(dw);
    return -1.f;
  }

  // Warmup
  op();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
  }

  // Timer t;
  // t.start();
  // for (int i = 0; i < reps; i++) op();
  // float ms = t.stop() / reps;

  if (dw) cudaFree(dw);
  return 0.0;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Example usage
// ═══════════════════════════════════════════════════════════════════════════
//
//  // 3-stage multistage, 128x128x8 threadblock, 32x64x8 warp
//  float ms = run_lof_gemm_multistage<128, 128, 8, 32, 64, 8, 3>(
//      M, N, K, 1.0f, 0.0f, dA, dB, dC, dD,
//      /*accum_mant_bits=*/10,
//      lo_float::Rounding_Mode::RoundToNearestEven,
//      /*stochastic_rounding_bits=*/0);
//
//  // 4-stage for even deeper pipelining (uses more shared memory)
//  float ms = run_lof_gemm_multistage<128, 128, 8, 32, 64, 8, 4>(
//      M, N, K, 1.0f, 0.0f, dA, dB, dC, dD,
//      /*accum_mant_bits=*/10,
//      lo_float::Rounding_Mode::Stochastic,
//      /*stochastic_rounding_bits=*/8);
//




#pragma once

#include <type_traits>

namespace lo_float {

// ─────────────────────────────────────────────────────────────────────────
//  Layout mapping: lo_float::Layout → cutlass::layout
// ─────────────────────────────────────────────────────────────────────────

namespace detail {

template <Layout L>
struct LayoutToCutlass;

template <>
struct LayoutToCutlass<ColMajor> {
  using type = cutlass::layout::ColumnMajor;
};

template <>
struct LayoutToCutlass<RowMajor> {
  using type = cutlass::layout::RowMajor;
};

template <Layout L>
using CutlassLayout = typename LayoutToCutlass<L>::type;

}  // namespace detail


// ─────────────────────────────────────────────────────────────────────────
//  lo_float::Gemm
//
//  Computes:  D = alpha * A * B + beta * C
//
//  Template parameters:
//    MatrixA, MatrixB, MatrixC, MatrixD  — lo_float::Matrix types
//      (carry element type and layout; accum type is always ElementC)
//    TbM, TbN, TbK    — Threadblock tile shape
//    WpM, WpN, WpK    — Warp tile shape
//    Stages            — Pipeline depth (2 = double-buffered,
//                        3+ = multistage with cp.async, requires SM80+)
//    SplitKSerial      — Enable split-K with serial reduction
//    Swizzle           — Threadblock swizzle functor
//
//  Runtime parameters (constructor):
//    accum_mantissa_bits, rounding_mode, stochastic_rounding_bits
//      — control lo_float accumulation rounding, independent of A/B/C/D
//
//  Always uses:  OpClassSimt, Sm80, OpLoFMultiplyAdd, InstructionShape<1,1,1>
// ─────────────────────────────────────────────────────────────────────────

template <
    typename MatrixA_,
    typename MatrixB_,
    typename MatrixC_,
    typename MatrixD_,
    int TbM = 128,
    int TbN = 128,
    int TbK = 8,
    int WpM = 32,
    int WpN = 64,
    int WpK = 8,
    int Stages = 3,
    bool SplitKSerial = false,
    typename Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
class Gemm {
public:

  // ── Matrix types ──────────────────────────────────────────────────────

  using MatrixA = MatrixA_;
  using MatrixB = MatrixB_;
  using MatrixC = MatrixC_;
  using MatrixD = MatrixD_;

  // ── Element types extracted from matrices ─────────────────────────────

  using ElementA = typename MatrixA::scalar_type;
  using ElementB = typename MatrixB::scalar_type;
  using ElementC = typename MatrixC::scalar_type;
  using ElementD = typename MatrixD::scalar_type;

  // ── CUTLASS layout types ──────────────────────────────────────────────

  using LayoutA = detail::CutlassLayout<MatrixA::layout>;
  using LayoutB = detail::CutlassLayout<MatrixB::layout>;
  using LayoutC = detail::CutlassLayout<MatrixC::layout>;
  using LayoutD = detail::CutlassLayout<MatrixD::layout>;


  static_assert(std::is_same<LayoutC, LayoutD>::value,
      "lo_float::Gemm requires C and D to share the same layout "
      "(CUTLASS 2.x constraint).");
  static_assert(std::is_same<ElementC, ElementD>::value,
      "lo_float::Gemm requires C and D to share the same element type "
      "(CUTLASS 2.x constraint).");

  // ── Compile-time constants ────────────────────────────────────────────

  using ThreadblockShape = cutlass::gemm::GemmShape<TbM, TbN, TbK>;
  using WarpShape        = cutlass::gemm::GemmShape<WpM, WpN, WpK>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  static constexpr int kStages        = Stages;
  static constexpr bool kSplitKSerial = SplitKSerial;

  // ── Underlying CUTLASS GEMM ───────────────────────────────────────────

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,   // output element type
      1,          // elements per access
      ElementC,   // accumulator type
      ElementC>;  // compute type

  using CutlassGemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementC,                          // accumulator
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      Swizzle,
      kStages,
      1,                                 // AlignmentA
      1,                                 // AlignmentB
      kSplitKSerial,
      cutlass::arch::OpLoFMultiplyAdd>;

  using Arguments = typename CutlassGemm::Arguments;

  // ── Status codes ──────────────────────────────────────────────────────

  enum class Status {
    kSuccess             =  0,
    kErrorNotSupported   = -1,
    kErrorAllocation     = -2,
    kErrorInitialize     = -3,
    kErrorExecution      = -4
  };

private:

  int          accum_mantissa_bits_;
  Rounding_Mode rounding_mode_;
  int          stochastic_rounding_bits_;

public:

  // ── Construction ──────────────────────────────────────────────────────

  Gemm(int accum_mantissa_bits         = 23,
       Rounding_Mode rounding_mode      = Rounding_Mode::RoundToNearestEven,
       int stochastic_rounding_bits    = 0)
    : accum_mantissa_bits_(accum_mantissa_bits)
    , rounding_mode_(rounding_mode)
    , stochastic_rounding_bits_(stochastic_rounding_bits)
  {}

  // ── Accessors ─────────────────────────────────────────────────────────

  void set_accum_mantissa_bits(int bits)        { accum_mantissa_bits_    = bits; }
  void set_rounding_mode(Rounding_Mode mode)     { rounding_mode_         = mode; }
  void set_stochastic_rounding_bits(int bits)   { stochastic_rounding_bits_ = bits; }

  int          get_accum_mantissa_bits()    const { return accum_mantissa_bits_; }
  Rounding_Mode get_rounding_mode()          const { return rounding_mode_; }
  int          get_stochastic_rounding_bits() const { return stochastic_rounding_bits_; }

  // ── Query ─────────────────────────────────────────────────────────────

  /// Check whether the problem configuration is supported.
  Status can_implement(
      MatrixA const& A, MatrixB const& B,
      MatrixC const& C, MatrixD const& D) const
  {
    auto args = make_arguments(1, 0, A, B, C, D, 1);
    if (CutlassGemm::can_implement(args) != cutlass::Status::kSuccess)
      return Status::kErrorNotSupported;
    return Status::kSuccess;
  }

  /// Return workspace bytes needed for a given split-K count.
  static size_t get_workspace_size(
      MatrixA const& A, MatrixB const& B,
      MatrixC const& C, MatrixD const& D,
      int split_k_slices = 1)
  {
    Arguments args({(int)A.m, (int)B.n, (int)A.n},
                   {A.data, (int)A.ld}, {B.data, (int)B.ld},
                   {C.data, (int)C.ld}, {D.data, (int)D.ld},
                   {1, 0}, split_k_slices);
    return CutlassGemm::get_workspace_size(args);
  }

  // ── Execute ───────────────────────────────────────────────────────────

  /// Run D = alpha * A * B + beta * C (single execution, no timing).
  Status operator()(
      ElementC alpha, ElementC beta,
      MatrixA const& A, MatrixB const& B,
      MatrixC const& C, MatrixD&       D,
      int split_k_slices       = 1,
      cudaStream_t stream      = nullptr) const
  {
    auto args = make_arguments(alpha, beta, A, B, C, D, split_k_slices);

    CutlassGemm op;

    if (CutlassGemm::can_implement(args) != cutlass::Status::kSuccess)
      return Status::kErrorNotSupported;

    size_t ws_bytes = CutlassGemm::get_workspace_size(args);
    void*  workspace = nullptr;
    if (ws_bytes) {
      if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess)
        return Status::kErrorAllocation;
    }

    if (op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      if (workspace) cudaFree(workspace);
      return Status::kErrorInitialize;
    }

    cutlass::Status s = op(stream);
    if (workspace) cudaFree(workspace);

    return (s == cutlass::Status::kSuccess)
        ? Status::kSuccess
        : Status::kErrorExecution;
  }

  /// Run the GEMM and return average kernel time in milliseconds.
  /// Returns a negative value on error (cast of Status).
  float benchmark(
      ElementC alpha, ElementC beta,
      MatrixA const& A, MatrixB const& B,
      MatrixC const& C, MatrixD&       D,
      int split_k_slices       = 1,
      int reps                 = 20,
      cudaStream_t stream      = nullptr) const
  {
    auto args = make_arguments(alpha, beta, A, B, C, D, split_k_slices);

    CutlassGemm op;

    if (CutlassGemm::can_implement(args) != cutlass::Status::kSuccess)
      return static_cast<float>(Status::kErrorNotSupported);

    size_t ws_bytes = CutlassGemm::get_workspace_size(args);
    void*  workspace = nullptr;
    if (ws_bytes) {
      if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess)
        return static_cast<float>(Status::kErrorAllocation);
    }

    if (op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      if (workspace) cudaFree(workspace);
      return static_cast<float>(Status::kErrorInitialize);
    }

    // Warmup
    op(stream);
    cudaDeviceSynchronize();

    // Timer t;
    // t.start();
    // for (int i = 0; i < reps; ++i) op(stream);
    // cudaDeviceSynchronize();
    // float ms = t.stop() / static_cast<float>(reps);

    if (workspace) cudaFree(workspace);
    return 0.0;
  }

private:

  // ── Helpers ───────────────────────────────────────────────────────────

  Arguments make_arguments(
      ElementC alpha, ElementC beta,
      MatrixA const& A, MatrixB const& B,
      MatrixC const& C, MatrixD const& D,
      int split_k_slices) const
  {
    int M = static_cast<int>(A.m);
    int N = static_cast<int>(B.n);
    int K = static_cast<int>(A.n);

    return Arguments(
        {M, N, K},
        {A.data, static_cast<int>(A.ld)},
        {B.data, static_cast<int>(B.ld)},
        {C.data, static_cast<int>(C.ld)},
        {D.data, static_cast<int>(D.ld)},
        {alpha, beta},
        split_k_slices,
        nullptr,                     // gather_A_indices
        nullptr,                     // gather_B_indices
        nullptr,                     // scatter_D_indices
        accum_mantissa_bits_,
        rounding_mode_,
        stochastic_rounding_bits_);
  }
};

}  // namespace lo_float
