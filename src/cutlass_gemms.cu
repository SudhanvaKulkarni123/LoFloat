#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "lo_float.h"
#include "layouts.h"
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
                    FragmentB const& B, FragmentC const& C, int accum_mant_bits, lo_float::RoundingMode rounding_mode, int stochastic_bits) const {
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
  lo_float::RoundingMode rounding_mode;
  int stochastic_bits;


  CUTLASS_DEVICE
  LoFMma(int accum_mant_bits, lo_float::RoundingMode rounding_mode, int stochastic_bits) : accum_mant_bits(accum_mant_bits), rounding_mode(rounding_mode), stochastic_bits(stochastic_bits) {}

  CUTLASS_HOST_DEVICE void set_accum_bits(int bits) { accum_mant_bits = bits; }
  CUTLASS_HOST_DEVICE void set_rounding_mode(lo_float::RoundingMode mode) { rounding_mode = mode; }
  CUTLASS_HOST_DEVICE void set_stochastic_bits(int bits) { stochastic_bits = bits; }
  CUTLASS_HOST_DEVICE int get_accum_bits() const { return accum_mant_bits; }
  CUTLASS_HOST_DEVICE lo_float::RoundingMode get_rounding_mode() const { return rounding_mode; }
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
  Timer t; t.start();
  for(int i=0;i<reps;i++) op();
  float ms = t.stop() / reps;
  if(dw) cudaFree(dw);
  return ms;
}



