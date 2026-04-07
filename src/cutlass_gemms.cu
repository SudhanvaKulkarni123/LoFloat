/******************************************************************************
 * relu_mma_cutlass_bench.cu
 *
 * Custom SIMT MMA via CUTLASS:  C += max(0, A * B)
 *
 * Strategy:
 *   1. Define OpMultiplyAddRelu tag
 *   2. Specialize thread::Mma for OpMultiplyAddRelu (the inner triple-loop)
 *   3. Subclass warp::MmaSimt to override ThreadMma with OpMultiplyAddRelu
 *   4. Specialize DefaultMmaCore for OpMultiplyAddRelu to wire it all up
 *
 * Build:
 *   nvcc -std=c++17 -O3 -I ../cutlass/include -I ../cutlass/tools/util/include -arch=sm_80 -lcublas custom_gemm.cu -o custom_gemm
 ******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "lo_float.h"
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

// ═══════════════════════════════════════════════════════════════════════════
//  Step 2: thread::Mma specialization for OpLoFMultiplyAdd
//
//  This is the innermost compute: the per-thread triple loop that does
//  D[m,n] += max(0, A[m,k] * B[n,k])
//
//  Opt 1: uses fmaxf() instead of ternary to guarantee FMNMX instruction
// ═══════════════════════════════════════════════════════════════════════════

namespace lo_float
{
    namespace lo_float_internal {

            template <typename Shape_, typename LayoutA_, typename LayoutB_, typename Enable>
    struct LoF_Mma<Shape_, float, LayoutA_, float, LayoutB_,
            float, layout::RowMajor,
            arch::OpMultiplyAddRelu, Enable> {

    using Shape = Shape_;
    using Operator = arch::OpMultiplyAddRelu;
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
                    FragmentB const& B, FragmentC const& C) const {
        D = C;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Shape::kK; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; n++) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Shape::kM; m++) {
            float p = A[m * Shape::kK + k] * B[n * Shape::kK + k];
            unsigned int p_as_int = __float_as_uint(p);
            p_as_int = RoundMantissa(p_as_int);
            D[m * Shape::kN + n] += __uint_as_float(p_as_int);
            }
        }
        }
    }
    };

        template<MatrixType MatrixA, MatrixType MatrixB, MatrixType MatrixC, 
         typename TypeAccum1, MatrixType MatrixAccum2, 
         std::size_t MR = 4, std::size_t NR = 4, std::size_t KR = 2>
        class Gemm {



            }
        }
    
} // namespace lofloat

 

namespace cutlass {
namespace gemm {
namespace thread {

template <typename Shape_, typename LayoutA_, typename LayoutB_, typename Enable>
struct Mma<Shape_, float, LayoutA_, float, LayoutB_,
           float, layout::RowMajor,
           arch::OpMultiplyAddRelu, Enable> {

  using Shape = Shape_;
  using Operator = arch::OpMultiplyAddRelu;
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
                  FragmentB const& B, FragmentC const& C) const {
    D = C;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; n++) {
        CUTLASS_PRAGMA_UNROLL
         for (int m = 0; m < Shape::kM; m++) {
          float p = A[m * Shape::kK + k] * B[n * Shape::kK + k];
         unsigned int p_as_int = __float_as_uint(p);
        p_as_int = RoundMantissa(p_as_int);
        D[m * Shape::kN + n] += __uint_as_float(p_as_int);
        }
      }
    }
  }
};

}  // namespace thread
}  // namespace gemm
}  // namespace cutlass

// ═══════════════════════════════════════════════════════════════════════════
//  Step 3: MmaSimtRelu — warp-level MMA that uses OpMultiplyAddRelu
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
class MmaSimtRelu {
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

  // *** THE KEY DIFFERENCE: OpMultiplyAddRelu instead of OpMultiplyAdd ***
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
    arch::OpMultiplyAddRelu
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

  CUTLASS_DEVICE
  MmaSimtRelu() {}

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, FragmentA a, FragmentB b,
    FragmentC const &c, int group_idx = 0) const {
    ThreadMma mma;
    mma(d, a, b, c);
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

// ═══════════════════════════════════════════════════════════════════════════
//  Step 4: Specialize DefaultMmaCore for OpMultiplyAddRelu + SIMT
//
//  For this benchmark: A=RowMajor, B=ColumnMajor.
//  We delegate all the boring layout/iterator math to the standard
//  OpMultiplyAdd DefaultMmaCore, then swap in MmaSimtRelu.
// ═══════════════════════════════════════════════════════════════════════════
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
    arch::OpMultiplyAddRelu,
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
  using Operator = arch::OpMultiplyAddRelu;

  using Base = DefaultMmaCore<
    Shape_, WarpShape_, gemm::GemmShape<1,1,1>,
    float, layout::RowMajor,
    float, layout::ColumnMajor,
    float, LayoutC_,
    arch::OpClassSimt, 2,
    arch::OpMultiplyAdd>;

  using WarpCount = typename Base::WarpCount;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;
  using SmemIteratorA = typename Base::SmemIteratorA;
  using SmemIteratorB = typename Base::SmemIteratorB;
  static int const kThreads = Base::kThreads;

  using MmaWarpSimt = warp::MmaSimtRelu<
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

// ═══════════════════════════════════════════════════════════════════════════
//  Step 5: DefaultMmaCoreRelu — standalone helper for multistage path
//
//  The multistage DefaultMmaCore specialization in cutlass doesn't pick up
//  our OpMultiplyAddRelu. Instead of fighting partial specialization, we
//  create a standalone struct that wraps the standard OpMultiplyAdd core
//  and swaps in MmaSimtRelu. Used only for stg >= 3.
// ═══════════════════════════════════════════════════════════════════════════

// template <
//     /// Shape of threadblock-scoped matrix multiply operator (concept:
//     /// GemmShape)
//     typename Shape_,
//     /// Shape of warp-level matrix multiply operator (concept: GemmShape)
//     typename WarpShape_,
//     /// Data type of A operand
//     typename ElementA_,
//     /// Data type of B operand
//     typename ElementB_,
//     /// Data type of accumulator
//     typename ElementC_,
//     /// Layout of accumulator
//     typename LayoutC_,
//     /// Operation performed by GEMM
//     typename Operator_>
// struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
//                       layout::RowMajor, ElementB_, layout::ColumnMajor,
//                       ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
//                      > {
//   using Shape = Shape_;
//   using WarpShape = WarpShape_;
//   using InstructionShape = GemmShape<1, 1, 1>;
//   using ElementA = ElementA_;
//   using LayoutA = layout::RowMajor;
//   using ElementB = ElementB_;
//   using LayoutB = layout::ColumnMajor;
//   using ElementC = ElementC_;
//   using LayoutC = LayoutC_;
//   using OperatorClass = arch::OpClassSimt;
//   static int const PartitionsK = Shape::kK / WarpShape::kK;

//   /// Default Operator
//   using Operator = Operator_;

//   /// Number of warps present
//   using WarpCount = GemmShape<
//     Shape::kM / WarpShape::kM,
//     Shape::kN / WarpShape::kN,
//     PartitionsK
//   >;

//   // Divisility requirements
//   static_assert(
//     !(Shape::kM % WarpShape::kM) &&
//     !(Shape::kN % WarpShape::kN),
//     "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
//   );

//   /// Number of threads per warp
//   static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

//   /// Number of threads total
//   static int const kThreads = WarpCount::kCount * kWarpSize;
  
//   static int const kElementsPerAccess = 1;

//   //
//   // Shared memory layouts
//   //

//   using SmemLayoutA = layout::ColumnMajor;
//   using SmemLayoutB = layout::RowMajor;

//   //
//   // Iterators to write to shared memory
//   //

//   /// ThreadMap of iterator A
//   using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
//     layout::PitchLinearShape<Shape::kK, Shape::kM>,
//     kThreads,
//     kElementsPerAccess
//   >;

//   /// Transpose the ThreadMap of iterator A
//   using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

//   /// Shared memory iterator to A operand
//   using SmemIteratorA = transform::threadblock::RegularTileIterator<
//     MatrixShape<Shape::kM, Shape::kK>, 
//     ElementA, 
//     SmemLayoutA,
//     1,
//     SmemThreadMapA // was IteratorThreadMapA
//   >;

//   /// ThreadMap of iterator B
//   using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
//     layout::PitchLinearShape<Shape::kK, Shape::kN>,
//     kThreads,
//     kElementsPerAccess
//   >;

//   /// Transpose the ThreadMap of iterator A
//   using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

//   /// Shared memory iterator to B operand
//   using SmemIteratorB = transform::threadblock::RegularTileIterator<
//     MatrixShape<Shape::kK, Shape::kN>, 
//     ElementB, 
//     SmemLayoutB,
//     0,
//     SmemThreadMapB // was IteratorThreadMapA
//   >;

//   //
//   // Warp-level matrix multiply operator
//   //

//   // Define the warp-level op
//   static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
//   static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
//   static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
//   static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
//   static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
//       "WarpShape must be divisible by ThreadTile shape.");
//   static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
//   static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
//   static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
//   static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
//   static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

//   static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
//   static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

//   static_assert(!(kPaddingM % LaneM) && !(kPaddingN % LaneN),
//                 "Padding must be divisible by Lane");

//   // these should have max of thread tile also
//   using LaneMmaShape = cutlass::gemm::GemmShape<
//       LaneM,
//       LaneN,
//       1>;
//   using Policy = cutlass::gemm::warp::MmaSimtPolicy<
//       cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
//       cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
//       LaneMmaShape
//   >;

//   using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
//       WarpShape,      /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
//       ElementA,       /// Data type of A elements
//       SmemLayoutA,    /// Layout of A matrix (concept: MatrixLayout)
//       ElementB,       /// Data type of B elements
//       SmemLayoutB,    /// Layout of B matrix (concept: MatrixLayout)
//       ElementC,       /// Element type of C matrix
//       LayoutC,        /// Layout of C matrix (concept: MatrixLayout)
//       Policy          /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
//   >;

//   /// Policy used to define MmaPipelined 
//   using MmaPolicy = MmaPolicy<
//     MmaWarpSimt,
//     MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
//     MatrixShape<0, kPaddingN>,    // skew for B matrix to avoid SMEM bank conflicts
//     WarpCount::kK
//   >;
// };

template <
  typename Shape_, typename WarpShape_, typename InstructionShape_,
  typename ElementA_, typename LayoutA_,
  typename ElementB_, typename LayoutB_,
  typename ElementC_, typename LayoutC_,
  int Stages>
struct DefaultMmaCoreRelu {
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

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimtRelu<
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
//  Includes for manual multistage kernel assembly
// ═══════════════════════════════════════════════════════════════════════════
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

// ═══════════════════════════════════════════════════════════════════════════
//  GEMM runners
// ═══════════════════════════════════════════════════════════════════════════

// Stages == 2: device::Gemm works (our DefaultMmaCore spec is picked up)
template<int TbM,int TbN,int TbK, int WpM,int WpN,int WpK>
float run_relu_gemm_2stg(int M,int N,int K,float alpha,float beta,
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
    cutlass::arch::OpMultiplyAddRelu>;

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

// Stages >= 3: manual assembly using DefaultMmaCoreRelu + MmaMultistage
template<int TbM,int TbN,int TbK, int WpM,int WpN,int WpK, int Stg>
float run_relu_gemm_multistage(int M,int N,int K,float alpha,float beta,
    const float* dA,const float* dB,const float* dC,float* dD,int reps=20)
{
  using ThreadblockShape = cutlass::gemm::GemmShape<TbM,TbN,TbK>;
  using WarpShape = cutlass::gemm::GemmShape<WpM,WpN,WpK>;
  using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<float,1,float,float>;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using MmaCore = DefaultMmaCoreRelu<
    ThreadblockShape, WarpShape, InstructionShape,
    float, LayoutA, float, LayoutB, float, LayoutC, Stg>;

  using AccessTypeA = cutlass::Array<float, 1>;
  using AccessTypeB = cutlass::Array<float, 1>;

  using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
    float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, AccessTypeA>;

  using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
    float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, AccessTypeB>;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
    ThreadblockShape,
    IteratorA, typename MmaCore::SmemIteratorA,
    cutlass::arch::CacheOperation::Always,
    IteratorB, typename MmaCore::SmemIteratorB,
    cutlass::arch::CacheOperation::Always,
    float, LayoutC,
    typename MmaCore::MmaPolicy,
    Stg>;

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape, typename MmaCore::MmaWarpSimt, EpilogueOp,
    EpilogueOp::kCount>::Epilogue;

  using GemmKernel = cutlass::gemm::kernel::Gemm<
    ThreadblockMma, Epilogue, Swizzle, false>;

   static_assert(Stg == ThreadblockMma::kStages, "multistage path not selected!");


  // Direct launch — kernel::Gemm has its own Params struct, not Arguments.
  // We build Params manually and launch via cutlass::Kernel<>.
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Compute grid dimensions
  Swizzle swizzle;
  cutlass::gemm::GemmCoord tiled_shape(
    (M + TbM - 1) / TbM,
    (N + TbN - 1) / TbN,
    1);
  dim3 grid = swizzle.get_grid_shape(tiled_shape);
  dim3 block(MmaCore::kThreads, 1, 1);

  // Build kernel params
  using TensorRefA = typename GemmKernel::Mma::IteratorA::TensorRef;
  using TensorRefB = typename GemmKernel::Mma::IteratorB::TensorRef;
  using TensorRefC = typename GemmKernel::Epilogue::OutputTileIterator::TensorRef;
  using EpilogueParams = typename GemmKernel::Epilogue::OutputOp::Params;

  typename GemmKernel::Params params(
    problem_size,
    tiled_shape,
    TensorRefA((float*)dA, K),
    TensorRefB((float*)dB, K),
    TensorRefC((float*)dC, N),
    TensorRefC((float*)dD, N),
    EpilogueParams(alpha, beta)
  );

  // Check shared memory
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
      cutlass::Kernel<GemmKernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (result != cudaSuccess) return -1.f;
  }

  // Warmup
  cutlass::Kernel<GemmKernel><<<grid, block, smem_size>>>(params);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer t; t.start();
  for(int i=0;i<reps;i++)
    cutlass::Kernel<GemmKernel><<<grid, block, smem_size>>>(params);
  float ms = t.stop() / reps;
  return ms;
}

// No dispatch wrapper — call run_relu_gemm_2stg or run_relu_gemm_multistage
// directly from the macros below to avoid nvcc instantiating both branches.

template<int TbM,int TbN,int TbK, int WpM,int WpN,int WpK, int Stg>
float run_std_gemm(int M,int N,int K,float alpha,float beta,
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
    Stg, 1, 1, false, cutlass::arch::OpMultiplyAdd>;

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
  if(dw)cudaFree(dw);
  return ms;
}

void cpu_relu_gemm(int M,int N,int K,float alpha,float beta,
    const float* A,const float* B,const float* C,float* D){
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){
    float acc=0;
    for(int k=0;k<K;k++){float p = A[m * K + k] * B[k + n * K];
unsigned int p_as_int = RoundMantissa((*((unsigned int*)(&p))));
acc += *((float*)(&p_as_int));}
    D[m*N+n]=alpha*acc+beta*C[m*N+n];
  }
}
