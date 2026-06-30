#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ── CUTLASS conv ─────────────────────────────────────────────────────────────
// The thread::Mma<OpLoFMultiplyAdd> / warp::LoFMma / DefaultMmaCore<OpLoFMultiplyAdd>
// specializations live in cutlass_gemms.cuh. Conv2d's OpClassSimt mainloop
// (cutlass/conv/kernel/default_conv2d_fprop.h) instantiates the exact same
// cutlass::gemm::threadblock::DefaultMmaCore template GEMM does (see the
// "Define the core components from GEMM" comment in that header) — so the leaf
// is shared and no new math is written here. See CUTLASS_plumbing.md.
#include "cutlass_gemms.cuh"

#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

namespace lo_float {

// ─────────────────────────────────────────────────────────────────────────
//  lo_float::Conv2d — forward convolution (Fprop) via CUTLASS implicit GEMM.
//
//  Computes:  D = alpha * conv(A, B) + beta * C     (NHWC activation/output,
//                                                      KRSC filter, both laid
//                                                      out as TensorNHWC)
//
//  Template parameters mirror lo_float::Gemm (cutlass_gemms.cuh):
//    TbM, TbN, TbK    — Threadblock tile shape (M=N*P*Q output rows, N=K
//                        output channels, K=C*R*S reduction)
//    WpM, WpN, WpK    — Warp tile shape
//    Stages            — Pipeline depth (>=3 for cp.async multistage, SM80+)
//
//  Runtime parameters (constructor): accum_mantissa_bits, rounding_mode,
//  stochastic_rounding_bits — control lo_float accumulation rounding,
//  plumbed through Arguments -> Params -> mainloop ctor -> warp_mma, exactly
//  as documented in CUTLASS_plumbing.md §3.
//
//  Always uses: OpClassSimt, Sm80, OpLoFMultiplyAdd, InstructionShape<1,1,1>,
//  groups=1 (matches the GEMM leaf's float-only, ungrouped specializations).
// ─────────────────────────────────────────────────────────────────────────

template <
    int TbM = 128,
    int TbN = 128,
    int TbK = 8,
    int WpM = 32,
    int WpN = 64,
    int WpK = 8,
    int Stages = 3>
class Conv2d {
public:

  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::TensorNHWC;
  using LayoutB = cutlass::layout::TensorNHWC;   // filter, read as K-R-S-C
  using LayoutC = cutlass::layout::TensorNHWC;

  using ThreadblockShape = cutlass::gemm::GemmShape<TbM, TbN, TbK>;
  using WarpShape        = cutlass::gemm::GemmShape<WpM, WpN, WpK>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  static constexpr int kStages = Stages;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementC, ElementCompute>;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages,
      cutlass::arch::OpLoFMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kAnalytic,
      cutlass::conv::StrideSupport::kStrided
  >::Kernel;

  using UnderlyingConv = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
  using Arguments = typename UnderlyingConv::Arguments;

  enum class Status {
    kSuccess           =  0,
    kErrorNotSupported = -1,
    kErrorAllocation   = -2,
    kErrorInitialize   = -3,
    kErrorExecution    = -4
  };

private:

  int           accum_mantissa_bits_;
  lo_float::ProjSpec proj_spec_;

public:

  Conv2d(int accum_mantissa_bits        = 23,
         lo_float::ProjSpec proj_spec    = lo_float::ProjSpec{})
    : accum_mantissa_bits_(accum_mantissa_bits)
    , proj_spec_(proj_spec)
  {}

  void set_accum_mantissa_bits(int bits)      { accum_mantissa_bits_      = bits; }
  void set_proj_spec(lo_float::ProjSpec spec)  { proj_spec_               = spec; }

  /// Run D = alpha * conv(A, B) + beta * C.
  /// A: NHWC activation (N, H, W, C_in); B: NHWC-laid-out filter (K, R, S, C_in);
  /// C, D: NHWC output (N, P, Q, K). problem_size.groups must be 1.
  Status operator()(
      cutlass::conv::Conv2dProblemSize const & problem_size,
      ElementA* dA, ElementB* dB,
      ElementC* dC, ElementC* dD,
      ElementCompute alpha = ElementCompute(1), ElementCompute beta = ElementCompute(0),
      cudaStream_t stream = nullptr) const
  {
    if (problem_size.groups != 1) {
      return Status::kErrorNotSupported;
    }

    cutlass::Tensor4DCoord input_extent{problem_size.N, problem_size.H, problem_size.W, problem_size.C};
    cutlass::Tensor4DCoord filter_extent{problem_size.K, problem_size.R, problem_size.S, problem_size.C};
    cutlass::Tensor4DCoord output_extent{problem_size.N, problem_size.P, problem_size.Q, problem_size.K};

    // Conv's kernel-level Arguments reuses Mma::IteratorA/B::TensorRef verbatim
    // (cutlass::TensorRef<Element, Layout>, no const) — unlike GEMM's
    // hand-written device::Gemm::Arguments, which const-qualifies A/B/C.
    cutlass::TensorRef<ElementA, LayoutA> ref_A(dA, LayoutA::packed(input_extent));
    cutlass::TensorRef<ElementB, LayoutB> ref_B(dB, LayoutB::packed(filter_extent));
    cutlass::TensorRef<ElementC, LayoutC> ref_C(dC, LayoutC::packed(output_extent));
    cutlass::TensorRef<ElementC, LayoutC> ref_D(dD, LayoutC::packed(output_extent));

    Arguments args(
        problem_size,
        ref_A, ref_B, ref_C, ref_D,
        {alpha, beta},
        cutlass::conv::SplitKMode::kSerial,
        accum_mantissa_bits_,
        proj_spec_.rounding_mode,
        proj_spec_.stoch_length);

    UnderlyingConv op;

    if (UnderlyingConv::can_implement(args) != cutlass::Status::kSuccess) {
      return Status::kErrorNotSupported;
    }

    size_t ws_bytes = UnderlyingConv::get_workspace_size(args);
    void*  workspace = nullptr;
    if (ws_bytes) {
      if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess) {
        return Status::kErrorAllocation;
      }
    }

    if (op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      if (workspace) cudaFree(workspace);
      return Status::kErrorInitialize;
    }

    cutlass::Status s = op(stream);
    if (workspace) cudaFree(workspace);

    return (s == cutlass::Status::kSuccess) ? Status::kSuccess : Status::kErrorExecution;
  }
};

}  // namespace lo_float
