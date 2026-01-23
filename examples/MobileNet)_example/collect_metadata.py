from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class ExtractedGraph:
    exported_program: Optional["torch.export.ExportedProgram"] = None
    fx_graph_module: Optional[fx.GraphModule] = None
    kind: str = "unknown"


def extract_graph(
    model: nn.Module,
    example_inputs: Union[Tuple[Any, ...], Sequence[Any]],
    *,
    prefer: str = "export",
    training: bool = False,
    strict_export: bool = True,
) -> ExtractedGraph:
    """Extract a computation graph from a generic nn.Module."""
    if not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)

    model = model.train() if training else model.eval()

    # Try torch.export first
    if prefer in ("export", "auto"):
        try:
            import torch.export
            ep = torch.export.export(model, example_inputs, strict=strict_export)
            return ExtractedGraph(exported_program=ep, kind="export")
        except Exception:
            pass

    # FX fallback
    try:
        gm = fx.symbolic_trace(model)
        return ExtractedGraph(fx_graph_module=gm, kind="fx")
    except Exception as e:
        raise RuntimeError(
            "Failed to extract a graph via torch.export and torch.fx.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def iter_nodes(extracted: ExtractedGraph):
    """Yield nodes from whichever graph representation was produced."""
    if extracted.kind == "fx":
        yield from extracted.fx_graph_module.graph.nodes
    elif extracted.kind == "export":
        ep = extracted.exported_program
        gm = ep.graph_module
        yield from gm.graph.nodes


# -------------------------
# FLOP Formulas - Fully Corrected
# -------------------------

def conv2d_flops(module: nn.Conv2d, inp_shape, out_shape):
    """
    Calculate FLOPs for Conv2d.
    
    Formula: FLOPs = 2 * MACs + bias_ops
    where MACs = Cout * Hout * Wout * (Cin/groups * Kh * Kw)
    """
    Cin = module.in_channels
    Cout = module.out_channels
    
    # Handle kernel_size as int or tuple
    if isinstance(module.kernel_size, tuple):
        Kh, Kw = module.kernel_size
    else:
        Kh = Kw = module.kernel_size
    
    # Output spatial dimensions
    Hout, Wout = out_shape[2], out_shape[3]
    groups = module.groups

    # Multiply-accumulate operations
    macs = Cout * Hout * Wout * (Cin // groups * Kh * Kw)
    flops = 2 * macs  # Each MAC = 1 multiply + 1 add = 2 FLOPs
    
    # Add bias operations (1 add per output element)
    if module.bias is not None:
        flops += Cout * Hout * Wout
    
    return flops


def linear_flops(module: nn.Linear, inp_shape, out_shape):
    """
    Calculate FLOPs for Linear layer (GEMM).
    
    For matrix multiply: (batch, in_features) @ (in_features, out_features)
    Formula: FLOPs = 2 * batch_size * in_features * out_features + bias_ops
    
    General GEMM (m×k @ k×n): FLOPs = 2mkn
    Square matrices (n×n @ n×n): FLOPs = 2n³
    """
    # Calculate total batch size (all dimensions except last)
    batch_size = 1
    for dim in inp_shape[:-1]:
        batch_size *= dim
    
    # Matrix multiplication: 2 FLOPs per MAC
    flops = 2 * batch_size * module.in_features * module.out_features
    
    # Bias addition: 1 add per output element
    if module.bias is not None:
        flops += batch_size * module.out_features
    
    return flops


def batchnorm_flops(module: Union[nn.BatchNorm1d, nn.BatchNorm2d], inp_shape, out_shape):
    """
    Calculate FLOPs for BatchNorm (inference mode).
    
    Operations per element:
    1. Subtract mean: 1 FLOP
    2. Divide by std (or multiply by 1/std): 1 FLOP
    3. Multiply by gamma: 1 FLOP
    4. Add beta: 1 FLOP
    Total: 4 FLOPs per element
    """
    numel = 1
    for dim in out_shape:
        numel *= dim
    return 4 * numel


def relu_flops(module: Union[nn.ReLU, nn.ReLU6], inp_shape, out_shape):
    """
    Calculate FLOPs for ReLU activation.
    
    ReLU performs: max(0, x) → 1 comparison per element
    """
    numel = 1
    for dim in out_shape:
        numel *= dim
    return numel


def adaptive_avg_pool2d_flops(module: nn.AdaptiveAvgPool2d, inp_shape, out_shape):
    """
    Calculate FLOPs for AdaptiveAvgPool2d.
    
    Each output element is computed by averaging a region of input elements.
    Operations: sum of region + 1 division
    """
    N, C, H_in, W_in = inp_shape
    H_out, W_out = out_shape[2], out_shape[3]
    
    # Size of pooling kernel for each output
    kernel_h = H_in // H_out if H_out > 0 else H_in
    kernel_w = W_in // W_out if W_out > 0 else W_in
    
    # Operations per output element:
    # - Sum: (kernel_h * kernel_w - 1) additions
    # - Average: 1 division
    # Total ≈ kernel_h * kernel_w
    ops_per_element = kernel_h * kernel_w
    
    total_output_elements = N * C * H_out * W_out
    return total_output_elements * ops_per_element


def maxpool2d_flops(module: nn.MaxPool2d, inp_shape, out_shape):
    """
    Calculate FLOPs for MaxPool2d.
    
    Finding max of K elements requires K-1 comparisons.
    """
    if isinstance(module.kernel_size, tuple):
        kernel_h, kernel_w = module.kernel_size
    else:
        kernel_h = kernel_w = module.kernel_size
    
    # Comparisons needed to find max
    comparisons_per_element = kernel_h * kernel_w - 1
    
    # Total output elements
    total_elements = 1
    for dim in out_shape:
        total_elements *= dim
    
    return total_elements * comparisons_per_element


# Comprehensive registry
FLOP_HANDLERS = {
    nn.Conv2d: conv2d_flops,
    nn.Linear: linear_flops,
    nn.BatchNorm1d: batchnorm_flops,
    nn.BatchNorm2d: batchnorm_flops,
    nn.ReLU: relu_flops,
    nn.ReLU6: relu_flops,
    nn.AdaptiveAvgPool2d: adaptive_avg_pool2d_flops,
    nn.MaxPool2d: maxpool2d_flops,
}


# -------------------------
# Analysis Functions
# -------------------------

def print_layer_flops(
    model: nn.Module,
    example_input: torch.Tensor,
):
    """Print aggregated FLOPs by module name."""
    model.eval()

    # Trace and propagate shapes
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(example_input)

    # Accumulate FLOPs per module
    flops_by_module: Dict[str, int] = defaultdict(int)

    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue

        module = gm.get_submodule(node.target)

        # Get output metadata
        out_meta = node.meta.get("tensor_meta")
        if out_meta is None:
            continue

        # Get input metadata
        inp_meta = None
        for arg in node.args:
            if isinstance(arg, fx.Node):
                inp_meta = arg.meta.get("tensor_meta")
                if inp_meta is not None:
                    break
        
        if inp_meta is None:
            continue

        # Calculate FLOPs using appropriate handler
        for mod_type, handler in FLOP_HANDLERS.items():
            if isinstance(module, mod_type):
                flops = handler(module, inp_meta.shape, out_meta.shape)
                flops_by_module[node.target] += flops
                break

    # Print results
    print(f"{'Layer':50s} {'FLOPs':>20s}")
    print("-" * 75)
    total_flops = 0
    for name, flops in sorted(flops_by_module.items()):
        print(f"{name:50s} {flops:>20,}")
        total_flops += flops
    print("-" * 75)
    print(f"{'TOTAL':50s} {total_flops:>20,}")
    print(f"\nTotal: {total_flops/1e9:.3f} GFLOPs")


def print_graph(extracted: ExtractedGraph, max_nodes: int = 50):
    """Print a readable listing of nodes with op + target."""
    for i, n in enumerate(iter_nodes(extracted)):
        if i >= max_nodes:
            print(f"... ({max_nodes} shown)")
            break
        print(f"{i:03d}  {n.op:12s}  {str(n.target)}")


def per_layer_flops(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    *,
    include_zero: bool = False,
) -> List[Tuple[str, int]]:
    """
    Returns FLOPs per invocation (FX node), not aggregated.
    
    Label format: "<module_path> :: <node_name> (<ModuleType>)"
    """
    model.eval()
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*example_inputs)

    results: List[Tuple[str, int]] = []

    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue

        mod = gm.get_submodule(node.target)
        out_meta = node.meta.get("tensor_meta")
        if out_meta is None:
            continue

        # Get input metadata
        inp_meta = None
        for a in node.args:
            if isinstance(a, fx.Node):
                inp_meta = a.meta.get("tensor_meta")
                if inp_meta is not None:
                    break

        flops = 0
        for t, handler in FLOP_HANDLERS.items():
            if isinstance(mod, t):
                if inp_meta is None:
                    break
                flops = handler(mod, inp_meta.shape, out_meta.shape)
                break

        if flops == 0 and not include_zero:
            continue

        label = f"{node.target} :: {node.name} ({type(mod).__name__})"
        results.append((label, flops))

    return results


def print_per_layer_flops(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    *,
    include_zero: bool = False,
    top_k: Optional[int] = None,
):
    """Print per-layer FLOP counts."""
    rows = per_layer_flops(model, example_inputs, include_zero=include_zero)

    if top_k is not None:
        rows = rows[:top_k]

    print(f"{'Layer (per invocation)':70s} {'FLOPs':>20s}")
    print("-" * 95)
    total_flops = 0
    for name, flops in rows:
        print(f"{name:70s} {flops:>20,}")
        total_flops += flops
    print("-" * 95)
    print(f"{'TOTAL':70s} {total_flops:>20,}")
    print(f"\nTotal: {total_flops/1e9:.3f} GFLOPs")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, example_inputs: Tuple[torch.Tensor, ...]):
    """Print comprehensive model summary with FLOPs and parameters."""
    print("=" * 95)
    print("MODEL SUMMARY")
    print("=" * 95)
    
    # Parameters
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # FLOPs
    print("\n" + "=" * 95)
    print("PER-LAYER FLOPS")
    print("=" * 95)
    print_per_layer_flops(model, example_inputs)