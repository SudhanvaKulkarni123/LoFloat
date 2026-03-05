import copy
import torch
import torch.nn as nn
import LoFloat as lof

def lofloatify(model: nn.Module, rounding_mode: lof.RoundingMode = lof.RoundingMode.RoundToNearestEven) -> nn.Module:
    model = copy.deepcopy(model)
    single_params = lof.create_single_params()
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = lof.LoF_Linear.from_linear(
                module,
                weight_params=single_params,
                act_params=single_params,
                bias_params=single_params,
                rounding_mode=rounding_mode
            )
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        elif isinstance(module, nn.Conv2d):
            new_layer = lof.LoF_Conv2d.from_conv2d(
                module,
                weight_params=single_params,
                act_params=single_params,
                bias_params=single_params,
                rounding_mode=rounding_mode
            )
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        else:
            if isinstance(model, nn.Sequential):
                model[int(name)] = lofloatify(module, rounding_mode=rounding_mode)
            else:
                setattr(model, name, lofloatify(module, rounding_mode=rounding_mode))
    return model

#set mantissa_bits for activation, weight or bias of all layers
def set_mantissa_fields(model, activation_mantissa_bits: dict, weight_mantissa_bits: dict, bias_mantissa_bits: dict):
    for name, module in model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        act_mant    = activation_mantissa_bits.get(name)
        weight_mant = weight_mantissa_bits.get(name)
        bias_mant   = bias_mantissa_bits.get(name)

        if act_mant is None or weight_mant is None or bias_mant is None:
            print(f"[set_mantissa_fields] skipping '{name}': missing entry in one or more dicts")
            continue

        module.set_mantissa(act_mant, weight_mant, bias_mant)


def set_exponent_fields(model, activation_exp_bits: dict, weight_exp_bits: dict, bias_exp_bits: dict):
    for name, module in model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        act_exp    = activation_exp_bits.get(name)
        weight_exp = weight_exp_bits.get(name)
        bias_exp   = bias_exp_bits.get(name)

        if act_exp is None or weight_exp is None or bias_exp is None:
            print(f"[set_exponent_fields] skipping '{name}': missing entry in one or more dicts")
            continue

        module.set_exponent(act_exp, weight_exp, bias_exp)

def set_exponentbias_fields(model, activation_expbias: dict, weight_expbias: dict, bias_expbias: dict):
    for name, module in model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        act_expbias    = activation_expbias.get(name)
        w_expbias = weight_expbias.get(name)
        b_expbias   = bias_expbias.get(name)

        if act_expbias is None or w_expbias is None or b_expbias is None:
            print(f"[set_exponentbias_fields] skipping '{name}': missing entry in one or more dicts")
            continue

        module.set_exponentbias(act_expbias, w_expbias, b_expbias)

def set_all_to_half(model):
    half_params = lof.create_half_params()
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = lof.LoF_Linear.from_linear(
                module,
                weight_params=half_params,
                act_params=half_params,
                bias_params=half_params
            )
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        elif isinstance(module, nn.Conv2d):
            new_layer = lof.LoF_Conv2d.from_conv2d(
                module,
                weight_params=half_params,
                act_params=half_params,
                bias_params=half_params)
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        else:
            if isinstance(model, nn.Sequential):
                model[int(name)] = set_all_to_half(module)
            else:
                setattr(model, name, set_all_to_half(module))
    return model


def set_all_to_3109(model, k, p):
    p3109_params = lof.create_p3109_params(k=k, p=p, is_signed=True, saturating=True)
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = lof.LoF_Linear.from_linear(
                module,
                weight_params=p3109_params,
                act_params=p3109_params,
                bias_params=p3109_params
            )
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        elif isinstance(module, nn.Conv2d):
            new_layer = lof.LoF_Conv2d.from_conv2d(
                module,
                weight_params=p3109_params,
                act_params=p3109_params,
                bias_params=p3109_params)
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        else:
            if isinstance(model, nn.Sequential):
                model[int(name)] = set_all_to_3109(module, k=k, p=p)
            else:
                setattr(model, name, set_all_to_3109(module, k=k, p=p))
    return model


def print_exp_mant(model):
    for name, module in model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            print(f"Layer: {name}")
            act_total = module.act_params.total_bits
            weight_total = module.weight_params.total_bits
            bias_total = module.bias_params.total_bits
            act_mant = module.act_params.mantissa_bits
            weight_mant = module.weight_params.mantissa_bits
            bias_mant = module.bias_params.mantissa_bits
            print(f"  Activation: mantissa bits={act_mant}, exponent bits={act_total - act_mant - 1}")
            print(f"  Weight: mantissa bits={weight_mant}, exponent bits={weight_total - weight_mant - 1}")
            print(f"  Bias: mantissa bits={bias_mant}, exponent bits={bias_total - bias_mant - 1}")

def name_type(total_bits, mantissa_bits):
    return f"binary{total_bits}p{mantissa_bits}"


def record_formats(model):
    formats_flops = {}

    for name, module in model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            act_format = name_type(module.act_params.total_bits, module.act_params.mantissa_bits)
            weight_format = name_type(module.weight_params.total_bits, module.weight_params.mantissa_bits)
            bias_format = name_type(module.bias_params.total_bits, module.bias_params.mantissa_bits)

            if isinstance(module, lof.LoF_Linear):
                num_flops_of_layer = 2 * module.in_features * module.out_features
            elif isinstance(module, lof.LoF_Conv2d):
                out_h, out_w = module.output_size if hasattr(module, 'output_size') else (1, 1)
                num_flops_of_layer = (
                    2 * module.in_channels * module.out_channels
                    * module.kernel_size[0] * module.kernel_size[1]
                    * out_h * out_w // module.groups
                )

            format_key = (act_format, weight_format, bias_format)
            if format_key not in formats_flops:
                formats_flops[format_key] = 0
            formats_flops[format_key] += num_flops_of_layer

        else:
            # All non-LoF layers counted as fp32
            num_flops_of_layer = 0

            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Normalize: subtract mean, divide std -> 2 FLOPs per element
                # Scale and shift (gamma, beta) -> 2 FLOPs per element
                # Total: 4 * num_features * spatial_size
                out_h, out_w = module.output_size if hasattr(module, 'output_size') else (1, 1)
                spatial = out_h * out_w if isinstance(module, nn.BatchNorm2d) else 1
                num_flops_of_layer = 4 * module.num_features * spatial

            # elif isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
            #     # 1 comparison per element
            #     if hasattr(module, 'output_size'):
            #         out_h, out_w = module.output_size
            #         num_flops_of_layer = module.inplace * out_h * out_w  # approximate
            #     # Often ignored or estimated from previous layer

            elif isinstance(module, nn.LeakyReLU):
                # 1 comparison + 1 multiply per element
                if hasattr(module, 'output_size') and hasattr(module, 'num_channels'):
                    out_h, out_w = module.output_size
                    num_flops_of_layer = 2 * module.num_channels * out_h * out_w

            elif isinstance(module, nn.SiLU):
                # sigmoid (4 FLOPs) + multiply (1 FLOP) = ~5 per element
                if hasattr(module, 'output_size') and hasattr(module, 'num_channels'):
                    out_h, out_w = module.output_size
                    num_flops_of_layer = 5 * module.num_channels * out_h * out_w

            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Comparisons/additions over the pooling window per output element
                if hasattr(module, 'output_size') and hasattr(module, 'num_channels'):
                    k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0] * module.kernel_size[1]
                    k_area = k * k if isinstance(module.kernel_size, int) else k
                    out_h, out_w = module.output_size
                    # (k_area - 1) comparisons or additions per output element
                    num_flops_of_layer = (k_area - 1) * module.num_channels * out_h * out_w

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                # Similar to AvgPool but kernel is inferred
                if hasattr(module, 'output_size') and hasattr(module, 'input_size') and hasattr(module, 'num_channels'):
                    in_h, in_w = module.input_size
                    out_h, out_w = module.output_size
                    k_h, k_w = in_h // out_h, in_w // out_w
                    num_flops_of_layer = (k_h * k_w - 1) * module.num_channels * out_h * out_w

            elif isinstance(module, nn.Upsample) or isinstance(module, nn.UpsamplingNearest2d):
                # Nearest: ~0 FLOPs (just copy)
                # Bilinear: ~4 multiplies + 3 adds per output element
                if hasattr(module, 'output_size') and hasattr(module, 'num_channels'):
                    out_h, out_w = module.output_size
                    if module.mode == 'bilinear':
                        num_flops_of_layer = 7 * module.num_channels * out_h * out_w
                    else:
                        num_flops_of_layer = 0

            elif isinstance(module, nn.Sigmoid):
                # exp + add + div ~ 4 FLOPs per element
                if hasattr(module, 'output_size') and hasattr(module, 'num_channels'):
                    out_h, out_w = module.output_size
                    num_flops_of_layer = 4 * module.num_channels * out_h * out_w

            elif isinstance(module, nn.Softmax):
                # exp per element + sum + div per element ~ 3n
                if hasattr(module, 'num_elements'):
                    num_flops_of_layer = 3 * module.num_elements

            elif isinstance(module, nn.Dropout):
                num_flops_of_layer = 0  # No FLOPs at inference

            if num_flops_of_layer > 0:
                fp32_key = ("fp32", "fp32", "fp32")
                if fp32_key not in formats_flops:
                    formats_flops[fp32_key] = 0
                formats_flops[fp32_key] += num_flops_of_layer

    return formats_flops
            
