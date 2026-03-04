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