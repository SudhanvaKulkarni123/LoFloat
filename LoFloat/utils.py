import copy
import torch
import torch.nn as nn
import LoFloat as lof

def lofloatify(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)

    single_params = lof.create_single_params()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = lof.LoF_Linear(
                module.in_features,
                module.out_features,
                act_params=single_params, bias_params=single_params, weight_params=single_params
            )
            # Handle Sequential (integer-keyed) vs named modules
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        elif isinstance(module, nn.Conv2d):
            new_layer = lof.LoF_Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                act_params=single_params, bias_params=single_params, weight_params=single_params
            )
            if isinstance(model, nn.Sequential):
                model[int(name)] = new_layer
            else:
                setattr(model, name, new_layer)
        else:
            if isinstance(model, nn.Sequential):
                model[int(name)] = lofloatify(module)
            else:
                setattr(model, name, lofloatify(module))

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

