from rl.net import static_module_wrapper
import warnings
import inspect
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from numbers import Number


def soft_update(target_net: nn.Module, source_net: nn.Module, update_rate) -> None:
    """
    Soft update target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
        update_rate: Update rate.

    Returns:
        None
    """
    with torch.no_grad():
        for target_param, param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - update_rate)
                + param.data.to(target_param.device) * update_rate
            )


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    """
    Hard update (directly copy) target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
    """

    for target_buffer, buffer in zip(target_net.buffers(), source_net.buffers()):
        target_buffer.data.copy_(buffer.data)
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def determine_device(model):
    devices = set()
    for k, v in model.named_parameters():
        devices.add(str(v.device))
    return list(devices)


def move_to_device(data, device):
    """
    Recursively move all torch tensors in a nested data structure to a specified device.

    Args:
        data (dict, list, torch.Tensor, or any other type): The data structure containing tensors.
        device (str or torch.device): The device to move the tensors to.

    Returns:
        The same data structure with all tensors moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def safe_call(
    model, *named_args, method="__call__", call_dp_or_ddp_internal_module=False
):
    """
    Call a model and discard unnecessary arguments. safe_call will automatically
    move tensors in named_args to the input device of the model

    Any input tensor in named_args must not be contained inside any container,
    such as list, dict, tuple, etc. Because they will be automatically moved
    to the input device of the specified model.

    Args:
        model: Model to be called, must be a wrapped nn.Module or an instance of
               NeuralNetworkModule.
        named_args: A dictionary of argument, key is argument's name, value is
                    argument's value.
        method: Method to invoke.
        call_dp_or_ddp_internal_module: Whether to call the module encapsulated
            in model when model is DP or DDP.

    Returns:
        Whatever returned by your module. If result is not a tuple, always
        wrap results inside a tuple
    """
    org_model = None
    if isinstance(
        model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
    ):
        org_model = model
        model = model.module
    if not hasattr(model, "input_device") or not hasattr(model, "output_device"):
        # try to automatically determine the input & output device of the model
        model_type = type(model)
        device = determine_device(model)
        if len(device) > 1:
            raise RuntimeError(
                f"""\
                Failed to automatically determine i/o device of your model: {model_type}
                Detected multiple devices: {device}

                You need to manually specify i/o device of your model.

                Either Wrap your model of type nn.Module with one of:
                1. static_module_wrapper from rl.net
                2. dynamic_module_wrapper from rl.net 
                
                Or construct your own module & model with: 
                NeuralNetworkModule from rl.net"""
            )
        else:
            # assume that i/o devices are the same as parameter device
            # print a warning
            warnings.warn(
                f"""\
                
                You have not specified the i/o device of your model {model_type}
                Automatically determined and set to: {device[0]}

                The framework is not responsible for any un-matching device issues 
                caused by this operation."""
            )
            model = static_module_wrapper(model, device[0], device[0])

    input_device = model.input_device
    if method == "__call__":
        arg_spec = inspect.getfullargspec(model.forward)
    else:
        arg_spec = inspect.getfullargspec(getattr(model, method))
    # exclude self in arg_spec.args
    args = arg_spec.args[1:] + arg_spec.kwonlyargs
    if arg_spec.defaults is not None:
        args_with_defaults = args[-len(arg_spec.defaults) :]
    else:
        args_with_defaults = []
    required_args = (
        set(args)
        - set(args_with_defaults)
        - set(
            arg_spec.kwonlydefaults.keys()
            if arg_spec.kwonlydefaults is not None
            else []
        )
    )
    args_dict = {}

    # fill in args
    for na in named_args:
        for k, v in na.items():
            if k in args or arg_spec.varargs is not None or arg_spec.varkw is not None:
                args_dict[k] = move_to_device(v, input_device)

    # check for necessary args
    missing = required_args - set(args_dict.keys())
    if len(missing) > 0:
        raise RuntimeError(
            f"""\
            Required arguments of the forward function of Model {type(model)} 
            is {required_args}, missing required arguments: {missing}

            Check your storage functions.
            """
        )

    if org_model is not None and not call_dp_or_ddp_internal_module:
        result = getattr(org_model, method)(**args_dict)
    else:
        result = getattr(model, method)(**args_dict)

    if isinstance(result, tuple):
        return result
    else:
        return (result,)


def safe_return(result):
    if len(result) == 1:
        return result[0]
    else:
        return result


def safe_import(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_globals_from_stack():
    frames = inspect.stack()
    global_vars = {}
    for frame in frames:
        for k, v in frame[0].f_globals.items():
            if not k.startswith("__"):
                global_vars[k] = v
    return global_vars


def assert_output_is_probs(tensor):
    if (
        tensor.dim() == 2
        and torch.all(torch.abs(torch.sum(tensor, dim=1) - 1.0) < 1e-5)
        and torch.all(tensor >= 0)
    ):
        return
    else:
        print(tensor)
        raise ValueError(
            "Input tensor is not a probability tensor, it must "
            "have 2 dimensions (0 and 1), a sum of 1.0 for each "
            "row in dimension 1, and a positive value for each "
            "element."
        )


def batch_tensor_dicts(
    list_of_dicts: List[Dict[str, torch.Tensor]], concatenate_samples=True
):
    if len(list_of_dicts) == 0:
        return {}
    # check keys
    keys = set(list_of_dicts[0].keys())
    value_is_tensor = {}
    for k, v in list_of_dicts[0].items():
        if not torch.is_tensor(v) and not isinstance(v, Number):
            raise ValueError(
                "Values of dictionaries must be tensors or scalars (int, float, bool)."
            )
        value_is_tensor[k] = torch.is_tensor(v)
    dict_of_lists = {k: [] for k in keys}
    for d in list_of_dicts:
        if set(d.keys()) != keys:
            raise ValueError("Dictionaries must have the same keys.")
        for k, v in d.items():
            if not torch.is_tensor(v) and not isinstance(v, Number):
                raise ValueError(
                    "Values of dictionaries must be tensors or scalars (int, float, bool)."
                )
            elif torch.is_tensor(v) != value_is_tensor[k]:
                raise ValueError(
                    "Values of dictionaries must be of same type under the same key."
                )
            dict_of_lists[k].append(v)
    if concatenate_samples:
        for k in dict_of_lists.keys():
            if value_is_tensor[k]:
                dict_of_lists[k] = torch.cat(dict_of_lists[k], dim=0)
            else:
                dict_of_lists[k] = torch.tensor(dict_of_lists[k])
    return dict_of_lists


def find_shared_parameters(
    models: List[nn.Module],
) -> Tuple[List[nn.Parameter], List[List[nn.Parameter]]]:
    """
    Args:
        models: A list of nn.Modules to check

    Returns:
        A list of shared parameters between all models.
        A list, with each sub list made of parameters that excludes the shared parameters for every model.
    """
    if len(models) == 1:
        return list(models[0].parameters()), [[]]
    else:
        parameters = [list(m.parameters()) for m in models]
        parameters_ids_maps = [{id(p): p for p in params} for params in parameters]
        parameter_ids = [set(id(p) for p in params) for params in parameters]
        shared_ids = parameter_ids[0]
        for ids in parameter_ids[1:]:
            shared_ids = shared_ids.intersection(ids)
        shared_parameters = [parameters_ids_maps[0][id_] for id_ in shared_ids]
        other_parameters = [
            [
                parameters_ids_maps[idx][id_]
                for id_ in parameter_ids[idx].difference(shared_ids)
            ]
            for idx in range(len(models))
        ]
        return shared_parameters, other_parameters
