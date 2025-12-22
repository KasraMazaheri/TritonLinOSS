import logging

import jax
import torch
import numpy as np
import equinox as eqx
from linoss.models.LinOSS import LinOSS as LinOSSJax
from linoss.models.TorchLinOSS import LinOSS as LinOSSTorch
from jax.tree_util import SequenceKey, DictKey, GetAttrKey, FlattenedIndexKey

from typing import Tuple

logger = logging.getLogger(__name__)


def pytree_path_to_str(
    path: Tuple[SequenceKey | DictKey | GetAttrKey | FlattenedIndexKey, ...],
) -> str:
    """
    Recursive function to convert a JAX pytree path to a string representation.
    """
    if len(path) == 0:
        return ""
    if isinstance(path[0], SequenceKey):
        rest = pytree_path_to_str(path[1:])
        return f"{path[0].idx}" + "." + rest if rest else f"{path[0].idx}"
    elif isinstance(path[0], GetAttrKey):
        rest = pytree_path_to_str(path[1:])
        return f"{path[0].name}" + "." + rest if rest else f"{path[0].name}"
    else:
        raise NotImplementedError(f"Unsupported key type: {type(path[0])}")


def map_jax_parameter_name_to_torch(param_name: str) -> str:
    """
    Map JAX parameter names to Torch parameter names.
    """
    jax_to_torch_mapping = {
        "batch_counter.init": "num_batches_tracked",
        "batch_state_index.init.0": "running_mean",
        "batch_state_index.init.1": "running_var",
    }
    # check if param_name ends in any of the keys in mapping
    for jax_name, torch_name in jax_to_torch_mapping.items():
        if param_name.endswith(jax_name):
            return param_name[: -len(jax_name)] + torch_name

    return param_name


def from_jax_to_torch(jax_model: LinOSSJax, torch_model: LinOSSTorch) -> None:
    """
    Copy parameters from a JAX LinOSS model to a Torch LinOSS model.
    """
    jax_model, _ = eqx.partition(jax_model, eqx.is_array)

    torch_model_params = {}
    jax_model_params = {}

    for param in torch_model.state_dict().keys():
        torch_model_params[param] = torch_model.state_dict()[param].shape

    for path, jax_param_value in jax.tree.leaves_with_path(jax_model):
        param_name = pytree_path_to_str(path)
        jax_model_params[param_name] = jax_param_value.shape

    for path, jax_param_value in jax.tree.leaves_with_path(jax_model):
        param_name = pytree_path_to_str(path)
        param_name_torch = map_jax_parameter_name_to_torch(param_name)

        if param_name_torch not in torch_model_params:
            logger.warning(
                f"Parameter {param_name_torch} not found in Torch model. Skipping."
            )

        torch_param = torch_model
        for key in param_name_torch.split(".")[:-1]:
            torch_param = getattr(torch_param, key)
        torch_value = getattr(torch_param, param_name_torch.split(".")[-1])
        device = torch_value.data.device
        torch_value.data = torch.from_numpy(np.array(jax_param_value)).to(device)
        logger.info(f"Copied parameter: {param_name_torch}")
        torch_model_params[param_name_torch] = True

    # print if there are any parameters in torch model that were not copied
    for param in torch_model_params.keys():
        if torch_model_params[param] is not True:
            logger.warning(f"Parameter {param} was not copied from JAX model.")
