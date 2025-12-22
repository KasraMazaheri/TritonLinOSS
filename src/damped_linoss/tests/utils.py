import jax.numpy as jnp

import jax
import torch
import equinox as eqx
import functools as ft
from typing import Tuple

from ..models.LinOSS import LinOSS as LinOSSJax
from ..models.TorchLinOSS import LinOSS as LinOSSTorch
from ..utils.from_jax import from_jax_to_torch


def compute_differences(orig_out, trit_out):
    """Compute discrepancy metrics between outputs."""
    max_diff = jnp.max(jnp.abs(orig_out - trit_out))
    mean_diff = jnp.mean(jnp.abs(orig_out - trit_out))
    rel_diff = jnp.mean(jnp.abs(orig_out - trit_out) / (jnp.abs(orig_out) + 1e-8))

    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rel_diff": float(rel_diff),
    }


def create_model_pair(
    layer_name: str,
    input_dim: int,
    hidden_dim: int,
    ssm_size: int,
    output_dim: int,
    num_blocks: int,
    classification: bool,
    tanh_output: bool,
    output_step: int,
    key,
    device: torch.device,
    inference: bool = True,
) -> Tuple:
    """Create jax and torch LinOSS models with same parameters."""
    model_params = {
        "layer_name": layer_name,
        "input_dim": input_dim,
        "state_dim": ssm_size,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_blocks": num_blocks,
        "classification": classification,
        "tanh_output": tanh_output,
        "output_step": output_step,
        "key": key,
    }

    jax_model = LinOSSJax(**model_params)

    torch_model = LinOSSTorch(**model_params).to(device)

    if inference:
        jax_model = eqx.tree_inference(jax_model, value=True)
        torch_model = torch_model.eval()

    from_jax_to_torch(jax_model, torch_model)

    def run_batched_forward_pass(X, model, state, key):
        return jax.vmap(
            model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
        )(X, state, key)[0]

    state = eqx.nn.State(jax_model)
    batched_jax_model = jax.jit(
        ft.partial(run_batched_forward_pass, model=jax_model, state=state, key=key)
    )

    return batched_jax_model, torch_model
