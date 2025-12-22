"""
Pytest-based correctness tests for TritonLinOSS implementation
"""

import pytest
import numpy as np
import jax.random as jr
import jax
import torch
from utils import compute_differences, create_model_pair


SEED = 42


@pytest.mark.parametrize("layer_name", ["IM", "IMEX", "Damped"])
@pytest.mark.parametrize(
    "batch_size,seq_length,input_dim,hidden_dim,ssm_size,output_dim",
    [
        (2, 64, 8, 32, 64, 10),  # small
        (4, 256, 16, 64, 128, 20),  # medium
        (8, 512, 32, 128, 256, 40),  # large
    ],
    ids=["small", "medium", "large"],
)
@pytest.mark.parametrize(
    "max_atol,mean_atol,rel_tol",
    [
        (1e-2, 1e-3, 1e-1),  # relaxed
        (1e-3, 1e-4, 1e-2),  # strict
    ],
    ids=["relaxed", "strict"],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_initialization_fwd(
    layer_name: str,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    hidden_dim: int,
    ssm_size: int,
    output_dim: int,
    max_atol: float,
    mean_atol: float,
    rel_tol: float,
    device: str,
):
    """ """
    key = jr.PRNGKey(SEED)
    device = device
    data_key, model_key, state_key = jr.split(key, 3)
    test_data_jax = jr.normal(data_key, (batch_size, seq_length, input_dim))
    test_data_torch = torch.tensor(np.array(test_data_jax), dtype=torch.float32).to(
        device
    )

    jax_model, torch_model = create_model_pair(
        layer_name=layer_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ssm_size=ssm_size,
        output_dim=output_dim,
        num_blocks=1,
        classification=True,
        tanh_output=False,
        output_step=1,
        key=model_key,
        device=device,
    )

    jax_forward = jax_model(test_data_jax)
    torch_forward = torch_model(test_data_torch)

    diffs = compute_differences(jax_forward, torch_forward.detach().cpu().numpy())

    assert diffs["max_diff"] < max_atol, (
        f"Max difference {diffs['max_diff']:.6e} exceeds tolerance {max_atol:.6e}"
    )
    assert diffs["mean_diff"] < mean_atol, (
        f"Mean difference {diffs['mean_diff']:.6e} exceeds tolerance {mean_atol:.6e}"
    )
    assert diffs["rel_diff"] < rel_tol, (
        f"Relative difference {diffs['rel_diff']:.6e} exceeds tolerance {rel_tol:.6e}"
    )


@pytest.mark.parametrize("layer_name", ["IM", "IMEX", "Damped"])
@pytest.mark.parametrize(
    "batch_size,seq_length,input_dim,hidden_dim,ssm_size,output_dim",
    [
        (2, 64, 8, 32, 64, 10),  # small
        (4, 256, 16, 64, 128, 20),  # medium
        (8, 512, 32, 128, 256, 40),  # large
    ],
    ids=["small", "medium", "large"],
)
@pytest.mark.parametrize(
    "max_atol,mean_atol,rel_tol",
    [
        (1e-2, 1e-2, 1e-2),  # relaxed
        (1e-3, 1e-3, 5e-3),  # strict
    ],
    ids=["relaxed", "strict"],
)
def test_initialization_bwd(
    layer_name: str,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    hidden_dim: int,
    ssm_size: int,
    output_dim: int,
    max_atol: float,
    mean_atol: float,
    rel_tol: float,
):
    """ """
    key = jr.PRNGKey(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_key, model_key, state_key = jr.split(key, 3)
    test_data_jax = jr.normal(data_key, (batch_size, seq_length, input_dim))
    test_data_torch = torch.tensor(
        np.array(test_data_jax), dtype=torch.float32, requires_grad=True
    ).to(device)
    test_data_torch.retain_grad()

    jax_model, torch_model = create_model_pair(
        layer_name=layer_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ssm_size=ssm_size,
        output_dim=output_dim,
        num_blocks=1,
        classification=True,
        tanh_output=False,
        output_step=1,
        key=model_key,
        device=device,
        inference=False,
    )

    torch_forward_mean = torch_model(test_data_torch).mean()
    torch_forward_mean.backward(retain_graph=True)
    grad_torch = test_data_torch.grad

    grad_jax = jax.grad(lambda x: jax_model(x).mean())(test_data_jax)

    diffs = compute_differences(grad_jax, grad_torch.detach().cpu().numpy())

    assert diffs["max_diff"] < max_atol, (
        f"Max difference {diffs['max_diff']:.6e} exceeds tolerance {max_atol:.6e}"
    )
    assert diffs["mean_diff"] < mean_atol, (
        f"Mean difference {diffs['mean_diff']:.6e} exceeds tolerance {mean_atol:.6e}"
    )
    assert diffs["rel_diff"] < rel_tol, (
        f"Relative difference {diffs['rel_diff']:.6e} exceeds tolerance {rel_tol:.6e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
