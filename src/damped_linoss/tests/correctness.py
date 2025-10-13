"""
Pytest-based correctness tests for TritonLinOSS implementation
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Tuple
from src.damped_linoss.models.LinOSS import LinOSS
from src.damped_linoss.models.TritonLinOSS import LinOSS as TritonLinOSS


SEED = 42


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
    key
) -> Tuple:
    """Create original and triton models with same parameters."""
    model_params = {
        'layer_name': layer_name,
        'input_dim': input_dim,
        'state_dim': ssm_size,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_blocks': num_blocks,
        'classification': classification,
        'tanh_output': tanh_output,
        'output_step': output_step,
        'key': key,
    }
    
    original_model = LinOSS(**model_params)
    triton_model = TritonLinOSS(**model_params)
    return original_model, triton_model


def run_forward_pass(model, test_data, state, state_key):
    """Run forward pass on a model."""
    return jax.vmap(
        model, 
        axis_name="batch",
        in_axes=(0, None, None),
        out_axes=(0, None)
    )(test_data, state, state_key)


def compute_differences(orig_out, trit_out):
    """Compute discrepancy metrics between outputs."""
    max_diff = jnp.max(jnp.abs(orig_out - trit_out))
    mean_diff = jnp.mean(jnp.abs(orig_out - trit_out))
    rel_diff = jnp.mean(jnp.abs(orig_out - trit_out) / (jnp.abs(orig_out) + 1e-8))
    
    return {
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'rel_diff': float(rel_diff)
    }


@pytest.mark.parametrize("layer_name", ['IM', 'IMEX', 'Damped'])
@pytest.mark.parametrize(
    "batch_size,seq_length,input_dim,hidden_dim,ssm_size,output_dim",
    [
        (2, 64, 8, 32, 64, 20),      # small
        (4, 256, 16, 64, 128, 40),   # medium
        (8, 512, 32, 128, 256, 80),  # large
    ],
    ids=["small", "medium", "large"]
)
@pytest.mark.parametrize(
    "max_atol,mean_atol,rel_tol",
    [
        (1e-5, 1e-6, 1e-4),  # strict
        (1e-2, 1e-3, 1e-1),  # relaxed
    ],
    ids=["strict", "relaxed"]
)
def test_output_correctness(
    layer_name: str,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    hidden_dim: int,
    ssm_size: int,
    output_dim: int,
    max_atol: float,
    mean_atol: float,
    rel_tol: float
):
    """
    Unified test for correctness of Triton LinOSS implementation.
    
    Tests that the Triton and original implementations produce similar outputs
    across different layer types, model sizes, and tolerance levels.
    
    Args:
        layer_name: Type of layer to test ('IM', 'IMEX', or 'Damped')
        batch_size: Batch size for test data
        seq_length: Sequence length for test data
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        ssm_size: SSM state dimension
        output_dim: Output dimension
        max_atol: Absolute tolerance for maximum difference
        mean_atol: Absolute tolerance for mean difference
        rel_tol: Relative tolerance threshold
    """
    key = jr.PRNGKey(SEED)
    data_key, model_key, state_key = jr.split(key, 3)
    test_data = jr.normal(data_key, (batch_size, seq_length, input_dim))
    
    original_model, triton_model = create_model_pair(
        layer_name=layer_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ssm_size=ssm_size,
        output_dim=output_dim,
        num_blocks=2,
        classification=True,
        tanh_output=False,
        output_step=1,
        key=model_key
    )
    
    orig_state = eqx.nn.State(original_model)
    trit_state = eqx.nn.State(triton_model)
    orig_out, _ = run_forward_pass(original_model, test_data, orig_state, state_key)
    trit_out, _ = run_forward_pass(triton_model, test_data, trit_state, state_key)
    
    assert orig_out.shape == trit_out.shape, (
        f"Shape mismatch: original {orig_out.shape} vs triton {trit_out.shape}"
    )
    assert orig_out.dtype == trit_out.dtype, (
        f"Dtype mismatch: original {orig_out.dtype} vs triton {trit_out.dtype}"
    )
    
    diffs = compute_differences(orig_out, trit_out)
    
    assert diffs['max_diff'] < max_atol, (
        f"Max difference {diffs['max_diff']:.6e} exceeds tolerance {max_atol:.6e}"
    )
    assert diffs['mean_diff'] < mean_atol, (
        f"Mean difference {diffs['mean_diff']:.6e} exceeds tolerance {mean_atol:.6e}"
    )
    assert diffs['rel_diff'] < rel_tol, (
        f"Relative difference {diffs['rel_diff']:.6e} exceeds tolerance {rel_tol:.6e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
