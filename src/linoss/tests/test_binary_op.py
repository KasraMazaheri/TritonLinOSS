import pytest
import jax.numpy as jnp
import torch
import numpy as np

from src.damped_linoss.models.LinOSS import binary_operator as jax_binary_operator
from src.damped_linoss.models.TorchLinOSS import (
    binary_operator as torch_binary_operator,
)


def generate_test_data(batch_size, P, seed=42):
    """Generate random test data for binary operator."""
    np.random.seed(seed)

    A_i = np.random.randn(batch_size, 4 * P).astype(np.float32)
    A_j = np.random.randn(batch_size, 4 * P).astype(np.float32)

    b_i_real = np.random.randn(batch_size, 2 * P).astype(np.float32)
    b_i_imag = np.random.randn(batch_size, 2 * P).astype(np.float32)
    b_j_real = np.random.randn(batch_size, 2 * P).astype(np.float32)
    b_j_imag = np.random.randn(batch_size, 2 * P).astype(np.float32)

    return {
        "A_i": A_i,
        "A_j": A_j,
        "b_i_real": b_i_real,
        "b_i_imag": b_i_imag,
        "b_j_real": b_j_real,
        "b_j_imag": b_j_imag,
    }


def prepare_jax_inputs(data):
    """Convert numpy data to JAX format."""
    b_i = data["b_i_real"] + 1j * data["b_i_imag"]
    b_j = data["b_j_real"] + 1j * data["b_j_imag"]
    return (jnp.array(data["A_i"]), jnp.array(b_i)), (
        jnp.array(data["A_j"]),
        jnp.array(b_j),
    )


def prepare_torch_inputs(data):
    """Convert numpy data to PyTorch format."""
    b_i = np.stack([data["b_i_real"], data["b_i_imag"]], axis=-1)
    b_j = np.stack([data["b_j_real"], data["b_j_imag"]], axis=-1)
    return (torch.from_numpy(data["A_i"]), torch.from_numpy(b_i)), (
        torch.from_numpy(data["A_j"]),
        torch.from_numpy(b_j),
    )


def compare_outputs(jax_out, torch_out, rtol=1e-5, atol=1e-6):
    """Compare JAX and PyTorch outputs."""
    A_jax, b_jax = jax_out
    A_torch, b_torch = torch_out

    np.testing.assert_allclose(
        np.array(A_jax),
        A_torch.numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="A outputs don't match",
    )

    np.testing.assert_allclose(
        np.array(b_jax).real,
        b_torch.numpy()[..., 0],
        rtol=rtol,
        atol=atol,
        err_msg="b real parts don't match",
    )
    np.testing.assert_allclose(
        np.array(b_jax).imag,
        b_torch.numpy()[..., 1],
        rtol=rtol,
        atol=atol,
        err_msg="b imaginary parts don't match",
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("P", [1, 8, 16, 32, 64])
def test_binary_operator_equivalence(batch_size, P):
    """Test that JAX and PyTorch binary operators."""
    data = generate_test_data(batch_size=batch_size, P=P)

    q_i_jax, q_j_jax = prepare_jax_inputs(data)
    q_i_torch, q_j_torch = prepare_torch_inputs(data)

    jax_out = jax_binary_operator(q_i_jax, q_j_jax)
    torch_out = torch_binary_operator(q_i_torch, q_j_torch)

    compare_outputs(jax_out, torch_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
