import pytest
import torch
import jax
import jax.numpy as jnp
from src.damped_linoss.models.LinOSS import binary_operator
from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction

# Set random seeds for reproducibility
torch.manual_seed(0)

def generate_random_torch_tensors(L, P, variance=1e-1):
    """Generate random torch tensors for testing parallel scan."""
    M = torch.randn(P * 4, device='cuda') * variance
    F = torch.randn(L, 2 * P, 2, device='cuda') * variance
    return M, F

def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    return jax.device_put(tensor.cpu().numpy())

def real_imag_to_complex(tensor):
    """Convert real-imaginary tensor to complex array."""
    return tensor[..., 0] + 1j * tensor[..., 1]

@pytest.mark.parametrize("L", [1, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("P", [1, 8, 16, 32, 64])
def test_parallel_scan_different_sizes(L, P):
    """Test parallel scan implementation with different dimensions."""
    M, F = generate_random_torch_tensors(L, P)
    torch_output_M, torch_output_F = ParallelScanFunction.apply(M, F)

    jax_M, jax_F = torch_to_jax(M), torch_to_jax(F)
    jax_output_M, jax_output_F = jax.lax.associative_scan(
        binary_operator, 
        (jax_M * jnp.ones((L, 4 * P)), real_imag_to_complex(jax_F))
    )

    assert jnp.allclose(jax_output_M, torch_to_jax(torch_output_M), atol=1e-5)
    assert jnp.allclose(jax_output_F, real_imag_to_complex(torch_to_jax(torch_output_F)), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])