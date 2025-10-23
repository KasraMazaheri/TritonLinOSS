import pytest
import torch
import jax
import jax.numpy as jnp

import time
from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction
from src.damped_linoss.parallel_scan.jax_version import jax_scan

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

@pytest.mark.parametrize("L", [1, 16, 32, 64, 128, 256, 512, 4096, 8192, 16384])
@pytest.mark.parametrize("P", [1, 8, 16, 32, 64])
def test_parallel_scan_different_sizes(L, P):
    """Test parallel scan implementation with different dimensions."""
    M, F = generate_random_torch_tensors(L, P)
    torch_output_M, torch_output_F = ParallelScanFunction.apply(M, F)
    start_time = time.perf_counter()
    torch_output_M, torch_output_F = ParallelScanFunction.apply(M, F)
    end_time = time.perf_counter()
    print(f"Parallel scan with L={L}, P={P} took {end_time - start_time:.6f} seconds.")

    jax_M, jax_F = torch_to_jax(M), torch_to_jax(F)
    jax_F_complex = real_imag_to_complex(jax_F)
    jax_M = jax_M * jnp.ones((L, 4 * P))
    jax_output_M, jax_output_F = jax_scan(jax_M, jax_F_complex)
    start_time = time.perf_counter()
    jax_output_M, jax_output_F = jax_scan(jax_M, jax_F_complex)
    end_time = time.perf_counter()
    print(f"JAX parallel scan with L={L}, P={P} took {end_time - start_time:.6f} seconds.")

    assert jnp.allclose(jax_output_M, torch_to_jax(torch_output_M), atol=1e-5)
    assert jnp.allclose(jax_output_F, real_imag_to_complex(torch_to_jax(torch_output_F)), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])