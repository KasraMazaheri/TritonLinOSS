import pytest
import torch
import jax
import jax.numpy as jnp
from src.damped_linoss.models.LinOSS import binary_operator
from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction

# Set random seeds for reproducibility
torch.manual_seed(0)

def generate_random_torch_tensors(L, P, variance=1e-3, requires_grad=False):
    """Generate random torch tensors for testing parallel scan.

    If requires_grad is True, tensors are created with requires_grad and
    .retain_grad() is called so their .grad fields are populated after
    backward().
    """
    M = torch.randn(P * 4, device='cuda', requires_grad=requires_grad) * variance
    F = torch.randn(L, 2 * P, 2, device='cuda', requires_grad=requires_grad) * variance
    if requires_grad:
        # Ensure we keep gradients on leaf tensors
        M.retain_grad()
        F.retain_grad()
    return M, F

def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    return jax.device_put(tensor.detach().cpu().numpy())

def real_imag_to_complex(tensor):
    """Convert real-imaginary tensor to complex array."""
    return tensor[..., 0] + 1j * tensor[..., 1]

@pytest.mark.parametrize("L", [1, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("P", [1, 8, 16, 32, 64])
def test_parallel_scan_fwd_bwd(L, P):
    """Test that forward and backward passes produce correct results matching the JAX implementation."""
    M, F = generate_random_torch_tensors(L, P, requires_grad=True)

    torch_output_M, torch_output_F = ParallelScanFunction.apply(M, F)
    torch_output_M.retain_grad()
    torch_output_F.retain_grad()
    loss = torch_output_M.sum() + torch_output_F[..., 0].sum()
    loss.backward()

    @jax.jit
    def jax_forward(jax_M, jax_F):
        jax_output_M, jax_output_F = jax.lax.associative_scan(
            binary_operator, (jax_M * jnp.ones((L, 4 * P)), real_imag_to_complex(jax_F))
        )
        return jax_output_M.sum() + jax_output_F.sum().real
    
    jax_M, jax_F = torch_to_jax(M), torch_to_jax(F)
    grad_func = jax.grad(jax_forward, argnums=(0, 1))
    jax_loss = jax_forward(jax_M, jax_F)
    jax_M_grad, jax_F_grad = grad_func(jax_M, jax_F)

    assert jnp.allclose(jax_loss,   torch_to_jax(loss),   atol=1e-2)
    assert jnp.allclose(jax_M_grad, torch_to_jax(M.grad), atol=1e-2)
    assert jnp.allclose(jax_F_grad, torch_to_jax(F.grad), atol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
