import pytest
import torch
import jax
import jax.numpy as jnp
from damped_linoss.models.LinOSS import binary_operator
from damped_linoss.models.TorchLinOSS import (
    binary_operator as torch_binary_operator,
)
from damped_linoss.parallel_scan.torch_associative_scan import (
    associative_scan as torch_associative_scan,
)


def generate_random_torch_tensors(
    B, L, P, variance=1e-3, requires_grad=False, device="cpu"
):
    """Generate random torch tensors for testing parallel scan.

    If requires_grad is True, tensors are created with requires_grad and
    .retain_grad() is called so their .grad fields are populated after
    backward().
    """
    M = torch.randn(B, P * 4, device=device, requires_grad=requires_grad) * variance
    F = (
        torch.randn(B, L, 2 * P, 2, device=device, requires_grad=requires_grad)
        * variance
    )
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


@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("L", [1, 16, 32, 64, 100, 128, 256])
@pytest.mark.parametrize("P", [1, 8, 16, 32, 35, 64])
def test_parallel_scan_fwd_bwd(B, L, P):
    """Test that forward and backward passes produce correct results matching the JAX implementation."""

    # Set random seeds for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    M, F = generate_random_torch_tensors(
        B, L, P, requires_grad=True, device=device
    )

    # Expand M to match the expected input shape for torch_associative_scan
    M_expanded = M.unsqueeze(1).expand(B, L, 4 * P)

    torch_output_M, torch_output_F = torch_associative_scan(
        torch_binary_operator, (M_expanded, F), reverse=False, axis=1
    )
    torch_output_M.retain_grad()
    torch_output_F.retain_grad()
    loss = torch_output_M.sum() + torch_output_F[..., 0].sum()
    loss.backward()

    @jax.vmap
    def jax_forward(jax_M, jax_F):
        return jax.lax.associative_scan(
            binary_operator, (jax_M * jnp.ones((L, 4 * P)), real_imag_to_complex(jax_F))
        )

    @jax.jit
    def jax_loss_fn(jax_M, jax_F):
        jax_output_M, jax_output_F = jax_forward(jax_M, jax_F)
        return jax_output_M.sum() + jax_output_F.sum().real

    jax_M, jax_F = torch_to_jax(M), torch_to_jax(F)
    grad_func = jax.grad(jax_loss_fn, argnums=(0, 1))
    jax_loss = jax_loss_fn(jax_M, jax_F)
    jax_M_grad, jax_F_grad = grad_func(jax_M, jax_F)

    assert jnp.allclose(jax_loss, torch_to_jax(loss), atol=1e-1)
    assert jnp.allclose(jax_M_grad, torch_to_jax(M.grad), atol=1e-1)
    assert jnp.allclose(jax_F_grad, torch_to_jax(F.grad), atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
