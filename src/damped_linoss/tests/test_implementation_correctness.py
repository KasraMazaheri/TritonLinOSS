import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from ..models.LinOSS import binary_operator
from ..models.TorchLinOSS import (
    binary_operator as torch_binary_operator,
)
from ..parallel_scan.torch_interface import ParallelScanFunction
from ..parallel_scan.torch_associative_scan import (
    associative_scan as torch_associative_scan,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("B,L,P", [(2, 128, 64), (4, 256, 32)])
def test_implementations_match(B, L, P, dtype):
    """Test that Triton, Torch Compile, and JAX implementations all produce matching results."""

    # Set seed for reproducibility
    torch.manual_seed(42)
    jax_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
    TILE_L = 128
    VAR = 1e-3

    # Generate torch data
    M_torch = torch.randn(B, 4 * P, dtype=dtype, device="cuda") * VAR
    F_torch = torch.randn(B, L, 2 * P, 2, dtype=dtype, device="cuda") * VAR

    # Generate JAX data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    M_jax = jax.random.normal(k1, (B, 4 * P), dtype=jax_dtype) * VAR
    F_jax = jax.random.normal(k2, (B, L, 2 * P, 2), dtype=jax_dtype) * VAR
    F_jax = F_jax[..., 0] + 1j * F_jax[..., 1]

    # 1. Triton implementation
    OM_triton, OF_triton = ParallelScanFunction.apply(
        M_torch.clone(), F_torch.clone(), TILE_L
    )

    # 2. Torch Compile implementation
    def _scan(m, f):
        return torch_associative_scan(
            torch_binary_operator,
            (m.unsqueeze(1).expand(B, L, 4 * P), f),
            reverse=False,
            axis=1,
        )

    compile_scan = torch.compile(_scan)
    OM_comp, OF_comp = compile_scan(M_torch.clone(), F_torch.clone())

    # 3. JAX implementation
    @jax.vmap
    def jax_scan(m, f):
        return jax.lax.associative_scan(binary_operator, (m * jnp.ones((L, 4 * P)), f))

    OM_jax, OF_jax = jax_scan(M_jax, F_jax)

    # Convert to numpy for comparison
    OM_t_np = OM_triton.detach().cpu().float().numpy()
    OF_t_np = OF_triton.detach().cpu().float().numpy()
    OM_c_np = OM_comp.detach().cpu().float().numpy()
    OF_c_np = OF_comp.detach().cpu().float().numpy()
    OM_j_np = np.array(OM_jax, dtype=np.float32)
    OF_j_np = np.stack([OF_jax.real, OF_jax.imag], axis=-1).astype(np.float32)

    # Check all implementations match within tolerance
    tol = 1e-2

    # Triton vs JAX
    assert np.abs(OM_t_np - OM_j_np).max() < tol, "Triton vs JAX OM mismatch"
    assert np.abs(OF_t_np - OF_j_np).max() < tol, "Triton vs JAX OF mismatch"

    # Compile vs JAX
    assert np.abs(OM_c_np - OM_j_np).max() < tol, "Compile vs JAX OM mismatch"
    assert np.abs(OF_c_np - OF_j_np).max() < tol, "Compile vs JAX OF mismatch"

    # Triton vs Compile
    assert np.abs(OM_t_np - OM_c_np).max() < tol, "Triton vs Compile OM mismatch"
    assert np.abs(OF_t_np - OF_c_np).max() < tol, "Triton vs Compile OF mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
