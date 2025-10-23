import jax
import jax.numpy as jnp
from src.damped_linoss.models.LinOSS import binary_operator

@jax.jit
def jax_scan(jax_M, jax_F_complex):
    return jax.lax.associative_scan(
            binary_operator, 
            (jax_M, jax_F_complex)
        )

