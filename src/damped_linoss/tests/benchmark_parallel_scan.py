import time
import itertools
import torch
import jax
import jax.numpy as jnp
import pandas as pd
from functools import partial

from src.damped_linoss.models.LinOSS import binary_operator
from src.damped_linoss.models.TorchLinOSS import (
    binary_operator as torch_binary_operator,
)
from src.damped_linoss.parallel_scan.torch_associative_scan import (
    associative_scan as torch_associative_scan,
)
from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_torch_data(B, L, P, dtype=torch.bfloat16, VAR=1e-3, requires_grad=False):
    device = get_device()
    M = (
        torch.randn(B, 4 * P, dtype=dtype, device=device, requires_grad=requires_grad)
        * VAR
    )
    F = (
        torch.randn(
            B, L, 2 * P, 2, dtype=dtype, device=device, requires_grad=requires_grad
        )
        * VAR
    )
    if requires_grad:
        M.retain_grad()
        F.retain_grad()
    return M, F


def generate_jax_data(B, L, P, dtype=jnp.bfloat16, VAR=1e-3, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    M = jax.random.normal(k1, (B, 4 * P), dtype=dtype) * VAR
    F = jax.random.normal(k2, (B, L, 2 * P, 2), dtype=dtype) * VAR
    F = F[..., 0] + 1j * F[..., 1]
    return M, F


def benchmark_callable(name, func, data_gen, n_warmup, n_repeat, sync_type="torch"):
    """Utility to benchmark a callable."""
    # Warmup
    for _ in range(n_warmup):
        args = data_gen()
        out = func(*args)
        if sync_type == "jax":
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

    if sync_type == "torch" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        args = data_gen()
        if sync_type == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        out = func(*args)

        if sync_type == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif sync_type == "jax":
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

        times.append(time.time() - t0)

    avg_time = sum(times) / n_repeat
    print(f"\t{name}: {avg_time * 1000:.2f} ms")
    return avg_time


def benchmark_config(
    B,
    L,
    P,
    TILE_L,
    n_warmup,
    n_repeat,
    torch_dtype=torch.bfloat16,
    jax_dtype=jnp.bfloat16,
):
    results = {}

    # --- Torch ---
    torch_gen = partial(generate_torch_data, B, L, P, dtype=torch_dtype)
    torch_gen_grad = partial(
        generate_torch_data, B, L, P, dtype=torch_dtype, requires_grad=True
    )

    results["torch_fwd"] = benchmark_callable(
        "Torch forward",
        lambda m, f: ParallelScanFunction.apply(m, f, TILE_L),
        torch_gen,
        n_warmup,
        n_repeat,
        "torch",
    )

    def torch_fwdbwd(m, f):
        om, of = ParallelScanFunction.apply(m, f, TILE_L)
        (om.sum() + of.sum()).backward()
        return om, of

    results["torch_fwdbwd"] = benchmark_callable(
        "Torch fwd+bwd", torch_fwdbwd, torch_gen_grad, n_warmup, n_repeat, "torch"
    )

    # --- Torch Compile ---
    def _scan_impl(m, f):
        return torch_associative_scan(
            torch_binary_operator,
            (m.unsqueeze(1).expand(B, L, 4 * P), f),
            reverse=False,
            axis=1,
        )

    compiled_scan = torch.compile(_scan_impl)

    results["torch_compile_fwd"] = benchmark_callable(
        "Compile forward", compiled_scan, torch_gen, n_warmup, n_repeat, "torch"
    )

    def comp_fwdbwd(m, f):
        om, of = compiled_scan(m, f)
        (om.sum() + of.sum()).backward()
        return om, of

    results["torch_compile_fwdbwd"] = benchmark_callable(
        "Compile fwd+bwd", comp_fwdbwd, torch_gen_grad, n_warmup, n_repeat, "torch"
    )

    # --- JAX ---
    key = jax.random.PRNGKey(0)

    def jax_gen():
        nonlocal key
        key, k = jax.random.split(key)
        return generate_jax_data(B, L, P, dtype=jax_dtype, key=k)

    @jax.jit
    @jax.vmap
    def jax_scan(m, f):
        return jax.lax.associative_scan(binary_operator, (m * jnp.ones((L, 4 * P)), f))

    results["jax_fwd"] = benchmark_callable(
        "JAX forward", jax_scan, jax_gen, n_warmup, n_repeat, "jax"
    )

    @jax.jit
    def jax_loss(m, f):
        om, of = jax_scan(m, f)
        return om.sum() + of.real.sum() + of.imag.sum()

    grad_func = jax.jit(jax.grad(jax_loss, argnums=(0, 1)))

    results["jax_fwdbwd"] = benchmark_callable(
        "JAX fwd+bwd", grad_func, jax_gen, n_warmup, n_repeat, "jax"
    )

    return results


def run_benchmarks(configs, n_warmup=3, n_repeat=10, dtypes=None):
    if dtypes is None:
        dtypes = [(torch.bfloat16, jnp.bfloat16)]

    rows = []
    for t_dtype, j_dtype in dtypes:
        for cfg in configs:
            L, P, TILE_L = cfg
            B = max(2**20 // (L * P), 1)
            print(
                f"Benchmarking: B={B}, L={L}, P={P}, TILE_L={TILE_L}, dtype={t_dtype}"
            )

            try:
                res = benchmark_config(
                    B, L, P, TILE_L, n_warmup, n_repeat, t_dtype, j_dtype
                )
                row = {
                    "B": B,
                    "L": L,
                    "P": P,
                    "TILE_L": TILE_L,
                    "dtype": str(t_dtype),
                    **res,
                    "fwd_speedup": res["jax_fwd"] / res["torch_fwd"],
                    "fwdbwd_speedup": res["jax_fwdbwd"] / res["torch_fwdbwd"],
                }
                rows.append(row)
            except Exception as e:
                print(f"Failed: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    L_vals = [2**p for p in range(12, 17, 2)]
    P_vals = [2**p for p in range(6, 9, 2)]
    TILE_L_vals = [2**p for p in range(7, 9)]
    configs = list(itertools.product(L_vals, P_vals, TILE_L_vals))

    # Example of running with multiple dtypes
    dtypes_to_test = [
        (torch.bfloat16, jnp.bfloat16),
        (torch.float32, jnp.float32),  # Uncomment to test float32
    ]

    print(
        f"Running benchmarks for {len(configs)} configs x {len(dtypes_to_test)} dtypes..."
    )
    df = run_benchmarks(configs, n_warmup=2, n_repeat=5, dtypes=dtypes_to_test)
    print(df)
    df.to_csv("benchmark_parallel_scan_results.csv", index=False)
