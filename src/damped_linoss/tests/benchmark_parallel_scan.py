import time
import torch
import jax
import jax.numpy as jnp
import pandas as pd
import itertools
from src.damped_linoss.models.LinOSS import binary_operator
from src.damped_linoss.models.TorchLinOSS import (
    binary_operator as torch_binary_operator,
)
from src.damped_linoss.parallel_scan.torch_associative_scan import (
    associative_scan as torch_associative_scan,
)
from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction


def generate_torch_data(B, L, P, VAR=1e-3, requires_grad=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = (
        torch.randn(
            B, P * 4, dtype=torch.bfloat16, device=device, requires_grad=requires_grad
        )
        * VAR
    )
    F = (
        torch.randn(
            B,
            L,
            2 * P,
            2,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=requires_grad,
        )
        * VAR
    )
    if requires_grad:
        M.retain_grad()
        F.retain_grad()
    return M, F


def generate_jax_data(B, L, P, VAR=1e-3, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    key_M, key_F = jax.random.split(key)
    M = (
        jax.random.normal(
            key_M,
            (
                B,
                4 * P,
            ),
            dtype=jnp.bfloat16,
        )
        * VAR
    )
    F = jax.random.normal(key_F, (B, L, 2 * P, 2), dtype=jnp.bfloat16) * VAR
    F = F[..., 0] + 1j * F[..., 1]
    return M, F


def benchmark_config(B, L, P, TILE_L, n_warmup, n_repeat):
    results = {}

    jax_key_seed = 0
    torch.manual_seed(0)

    # Torch Forward
    torch_times_fwd = []
    for _ in range(n_warmup):
        M, F = generate_torch_data(B, L, P)
        _ = ParallelScanFunction.apply(M, F, TILE_L)
    for _ in range(n_repeat):
        M, F = generate_torch_data(B, L, P)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        _ = ParallelScanFunction.apply(M, F, TILE_L)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_times_fwd.append(time.time() - t0)
    results["torch_fwd"] = sum(torch_times_fwd) / n_repeat

    print(f"\tTorch forward:          {results['torch_fwd'] * 1000:.2f} ms")

    # Torch Forward+Backward
    torch_times_fwbw = []
    tOM, tOF = None, None
    for _ in range(n_warmup):
        M, F = generate_torch_data(B, L, P, requires_grad=True)
        OM, OF = ParallelScanFunction.apply(M, F, TILE_L)
        tOM, tOF = OM.detach().clone(), OF.detach().clone()
        loss = OM.sum() + OF.sum()
        loss.backward()
    for _ in range(n_repeat):
        M, F = generate_torch_data(B, L, P, requires_grad=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        OM, OF = ParallelScanFunction.apply(M, F, TILE_L)
        loss = OM.sum() + OF.sum()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_times_fwbw.append(time.time() - t0)
    results["torch_fwdbwd"] = sum(torch_times_fwbw) / n_repeat

    print(f"\tTorch forward+backward: {results['torch_fwdbwd'] * 1000:.2f} ms")

    # Torch Forward (torch.compile)
    def _torch_compile_fwd(M, F):
        return torch_associative_scan(
            torch_binary_operator,
            (M.unsqueeze(1).expand(B, L, 4 * P), F),
            reverse=False,
            axis=0,
        )

    torch_compile_fwd = torch.compile(_torch_compile_fwd)

    torch_times_fwd = []
    for _ in range(n_warmup):
        M, F = generate_torch_data(B, L, P)
        tcOM, tcOF = torch_compile_fwd(M, F)
        print("OM diff", ((tOM - tcOM).mean() / tOM.mean()).cpu())
        print("OF Diff", ((tOF - tcOF).mean() / tOF.mean()).cpu())
    for _ in range(n_repeat):
        M, F = generate_torch_data(B, L, P)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        _ = torch_compile_fwd(M, F)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_times_fwd.append(time.time() - t0)
    results["torch_compile_fwd"] = sum(torch_times_fwd) / n_repeat

    print(
        f"\tTorch (torch.compile) forward: {results['torch_compile_fwd'] * 1000:.2f} ms"
    )

    # Torch Forward+Backward (torch.compile)
    torch_times_fwbw = []
    for _ in range(n_warmup):
        M, F = generate_torch_data(B, L, P, requires_grad=True)
        OM, OF = torch_compile_fwd(M, F)
        loss = OM.sum() + OF.sum()
        loss.backward()
    for _ in range(n_repeat):
        M, F = generate_torch_data(B, L, P, requires_grad=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        OM, OF = torch_compile_fwd(M, F)
        loss = OM.sum() + OF.sum()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_times_fwbw.append(time.time() - t0)
    results["torch_compile_fwdbwd"] = sum(torch_times_fwbw) / n_repeat

    print(
        f"\tTorch (torch.compile) forward+backward: {results['torch_compile_fwdbwd'] * 1000:.2f} ms"
    )

    # JAX wrappers
    @jax.jit
    @jax.vmap
    def jax_forward(jM, jF):
        return jax.lax.associative_scan(
            binary_operator, (jM * jnp.ones((L, 4 * P)), jF)
        )

    @jax.jit
    def jax_loss(jM, jF):
        jOM, jOF = jax_forward(jM, jF)
        return jOM.sum() + jOF.real.sum() + jOF.imag.sum()

    grad_func = jax.jit(jax.grad(jax_loss, argnums=(0, 1)))

    # JAX Forward
    jax_times_fwd = []
    key = jax.random.PRNGKey(jax_key_seed)
    for _ in range(n_warmup):
        key, subkey = jax.random.split(key)
        jM, jF = generate_jax_data(B, L, P, key=subkey)
        jOM, jOF = jax_forward(jM, jF)
        jOM.block_until_ready()
        jOF.block_until_ready()
    for _ in range(n_repeat):
        key, subkey = jax.random.split(key)
        jM, jF = generate_jax_data(B, L, P, key=subkey)
        t0 = time.time()
        jOM, jOF = jax_forward(jM, jF)
        jOM.block_until_ready()
        jOF.block_until_ready()
        jax_times_fwd.append(time.time() - t0)
    results["jax_fwd"] = sum(jax_times_fwd) / n_repeat

    print(f"\tJax forward:          {results['jax_fwd'] * 1000:.2f} ms")

    # JAX Forward+Backward
    jax_times_fwbw = []
    key = jax.random.PRNGKey(jax_key_seed + 42)
    for _ in range(n_warmup):
        key, subkey = jax.random.split(key)
        jM, jF = generate_jax_data(B, L, P, key=subkey)
        jgM, jgF = grad_func(jM, jF)
        jgM.block_until_ready()
        jgF.block_until_ready()
    for _ in range(n_repeat):
        key, subkey = jax.random.split(key)
        jM, jF = generate_jax_data(B, L, P, key=subkey)
        t0 = time.time()
        jgM, jgF = grad_func(jM, jF)
        jgM.block_until_ready()
        jgF.block_until_ready()
        jax_times_fwbw.append(time.time() - t0)
    results["jax_fwdbwd"] = sum(jax_times_fwbw) / n_repeat

    print(f"\tJax forward+backward: {results['jax_fwdbwd'] * 1000:.2f} ms")

    return results


def run_benchmarks(configs, n_warmup=3, n_repeat=10):
    rows = []
    for cfg in configs:
        L, P, TILE_L = cfg
        B = max(2**20 // (L * P), 1)
        print(f"Benchmarking: B={B}, L={L}, P={P}, TILE_L={TILE_L}")
        res = benchmark_config(B, L, P, TILE_L, n_warmup, n_repeat)
        row = {
            "B": B,
            "L": L,
            "P": P,
            "TILE_L": TILE_L,
            "torch_fwd": res["torch_fwd"],
            "torch_compile_fwd": res["torch_compile_fwd"],
            "jax_fwd": res["jax_fwd"],
            "torch_fwdbwd": res["torch_fwdbwd"],
            "torch_compile_fwdbwd": res["torch_compile_fwdbwd"],
            "jax_fwdbwd": res["jax_fwdbwd"],
            "fwd_speedup": res["jax_fwd"] / res["torch_fwd"],
            "fwdbwd_speedup": res["jax_fwdbwd"] / res["torch_fwdbwd"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    L_vals = [2**p for p in range(12, 17, 2)]
    P_vals = [2**p for p in range(6, 9, 2)]
    TILE_L_vals = [2**p for p in range(7, 9)]
    configs = [cfg for cfg in itertools.product(L_vals, P_vals, TILE_L_vals)][:1]
    print(f"Running benchmarks for {len(configs)} configurations...")
    df = run_benchmarks(configs, n_warmup=2, n_repeat=5)
    print(df)
