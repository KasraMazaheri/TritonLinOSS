"""Benchmark LinOSS scan/model backends.

This script is intentionally not a pytest module. It measures:
- Triton PyTorch scan backend
- native PyTorch associative-scan backend
- optional torch.compile native scan
- JAX associative_scan reference
- Torch LinOSS model forward with native vs Triton scan
- JAX LinOSS model forward
"""

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def torch_timer(fn, warmup=10, repeat=50):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def jax_timer(fn, warmup=5, repeat=30):
    import jax

    for _ in range(warmup):
        out = fn()
        jax.tree.map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.tree.map(lambda x: x.block_until_ready(), out)
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def make_scan_data(batch, length, state, torch_dtype):
    import torch

    torch.manual_seed(0)
    M = torch.randn(batch, 4 * state, device="cuda", dtype=torch_dtype) * 1e-3
    F = (
        torch.randn(batch, length, 2 * state, 2, device="cuda", dtype=torch_dtype)
        * 1e-3
    )
    return M, F


def torch_native_scan(M, F):
    from damped_linoss.models.TorchLinOSS import (
        binary_operator as torch_binary_operator,
    )
    from damped_linoss.parallel_scan.torch_associative_scan import associative_scan

    batch, length = F.shape[:2]
    state = F.shape[2] // 2
    return associative_scan(
        torch_binary_operator,
        (M.unsqueeze(1).expand(batch, length, 4 * state), F),
        reverse=False,
        axis=1,
    )


def make_jax_scan(batch, length, state, jax_dtype):
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from damped_linoss.models.LinOSS import binary_operator as jax_binary_operator

    key = jr.PRNGKey(0)
    m_key, f_key = jr.split(key)
    M = jax.random.normal(m_key, (batch, 4 * state), dtype=jax_dtype) * 1e-3
    F_parts = (
        jax.random.normal(f_key, (batch, length, 2 * state, 2), dtype=jax_dtype)
        * 1e-3
    )
    F = F_parts[..., 0] + 1j * F_parts[..., 1]

    @jax.jit
    @jax.vmap
    def scan(m, f):
        return jax.lax.associative_scan(
            jax_binary_operator,
            (m * jnp.ones((length, 4 * state), dtype=jax_dtype), f),
        )

    @jax.jit
    def grad_scan(m, f):
        def loss_fn(mm, ff):
            OM, OF = scan(mm, ff)
            return OM.sum() + OF.real.sum() + OF.imag.sum()

        return jax.grad(loss_fn, argnums=(0, 1))(m, f)

    return M, F, scan, grad_scan


def bench_scan(configs, out_path, include_compile, backend):
    rows = []
    if backend == "jax":
        import jax.numpy as jnp

        dtype_pairs = [
            ("float32", None, jnp.float32),
            ("bfloat16", None, jnp.bfloat16),
        ]
    else:
        import torch

        from damped_linoss.parallel_scan.torch_interface import ParallelScanFunction

        dtype_pairs = [
            ("float32", torch.float32, None),
            ("bfloat16", torch.bfloat16, None),
        ]
    tile_values = [None, 64, 128, 256, 512]
    for batch, length, state in configs:
        for dtype_name, torch_dtype, jax_dtype in dtype_pairs:
            if backend == "jax":
                jM, jF, jax_scan, jax_grad_scan = make_jax_scan(
                    batch, length, state, jax_dtype
                )
                row = {
                    "kind": "scan",
                    "backend": "jax",
                    "B": batch,
                    "L": length,
                    "P": state,
                    "dtype": dtype_name,
                    "tile": "",
                    "fwd_ms": jax_timer(lambda: jax_scan(jM, jF), warmup=5, repeat=30),
                    "fwdbwd_ms": jax_timer(
                        lambda: jax_grad_scan(jM, jF), warmup=3, repeat=20
                    ),
                }
                rows.append(row)
                print(row, flush=True)
                continue

            M, F = make_scan_data(batch, length, state, torch_dtype)
            M_grad = M.detach().clone().requires_grad_(True)
            F_grad = F.detach().clone().requires_grad_(True)

            native_fwd_ms = torch_timer(
                lambda: torch_native_scan(M, F), warmup=5, repeat=20
            )

            def native_fwdbwd():
                M_grad.grad = None
                F_grad.grad = None
                OM, OF = torch_native_scan(M_grad, F_grad)
                (OM.float().sum() + OF.float().sum()).backward()

            native_fwdbwd_ms = torch_timer(native_fwdbwd, warmup=3, repeat=10)
            rows.append(
                {
                    "kind": "scan",
                    "backend": "torch_native",
                    "B": batch,
                    "L": length,
                    "P": state,
                    "dtype": dtype_name,
                    "tile": "",
                    "fwd_ms": native_fwd_ms,
                    "fwdbwd_ms": native_fwdbwd_ms,
                }
            )
            print(rows[-1], flush=True)

            if include_compile:
                row = {
                    "kind": "scan",
                    "backend": "torch_compile",
                    "B": batch,
                    "L": length,
                    "P": state,
                    "dtype": dtype_name,
                    "tile": "",
                    "fwd_ms": None,
                    "fwdbwd_ms": None,
                    "error": "",
                }
                try:
                    compiled_scan = torch.compile(torch_native_scan)
                    row["fwd_ms"] = torch_timer(
                        lambda: compiled_scan(M, F), warmup=10, repeat=30
                    )

                    def compiled_fwdbwd():
                        M_grad.grad = None
                        F_grad.grad = None
                        OM, OF = compiled_scan(M_grad, F_grad)
                        (OM.float().sum() + OF.float().sum()).backward()

                    row["fwdbwd_ms"] = torch_timer(
                        compiled_fwdbwd, warmup=5, repeat=10
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                print(rows[-1], flush=True)

            for tile in tile_values:
                fwd = torch_timer(
                    lambda tile=tile: ParallelScanFunction.apply(M, F)
                    if tile is None
                    else ParallelScanFunction.apply(M, F, tile),
                    warmup=10,
                    repeat=50,
                )

                def triton_fwdbwd(tile=tile):
                    M_grad.grad = None
                    F_grad.grad = None
                    if tile is None:
                        OM, OF = ParallelScanFunction.apply(M_grad, F_grad)
                    else:
                        OM, OF = ParallelScanFunction.apply(M_grad, F_grad, tile)
                    (OM.float().sum() + OF.float().sum()).backward()

                fwdbwd = torch_timer(triton_fwdbwd, warmup=5, repeat=20)
                row = (
                    {
                        "kind": "scan",
                        "backend": "torch_triton",
                        "B": batch,
                        "L": length,
                        "P": state,
                        "dtype": dtype_name,
                        "tile": "default" if tile is None else tile,
                        "fwd_ms": fwd,
                        "fwdbwd_ms": fwdbwd,
                    }
                )
                rows.append(row)
                print(row, flush=True)

    write_rows(out_path, rows)


def run_batched_jax_forward(X, model, state, key):
    import jax

    return jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(X, state, key)[0]


def make_model_pair(layer_name, batch, length, input_dim, state_dim, hidden_dim):
    import functools as ft

    import equinox as eqx
    import jax
    import jax.random as jr
    import numpy as np
    import torch

    from damped_linoss.models.LinOSS import LinOSS as JaxLinOSS
    from damped_linoss.models.TorchLinOSS import LinOSS as TorchLinOSS
    from damped_linoss.utils.from_jax import from_jax_to_torch

    key = jr.PRNGKey(42)
    model = JaxLinOSS(
        layer_name=layer_name,
        input_dim=input_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=8,
        num_blocks=2,
        classification=True,
        tanh_output=False,
        output_step=1,
        key=key,
    )
    model = eqx.tree_inference(model, value=True)
    state = eqx.nn.State(model)
    jax_forward = jax.jit(
        ft.partial(run_batched_jax_forward, model=model, state=state, key=key)
    )

    torch_triton = TorchLinOSS(
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        8,
        2,
        classification=True,
        use_triton=True,
    ).cuda().eval()
    torch_native = TorchLinOSS(
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        8,
        2,
        classification=True,
        use_triton=False,
    ).cuda().eval()
    from_jax_to_torch(model, torch_triton)
    from_jax_to_torch(model, torch_native)

    x_key = jr.PRNGKey(7)
    X_jax = jr.normal(x_key, (batch, length, input_dim))
    X_torch = torch.tensor(np.array(X_jax), dtype=torch.float32, device="cuda")
    return X_jax, X_torch, jax_forward, torch_triton, torch_native


def make_jax_model(layer_name, batch, length, input_dim, state_dim, hidden_dim):
    import functools as ft

    import equinox as eqx
    import jax
    import jax.random as jr

    from damped_linoss.models.LinOSS import LinOSS as JaxLinOSS

    key = jr.PRNGKey(42)
    model = JaxLinOSS(
        layer_name=layer_name,
        input_dim=input_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        output_dim=8,
        num_blocks=2,
        classification=True,
        tanh_output=False,
        output_step=1,
        key=key,
    )
    model = eqx.tree_inference(model, value=True)
    state = eqx.nn.State(model)
    jax_forward = jax.jit(
        ft.partial(run_batched_jax_forward, model=model, state=state, key=key)
    )
    X_jax = jr.normal(jr.PRNGKey(7), (batch, length, input_dim))
    return X_jax, jax_forward


def make_torch_models(layer_name, batch, length, input_dim, state_dim, hidden_dim):
    import torch

    from damped_linoss.models.TorchLinOSS import LinOSS as TorchLinOSS

    torch_triton = TorchLinOSS(
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        8,
        2,
        classification=True,
        use_triton=True,
    ).cuda().eval()
    torch_native = TorchLinOSS(
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        8,
        2,
        classification=True,
        use_triton=False,
    ).cuda().eval()
    torch.manual_seed(7)
    X_torch = torch.randn(batch, length, input_dim, device="cuda")
    return X_torch, torch_triton, torch_native


def bench_model(configs, out_path, backend):
    rows = []
    for layer_name, batch, length, input_dim, state_dim, hidden_dim in configs:
        if backend == "jax":
            X_jax, jax_forward = make_jax_model(
                layer_name, batch, length, input_dim, state_dim, hidden_dim
            )
            row = {
                "kind": "model",
                "backend": "jax",
                "layer": layer_name,
                "B": batch,
                "L": length,
                "input_dim": input_dim,
                "P": state_dim,
                "hidden_dim": hidden_dim,
                "fwd_ms": jax_timer(lambda: jax_forward(X_jax), warmup=5, repeat=30),
            }
            rows.append(row)
            print(row, flush=True)
            continue

        X_torch, torch_triton, torch_native = make_torch_models(
            layer_name, batch, length, input_dim, state_dim, hidden_dim
        )
        import torch

        with torch.no_grad():
            rows.append(
                {
                    "kind": "model",
                    "backend": "torch_triton",
                    "layer": layer_name,
                    "B": batch,
                    "L": length,
                    "input_dim": input_dim,
                    "P": state_dim,
                    "hidden_dim": hidden_dim,
                    "fwd_ms": torch_timer(
                        lambda: torch_triton(X_torch), warmup=10, repeat=50
                    ),
                }
            )
            print(rows[-1], flush=True)
            rows.append(
                {
                    "kind": "model",
                    "backend": "torch_native",
                    "layer": layer_name,
                    "B": batch,
                    "L": length,
                    "input_dim": input_dim,
                    "P": state_dim,
                    "hidden_dim": hidden_dim,
                    "fwd_ms": torch_timer(
                        lambda: torch_native(X_torch), warmup=5, repeat=20
                    ),
                }
            )
            print(rows[-1], flush=True)
    write_rows(out_path, rows)


def write_rows(out_path, rows):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scan", "model"], required=True)
    parser.add_argument("--backend", choices=["torch", "jax"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this benchmark")
        print("Torch:", torch.__version__)
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        import jax

        print("JAX:", jax.__version__, jax.devices()[0])
    if args.mode == "scan":
        configs = [
            (16, 128, 32),
            (8, 512, 64),
            (4, 2048, 128),
            (2, 8192, 128),
            (1, 16384, 256),
        ]
        bench_scan(configs, args.out, args.include_compile, args.backend)
    else:
        configs = [
            ("IM", 16, 256, 8, 64, 64),
            ("IMEX", 16, 256, 8, 64, 64),
            ("Damped", 16, 256, 8, 64, 64),
            ("IM", 8, 1024, 16, 128, 128),
            ("Damped", 8, 1024, 16, 128, 128),
        ]
        bench_model(configs, args.out, args.backend)


if __name__ == "__main__":
    main()
