"""
This module contains a function for generating the coefficients for a Hermite cubic spline with backwards differences.
"""

import numpy as np
import diffrax
import jax
import jax.numpy as jnp


def calc_coeffs(data, include_time, T):
    if include_time:
        ts = data[:, :, 0]
    else:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, data)
    return coeffs


def batch_calc_coeffs(data, include_time, T, inmemory=True):
    N = len(data)
    batchsize = 256
    num_batches = N // batchsize
    remainder = N % batchsize
    coeffs = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        coeffs.append(
            out_func(
                calc_coeffs(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), include_time, T
                )
            )
        )
    if remainder > 0:
        coeffs.append(
            out_func(calc_coeffs(in_func(data[-remainder:]), include_time, T))
        )
    if inmemory:
        coeffs = jnp.concatenate(coeffs)
    else:
        coeffs = np.concatenate(coeffs)
    return coeffs
