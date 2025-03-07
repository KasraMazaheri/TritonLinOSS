"""
This module contains functions for generating log-signature of paths over intervals of length stepsize.
"""

import numpy as np
import jax
import jax.numpy as jnp
from signax.signature import signature
from signax.signature_flattened import flatten
from signax.tensor_ops import log

from linoss.data_processing.hall_set import HallSet


def hall_basis_logsig(x, depth, t2l):
    logsig = flatten(log(signature(x, depth)))
    if depth == 1:
        return jnp.concatenate((jnp.array([0]), logsig))
    else:
        return t2l[:, 1:] @ logsig


def calc_paths(data, stepsize, depth):
    """
    Generate log-signature objects from data.

    In the future, this function will use RoughPy, and return path objects,
    which can be queried over any interval for the log-signature. Right now,
    it is necessary to specify the stepsize and depth ahead of time.
    """
    data = jnp.concatenate(
        (jnp.zeros((data.shape[0], 1, data.shape[-1])), data), axis=1
    )

    if depth == 2:
        hs = HallSet(data.shape[-1], depth)
        t2l = hs.t2l_matrix(depth)
    else:
        t2l = None

    prepend = lambda x: jnp.concatenate(
        (
            jnp.concatenate((jnp.zeros((1, data.shape[-1])), x[:-1, -1, :]))[
                :, None, :
            ],
            x,
        ),
        axis=1,
    )

    if stepsize > data.shape[1]:
        stepsize = data.shape[1]

    if data.shape[1] % stepsize != 0:
        final_data = data[:, -(data.shape[1] % stepsize) - 1 :, ...]
        data = data[:, : -(data.shape[1] % stepsize), ...].reshape(
            data.shape[0], -1, stepsize, data.shape[-1]
        )
        data = jax.vmap(prepend)(data)
        final_data = prepend(final_data)
    else:
        data = data.reshape(data.shape[0], -1, stepsize, data.shape[-1])
        data = jax.vmap(prepend)(data)
        final_data = None

    vmap_calc_logsig = jax.vmap(hall_basis_logsig, in_axes=(0, None, None))
    logsigs = jax.vmap(vmap_calc_logsig, in_axes=(0, None, None))(data, depth, t2l)
    if final_data is not None:
        final_logsigs = vmap_calc_logsig(final_data, depth, t2l)[:, None, :]
        logsigs = jnp.concatenate(
            (
                logsigs,
                final_logsigs,
            ),
            axis=1,
        )

    return logsigs


def batch_calc_paths(data, stepsize, depth, inmemory=True):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        path_data.append(
            out_func(
                calc_paths(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), stepsize, depth
                )
            )
        )
    if remainder > 0:
        path_data.append(
            out_func(calc_paths(in_func(data[-remainder:]), stepsize, depth))
        )
    if inmemory:
        path_data = jnp.concatenate(path_data)
    else:
        path_data = np.concatenate(path_data)
    return path_data
