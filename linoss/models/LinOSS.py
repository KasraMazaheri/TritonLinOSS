import os
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random
import numpy as np

from linoss.models.LRU import GLU


def simple_uniform_init(rng, shape, std=1.0):
    weights = random.uniform(rng, shape) * 2.0 * std - std
    return weights


def map_theta_to_A(thetas, G_diag, steps):
    A_plus = (
        4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2)
            + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))
    A_minus = (
        -4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2)
            + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))

    A_diag = jnp.where(thetas > jnp.pi / 2, A_plus, A_minus)

    return A_diag


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.
    Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def linoss_im(A_diag, B, input_sequence, step):
    """Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        ys (float32): the SSM states (LinOSS_IMEX layer pre-output pre-activations)      (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_12 = -1.0 * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = jnp.concatenate([M_11, M_12, M_21, M_22])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = M_11 * Bu_elements * step
    F2 = M_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def linoss_imex(A_diag, B, input_sequence, step):
    """Compute the LxH output of of LinOSS-IMEX given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        ys (float32): the SSM states (LinOSS_IMEX layer pre-output pre-activations)      (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    M = jnp.concatenate([A_, B_, C_, D_])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def damped_linoss_imex(A_diag, G_diag, B, input_sequence, step):
    """Compute the LxH output of of Damped LinOSS-IMEX given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        G_diag  (float32):   diagonal damping matrix (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        ys (float32): the SSM states (LinOSS_IMEX layer pre-output pre-activations)      (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = Identity - step**2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


class LinOSSLayer(eqx.Module):
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str
    damping: bool

    def __init__(self, ssm_size, H, discretization, damping, r_min, theta_max, *, key):
        A_key, G_key, B_key, C_key, D_key, step_key, key = jr.split(key, 7)

        self.steps = normal(stddev=0.5)(step_key, (ssm_size,))
        steps = nn.sigmoid(self.steps)

        if discretization == "IMEX" and damping:
            r_max = 1.0
            mags = jnp.sqrt(
                random.uniform(G_key, shape=(ssm_size,)) * (r_max**2 - r_min**2)
                + r_min**2
            )
            self.G_diag = (1 - mags**2) / (steps * mags**2)
            G_diag = nn.relu(self.G_diag)

            theta = random.uniform(A_key, shape=(ssm_size,)) * theta_max
            self.A_diag = map_theta_to_A(theta, G_diag, steps)
        else:
            self.G_diag = None
            self.A_diag = random.uniform(A_key, shape=(ssm_size,))

        self.B = simple_uniform_init(
            B_key, shape=(ssm_size, H, 2), std=1.0 / math.sqrt(H)
        )
        self.C = simple_uniform_init(
            C_key, shape=(H, ssm_size, 2), std=1.0 / math.sqrt(ssm_size)
        )
        self.D = normal(stddev=1.0)(D_key, (H,))

        self.discretization = discretization
        self.damping = damping

    def __call__(self, input_sequence):
        steps = nn.sigmoid(self.steps)

        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        if self.discretization == "IM":
            if self.damping:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(
                        self.discretization, self.damping
                    )
                )
            else:
                A_diag = nn.relu(self.A_diag)
                ys = linoss_im(A_diag, B_complex, input_sequence, steps)
        elif self.discretization == "IMEX":
            if self.damping:
                G_diag = nn.relu(self.G_diag)
                A_boundary_low = (
                    2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)
                ) / steps**2
                A_boundary_high = (
                    2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)
                ) / steps**2
                A_diag = (
                    A_boundary_low
                    + nn.relu(self.A_diag - A_boundary_low)
                    - nn.relu(self.A_diag - A_boundary_high)
                )
                ys = damped_linoss_imex(
                    A_diag, G_diag, B_complex, input_sequence, steps
                )
            else:
                A_diag = nn.relu(self.A_diag)
                ys = linoss_imex(A_diag, B_complex, input_sequence, steps)
        else:
            raise NotImplementedError(
                "Discretization {} not implemented".format(self.discretization)
            )

        # Apply SSM Output Operations Cx + Du
        Cy = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        xs = Cy + Du

        return xs


class LinOSSBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    ssm: LinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        damping,
        r_min,
        theta_max,
        drop_rate=0.05,
        *,
        key,
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.ssm = LinOSSLayer(
            ssm_size,
            H,
            discretization,
            damping,
            r_min,
            theta_max,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    linear_output: bool
    output_step: int
    use_last_output: bool
    discretization: str
    damping: bool
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        classification,
        linear_output,
        output_step,
        use_last_output,
        discretization,
        damping,
        r_min,
        theta_max,
        *,
        key,
    ):
        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(
            key, num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                ssm_size,
                H,
                discretization,
                damping,
                r_min,
                theta_max,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.linear_output = linear_output
        self.output_step = output_step
        self.use_last_output = use_last_output
        self.discretization = discretization
        self.damping = damping

    def __call__(self, x, state, key):
        """Compute LinOSS."""
        sql, _ = x.shape
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)

        if self.classification:
            if self.use_last_output:
                x = x[-1, :]
            else:
                x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            if self.use_last_output:
                x = x[-1, :]
                x = self.linear_layer(x)
                x = x.reshape(1, -1)
            else:
                x = x[self.output_step - 1 :: self.output_step]
                x = jax.vmap(self.linear_layer)(x)
                x = x.reshape(sql // self.output_step, -1)
            if not self.linear_output:
                x = jax.nn.tanh(x)

        return x, state