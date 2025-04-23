"""
Code modified from https://gist.github.com/Ryu1845/7e78da4baa8925b4de482969befa949d

This module implements the `LRU` class, a model architecture using JAX and Equinox.

Attributes of the `LRU` class:
- `linear_encoder`: The linear encoder applied to the input time series data.
- `blocks`: A list of `LRUBlock` instances, each containing the LRU layer, normalization, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the model predictions.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes the following classes and functions:
- `GLU`: Implements a Gated Linear Unit for non-linear transformations within the model.
- `LRULayer`: A single LRU layer that applies complex-valued transformations and projections to the input.
- `LRUBlock`: A block consisting of normalization, LRU layer, GLU, and dropout, used as a building block for the `LRU`
              model.
- `binary_operator_diag`: A helper function used in the associative scan operation within `LRULayer` to process diagonal
                          elements.
"""

import os
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


class LRULayer(eqx.Module):
    nu_log: jnp.ndarray
    theta_log: jnp.ndarray
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray
    gamma_log: jnp.ndarray

    def __init__(self, N, H, r_min=0.9, r_max=1, max_phase=6.28, *, key):
        u1_key, u2_key, B_re_key, B_im_key, C_re_key, C_im_key, D_key = jr.split(key, 7)

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = jr.uniform(u1_key, shape=(N,))
        u2 = jr.uniform(u2_key, shape=(N,))
        self.nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        self.theta_log = jnp.log(max_phase * u2)

        # Glorot initialized Input/Output projection matrices
        self.B_re = jr.normal(B_re_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.B_im = jr.normal(B_im_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.C_re = jr.normal(C_re_key, shape=(H, N)) / jnp.sqrt(N)
        self.C_im = jr.normal(C_im_key, shape=(H, N)) / jnp.sqrt(N)
        self.D = jr.normal(D_key, shape=(H,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    def __call__(self, x, save_dir=None):
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im
        # Running the LRU + output projection
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.associative_scan(
            binary_operator_diag, elements
        )  # all x_k
        y = jax.vmap(lambda z, u: (C @ z).real + (self.D * u))(inner_states, x)

        # Save weights, SSM states, SSM outputs
        if save_dir is not None:
            jnp.save(save_dir + "ssm_states.npy", inner_states)
            jnp.save(save_dir + "ssm_outputs.npy", y)

        return y


class LRUBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    lru: LRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, N, H, r_min=0.9, r_max=1, max_phase=6.28, drop_rate=0.1, *, key):
        lrukey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.lru = LRULayer(N, H, r_min, r_max, max_phase, key=lrukey)
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key, save_dir=None):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.lru(x, save_dir=save_dir)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        # Save activations
        if save_dir is not None:
            jnp.save(save_dir + "activations.npy", x)

        return x, state


class LRU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LRUBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    linear_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        data_dim,
        N,
        H,
        output_dim,
        classification,
        linear_output,
        output_step,
        r_min=0.9,
        r_max=1,
        max_phase=6.28,
        drop_rate=0.1,
        *,
        key,
    ):
        linear_encoder_key, *block_keys, linear_layer_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(data_dim, H, key=linear_encoder_key)
        self.blocks = [
            LRUBlock(N, H, r_min, r_max, max_phase, drop_rate, key=key)
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.linear_output = linear_output
        self.output_step = output_step

    def __call__(self, x, state, key, save_dir=None):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            jnp.save(save_dir + "/input.npy", x)

        for i, (block, key) in enumerate(zip(self.blocks, dropkeys)):
            if save_dir is not None:
                block_dir = save_dir + f"/block_{i}/"
                os.makedirs(block_dir, exist_ok=True)
                x, state = block(x, state, key=key, save_dir=block_dir)
            else:
                x, state = block(x, state, key=key, save_dir=None)

        if self.classification:
            x = jnp.mean(x, axis=0)
            if save_dir is not None:
                jnp.save(save_dir + "/output.npy", self.linear_layer(x))
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            if save_dir is not None:
                jnp.save(save_dir + "/output.npy", jax.vmap(self.linear_layer)(x))
            x = jax.vmap(self.linear_layer)(x)
            if not self.linear_output:
                x = jax.nn.tanh(x)

        return x, state

    def save_params(self, save_dir):
        """Saves parameters as directory tree"""
        os.makedirs(save_dir + "/input/", exist_ok=True)
        os.makedirs(save_dir + "/output/", exist_ok=True)
        jnp.save(save_dir + "/input/weight.npy", self.linear_encoder.weight)
        jnp.save(save_dir + "/input/bias.npy", self.linear_encoder.bias)
        jnp.save(save_dir + "/output/weight.npy", self.linear_layer.weight)
        jnp.save(save_dir + "/output/bias.npy", self.linear_layer.bias)

        for i, block in enumerate(self.blocks):
            os.makedirs(save_dir + f"/block_{i}/glu/w1/", exist_ok=True)
            os.makedirs(save_dir + f"/block_{i}/glu/w2/", exist_ok=True)
            jnp.save(save_dir + f"/block_{i}/glu/w1/weight.npy", block.glu.w1.weight)
            jnp.save(save_dir + f"/block_{i}/glu/w1/bias.npy", block.glu.w1.bias)
            jnp.save(save_dir + f"/block_{i}/glu/w2/weight.npy", block.glu.w2.weight)
            jnp.save(save_dir + f"/block_{i}/glu/w2/bias.npy", block.glu.w2.bias)

            Lambda = jnp.exp(
                -jnp.exp(block.lru.nu_log) + 1j * jnp.exp(block.lru.theta_log)
            )
            B_norm = (block.lru.B_re + 1j * block.lru.B_im) * jnp.expand_dims(
                jnp.exp(block.lru.gamma_log), axis=-1
            )
            C = block.lru.C_re + 1j * block.lru.C_im

            jnp.save(save_dir + f"/block_{i}/M.npy", jnp.diag(Lambda))
            jnp.save(save_dir + f"/block_{i}/B.npy", B_norm)
            jnp.save(save_dir + f"/block_{i}/C.npy", C)
            jnp.save(save_dir + f"/block_{i}/D.npy", jnp.diag(block.lru.D))
