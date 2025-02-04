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


def simple_uniform_init(rng, shape, std=1.):
    weights = random.uniform(rng, shape)*2.*std - std
    return weights


def map_theta_to_A(thetas, G_diag, steps):
    A_plus = (4 * jnp.sqrt(steps ** 4 * jnp.cos(thetas) ** (-2) + steps ** 5 * G_diag * jnp.cos(thetas) ** (-2)) \
        - steps ** 2 * (- 4 - 2 * steps * G_diag - 4 * jnp.tan(thetas) ** 2 - 2 * steps * G_diag * jnp.tan(thetas) ** 2)) \
        / (2 * steps ** 4 * (1 + jnp.tan(thetas) ** 2))
    A_minus = (- 4 * jnp.sqrt(steps ** 4 * jnp.cos(thetas) ** (-2) + steps ** 5 * G_diag * jnp.cos(thetas) ** (-2)) \
        - steps ** 2 * (- 4 - 2 * steps * G_diag - 4 * jnp.tan(thetas) ** 2 - 2 * steps * G_diag * jnp.tan(thetas) ** 2)) \
        / (2 * steps ** 4 * (1 + jnp.tan(thetas) ** 2))

    A_diag = jnp.where(thetas > jnp.pi / 2, A_plus, A_minus)

    return A_diag


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N: 1 * N]
    iB_ = A_i[1 * N: 2 * N]
    iC_ = A_i[2 * N: 3 * N]
    iD_ = A_i[3 * N: 4 * N]
    jA_ = A_j[0 * N: 1 * N]
    jB_ = A_j[1 * N: 2 * N]
    jC_ = A_j[2 * N: 3 * N]
    jD_ = A_j[3 * N: 4 * N]
    A_new = iA_ * jA_ + iB_ * jC_
    B_new = iA_ * jB_ + iB_ * jD_
    C_new = iC_ * jA_ + iD_ * jC_
    D_new = iC_ * jB_ + iD_ * jD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def make_linoss_im_recurrence(A_diag, step):
    """Compute the PxP recurrent matrix M for LinOSS-IM
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        M    (float32): the recurrent matrix (P, P)
    """
    S = 1. / (1. + step ** 2. * A_diag)
    M_11 = jnp.diag(1. - step ** 2. * A_diag * S)
    M_12 = jnp.diag(-1. * step * A_diag * S)
    M_21 = jnp.diag(step * S)
    M_22 = jnp.diag(S)

    M = jnp.block([
        [M_11, M_12],
        [M_21, M_22]
    ])

    return M


def make_linoss_imex_recurrence(A_diag, step):
    """Compute the PxP recurrent matrix M for LinOSS-IMEX
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        M  (float32): the recurrent matrix (P, P)
    """
    M_11 = jnp.diag(jnp.ones_like(A_diag))
    M_12 = jnp.diag(-1. * step * A_diag)
    M_21 = jnp.diag(step)
    M_22 = jnp.diag(1. - (step ** 2.) * A_diag)

    M = jnp.block([
        [M_11, M_12],
        [M_21, M_22]
    ])

    return M


def make_damped_linoss_imex_recurrence(A_diag, G_diag, step):
    """Compute the PxP recurrent matrix M for Damped LinOSS-IMEX
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        G_diag  (float32):   diagonal damping matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        M    (float32): the recurrent matrix (P, P)
    """
    I = jnp.ones_like(A_diag)
    S = I + step * G_diag
    M_11 = jnp.diag(1. / S)
    M_12 = jnp.diag(- step / S * A_diag)
    M_21 = jnp.diag(step / S)
    M_22 = jnp.diag(I - step ** 2 / S * A_diag)

    M = jnp.block([
        [M_11, M_12],
        [M_21, M_22]
    ])

    return M


def apply_linoss_im(A_diag, B, input_sequence, step):
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

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_11 = 1. - step ** 2. * A_diag * schur_comp
    M_12 = -1. * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = jnp.concatenate([M_11, M_12, M_21, M_22])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = M_11 * Bu_elements * step
    F2 = M_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return ys


def apply_linoss_imex(A_diag, B, input_sequence, step):
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
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M = jnp.concatenate([A_, B_, C_, D_])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return ys


def apply_damped_linoss_imex(A_diag, G_diag, B, input_sequence, step):
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

    I = jnp.ones_like(A_diag)
    S = I + step * G_diag
    M_11 = 1. / S
    M_12 = - step / S * A_diag
    M_21 = step / S
    M_22 = I - step ** 2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))
    
    F1 = step * (1. / S) * Bu_elements
    F2 = step ** 2 * (1. / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0]:]

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

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        damping,
        r_min,
        *,
        key
    ):
        A_key, G_key, B_key, C_key, D_key, step_key, key = jr.split(key, 7)

        self.steps = normal(stddev=0.5)(step_key, (ssm_size,))
        steps = nn.sigmoid(self.steps)

        if discretization == "IMEX" and damping:
            r_max = 1.0
            mags = jnp.sqrt(random.uniform(G_key, shape=(ssm_size,)) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
            self.G_diag = (1 - mags ** 2) / (steps * mags ** 2)
            G_diag = nn.relu(self.G_diag)

            theta = random.uniform(A_key, shape=(ssm_size,)) * jnp.pi
            self.A_diag = map_theta_to_A(theta, G_diag, steps)
        else:
            self.G_diag = None
            self.A_diag = random.uniform(A_key, shape=(ssm_size,))   

        self.B = simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1./math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1./math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))

        self.discretization = discretization
        self.damping = damping

    def __call__(self, input_sequence, save_dir=None):
        steps = nn.sigmoid(self.steps)
        
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        if self.discretization == "IM":
            if self.damping:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(self.discretization, self.damping)
                )
            else:
                A_diag = nn.relu(self.A_diag)
                ys = apply_linoss_im(A_diag, B_complex, input_sequence, steps)
        elif self.discretization == "IMEX":
            if self.damping:
                G_diag = nn.relu(self.G_diag)
                A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps ** 2
                A_boundary_high = (2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)) / steps ** 2
                A_diag = A_boundary_low + nn.relu(self.A_diag - A_boundary_low) - nn.relu(self.A_diag - A_boundary_high)
                ys = apply_damped_linoss_imex(A_diag, G_diag, B_complex, input_sequence, steps)
            else:
                A_diag = nn.relu(self.A_diag)
                ys = apply_linoss_imex(A_diag, B_complex, input_sequence, steps)
        else:
            raise NotImplementedError(
                "Discretization {} not implemented".format(self.discretization)
            )

        # Apply SSM Output Operations Cx + Du
        Cy = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        xs = Cy + Du

        # Save weights, SSM states, SSM outputs
        if save_dir is not None:
            jnp.save(save_dir + 'ssm_states.npy', ys)
            jnp.save(save_dir + 'ssm_outputs.npy', xs)

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
        drop_rate=0.05,
        *,
        key
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
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key, save_dir=None):
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x, save_dir=save_dir)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        # Save activations
        if save_dir is not None:
            jnp.save(save_dir + 'activations.npy', x)

        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    linear_output: bool
    output_step: int
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
        discretization,
        damping,
        r_min,
        *,
        key
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
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.linear_output = linear_output
        self.output_step = output_step
        self.discretization = discretization
        self.damping = damping

    def __call__(self, x, state, key, save_dir=None):
        """Compute LinOSS."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            jnp.save(save_dir + 'input.npy', x)

        for i, (block, key) in enumerate(zip(self.blocks, dropkeys)):
            if save_dir is not None:
                block_dir = save_dir + f'block_{i}/'
                os.makedirs(block_dir, exist_ok=True)
                x, state = block(x, state, key=key, save_dir=block_dir)
            else:
                x, state = block(x, state, key=key, save_dir=None)

        if self.classification:
            x = jnp.mean(x, axis=0)
            if save_dir is not None:
                jnp.save(save_dir + 'output.npy', self.linear_layer(x)) 
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_layer)(x)
            if save_dir is not None:
                jnp.save(save_dir + 'output.npy', x) 
            if not self.linear_output:
                x = jax.nn.tanh(x)

        return x, state
    
    def save_params(self, save_dir):
        """Saves parameters as directory tree"""
        save_dir = save_dir + '/params/'
        os.makedirs(save_dir + 'input/', exist_ok=True)
        os.makedirs(save_dir + 'output/', exist_ok=True)
        jnp.save(save_dir + 'input/weight.npy', self.linear_encoder.weight)
        jnp.save(save_dir + 'input/bias.npy', self.linear_encoder.bias)
        jnp.save(save_dir + 'output/weight.npy', self.linear_layer.weight)
        jnp.save(save_dir + 'output/bias.npy', self.linear_layer.bias)

        for i, block in enumerate(self.blocks):
            os.makedirs(save_dir + f'block_{i}/glu/w1/', exist_ok=True)
            os.makedirs(save_dir + f'block_{i}/glu/w2/', exist_ok=True)
            jnp.save(save_dir + f'block_{i}/glu/w1/weight.npy', block.glu.w1.weight)
            jnp.save(save_dir + f'block_{i}/glu/w1/bias.npy', block.glu.w1.bias)
            jnp.save(save_dir + f'block_{i}/glu/w2/weight.npy', block.glu.w2.weight)
            jnp.save(save_dir + f'block_{i}/glu/w2/bias.npy', block.glu.w2.bias)

            steps = nn.sigmoid(block.ssm.steps)
            B_complex = block.ssm.B[..., 0] + 1j * block.ssm.B[..., 1]
            C_complex = block.ssm.C[..., 0] + 1j * block.ssm.C[..., 1]
            D = jnp.diag(block.ssm.D)

            if self.discretization == "IM":
                if self.damping:
                    raise NotImplementedError(
                        "Discretization {} and damping = {} not implemented".format(self.discretization, self.damping)
                    )
                else:
                    A_diag = nn.relu(block.ssm.A_diag)
                    M = make_linoss_im_recurrence(A_diag, steps)
            elif self.discretization == "IMEX":
                if self.damping:
                    G_diag = nn.relu(block.ssm.G_diag)
                    A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps ** 2
                    A_boundary_high = (2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)) / steps ** 2
                    A_diag = A_boundary_low + nn.relu(block.ssm.A_diag - A_boundary_low) - nn.relu(block.ssm.A_diag - A_boundary_high)
                    M = make_damped_linoss_imex_recurrence(A_diag, G_diag, steps)
                else:
                    A_diag = nn.relu(block.ssm.A_diag)
                    M = make_linoss_imex_recurrence(A_diag, steps)
            else:
                raise NotImplementedError(
                    "Discretization {} not implemented".format(self.discretization)
                )

            jnp.save(save_dir + f'block_{i}/M.npy', M)
            jnp.save(save_dir + f'block_{i}/B.npy', B_complex)
            jnp.save(save_dir + f'block_{i}/C.npy', C_complex)
            jnp.save(save_dir + f'block_{i}/D.npy', D)
            jnp.save(save_dir + f'block_{i}/steps.npy', steps)