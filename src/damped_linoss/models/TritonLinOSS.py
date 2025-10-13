import abc
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import equinox as eqx

from src.damped_linoss.models.common import GLU, simple_uniform_init

import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def prep_scan_kernel(
    # --- Input Pointers ---
    x_ptr,
    A_diag_ptr,
    steps_ptr,
    B_ptr,
    # --- Output Pointers ---
    F_ptr,
    BS_ptr,
    # --- Dimensions ---
    L, P, H,
    # --- Strides ---
    stride_x_l, stride_x_h,
    stride_b_p, stride_b_h, stride_b_c,
    stride_f_l, stride_f_p, stride_f_f, stride_f_c,
    stride_bs_l, stride_bs_p, stride_bs_f, stride_bs_c,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
    TILE_P: tl.constexpr,
    MICROTILE_H: tl.constexpr,
):
    """
    Triton kernel for the first phase of the LinOSS recurrence.
    - Grid: (cdiv(L, TILE_L), cdiv(P, TILE_P))
    - Each thread block computes a (TILE_L, TILE_P) portion of the state space.
    - It computes Bu and F on-the-fly, stores them in shared memory,
      performs a local scan, and writes only final accumulated state of the
      block and the F back to HBM.
    """
    # 1. ------------- GET PROGRAM IDs & DEFINE OFFSETS -----------------
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    p_offsets = bidp * TILE_P + tl.arange(0, TILE_P)
    p2_offsets = bidp * TILE_P + tl.arange(0, 2 * TILE_P) % TILE_P
    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    h_offsets = tl.arange(0, MICROTILE_H)
    c_offsets = tl.arange(0, 2 * TILE_P) // TILE_P # Handles complex parts

    p_mask = p_offsets < P
    l_mask = l_offsets < L

    # 2. ------------- LOAD BLOCK-CONSTANT PARAMETERS --------------------
    # These are constant for all `l` within this block's tile.
    # Shapes: (TILE_P,)
    A_diag = tl.load(A_diag_ptr + p_offsets, mask=p_mask, other=0.0)
    steps = tl.load(steps_ptr + p_offsets, mask=p_mask, other=0.0)
    
    # This is the 'A' part of our (A, b) scan tuple and is constant for this block
    # Shapes: (TILE_P,)
    schur_comp = 1.0 / (1.0 + steps * steps * A_diag)
    M_11 = 1.0 - steps * steps * A_diag * schur_comp
    M_12 = -1.0 * steps * A_diag * schur_comp
    M_21 = steps * schur_comp
    M_22 = schur_comp
    
    # 3. ------------- COMPUTE F ON-THE-FLY & STORE TO SHARED MEMORY -----
    # Shared memory to hold F values for the block
    # F is complex, so we store its real and imaginary parts.
    # So we need TILE_L x TILE_P x 2 elements for F1 and F2 each.
    F1 = tl.zeros((TILE_L, TILE_P, 2), dtype=tl.float32)
    F2 = tl.zeros((TILE_L, TILE_P, 2), dtype=tl.float32)
    
    p_complex_offsets = p2_offsets * stride_b_p + c_offsets * stride_b_c
    p_complex_mask = p2_offsets < P

    for h in range(0, H, MICROTILE_H):
        h_micro_tile_mask = (h + h_offsets[None, :]) < H
        B_mask = p_complex_mask[:, None] & h_micro_tile_mask
        x_mask = l_mask[:, None] & h_micro_tile_mask

        B_block_ptr = B_ptr + h_offsets[None, :] * stride_b_h + p_complex_offsets[:, None]
        x_block_ptr = x_ptr + h_offsets[None, :] * stride_x_h + l_offsets[:, None] * stride_x_l

        # We will load real and imag parts together
        B = tl.load(B_block_ptr, mask=B_mask, other=0.0)
        x = tl.load(x_block_ptr, mask=x_mask, other=0.0)

        # Compute Bu for the micro-tile. Shape: (TILE_L, TILE_P*2)
        Bu = tl.dot(x, B.T)

        # Apply the M matrix and steps to get F1 and F2 and accumulate in shared memory
        F1 += (M_11 * steps)[None, :, None] * Bu.reshape(TILE_L, TILE_P, 2)
        F2 += (M_21 * steps)[None, :, None] * Bu.reshape(TILE_L, TILE_P, 2)

    tl.debug_barrier() # Sync threads to ensure all of F is written

    # 4. ------------- STORE TO F_PTR ------------------------------------
    f_p2_offsets = p2_offsets * stride_f_p + c_offsets * stride_f_c
    f_mask = l_mask[:, None] & p_complex_mask[None, :]
    f_tile_ptr = F_ptr + l_offsets[:, None] * stride_f_l + f_p2_offsets[None, :]

    tl.store(f_tile_ptr + 0 * stride_f_f, F1.reshape(TILE_L, TILE_P * 2), mask=f_mask)
    tl.store(f_tile_ptr + 1 * stride_f_f, F2.reshape(TILE_L, TILE_P * 2), mask=f_mask)

    # 4. ------------- PERFORM INTRA-BLOCK SCAN (SEQUENTIAL) -------------
    # We now have the 'b' part of our sequence in shared memory. The 'A' part
    # is M_11, M_12, M_21, M_22, which is constant across a block.
    # We scan sequentially over the TILE_L dimension.
    f1_ptr = F_ptr + (bidl * TILE_L) * stride_f_l + p2_offsets * stride_f_p + c_offsets * stride_f_c
    f2_ptr = f1_ptr + 1 * stride_f_f

    b_acc_1 = tl.load(f1_ptr, mask=p_complex_mask).reshape(TILE_P, 2)
    b_acc_2 = tl.load(f2_ptr, mask=p_complex_mask).reshape(TILE_P, 2)
    A_acc_11, A_acc_12, A_acc_21, A_acc_22 = M_11, M_12, M_21, M_22
    
    for l_offset in range(1, min(TILE_L, L - bidl * TILE_L)):
        b_j1 = tl.load(f1_ptr + l_offset * stride_f_l, mask=p_complex_mask).reshape(TILE_P, 2)
        b_j2 = tl.load(f2_ptr + l_offset * stride_f_l, mask=p_complex_mask).reshape(TILE_P, 2)
        
        # This is the `binary_operator` logic, applied in parallel for TILE_P states
        # A_i is the accumulator (A_acc), b_i is the accumulator (b_acc)
        # A_j is the constant M matrix, b_j is the current element b_j   
        A_new_11 = M_11 * A_acc_11 + M_12 * A_acc_21
        A_new_12 = M_11 * A_acc_12 + M_12 * A_acc_22
        A_new_21 = M_21 * A_acc_11 + M_22 * A_acc_21
        A_new_22 = M_21 * A_acc_12 + M_22 * A_acc_22
        A_acc_11, A_acc_12, A_acc_21, A_acc_22 = A_new_11, A_new_12, A_new_21, A_new_22
        
        b_new_1 = M_11[:, None] * b_acc_1 + M_12[:, None] * b_acc_2 + b_j1
        b_new_2 = M_21[:, None] * b_acc_1 + M_22[:, None] * b_acc_2 + b_j2
        b_acc_1, b_acc_2 = b_new_1, b_new_2
        
    # 5. ------------- WRITE FINAL BLOCK STATE TO HBM --------------------
    BS_tile_ptr = BS_ptr + bidl * stride_bs_l + p2_offsets * stride_bs_p + c_offsets * stride_bs_c
    tl.store(BS_tile_ptr + 0 * stride_bs_f, b_acc_1.reshape(TILE_P * 2), mask=p_complex_mask)
    tl.store(BS_tile_ptr + 1 * stride_bs_f, b_acc_2.reshape(TILE_P * 2), mask=p_complex_mask)


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


class _AbstractLinOSSLayer(eqx.Module):
    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError
    

class IMLayer(_AbstractLinOSSLayer):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array

    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, *, key):
        A_key, B_key, C_key, D_key, step_key, key = jr.split(key, 6)

        self.steps = normal(stddev=0.5)(step_key, (state_dim,))
        self.A_diag = jr.uniform(A_key, shape=(state_dim,))
        self.B = simple_uniform_init(
            B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim)
        )
        self.C = simple_uniform_init(
            C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim)
        )
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _recurrence(self, A_diag, B_complex, input_sequence, step):
        """Compute the LxP output of LinOSS-IM given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            B_complex       (complex64):  input matrix              (P, H)
            input_sequence  (float32):    input sequence            (L, H)
            step            (float):      discretization time-step  (P,)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)

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

    def __call__(self, input_sequence):
        steps = nn.sigmoid(self.steps)
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]
        A_diag = nn.relu(self.A_diag)
            
        ys = self._recurrence(A_diag, B_complex, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        Cy = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        xs = Cy + Du

        return xs


class LinOSSBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    layer: _AbstractLinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        layer_name,
        state_dim,
        hidden_dim,
        r_min,
        r_max,
        theta_max,
        drop_rate,
        *,
        key,
    ):
        ssmkey, glukey = jr.split(key, 2)
        layer_map = {
            "IM": IMLayer,
            # "IMEX": IMEXLayer,
            # "Damped": DampedLayer,
        }
        if layer_name not in layer_map.keys():
            raise KeyError(f"Layer name {layer_name} not defined.")

        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        self.layer = layer_map[layer_name](
            state_dim,
            hidden_dim,
            r_min,
            r_max,
            theta_max,
            key=ssmkey,
        )
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.layer(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: list[LinOSSBlock]
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification,
        tanh_output,
        output_step,
        r_min=0.9,
        r_max=1.0,
        theta_max=jnp.pi,
        drop_rate=0.05,
        *,
        key,
    ):
        linear_encoder_key, *block_keys, linear_decoder_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                layer_name,
                state_dim,
                hidden_dim,
                r_min,
                r_max,
                theta_max,
                drop_rate,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_decoder = eqx.nn.Linear(hidden_dim, output_dim, key=linear_decoder_key)

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = self.linear_decoder(x)
            x = jax.nn.softmax(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_decoder)(x)
            if self.tanh_output:
                x = jax.nn.tanh(x)

        return x, state
    