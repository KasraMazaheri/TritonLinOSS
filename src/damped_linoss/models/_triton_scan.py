import torch
import triton
import triton.language as tl
import jax
import jax.numpy as jnp

torch.manual_seed(0)


@jax.vmap
def binary_operator_jax(q_i, q_j):
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


@triton.jit
def binary_operator(
    i_M_11,
    i_M_12,
    i_M_21,
    i_M_22,
    i_F1,
    i_F2,
    j_M_11,
    j_M_12,
    j_M_21,
    j_M_22,
    j_F1,
    j_F2,
):
    """"""
    new_M_11 = j_M_11 * i_M_11 + j_M_12 * i_M_21
    new_M_12 = j_M_11 * i_M_12 + j_M_12 * i_M_22
    new_M_21 = j_M_21 * i_M_11 + j_M_22 * i_M_21
    new_M_22 = j_M_21 * i_M_12 + j_M_22 * i_M_22

    new_F1 = j_M_11 * i_F1 + j_M_12 * i_F2 + j_F1
    new_F2 = j_M_21 * i_F1 + j_M_22 * i_F2 + j_F2

    return new_M_11, new_M_12, new_M_21, new_M_22, new_F1, new_F2


@triton.jit
def simple_ssm_tt(
    M_ptr,
    F_ptr,
    Out_ptr,
    # Dimensions of the input tensors
    Seq_len,
    P_dim,
    # Strides
    stride_m_seqlen,
    stride_m_pdim,
    stride_m_d1,
    stride_m_d2,
    stride_f_seqlen,
    stride_f_pdim,
    stride_f_d1,
    stride_out_seqlen,
    stride_out_pdim,
    stride_out_d1,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_PDIM: tl.constexpr,
):
    """
    Triton kernel to load two tensors M and F.
    M: (Seq_len, P_dim, 2, 2)
    F: (Seq_len, P_dim, 2)
    """
    pid_seq = tl.program_id(axis=0)
    pid_pdim = tl.program_id(axis=1)

    offs_seq = pid_seq * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    offs_pdim = pid_pdim * BLOCK_SIZE_PDIM + tl.arange(0, BLOCK_SIZE_PDIM)

    offs_m_seq = offs_seq[:, None, None, None] * stride_m_seqlen
    offs_m_pdim = offs_pdim[None, :, None, None] * stride_m_pdim
    offs_m_d1 = tl.arange(0, 2)[None, None, :, None] * stride_m_d1
    offs_m_d2 = tl.arange(0, 2)[None, None, None, :] * stride_m_d2

    m_ptrs = M_ptr + (offs_m_seq + offs_m_pdim + offs_m_d1 + offs_m_d2)

    mask_m = (offs_seq[:, None, None, None] < Seq_len) & (
        offs_pdim[None, :, None, None] < P_dim
    )

    M = tl.load(m_ptrs, mask=mask_m, other=0.0)

    offs_f_seq = offs_seq[:, None, None] * stride_f_seqlen
    offs_f_pdim = offs_pdim[None, :, None] * stride_f_pdim
    offs_f_d1 = tl.arange(0, 2)[None, None, :] * stride_f_d1

    f_ptrs = F_ptr + (offs_f_seq + offs_f_pdim + offs_f_d1)

    mask_f = (offs_seq[:, None, None] < Seq_len) & (offs_pdim[None, :, None] < P_dim)

    F = tl.load(f_ptrs, mask=mask_f, other=0.0)

    # Triton doesn't support indexing but we can use split along last dims
    M_11_21, M_12_22 = tl.split(M)
    M_11, M_21 = tl.split(M_11_21)
    M_12, M_22 = tl.split(M_12_22)

    F1, F2 = tl.split(F)

    *_, y_1, y_2 = tl.associative_scan(
        (M_11, M_12, M_21, M_22, F1, F2), 0, binary_operator
    )

    # recombne y_1 and y_2
    h2 = tl.join(y_1, y_2)

    offs_out_seq = offs_seq[:, None, None] * stride_out_seqlen
    offs_out_pdim = offs_pdim[None, :, None] * stride_out_pdim
    offs_out_d1 = tl.arange(0, 2)[None, None, :] * stride_out_d1
    outs_ptrs = Out_ptr + (offs_out_seq + offs_out_pdim + offs_out_d1)
    mask_out = (offs_seq[:, None, None] < Seq_len) & (offs_pdim[None, :, None] < P_dim)

    tl.store(outs_ptrs, h2, mask=mask_out)


# Constants
P = 2
K = 8
BLOCKS = 1
L = K * BLOCKS

M_11 = torch.ones((P)).to(torch.float32)
M_12 = torch.ones((P)).to(torch.float32) * 2.0
M_21 = torch.ones((P)).to(torch.float32) * 3.0
M_22 = torch.ones((P)).to(torch.float32) * 4.0

M = (
    torch.stack(
        [
            torch.stack([M_11, M_12], dim=-1),
            torch.stack([M_21, M_22], dim=-1),
        ],
        dim=-2,
    )
    .unsqueeze(0)
    .expand(L, -1, -1, -1)
    .contiguous()
)


# F1 is 1 and 3s
F1 = (torch.arange((P)).to(torch.float32) + 1) * 2.0 - 1.0
# F2 is 0 and 2s
F2 = torch.arange((P)).to(torch.float32) * 2

F = torch.stack([F1, F2], dim=-1).unsqueeze(0).expand(L, -1, -1).contiguous()

Y = torch.empty_like(F)

simple_ssm_tt[(BLOCKS,)](
    M,
    F,
    Y,
    M.shape[0],
    P,
    M.stride(0),
    M.stride(1),
    M.stride(2),
    M.stride(3),
    F.stride(0),
    F.stride(1),
    F.stride(2),
    Y.stride(0),
    Y.stride(1),
    Y.stride(2),
    BLOCK_SIZE_SEQ=K,
    BLOCK_SIZE_PDIM=P,
)

print(Y.shape)

# set jax device to cpu
jax.config.update("jax_platform_name", "cpu")

M_jax = jax.numpy.array(M.numpy())
F_jax = jax.numpy.array(F.numpy())

# reshape M to (L, 4*P)
M_jax = jax.numpy.concatenate(
    [
        M_jax[:, :, 0, 0],
        M_jax[:, :, 0, 1],
        M_jax[:, :, 1, 0],
        M_jax[:, :, 1, 1],
    ],
    axis=-1,
)

F_jax = jax.numpy.concatenate(
    [
        F_jax[:, :, 0],
        F_jax[:, :, 1],
    ],
    axis=-1,
)

_, ref_solution = jax.lax.associative_scan(binary_operator_jax, (M_jax, F_jax))

# resahpe Y to (L, P*2)
Y_jax = jax.numpy.concatenate(
    [
        Y[:, :, 0].numpy(),
        Y[:, :, 1].numpy(),
    ],
    axis=-1,
)
print("Max difference:", jax.numpy.max(jax.numpy.abs(ref_solution - Y_jax)))
print(ref_solution - Y_jax)
