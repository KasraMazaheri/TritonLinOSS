import triton
import triton.language as tl


@triton.jit
def associative_operator(
    i_M_11, i_M_12, i_M_21, i_M_22,
    i_F_r1, i_F_i1, i_F_r2, i_F_i2,
    j_M_11, j_M_12, j_M_21, j_M_22,
    j_F_r1, j_F_i1, j_F_r2, j_F_i2,
):
    M_11 = j_M_11 * i_M_11 + j_M_12 * i_M_21
    M_12 = j_M_11 * i_M_12 + j_M_12 * i_M_22
    M_21 = j_M_21 * i_M_11 + j_M_22 * i_M_21
    M_22 = j_M_21 * i_M_12 + j_M_22 * i_M_22

    F_r1 = j_M_11 * i_F_r1 + j_M_12 * i_F_r2 + j_F_r1
    F_i1 = j_M_11 * i_F_i1 + j_M_12 * i_F_i2 + j_F_i1
    F_r2 = j_M_21 * i_F_r1 + j_M_22 * i_F_r2 + j_F_r2
    F_i2 = j_M_21 * i_F_i1 + j_M_22 * i_F_i2 + j_F_i2
    return (M_11, M_12, M_21, M_22, F_r1, F_i1, F_r2, F_i2)

@triton.jit
def load_bulk(ptr_1, ptr_2, ptr_3, ptr_4, offsets, mask=None):
    other = None if mask is None else 0.0
    val1 = tl.load(ptr_1 + offsets, mask=mask, other=other)
    val2 = tl.load(ptr_2 + offsets, mask=mask, other=other)
    val3 = tl.load(ptr_3 + offsets, mask=mask, other=other)
    val4 = tl.load(ptr_4 + offsets, mask=mask, other=other)
    return val1, val2, val3, val4

@triton.jit
def store_bulk(ptr_1, ptr_2, ptr_3, ptr_4, offsets, val1, val2, val3, val4, mask=None):
    tl.store(ptr_1 + offsets, val1, mask=mask)
    tl.store(ptr_2 + offsets, val2, mask=mask)
    tl.store(ptr_3 + offsets, val3, mask=mask)
    tl.store(ptr_4 + offsets, val4, mask=mask)

@triton.jit
def parallel_scan(
    # --- Input Pointers ---
    M_11_ptr, M_12_ptr, M_21_ptr, M_22_ptr,
    F_r1_ptr, F_i1_ptr, F_r2_ptr, F_i2_ptr,
    # --- Output Pointers for the Cumalative Scan ---
    OM_11_ptr, OM_12_ptr, OM_21_ptr, OM_22_ptr,
    OF_r1_ptr, OF_i1_ptr, OF_r2_ptr, OF_i2_ptr,
    # --- Dimensions ---
    L, P,
    # --- Strides ---
    M_stride_l, M_stride_p,
    F_stride_l, F_stride_p,
    OM_stride_l, OM_stride_p,
    OF_stride_l, OF_stride_p,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
    TILE_P: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    p_offsets = bidp * TILE_P + tl.arange(0, TILE_P)
    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    p_mask = p_offsets < P
    l_mask = l_offsets < L

    M_offsets = l_offsets[:, None] * M_stride_l + p_offsets[None, :] * M_stride_p
    F_offsets = l_offsets[:, None] * F_stride_l + p_offsets[None, :] * F_stride_p
    mask      = l_mask[:, None] & p_mask[None, :]

    M_11, M_12, M_21, M_22 = load_bulk(M_11_ptr, M_12_ptr, M_21_ptr, M_22_ptr, M_offsets, mask=mask)
    F_r1, F_i1, F_r2, F_i2 = load_bulk(F_r1_ptr, F_i1_ptr, F_r2_ptr, F_i2_ptr, F_offsets, mask=mask)

    OM_11, OM_12, OM_21, OM_22, OF_r1, OF_i1, OF_r2, OF_i2 = tl.associative_scan(
        (M_11, M_12, M_21, M_22, F_r1, F_i1, F_r2, F_i2), axis=0, combine_fn=associative_operator
    )

    OM_offsets = l_offsets[:, None] * OM_stride_l + p_offsets[None, :] * OM_stride_p
    OF_offsets = l_offsets[:, None] * OF_stride_l + p_offsets[None, :] * OF_stride_p

    store_bulk(OM_11_ptr, OM_12_ptr, OM_21_ptr, OM_22_ptr, OM_offsets, OM_11, OM_12, OM_21, OM_22, mask=mask)
    store_bulk(OF_r1_ptr, OF_i1_ptr, OF_r2_ptr, OF_i2_ptr, OF_offsets, OF_r1, OF_i1, OF_r2, OF_i2, mask=mask)
    
@triton.jit
def inter_block_scan(
    # --- Input & Output Pointers for the Scan of the Entire Block ---
    BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr,
    BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr,
    # --- Dimensions ---
    L, # Different from the main L, this is number of blocks
    # --- Strides ---
    BM_stride_l, BM_stride_p,
    BF_stride_l, BF_stride_p,
):
    bidp = tl.program_id(0)

    CM_11, CM_12, CM_21, CM_22 = load_bulk(BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr, bidp * BM_stride_p)
    CF_r1, CF_i1, CF_r2, CF_i2 = load_bulk(BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr, bidp * BF_stride_p)
    for l_offset in range(1, L):
        BM_offsets = l_offset * BM_stride_l + bidp * BM_stride_p
        BF_offsets = l_offset * BF_stride_l + bidp * BF_stride_p

        M_11, M_12, M_21, M_22 = load_bulk(BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr, BM_offsets)
        F_r1, F_i1, F_r2, F_i2 = load_bulk(BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr, BF_offsets)

        CM_11, CM_12, CM_21, CM_22, CF_r1, CF_i1, CF_r2, CF_i2 = associative_operator(
            CM_11, CM_12, CM_21, CM_22, CF_r1, CF_i1, CF_r2, CF_i2,
                M_11, M_12, M_21, M_22, F_r1, F_i1, F_r2, F_i2
        )

        store_bulk(BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr, BM_offsets, CM_11, CM_12, CM_21, CM_22)
        store_bulk(BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr, BF_offsets, CF_r1, CF_i1, CF_r2, CF_i2)

@triton.jit
def parallel_scan_epilogue(
    # --- Input and Output Pointers for the Cumalative Scan ---
    OM_11_ptr, OM_12_ptr, OM_21_ptr, OM_22_ptr,
    OF_r1_ptr, OF_i1_ptr, OF_r2_ptr, OF_i2_ptr,
    # --- Output Pointers for the Scan of the Entire Block ---
    BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr,
    BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr,
    # --- Dimensions ---
    L, P,
    # --- Strides ---
    OM_stride_l, OM_stride_p,
    OF_stride_l, OF_stride_p,
    BM_stride_l, BM_stride_p,
    BF_stride_l, BF_stride_p,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
    TILE_P: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    p_offsets = bidp * TILE_P + tl.arange(0, TILE_P)
    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    p_mask = p_offsets < P
    l_mask = l_offsets < L

    OM_offsets = l_offsets[:, None] * OM_stride_l + p_offsets[None, :] * OM_stride_p
    OF_offsets = l_offsets[:, None] * OF_stride_l + p_offsets[None, :] * OF_stride_p
    mask       = l_mask[:, None] & p_mask[None, :]

    OM_11, OM_12, OM_21, OM_22 = load_bulk(OM_11_ptr, OM_12_ptr, OM_21_ptr, OM_22_ptr, OM_offsets, mask=mask)
    OF_r1, OF_i1, OF_r2, OF_i2 = load_bulk(OF_r1_ptr, OF_i1_ptr, OF_r2_ptr, OF_i2_ptr, OF_offsets, mask=mask)

    BM_offsets = bidl * BM_stride_l + p_offsets * BM_stride_p
    BF_offsets = bidl * BF_stride_l + p_offsets * BF_stride_p

    BM_11, BM_12, BM_21, BM_22 = load_bulk(BM_11_ptr, BM_12_ptr, BM_21_ptr, BM_22_ptr, BM_offsets, mask=p_mask)
    BF_r1, BF_i1, BF_r2, BF_i2 = load_bulk(BF_r1_ptr, BF_i1_ptr, BF_r2_ptr, BF_i2_ptr, BF_offsets, mask=p_mask)

    OM_11, OM_12, OM_21, OM_22, OF_r1, OF_i1, OF_r2, OF_i2 = associative_operator(
        BM_11, BM_12, BM_21, BM_22, BF_r1, BF_i1, BF_r2, BF_i2,
        OM_11, OM_12, OM_21, OM_22, OF_r1, OF_i1, OF_r2, OF_i2,
    )

    store_bulk(OM_11_ptr, OM_12_ptr, OM_21_ptr, OM_22_ptr, OM_offsets, OM_11, OM_12, OM_21, OM_22, mask=mask)
    store_bulk(OF_r1_ptr, OF_i1_ptr, OF_r2_ptr, OF_i2_ptr, OF_offsets, OF_r1, OF_i1, OF_r2, OF_i2, mask=mask)
