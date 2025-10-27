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
def load_bulk(ptr, offsets, d1_stride, d2_stride, mask=None):
    other = None if mask is None else 0.0
    val1 = tl.load(ptr + offsets, mask=mask, other=other)
    val2 = tl.load(ptr + offsets + d2_stride, mask=mask, other=other)
    val3 = tl.load(ptr + offsets + d1_stride, mask=mask, other=other)
    val4 = tl.load(ptr + offsets + d1_stride + d2_stride, mask=mask, other=other)
    return val1, val2, val3, val4

@triton.jit
def store_bulk(ptr, offsets, d1_stride, d2_stride, Vs, mask=None):
    tl.store(ptr + offsets, Vs[0], mask=mask)
    tl.store(ptr + offsets + d2_stride, Vs[1], mask=mask)
    tl.store(ptr + offsets + d1_stride, Vs[2], mask=mask)
    tl.store(ptr + offsets + d1_stride + d2_stride, Vs[3], mask=mask)

@triton.jit
def parallel_scan(
    # --- Input Pointers ---
    M_ptr, F_ptr,
    # --- Output Pointers for the Cumalative Scan ---
    OM_ptr, OF_ptr,
    # --- Dimensions ---
    L,
    # --- Strides ---
    M_stride_l, M_stride_p, M_stride_d1, M_stride_d2,
    F_stride_l, F_stride_p, F_stride_d1, F_stride_d2,
    OM_stride_l, OM_stride_p, OM_stride_d1, OM_stride_d2,
    OF_stride_l, OF_stride_p, OF_stride_d1, OF_stride_d2,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    l_mask = l_offsets < L

    M_offsets = l_offsets * M_stride_l + bidp * M_stride_p
    F_offsets = l_offsets * F_stride_l + bidp * F_stride_p
    mask      = l_mask

    Ms = load_bulk(M_ptr, M_offsets, M_stride_d1, M_stride_d2, mask)
    Fs = load_bulk(F_ptr, F_offsets, F_stride_d1, F_stride_d2, mask)

    Vs = tl.associative_scan(
        Ms + Fs, axis=0, combine_fn=associative_operator
    )
    OMs, OFs = Vs[:4], Vs[4:]

    OM_offsets = l_offsets * OM_stride_l + bidp * OM_stride_p
    OF_offsets = l_offsets * OF_stride_l + bidp * OF_stride_p

    store_bulk(OM_ptr, OM_offsets, OM_stride_d1, OM_stride_d2, OMs, mask=mask)
    store_bulk(OF_ptr, OF_offsets, OF_stride_d1, OF_stride_d2, OFs, mask=mask)
    
@triton.jit
def inter_block_scan(
    # --- Input & Output Pointers for the Scan of the Entire Block ---
    BM_ptr, BF_ptr,
    # --- Dimensions ---
    L, # Different from the main L, this is number of blocks
    # --- Strides ---
    BM_stride_l, BM_stride_p, BM_stride_d1, BM_stride_d2,
    BF_stride_l, BF_stride_p, BF_stride_d1, BF_stride_d2,
):
    bidp = tl.program_id(0)

    CMs = load_bulk(BM_ptr, bidp * BM_stride_p, BM_stride_d1, BM_stride_d2)
    CFs = load_bulk(BF_ptr, bidp * BF_stride_p, BF_stride_d1, BF_stride_d2)
    for l_offset in range(1, L):
        BM_offsets = l_offset * BM_stride_l + bidp * BM_stride_p
        BF_offsets = l_offset * BF_stride_l + bidp * BF_stride_p

        Ms = load_bulk(BM_ptr, BM_offsets, BM_stride_d1, BM_stride_d2)
        Fs = load_bulk(BF_ptr, BF_offsets, BF_stride_d1, BF_stride_d2)
        Vs = associative_operator(*CMs, *CFs, *Ms, *Fs)
        CMs, CFs = Vs[:4], Vs[4:]

        store_bulk(BM_ptr, BM_offsets, BM_stride_d1, BM_stride_d2, CMs)
        store_bulk(BF_ptr, BF_offsets, BF_stride_d1, BF_stride_d2, CFs)

@triton.jit
def parallel_scan_epilogue(
    # --- Input and Output Pointers for the Cumalative Scan ---
    OM_ptr, OF_ptr,
    # --- Output Pointers for the Scan of the Entire Block ---
    BM_ptr, BF_ptr,
    # --- Dimensions ---
    L,
    # --- Strides ---
    OM_stride_l, OM_stride_p, OM_stride_d1, OM_stride_d2,
    OF_stride_l, OF_stride_p, OF_stride_d1, OF_stride_d2,
    BM_stride_l, BM_stride_p, BM_stride_d1, BM_stride_d2,
    BF_stride_l, BF_stride_p, BF_stride_d1, BF_stride_d2,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    l_mask = l_offsets < L

    OM_offsets = l_offsets * OM_stride_l + bidp * OM_stride_p
    OF_offsets = l_offsets * OF_stride_l + bidp * OF_stride_p
    mask       = l_mask

    OMs = load_bulk(OM_ptr, OM_offsets, OM_stride_d1, OM_stride_d2, mask=mask)
    OFs = load_bulk(OF_ptr, OF_offsets, OF_stride_d1, OF_stride_d2, mask=mask)

    BM_offsets = bidl * BM_stride_l + bidp * BM_stride_p
    BF_offsets = bidl * BF_stride_l + bidp * BF_stride_p

    BMs = load_bulk(BM_ptr, BM_offsets, BM_stride_d1, BM_stride_d2)
    BFs = load_bulk(BF_ptr, BF_offsets, BF_stride_d1, BF_stride_d2)

    Vs = associative_operator(*BMs, *BFs, *OMs, *OFs)
    OMs, OFs = Vs[:4], Vs[4:]

    store_bulk(OM_ptr, OM_offsets, OM_stride_d1, OM_stride_d2, OMs, mask=mask)
    store_bulk(OF_ptr, OF_offsets, OF_stride_d1, OF_stride_d2, OFs, mask=mask)
