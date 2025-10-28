import triton
import triton.language as tl


@triton.jit
def associative_operator_fwd(
    i_M_11, i_M_12, i_M_21, i_M_22,
    i_F_11, i_F_12, i_F_21, i_F_22,
    j_M_11, j_M_12, j_M_21, j_M_22,
    j_F_11, j_F_12, j_F_21, j_F_22,
):
    # M = j_M * i_M
    M_11 = j_M_11 * i_M_11 + j_M_12 * i_M_21
    M_12 = j_M_11 * i_M_12 + j_M_12 * i_M_22
    M_21 = j_M_21 * i_M_11 + j_M_22 * i_M_21
    M_22 = j_M_21 * i_M_12 + j_M_22 * i_M_22

    # F = j_M * i_F + j_F
    F_11 = j_M_11 * i_F_11 + j_M_12 * i_F_21 + j_F_11
    F_12 = j_M_11 * i_F_12 + j_M_12 * i_F_22 + j_F_12
    F_21 = j_M_21 * i_F_11 + j_M_22 * i_F_21 + j_F_21
    F_22 = j_M_21 * i_F_12 + j_M_22 * i_F_22 + j_F_22
    return (M_11, M_12, M_21, M_22, F_11, F_12, F_21, F_22)

@triton.jit
def load_bulk(ptr, offsets, d1_stride, d2_stride, mask=None):
    other = None if mask is None else 0.0
    val1 = tl.load(ptr + offsets, mask=mask, other=other)
    val2 = tl.load(ptr + offsets + d2_stride, mask=mask, other=other)
    val3 = tl.load(ptr + offsets + d1_stride, mask=mask, other=other)
    val4 = tl.load(ptr + offsets + d1_stride + d2_stride, mask=mask, other=other)
    return val1, val2, val3, val4

@triton.jit
def load_bulk_bwd(ptr, offsets, d1_stride, d2_stride, mask, others):
    other1, other2, other3, other4 = others
    val1 = tl.load(ptr + offsets, mask=mask, other=other1)
    val2 = tl.load(ptr + offsets + d2_stride, mask=mask, other=other2)
    val3 = tl.load(ptr + offsets + d1_stride, mask=mask, other=other3)
    val4 = tl.load(ptr + offsets + d1_stride + d2_stride, mask=mask, other=other4)
    return val1, val2, val3, val4

@triton.jit
def store_bulk(ptr, offsets, d1_stride, d2_stride, Vs, mask=None):
    tl.store(ptr + offsets, Vs[0], mask=mask)
    tl.store(ptr + offsets + d2_stride, Vs[1], mask=mask)
    tl.store(ptr + offsets + d1_stride, Vs[2], mask=mask)
    tl.store(ptr + offsets + d1_stride + d2_stride, Vs[3], mask=mask)

@triton.jit
def parallel_scan_fwd(
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
        Ms + Fs, axis=0, combine_fn=associative_operator_fwd
    )
    OMs, OFs = Vs[:4], Vs[4:]

    OM_offsets = l_offsets * OM_stride_l + bidp * OM_stride_p
    OF_offsets = l_offsets * OF_stride_l + bidp * OF_stride_p

    store_bulk(OM_ptr, OM_offsets, OM_stride_d1, OM_stride_d2, OMs, mask=mask)
    store_bulk(OF_ptr, OF_offsets, OF_stride_d1, OF_stride_d2, OFs, mask=mask)
    
@triton.jit
def inter_block_scan_fwd(
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
        Vs = associative_operator_fwd(*CMs, *CFs, *Ms, *Fs)
        CMs, CFs = Vs[:4], Vs[4:]

        store_bulk(BM_ptr, BM_offsets, BM_stride_d1, BM_stride_d2, CMs)
        store_bulk(BF_ptr, BF_offsets, BF_stride_d1, BF_stride_d2, CFs)

@triton.jit
def parallel_scan_epilogue_fwd(
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

    Vs = associative_operator_fwd(*BMs, *BFs, *OMs, *OFs)
    OMs, OFs = Vs[:4], Vs[4:]

    store_bulk(OM_ptr, OM_offsets, OM_stride_d1, OM_stride_d2, OMs, mask=mask)
    store_bulk(OF_ptr, OF_offsets, OF_stride_d1, OF_stride_d2, OFs, mask=mask)

# ================================================================================
# Backward Pass for Parallel Scan
# ================================================================================

@triton.jit
def associative_operator_bwd(
    # Left element (j)
    j_M_11, j_M_12, j_M_21, j_M_22,
    j_gOM_11, j_gOM_12, j_gOM_21, j_gOM_22,
    j_gOF_11, j_gOF_12, j_gOF_21, j_gOF_22,
    # Right element (i)
    i_M_11, i_M_12, i_M_21, i_M_22,
    i_gOM_11, i_gOM_12, i_gOM_21, i_gOM_22,
    i_gOF_11, i_gOF_12, i_gOF_21, i_gOF_22,
):
    M_11 = i_M_11 * j_M_11 + i_M_12 * j_M_21
    M_12 = i_M_11 * j_M_12 + i_M_12 * j_M_22
    M_21 = i_M_21 * j_M_11 + i_M_22 * j_M_21
    M_22 = i_M_21 * j_M_12 + i_M_22 * j_M_22

    # Logic: gOF_new = j_gOF + j_M.T * i_gOF
    gOF_11 = j_gOF_11 + (j_M_11 * i_gOF_11 + j_M_21 * i_gOF_21)
    gOF_12 = j_gOF_12 + (j_M_11 * i_gOF_12 + j_M_21 * i_gOF_22)
    gOF_21 = j_gOF_21 + (j_M_12 * i_gOF_11 + j_M_22 * i_gOF_21)
    gOF_22 = j_gOF_22 + (j_M_12 * i_gOF_12 + j_M_22 * i_gOF_22)

    # Logic: gOM_new = j_gOM + j_M.T * i_gOM
    gOM_11 = j_gOM_11 + (j_M_11 * i_gOM_11 + j_M_21 * i_gOM_21)
    gOM_12 = j_gOM_12 + (j_M_11 * i_gOM_12 + j_M_21 * i_gOM_22)
    gOM_21 = j_gOM_21 + (j_M_12 * i_gOM_11 + j_M_22 * i_gOM_21)
    gOM_22 = j_gOM_22 + (j_M_12 * i_gOM_12 + j_M_22 * i_gOM_22)

    return (
        M_11, M_12, M_21, M_22,
        gOM_11, gOM_12, gOM_21, gOM_22,
        gOF_11, gOF_12, gOF_21, gOF_22,
    )

@triton.jit
def vjb(
    gOM_11, gOM_12, gOM_21, gOM_22, 
    gOF_11, gOF_12, gOF_21, gOF_22, 
    OM_prev_11, OM_prev_12, OM_prev_21, OM_prev_22, 
    OF_prev_11, OF_prev_12, OF_prev_21, OF_prev_22, 
):
    # gM = gOF * OF[i-1].T + gOM * OM[i-1].T
    gM_11 = (gOF_11 * OF_prev_11 + gOF_12 * OF_prev_12) + (gOM_11 * OM_prev_11 + gOM_12 * OM_prev_12)
    gM_12 = (gOF_11 * OF_prev_21 + gOF_12 * OF_prev_22) + (gOM_11 * OM_prev_21 + gOM_12 * OM_prev_22)
    gM_21 = (gOF_21 * OF_prev_11 + gOF_22 * OF_prev_12) + (gOM_21 * OM_prev_11 + gOM_22 * OM_prev_12)
    gM_22 = (gOF_21 * OF_prev_21 + gOF_22 * OF_prev_22) + (gOM_21 * OM_prev_21 + gOM_22 * OM_prev_22)
    return gM_11, gM_12, gM_21, gM_22

@triton.jit
def parallel_scan_bwd(
    # --- Input Pointers ---
    M_ptr,
    gOM_ptr, gOF_ptr,
    # --- Output Pointers for the Cumalative Scan ---
    RM_ptr, gM_ptr, gF_ptr,
    # --- Dimensions ---
    L,
    # --- Strides ---
    M_stride_l, M_stride_p, M_stride_d1, M_stride_d2,
    gOM_stride_l, gOM_stride_p, gOM_stride_d1, gOM_stride_d2,
    gOF_stride_l, gOF_stride_p, gOF_stride_d1, gOF_stride_d2,
    RM_stride_l, RM_stride_p, RM_stride_d1, RM_stride_d2,
    gM_stride_l, gM_stride_p, gM_stride_d1, gM_stride_d2,
    gF_stride_l, gF_stride_p, gF_stride_d1, gF_stride_d2,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    l_mask = l_offsets < L

    M_offsets   = l_offsets * M_stride_l   + bidp * M_stride_p
    gOM_offsets = l_offsets * gOM_stride_l + bidp * gOM_stride_p
    gOF_offsets = l_offsets * gOF_stride_l + bidp * gOF_stride_p
    RM_offsets  = l_offsets * RM_stride_l  + bidp * RM_stride_p
    gM_offsets  = l_offsets * gM_stride_l  + bidp * gM_stride_p
    gF_offsets  = l_offsets * gF_stride_l  + bidp * gF_stride_p

    Ms   = load_bulk_bwd(M_ptr, M_offsets, M_stride_d1, M_stride_d2, mask=l_mask, others=(1, 0, 0, 1))
    gOMs = load_bulk(gOM_ptr, gOM_offsets, gOM_stride_d1, gOM_stride_d2, mask=l_mask)
    gOFs = load_bulk(gOF_ptr, gOF_offsets, gOF_stride_d1, gOF_stride_d2, mask=l_mask)

    Vs = tl.associative_scan(
        Ms + gOMs + gOFs, axis=0, combine_fn=associative_operator_bwd, reverse=True
    )
    RMs, gMs, gFs = Vs[:4], Vs[4:8], Vs[8:12]

    store_bulk(RM_ptr, RM_offsets, RM_stride_d1, RM_stride_d2, RMs, mask=l_mask)
    store_bulk(gM_ptr, gM_offsets, gM_stride_d1, gM_stride_d2, gMs, mask=l_mask)
    store_bulk(gF_ptr, gF_offsets, gF_stride_d1, gF_stride_d2, gFs, mask=l_mask)

@triton.jit
def inter_block_scan_bwd(
    # --- Input & Output Pointers for the Scan of the Entire Block ---
    BM_ptr, gBM_ptr, gBF_ptr,
    # --- Dimensions ---
    L, # Different from the main L, this is number of blocks
    # --- Strides ---
    BM_stride_l, BM_stride_p, BM_stride_d1, BM_stride_d2,
    gBM_stride_l, gBM_stride_p, gBM_stride_d1, gBM_stride_d2,
    gBF_stride_l, gBF_stride_p, gBF_stride_d1, gBF_stride_d2,
):
    bidp = tl.program_id(0)

    BM_offset  = (L - 1) * BM_stride_l  + bidp * BM_stride_p
    gBM_offset = (L - 1) * gBM_stride_l + bidp * gBM_stride_p
    gBF_offset = (L - 1) * gBF_stride_l + bidp * gBF_stride_p

    CMs  = load_bulk(BM_ptr, BM_offset, BM_stride_d1, BM_stride_d2)
    gCMs = load_bulk(gBM_ptr, gBM_offset, gBM_stride_d1, gBM_stride_d2)
    gCFs = load_bulk(gBF_ptr, gBF_offset, gBF_stride_d1, gBF_stride_d2)
    for _ in range(L - 2, -1, -1):
        BM_offset  -= BM_stride_l
        gBM_offset -= gBM_stride_l
        gBF_offset -= gBF_stride_l

        BMs  = load_bulk(BM_ptr, BM_offset, BM_stride_d1, BM_stride_d2)
        gBMs = load_bulk(gBM_ptr, gBM_offset, gBM_stride_d1, gBM_stride_d2)
        gBFs = load_bulk(gBF_ptr, gBF_offset, gBF_stride_d1, gBF_stride_d2)
        Vs = associative_operator_bwd(*BMs, *gBMs, *gBFs, *CMs, *gCMs, *gCFs)
        CMs, gCMs, gCFs = Vs[:4], Vs[4:8], Vs[8:12]

        store_bulk(BM_ptr, BM_offset, BM_stride_d1, BM_stride_d2, CMs)
        store_bulk(gBM_ptr, gBM_offset, gBM_stride_d1, gBM_stride_d2, gCMs)
        store_bulk(gBF_ptr, gBF_offset, gBF_stride_d1, gBF_stride_d2, gCFs)

@triton.jit
def parallel_scan_epilogue_bwd(
    # --- Input and Output Pointers for the Cumalative Scan ---
    OM_ptr, OF_ptr,
    BM_ptr, gBM_ptr, gBF_ptr,
    RM_ptr, gM_ptr, gF_ptr,
    # --- Dimensions ---
    L,
    # --- Strides ---
    OM_stride_l, OM_stride_p, OM_stride_d1, OM_stride_d2,
    OF_stride_l, OF_stride_p, OF_stride_d1, OF_stride_d2,
    
    BM_stride_l, BM_stride_p, BM_stride_d1, BM_stride_d2,
    gBM_stride_l, gBM_stride_p, gBM_stride_d1, gBM_stride_d2,
    gBF_stride_l, gBF_stride_p, gBF_stride_d1, gBF_stride_d2,
    
    RM_stride_l, RM_stride_p, RM_stride_d1, RM_stride_d2,
    gM_stride_l, gM_stride_p, gM_stride_d1, gM_stride_d2,
    gF_stride_l, gF_stride_p, gF_stride_d1, gF_stride_d2,
    # --- Compile-time Constants ---
    TILE_L: tl.constexpr,
):
    bidl = tl.program_id(0)
    bidp = tl.program_id(1)

    l_offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    l_mask = l_offsets < L

    RM_offsets = l_offsets * RM_stride_l + bidp * RM_stride_p
    gM_offsets = l_offsets * gM_stride_l + bidp * gM_stride_p
    gF_offsets = l_offsets * gF_stride_l + bidp * gF_stride_p
    mask       = l_mask

    RMs = load_bulk(RM_ptr, RM_offsets, RM_stride_d1, RM_stride_d2, mask=mask)
    gMs = load_bulk(gM_ptr, gM_offsets, gM_stride_d1, gM_stride_d2, mask=mask)
    gFs = load_bulk(gF_ptr, gF_offsets, gF_stride_d1, gF_stride_d2, mask=mask)

    BM_offset  = (bidl + 1) * BM_stride_l + bidp * BM_stride_p
    gBM_offset = (bidl + 1) * gBM_stride_l + bidp * gBM_stride_p
    gBF_offset = (bidl + 1) * gBF_stride_l + bidp * gBF_stride_p
    b_mask     = (bidl + 1) < tl.num_programs(0)

    BMs  = load_bulk(BM_ptr, BM_offset, BM_stride_d1, BM_stride_d2, b_mask)
    gBMs = load_bulk(gBM_ptr, gBM_offset, gBM_stride_d1, gBM_stride_d2, b_mask)
    gBFs = load_bulk(gBF_ptr, gBF_offset, gBF_stride_d1, gBF_stride_d2, b_mask)

    Vs = associative_operator_bwd(*RMs, *gMs, *gFs, *BMs, *gBMs, *gBFs)
    gOMs, gOFs = Vs[4:8], Vs[8:12]

    l_offsets_prev = l_offsets - 1
    l_mask_prev = l_offsets_prev >= 0

    OM_prev_offsets = l_offsets_prev * OM_stride_l + bidp * OM_stride_p
    OF_prev_offsets = l_offsets_prev * OF_stride_l + bidp * OF_stride_p
    OMs_prev = load_bulk_bwd(OM_ptr, OM_prev_offsets, OM_stride_d1, OM_stride_d2, mask=l_mask_prev, others=(1, 0, 0, 1))
    OFs_prev = load_bulk(OF_ptr, OF_prev_offsets, OF_stride_d1, OF_stride_d2, mask=l_mask_prev)

    gFs = gOFs
    gMs = vjb(*gOMs, *gOFs, *OMs_prev, *OFs_prev)

    store_bulk(gM_ptr, gM_offsets, gM_stride_d1, gM_stride_d2, gMs, mask=l_mask)
    store_bulk(gF_ptr, gF_offsets, gF_stride_d1, gF_stride_d2, gFs, mask=l_mask)
