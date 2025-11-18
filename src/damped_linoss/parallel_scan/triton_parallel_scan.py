import triton
import triton.language as tl

# ===============================================
# Utility Functions for LinOSS Parallel Scan
# ===============================================


@triton.jit
def load_2x2(ptr, offsets, strides, mask=None, others=None):
    bidb, bidp = tl.program_id(0), tl.program_id(1)
    ptr = ptr + bidb * strides[0] + offsets * strides[1] + bidp * strides[2]
    others = (None,) * 4 if mask is None else ((0.0,) * 4 if others is None else others)
    return (
        tl.load(ptr, mask=mask, other=others[0]),
        tl.load(ptr + strides[4], mask=mask, other=others[1]),
        tl.load(ptr + strides[3], mask=mask, other=others[2]),
        tl.load(ptr + strides[3] + strides[4], mask=mask, other=others[3]),
    )


@triton.jit
def store_2x2(ptr, offsets, strides, Vs, mask=None):
    bidb, bidp = tl.program_id(0), tl.program_id(1)
    ptr = ptr + bidb * strides[0] + offsets * strides[1] + bidp * strides[2]
    tl.store(ptr, Vs[0], mask=mask)
    tl.store(ptr + strides[4], Vs[1], mask=mask)
    tl.store(ptr + strides[3], Vs[2], mask=mask)
    tl.store(ptr + strides[3] + strides[4], Vs[3], mask=mask)


# ===============================================
# Forward Pass for LinOSS Parallel Scan
# ===============================================


@triton.jit
def associative_operator_fwd(
    i_M_11, i_M_12, i_M_21, i_M_22,
    i_F_11, i_F_12, i_F_21, i_F_22,
    j_M_11, j_M_12, j_M_21, j_M_22,
    j_F_11, j_F_12, j_F_21, j_F_22,
):  # fmt: skip
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
def parallel_scan_fwd(
    M_ptr, F_ptr,
    OM_ptr, OF_ptr,
    L,
    M_strides, F_strides, OM_strides, OF_strides,
    TILE_L: tl.constexpr,
):  # fmt: skip
    bidl = tl.program_id(2)
    offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    mask = offsets < L

    Ms = load_2x2(M_ptr, offsets, M_strides, mask)
    Fs = load_2x2(F_ptr, offsets, F_strides, mask)

    Vs = tl.associative_scan(Ms + Fs, axis=0, combine_fn=associative_operator_fwd)

    store_2x2(OM_ptr, offsets, OM_strides, Vs[:4], mask)
    store_2x2(OF_ptr, offsets, OF_strides, Vs[4:], mask)


@triton.jit
def inter_block_scan_fwd(
    BM_ptr, BF_ptr,
    L, # Different from the main L, this is number of blocks
    BM_strides, BF_strides,
):  # fmt: skip
    CMs = load_2x2(BM_ptr, 0, BM_strides)
    CFs = load_2x2(BF_ptr, 0, BF_strides)
    for offset in range(1, L):
        Ms = load_2x2(BM_ptr, offset, BM_strides)
        Fs = load_2x2(BF_ptr, offset, BF_strides)

        Vs = associative_operator_fwd(*CMs, *CFs, *Ms, *Fs)
        CMs, CFs = Vs[:4], Vs[4:]

        store_2x2(BM_ptr, offset, BM_strides, CMs)
        store_2x2(BF_ptr, offset, BF_strides, CFs)


@triton.jit
def parallel_scan_epilogue_fwd(
    OM_ptr, OF_ptr,
    BM_ptr, BF_ptr,
    L,
    OM_strides, OF_strides, BM_strides, BF_strides,
    TILE_L: tl.constexpr,
):  # fmt: skip
    bidl = tl.program_id(2)
    offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    mask = offsets < L

    OMs = load_2x2(OM_ptr, offsets, OM_strides, mask=mask)
    OFs = load_2x2(OF_ptr, offsets, OF_strides, mask=mask)
    BMs = load_2x2(BM_ptr, bidl, BM_strides)
    BFs = load_2x2(BF_ptr, bidl, BF_strides)

    Vs = associative_operator_fwd(*BMs, *BFs, *OMs, *OFs)

    store_2x2(OM_ptr, offsets, OM_strides, Vs[:4], mask=mask)
    store_2x2(OF_ptr, offsets, OF_strides, Vs[4:], mask=mask)


# ===============================================
# Backward Pass for LinOSS Parallel Scan
# ===============================================


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
):  # fmt: skip
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
    )  # fmt: skip


@triton.jit
def vjb(
    gOM_11, gOM_12, gOM_21, gOM_22, 
    gOF_11, gOF_12, gOF_21, gOF_22, 
    OM_prev_11, OM_prev_12, OM_prev_21, OM_prev_22, 
    OF_prev_11, OF_prev_12, OF_prev_21, OF_prev_22, 
):  # fmt: skip
    # gM = gOF * OF[i-1].T + gOM * OM[i-1].T
    gM_11 = (gOF_11 * OF_prev_11 + gOF_12 * OF_prev_12) + (
        gOM_11 * OM_prev_11 + gOM_12 * OM_prev_12
    )
    gM_12 = (gOF_11 * OF_prev_21 + gOF_12 * OF_prev_22) + (
        gOM_11 * OM_prev_21 + gOM_12 * OM_prev_22
    )
    gM_21 = (gOF_21 * OF_prev_11 + gOF_22 * OF_prev_12) + (
        gOM_21 * OM_prev_11 + gOM_22 * OM_prev_12
    )
    gM_22 = (gOF_21 * OF_prev_21 + gOF_22 * OF_prev_22) + (
        gOM_21 * OM_prev_21 + gOM_22 * OM_prev_22
    )
    return gM_11, gM_12, gM_21, gM_22


@triton.jit
def parallel_scan_bwd(
    M_ptr, gOM_ptr, gOF_ptr,
    RM_ptr, gM_ptr, gF_ptr,
    L,
    M_strides, gOM_strides, gOF_strides, RM_strides, gM_strides, gF_strides,
    TILE_L: tl.constexpr,
):  # fmt: skip
    bidl = tl.program_id(2)
    offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    mask = offsets < L

    Ms = load_2x2(M_ptr, offsets, M_strides, mask=mask, others=(1, 0, 0, 1))
    gOMs = load_2x2(gOM_ptr, offsets, gOM_strides, mask=mask)
    gOFs = load_2x2(gOF_ptr, offsets, gOF_strides, mask=mask)

    Vs = tl.associative_scan(
        Ms + gOMs + gOFs, axis=0, combine_fn=associative_operator_bwd, reverse=True
    )

    store_2x2(RM_ptr, offsets, RM_strides, Vs[:4], mask=mask)
    store_2x2(gM_ptr, offsets, gM_strides, Vs[4:8], mask=mask)
    store_2x2(gF_ptr, offsets, gF_strides, Vs[8:], mask=mask)


@triton.jit
def inter_block_scan_bwd(
    BM_ptr, gBM_ptr, gBF_ptr,
    L, # Different from the main L, this is number of blocks
    BM_strides, gBM_strides, gBF_strides,
):  # fmt: skip
    CMs = load_2x2(BM_ptr, L - 1, BM_strides)
    gCMs = load_2x2(gBM_ptr, L - 1, gBM_strides)
    gCFs = load_2x2(gBF_ptr, L - 1, gBF_strides)

    for offset in range(L - 2, -1, -1):
        BMs = load_2x2(BM_ptr, offset, BM_strides)
        gBMs = load_2x2(gBM_ptr, offset, gBM_strides)
        gBFs = load_2x2(gBF_ptr, offset, gBF_strides)

        Vs = associative_operator_bwd(*BMs, *gBMs, *gBFs, *CMs, *gCMs, *gCFs)
        CMs, gCMs, gCFs = Vs[:4], Vs[4:8], Vs[8:]

        store_2x2(BM_ptr, offset, BM_strides, CMs)
        store_2x2(gBM_ptr, offset, gBM_strides, gCMs)
        store_2x2(gBF_ptr, offset, gBF_strides, gCFs)


@triton.jit
def parallel_scan_epilogue_bwd(
    OM_ptr, OF_ptr,
    BM_ptr, gBM_ptr, gBF_ptr,
    RM_ptr, gM_ptr, gF_ptr,
    L,
    OM_strides, OF_strides, 
    BM_strides, gBM_strides, gBF_strides, 
    RM_strides, gM_strides, gF_strides,
    TILE_L: tl.constexpr,
):  # fmt: skip
    bidl = tl.program_id(2)
    numl = tl.num_programs(2)
    offsets = bidl * TILE_L + tl.arange(0, TILE_L)
    l_mask = offsets < L
    b_mask = bidl + 1 < numl

    RMs = load_2x2(RM_ptr, offsets, RM_strides, mask=l_mask)
    gMs = load_2x2(gM_ptr, offsets, gM_strides, mask=l_mask)
    gFs = load_2x2(gF_ptr, offsets, gF_strides, mask=l_mask)
    BMs = load_2x2(BM_ptr, bidl + 1, BM_strides, mask=b_mask)
    gBMs = load_2x2(gBM_ptr, bidl + 1, gBM_strides, mask=b_mask)
    gBFs = load_2x2(gBF_ptr, bidl + 1, gBF_strides, mask=b_mask)

    Vs = associative_operator_bwd(*RMs, *gMs, *gFs, *BMs, *gBMs, *gBFs)
    gOMs, gOFs = Vs[4:8], Vs[8:12]

    prev_offsets = offsets - 1
    l_prev_mask = prev_offsets >= 0

    OMs_prev = load_2x2(
        OM_ptr, prev_offsets, OM_strides, mask=l_prev_mask, others=(1, 0, 0, 1)
    )
    OFs_prev = load_2x2(OF_ptr, prev_offsets, OF_strides, mask=l_prev_mask)

    gFs = gOFs
    gMs = vjb(*gOMs, *gOFs, *OMs_prev, *OFs_prev)

    store_2x2(gM_ptr, offsets, gM_strides, gMs, mask=l_mask)
    store_2x2(gF_ptr, offsets, gF_strides, gFs, mask=l_mask)
