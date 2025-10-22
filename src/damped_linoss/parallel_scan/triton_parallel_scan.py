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


# ================================================================================
# Backward Pass for Parallel Scan
# ================================================================================

@triton.jit
def vm_prod_a(
    # c_tuple: The vector for the vector-matrix product
    c_M_11, c_M_12, c_M_21, c_M_22,
    c_F_r1, c_F_i1, c_F_r2, c_F_i2,
    # x_tuple: The state that defines the Jacobian matrix J_a(x)
    x_M_11, x_M_12, x_M_21, x_M_22,
    # ... x_F values are not needed for J_a
):
    """Computes c * J_a(x), the vector-matrix product needed for the grad scan."""
    # Matrix J_a(x) has a block-diagonal structure where both blocks are identical.
    # We compute `res_M = c_M * J_MM` and `res_F = c_F * J_FF`
    
    # Result for the M part of the state
    res_M_11 = c_M_11 * x_M_11 + c_M_21 * x_M_21
    res_M_12 = c_M_12 * x_M_11 + c_M_22 * x_M_21
    res_M_21 = c_M_11 * x_M_12 + c_M_21 * x_M_22
    res_M_22 = c_M_12 * x_M_12 + c_M_22 * x_M_22

    # Result for the F part of the state
    res_F_r1 = c_F_r1 * x_M_11 + c_F_r2 * x_M_21
    res_F_i1 = c_F_i1 * x_M_11 + c_F_i2 * x_M_21
    res_F_r2 = c_F_r1 * x_M_12 + c_F_r2 * x_M_22
    res_F_i2 = c_F_i1 * x_M_12 + c_F_i2 * x_M_22

    return (res_M_11, res_M_12, res_M_21, res_M_22,
            res_F_r1, res_F_i1, res_F_r2, res_F_i2)

@triton.jit
def vjp_b(
    # g_out_tuple: Upstream gradients for the output
    g_M_11, g_M_12, g_M_21, g_M_22,
    g_F_r1, g_F_i1, g_F_r2, g_F_i2,
    # i_tuple: First input to the forward op, defines Jacobian J_b(i)
    i_M_11, i_M_12, i_M_21, i_M_22,
    i_F_r1, i_F_i1, i_F_r2, i_F_i2,
):
    """Computes J_b(i)^T * g_out, the VJP for the final gradient calculation."""
    # Gradient for the F part of the second input (j_F)
    g_j_F_r1 = g_F_r1
    g_j_F_i1 = g_F_i1
    g_j_F_r2 = g_F_r2
    g_j_F_i2 = g_F_i2

    # Gradient for the M part of the second input (j_M)
    g_j_M_11 = g_M_11 * i_M_11 + g_M_12 * i_M_12 + g_F_r1 * i_F_r1 + g_F_i1 * i_F_i1
    g_j_M_12 = g_M_11 * i_M_21 + g_M_12 * i_M_22 + g_F_r1 * i_F_r2 + g_F_i1 * i_F_i2
    g_j_M_21 = g_M_21 * i_M_11 + g_M_22 * i_M_12 + g_F_r2 * i_F_r1 + g_F_i2 * i_F_i1
    g_j_M_22 = g_M_21 * i_M_21 + g_M_22 * i_M_22 + g_F_r2 * i_F_r2 + g_F_i2 * i_F_i2

    return (g_j_M_11, g_j_M_12, g_j_M_21, g_j_M_22,
            g_j_F_r1, g_j_F_i1, g_j_F_r2, g_j_F_i2)

@triton.jit
def associative_operator_bwd(
    # First input pair (i_x, i_c)
    i_x_M_11, i_x_M_12, i_x_M_21, i_x_M_22,
    i_x_F_r1, i_x_F_i1, i_x_F_r2, i_x_F_i2,
    i_c_M_11, i_c_M_12, i_c_M_21, i_c_M_22,
    i_c_F_r1, i_c_F_i1, i_c_F_r2, i_c_F_i2,
    # Second input pair (j_x, j_c)
    j_x_M_11, j_x_M_12, j_x_M_21, j_x_M_22,
    j_x_F_r1, j_x_F_i1, j_x_F_r2, j_x_F_i2,
    j_c_M_11, j_c_M_12, j_c_M_21, j_c_M_22,
    j_c_F_r1, j_c_F_i1, j_c_F_r2, j_c_F_i2,
):
    # The output x-state combines using the original forward operator
    o_x_M_11, o_x_M_12, o_x_M_21, o_x_M_22, \
    o_x_F_r1, o_x_F_i1, o_x_F_r2, o_x_F_i2 = associative_operator(
        i_x_M_11, i_x_M_12, i_x_M_21, i_x_M_22,
        i_x_F_r1, i_x_F_i1, i_x_F_r2, i_x_F_i2,
        j_x_M_11, j_x_M_12, j_x_M_21, j_x_M_22,
        j_x_F_r1, j_x_F_i1, j_x_F_r2, j_x_F_i2,
    )

    # The output c-state (gradient) combines via c_new = c1 + c2 * J_a(x1)
    vm_prod_res = vm_prod_a(
        j_c_M_11, j_c_M_12, j_c_M_21, j_c_M_22,
        j_c_F_r1, j_c_F_i1, j_c_F_r2, j_c_F_i2,
        i_x_M_11, i_x_M_12, i_x_M_21, i_x_M_22,
    )

    o_c_M_11 = i_c_M_11 + vm_prod_res[0]
    o_c_M_12 = i_c_M_12 + vm_prod_res[1]
    o_c_M_21 = i_c_M_21 + vm_prod_res[2]
    o_c_M_22 = i_c_M_22 + vm_prod_res[3]
    o_c_F_r1 = i_c_F_r1 + vm_prod_res[4]
    o_c_F_i1 = i_c_F_i1 + vm_prod_res[5]
    o_c_F_r2 = i_c_F_r2 + vm_prod_res[6]
    o_c_F_i2 = i_c_F_i2 + vm_prod_res[7]
    
    return (
        o_x_M_11, o_x_M_12, o_x_M_21, o_x_M_22,
        o_x_F_r1, o_x_F_i1, o_x_F_r2, o_x_F_i2,
        o_c_M_11, o_c_M_12, o_c_M_21, o_c_M_22,
        o_c_F_r1, o_c_F_i1, o_c_F_r2, o_c_F_i2,
    )

@triton.jit
def parallel_scan_bwd(
    # --- Input Pointers ---
    # Original inputs to the forward scan (X)
    X_M_11_ptr, X_M_12_ptr, X_M_21_ptr, X_M_22_ptr,
    X_F_r1_ptr, X_F_i1_ptr, X_F_r2_ptr, X_F_i2_ptr,
    # Outputs of the forward scan (Y)
    Y_M_11_ptr, Y_M_12_ptr, Y_M_21_ptr, Y_M_22_ptr,
    Y_F_r1_ptr, Y_F_i1_ptr, Y_F_r2_ptr, Y_F_i2_ptr,
    # Upstream gradients w.r.t. Y (gY)
    gY_M_11_ptr, gY_M_12_ptr, gY_M_21_ptr, gY_M_22_ptr,
    gY_F_r1_ptr, gY_F_i1_ptr, gY_F_r2_ptr, gY_F_i2_ptr,
    # --- Output Pointers ---
    # Gradients w.r.t. X (gX)
    gX_M_11_ptr, gX_M_12_ptr, gX_M_21_ptr, gX_M_22_ptr,
    gX_F_r1_ptr, gX_F_i1_ptr, gX_F_r2_ptr, gX_F_i2_ptr,
    # --- Dimensions ---
    L, P,
    # --- Strides ---
    XM_stride_l, XM_stride_p, XF_stride_l, XF_stride_p,
    YM_stride_l, YM_stride_p, YF_stride_l, YF_stride_p,
    gYM_stride_l, gYM_stride_p, gYF_stride_l, gYF_stride_p,
    gXM_stride_l, gXM_stride_p, gXF_stride_l, gXF_stride_p,
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
    mask = l_mask[:, None] & p_mask[None, :]

    # --- Step 1: Load inputs for the reverse scan ---
    # The state for the reverse scan is (x, gY)
    XM_offsets = l_offsets[:, None] * XM_stride_l + p_offsets[None, :] * XM_stride_p
    XF_offsets = l_offsets[:, None] * XF_stride_l + p_offsets[None, :] * XF_stride_p
    X_M_tuple = load_bulk(X_M_11_ptr, X_M_12_ptr, X_M_21_ptr, X_M_22_ptr, XM_offsets, mask=mask)
    X_F_tuple = load_bulk(X_F_r1_ptr, X_F_i1_ptr, X_F_r2_ptr, X_F_i2_ptr, XF_offsets, mask=mask)
    
    gYM_offsets = l_offsets[:, None] * gYM_stride_l + p_offsets[None, :] * gYM_stride_p
    gYF_offsets = l_offsets[:, None] * gYF_stride_l + p_offsets[None, :] * gYF_stride_p
    gY_M_tuple = load_bulk(gY_M_11_ptr, gY_M_12_ptr, gY_M_21_ptr, gY_M_22_ptr, gYM_offsets, mask=mask)
    gY_F_tuple = load_bulk(gY_F_r1_ptr, gY_F_i1_ptr, gY_F_r2_ptr, gY_F_i2_ptr, gYF_offsets, mask=mask)

    # --- Step 2: Perform the reverse associative scan ---
    scan_result = tl.associative_scan(
        X_M_tuple + X_F_tuple + gY_M_tuple + gY_F_tuple, axis=0, combine_fn=associative_operator_bwd, reverse=True
    )
    # The result contains the scanned x and c values. We only need the scanned c (gradient).
    C_tuple = scan_result[8:16]

    # --- Step 3: Compute final gradients gX ---
    # gX_i = vjp_b(C_i, Y_{i-1})
    l_offsets_prev = l_offsets - 1
    l_mask_prev = l_offsets_prev >= 0
    mask_prev = l_mask_prev[:, None] & p_mask[None, :]
    
    YM_offsets_prev = l_offsets_prev[:, None] * YM_stride_l + p_offsets[None, :] * YM_stride_p
    YF_offsets_prev = l_offsets_prev[:, None] * YF_stride_l + p_offsets[None, :] * YF_stride_p
    Y_prev_M_tuple = load_bulk(Y_M_11_ptr, Y_M_12_ptr, Y_M_21_ptr, Y_M_22_ptr, YM_offsets_prev, mask=mask_prev)
    Y_prev_F_tuple = load_bulk(Y_F_r1_ptr, Y_F_i1_ptr, Y_F_r2_ptr, Y_F_i2_ptr, YF_offsets_prev, mask=mask_prev)
    
    # For the first element (i=0), Y_{-1} is the identity (M=I, F=0).
    Y_prev_M_11, Y_prev_M_12, Y_prev_M_21, Y_prev_M_22 = Y_prev_M_tuple
    Y_prev_F_r1, Y_prev_F_i1, Y_prev_F_r2, Y_prev_F_i2 = Y_prev_F_tuple
    
    Y_prev_M_11 = tl.where(l_mask_prev[:, None], Y_prev_M_11, 1.0)
    Y_prev_M_12 = tl.where(l_mask_prev[:, None], Y_prev_M_12, 0.0)
    Y_prev_M_21 = tl.where(l_mask_prev[:, None], Y_prev_M_21, 0.0)
    Y_prev_M_22 = tl.where(l_mask_prev[:, None], Y_prev_M_22, 1.0)
    Y_prev_F_r1 = tl.where(l_mask_prev[:, None], Y_prev_F_r1, 0.0)
    Y_prev_F_i1 = tl.where(l_mask_prev[:, None], Y_prev_F_i1, 0.0)
    Y_prev_F_r2 = tl.where(l_mask_prev[:, None], Y_prev_F_r2, 0.0)
    Y_prev_F_i2 = tl.where(l_mask_prev[:, None], Y_prev_F_i2, 0.0)
    
    Y_prev_tuple_identity = (
        Y_prev_M_11, Y_prev_M_12, Y_prev_M_21, Y_prev_M_22,
        Y_prev_F_r1, Y_prev_F_i1, Y_prev_F_r2, Y_prev_F_i2
    )

    # Calculate final gradient gX
    gX_tuple = vjp_b(*C_tuple, *Y_prev_tuple_identity)
    gX_M_tuple = (gX_tuple[0], gX_tuple[1], gX_tuple[2], gX_tuple[3])
    gX_F_tuple = (gX_tuple[4], gX_tuple[5], gX_tuple[6], gX_tuple[7])

    # Store the result
    gXM_offsets = l_offsets[:, None] * gXM_stride_l + p_offsets[None, :] * gXM_stride_p
    gXF_offsets = l_offsets[:, None] * gXF_stride_l + p_offsets[None, :] * gXF_stride_p
    store_bulk(gX_M_11_ptr, gX_M_12_ptr, gX_M_21_ptr, gX_M_22_ptr, gXM_offsets, *gX_M_tuple, mask=mask)
    store_bulk(gX_F_r1_ptr, gX_F_i1_ptr, gX_F_r2_ptr, gX_F_i2_ptr, gXF_offsets, *gX_F_tuple, mask=mask)