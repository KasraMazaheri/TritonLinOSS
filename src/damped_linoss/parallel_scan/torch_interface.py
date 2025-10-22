import torch
import triton

from .triton_parallel_scan import (
    parallel_scan,
    inter_block_scan,
    parallel_scan_epilogue,
)


def parallel_scan_wrapper(M, F, TILE_L=64, TILE_P=32):
    L = F.shape[0]
    P = F.shape[1] // 2 # F1 and F2 are hstacked
    assert M.shape == (4 * P,)
    assert F.shape == (L, 2 * P, 2)
    M = M.broadcast_to((L, 4 * P)).reshape(L, 4, P)
    F = F.reshape(L, 2, P, 2)

    M_11, M_12, M_21, M_22 = M[:, 0, :], M[:, 1, :], M[:, 2, :], M[:, 3, :]
    F1, F2 = F[:, 0], F[:, 1]
    F_r1, F_i1 = F1[..., 0], F1[..., 1]
    F_r2, F_i2 = F2[..., 0], F2[..., 1]

    # Allocate output tensors
    OM = torch.zeros_like(M)
    OF = torch.zeros_like(F)

    OM_11, OM_12, OM_21, OM_22 = OM[:, 0, :], OM[:, 1, :], OM[:, 2, :], OM[:, 3, :]
    OF_1, OF_2 = OF[:, 0], OF[:, 1]
    OF_r1, OF_i1 = OF_1[..., 0], OF_1[..., 1]
    OF_r2, OF_i2 = OF_2[..., 0], OF_2[..., 1]

    # First parallel scan
    grid = (triton.cdiv(L, TILE_L), triton.cdiv(P, TILE_P))
    parallel_scan[grid](
        # --- Input Pointers ---
        M_11_ptr=M_11, M_12_ptr=M_12, M_21_ptr=M_21, M_22_ptr=M_22,
        F_r1_ptr=F_r1, F_i1_ptr=F_i1, F_r2_ptr=F_r2, F_i2_ptr=F_i2,
        # --- Output Pointers for the Scan of the Entire Block ---
        OM_11_ptr=OM_11, OM_12_ptr=OM_12, OM_21_ptr=OM_21, OM_22_ptr=OM_22,
        OF_r1_ptr=OF_r1, OF_i1_ptr=OF_i1, OF_r2_ptr=OF_r2, OF_i2_ptr=OF_i2,
        # --- Dimensions ---
        L=L, P=P,
        # --- Strides ---
        M_stride_l=M_11.stride()[0], M_stride_p=M_11.stride()[1],
        F_stride_l=F_r1.stride()[0], F_stride_p=F_r1.stride()[1],
        OM_stride_l=OM_11.stride()[0], OM_stride_p=OM_11.stride()[1],
        OF_stride_l=OF_r1.stride()[0], OF_stride_p=OF_r1.stride()[1],
        # --- Compile-time Constants ---
        TILE_L=TILE_L,
        TILE_P=TILE_P,
    )

    BM = OM[TILE_L - 1 :: TILE_L].clone()
    BF = OF[TILE_L - 1 :: TILE_L].clone()

    BM_11, BM_12, BM_21, BM_22 = BM[:, 0, :], BM[:, 1, :], BM[:, 2, :], BM[:, 3, :]
    BF_1, BF_2 = BF[:, 0], BF[:, 1]
    BF_r1, BF_i1 = BF_1[..., 0], BF_1[..., 1]
    BF_r2, BF_i2 = BF_2[..., 0], BF_2[..., 1]

    # Compute partial sums
    grid = (P,)
    inter_block_scan[grid](
        # --- Input & Output Pointers for the Scan of the Entire Block ---
        BM_11_ptr=BM_11, BM_12_ptr=BM_12, BM_21_ptr=BM_21, BM_22_ptr=BM_22,
        BF_r1_ptr=BF_r1, BF_i1_ptr=BF_i1, BF_r2_ptr=BF_r2, BF_i2_ptr=BF_i2,
        # --- Dimensions ---
        L=BM_11.shape[0],
        # --- Strides ---
        BM_stride_l=BM_11.stride()[0], BM_stride_p=BM_11.stride()[1],
        BF_stride_l=BF_r1.stride()[0], BF_stride_p=BF_r1.stride()[1],
    )

    # Parallel scan epilogue to add partial sums to each block
    # Note that the first block does not need to be updated with the partial sums
    grid = (triton.cdiv(L, TILE_L) - 1, triton.cdiv(P, TILE_P))
    parallel_scan_epilogue[grid](
        # --- Input and Output Pointers for the Cumalative Scan ---
        OM_11_ptr=OM_11[TILE_L:], OM_12_ptr=OM_12[TILE_L:], OM_21_ptr=OM_21[TILE_L:], OM_22_ptr=OM_22[TILE_L:],
        OF_r1_ptr=OF_r1[TILE_L:], OF_i1_ptr=OF_i1[TILE_L:], OF_r2_ptr=OF_r2[TILE_L:], OF_i2_ptr=OF_i2[TILE_L:],
        # --- Output Pointers for the Scan of the Entire Block ---
        BM_11_ptr=BM_11, BM_12_ptr=BM_12, BM_21_ptr=BM_21, BM_22_ptr=BM_22,
        BF_r1_ptr=BF_r1, BF_i1_ptr=BF_i1, BF_r2_ptr=BF_r2, BF_i2_ptr=BF_i2,
        # --- Dimensions ---
        L = L - TILE_L, P=P,
        # --- Strides ---
        OM_stride_l=OM_11.stride()[0], OM_stride_p=OM_11.stride()[1],
        OF_stride_l=OF_r1.stride()[0], OF_stride_p=OF_r1.stride()[1],
        BM_stride_l=BM_11.stride()[0], BM_stride_p=BM_11.stride()[1],
        BF_stride_l=BF_r1.stride()[0], BF_stride_p=BF_r1.stride()[1],
        # --- Compile-time Constants ---
        TILE_L=TILE_L,
        TILE_P=TILE_P,
    )

    return OM.reshape(L, 4 * P), OF.reshape(L, 2 * P, 2)
