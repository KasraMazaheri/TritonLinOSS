import torch
import triton

from .triton_parallel_scan import (
    parallel_scan,
    inter_block_scan,
    parallel_scan_epilogue,
)


class ParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, F, TILE_L=64, TILE_P=32):
        """
        The forward pass is identical to your original wrapper function.
        We save the inputs and outputs for the backward pass.
        """
        L = F.shape[0]
        P = F.shape[1] // 2
        assert M.shape == (4 * P,)
        assert F.shape == (L, 2 * P, 2)
        M_b = M.broadcast_to((L, 4 * P)).reshape(L, 4, P)
        F_b = F.reshape(L, 2, P, 2)

        M_11, M_12, M_21, M_22 = M_b.unbind(1)
        F1, F2 = F_b.unbind(1)
        F_r1, F_i1 = F1.unbind(-1)
        F_r2, F_i2 = F2.unbind(-1)

        # Allocate output tensors
        OM = torch.zeros_like(M_b)
        OF = torch.zeros_like(F_b)
        
        OM_11, OM_12, OM_21, OM_22 = OM.unbind(1)
        OF_1, OF_2 = OF.unbind(1)
        OF_r1, OF_i1 = OF_1.unbind(-1)
        OF_r2, OF_i2 = OF_2.unbind(-1)

        # First parallel scan
        grid = (triton.cdiv(L, TILE_L), triton.cdiv(P, TILE_P))
        parallel_scan[grid](
            # --- Input Pointers ---
            M_11, M_12, M_21, M_22, F_r1, F_i1, F_r2, F_i2,
            # --- Output Pointers ---
            OM_11, OM_12, OM_21, OM_22, OF_r1, OF_i1, OF_r2, OF_i2,
            # --- Dimensions ---
            L, P,
            # --- Strides ---
            M_11.stride(0), M_11.stride(1), F_r1.stride(0), F_r1.stride(1),
            OM_11.stride(0), OM_11.stride(1), OF_r1.stride(0), OF_r1.stride(1),
            # --- Compile-time Constants ---
            TILE_L=TILE_L,
            TILE_P=TILE_P,
        )

        num_blocks_l = triton.cdiv(L, TILE_L)
        if num_blocks_l > 1:
            BM = OM[TILE_L - 1::TILE_L].clone()
            BF = OF[TILE_L - 1::TILE_L].clone()

            BM_11, BM_12, BM_21, BM_22 = BM.unbind(1)
            BF_1, BF_2 = BF.unbind(1)
            BF_r1, BF_i1 = BF_1.unbind(-1)
            BF_r2, BF_i2 = BF_2.unbind(-1)

            # Compute partial sums
            grid_inter = (P,)
            inter_block_scan[grid_inter](
                # --- Input & Output Pointers for the Scan of the Entire Block ---
                BM_11, BM_12, BM_21, BM_22, BF_r1, BF_i1, BF_r2, BF_i2,
                # --- Dimensions ---
                BM_11.shape[0],
                # --- Strides ---
                BM_11.stride(0), BM_11.stride(1), BF_r1.stride(0), BF_r1.stride(1),
            )

            # Parallel scan epilogue to add partial sums to each block
            # Note that the first block does not need to be updated with the partial sums
            grid_epilogue = (num_blocks_l - 1, triton.cdiv(P, TILE_P))
            parallel_scan_epilogue[grid_epilogue](
                # --- Input and Output Pointers for the Cumalative Scan ---
                OM_11[TILE_L:], OM_12[TILE_L:], OM_21[TILE_L:], OM_22[TILE_L:],
                OF_r1[TILE_L:], OF_i1[TILE_L:], OF_r2[TILE_L:], OF_i2[TILE_L:],
                # --- Output Pointers for the Scan of the Entire Block ---
                BM_11, BM_12, BM_21, BM_22, BF_r1, BF_i1, BF_r2, BF_i2,
                # --- Dimensions ---
                L - TILE_L, P,
                # --- Strides ---
                OM_11.stride(0), OM_11.stride(1), OF_r1.stride(0), OF_r1.stride(1),
                BM_11.stride(0), BM_11.stride(1), BF_r1.stride(0), BF_r1.stride(1),
                # --- Compile-time Constants ---
                TILE_L=TILE_L,
                TILE_P=TILE_P,
            )
        
        # Save tensors and constants for backward pass
        ctx.save_for_backward(M, F, OM, OF)
        ctx.TILE_L = TILE_L
        ctx.TILE_P = TILE_P

        return OM.reshape(L, 4 * P), OF.reshape(L, 2 * P, 2)

    @staticmethod
    def backward(ctx, gOM, gOF):
        """
        Backward pass using the three-stage reverse scan, mirroring the forward pass structure.
        """
        M, F, OM, OF = ctx.saved_tensors
        TILE_L = ctx.TILE_L
        TILE_P = ctx.TILE_P

        raise NotImplementedError("Backward pass is not implemented yet.")