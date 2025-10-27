import torch
import triton

from .triton_parallel_scan import (
    parallel_scan,
    inter_block_scan,
    parallel_scan_epilogue,
)


class ParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, F, TILE_L=64):
        """
        The forward pass is identical to your original wrapper function.
        We save the inputs and outputs for the backward pass.
        """
        L = F.shape[0]
        P = F.shape[1] // 2
        assert M.shape == (4 * P,)
        assert F.shape == (L, 2 * P, 2)

        # Restructure as 2x2 matrices
        M = M.reshape(2, 2, P).permute(2, 0, 1)
        F = F.reshape(L, 2, P, 2).permute(0, 2, 1, 3)

        M = M.broadcast_to((L, P, 2, 2))

        # Allocate output tensors
        OM = torch.zeros_like(M)
        OF = torch.zeros_like(F)

        # First parallel scan
        grid = (triton.cdiv(L, TILE_L), P)
        parallel_scan[grid](
            # --- Input Pointers ---
            M, F,
            # --- Output Pointers ---
            OM, OF,
            # --- Dimensions ---
            L,
            # --- Strides ---
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            F.stride(0), F.stride(1), F.stride(2), F.stride(3),
            OM.stride(0), OM.stride(1), OM.stride(2), OM.stride(3),
            OF.stride(0), OF.stride(1), OF.stride(2), OF.stride(3),
            # --- Compile-time Constants ---
            TILE_L=TILE_L,
        )

        num_blocks_l = triton.cdiv(L, TILE_L)
        if num_blocks_l > 1:
            BM = OM[TILE_L - 1::TILE_L].clone()
            BF = OF[TILE_L - 1::TILE_L].clone()

            # Compute partial sums
            grid_inter = (P,)
            inter_block_scan[grid_inter](
                # --- Input & Output Pointers for the Scan of the Entire Block ---
                BM, BF,
                # --- Dimensions ---
                BM.shape[0],
                # --- Strides ---
                BM.stride(0), BM.stride(1), BM.stride(2), BM.stride(3),
                BF.stride(0), BF.stride(1), BF.stride(2), BF.stride(3),
            )

            # Parallel scan epilogue to add partial sums to each block
            # Note that the first block does not need to be updated with the partial sums
            grid_epilogue = (num_blocks_l - 1, P)
            parallel_scan_epilogue[grid_epilogue](
                # --- Input and Output Pointers for the Cumalative Scan ---
                OM[TILE_L:], OF[TILE_L:],
                # --- Output Pointers for the Scan of the Entire Block ---
                BM, BF,
                # --- Dimensions ---
                L - TILE_L,
                # --- Strides ---
                OM.stride(0), OM.stride(1), OM.stride(2), OM.stride(3),
                OF.stride(0), OF.stride(1), OF.stride(2), OF.stride(3),
                BM.stride(0), BM.stride(1), BM.stride(2), BM.stride(3),
                BF.stride(0), BF.stride(1), BF.stride(2), BF.stride(3),
                # --- Compile-time Constants ---
                TILE_L=TILE_L,
            )
        
        # Save tensors and constants for backward pass
        ctx.save_for_backward(M, F, OM, OF)
        ctx.TILE_L = TILE_L

        OM = OM.permute(0, 2, 3, 1)
        OF = OF.permute(0, 2, 1, 3)

        return OM.reshape(L, 4 * P), OF.reshape(L, 2 * P, 2)

    @staticmethod
    def backward(ctx, gOM, gOF):
        """
        Backward pass using the three-stage reverse scan, mirroring the forward pass structure.
        """
        M, F, OM, OF = ctx.saved_tensors
        TILE_L = ctx.TILE_L

        raise NotImplementedError("Backward pass is not implemented yet.")