import torch
import triton

from .triton_parallel_scan import (
    parallel_scan_fwd,
    inter_block_scan_fwd,
    parallel_scan_epilogue_fwd,
    parallel_scan_bwd,
    inter_block_scan_bwd,
    parallel_scan_epilogue_bwd,
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
        parallel_scan_fwd[grid](
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
            inter_block_scan_fwd[grid_inter](
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
            parallel_scan_epilogue_fwd[grid_epilogue](
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

        L = F.shape[0]
        P = F.shape[1]
        assert M.shape == (L, P, 2, 2)
        assert F.shape == (L, P, 2, 2)
        assert gOM.shape == (L, 4 * P)
        assert gOF.shape == (L, 2 * P, 2)

        gOM = gOM.reshape(L, 2, 2, P).permute(0, 3, 1, 2)
        gOF = gOF.reshape(L, 2, P, 2).permute(0, 2, 1, 3)

        # Allocate output tensors
        RM = torch.zeros_like(M)
        gM = torch.zeros_like(M)
        gF = torch.zeros_like(F)

        # First parallel scan
        grid = (triton.cdiv(L, TILE_L), P)
        parallel_scan_bwd[grid](
            # --- Input Pointers ---
            M, gOM, gOF,
            # --- Output Pointers ---
            RM, gM, gF,
            # --- Dimensions ---
            L,
            # --- Strides ---
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            gOM.stride(0), gOM.stride(1), gOM.stride(2), gOM.stride(3),
            gOF.stride(0), gOF.stride(1), gOF.stride(2), gOF.stride(3),
            RM.stride(0), RM.stride(1), RM.stride(2), RM.stride(3),
            gM.stride(0), gM.stride(1), gM.stride(2), gM.stride(3),
            gF.stride(0), gF.stride(1), gF.stride(2), gF.stride(3),
            # --- Compile-time Constants ---
            TILE_L=TILE_L,
        )

        BM  = RM[0::TILE_L].clone()
        gBM = gM[0::TILE_L].clone()
        gBF = gF[0::TILE_L].clone()

        # Compute partial sums
        grid_inter = (P,)
        inter_block_scan_bwd[grid_inter](
            # --- Input & Output Pointers for the Scan of the Entire Block ---
            BM, gBM, gBF,
            # --- Dimensions ---
            BM.shape[0],
            # --- Strides ---
            BM.stride(0), BM.stride(1), BM.stride(2), BM.stride(3),
            gBM.stride(0), gBM.stride(1), gBM.stride(2), gBM.stride(3),
            gBF.stride(0), gBF.stride(1), gBF.stride(2), gBF.stride(3),
        )

        # Parallel scan epilogue to add partial sums to each block
        grid_epilogue = (triton.cdiv(L, TILE_L), P)
        parallel_scan_epilogue_bwd[grid_epilogue](
            # --- Input and Output Pointers for the Cumalative Scan ---
            OM, OF,
            BM, gBM, gBF,
            RM, gM, gF,
            # --- Dimensions ---
            L,
            # --- Strides ---
            OM.stride(0), OM.stride(1), OM.stride(2), OM.stride(3),
            OF.stride(0), OF.stride(1), OF.stride(2), OF.stride(3),
            BM.stride(0), BM.stride(1), BM.stride(2), BM.stride(3),
            gBM.stride(0), gBM.stride(1), gBM.stride(2), gBM.stride(3),
            gBF.stride(0), gBF.stride(1), gBF.stride(2), gBF.stride(3),
            RM.stride(0), RM.stride(1), RM.stride(2), RM.stride(3),
            gM.stride(0), gM.stride(1), gM.stride(2), gM.stride(3),
            gF.stride(0), gF.stride(1), gF.stride(2), gF.stride(3),
            # --- Compile-time Constants ---
            TILE_L=TILE_L,
        )

        gM = gM.permute(0, 2, 3, 1)
        gF = gF.permute(0, 2, 1, 3)

        return gM.reshape(L, 4 * P), gF.reshape(L, 2 * P, 2), None