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
        grid = (P, triton.cdiv(L, TILE_L))
        parallel_scan_fwd[grid](
            M, F,
            OM, OF,
            L,
            M.stride(), F.stride(), OM.stride(), OF.stride(),
            TILE_L=TILE_L,
        )

        num_blocks_l = triton.cdiv(L, TILE_L)
        if num_blocks_l > 1:
            BM = OM[TILE_L - 1::TILE_L].clone()
            BF = OF[TILE_L - 1::TILE_L].clone()

            # Compute partial sums
            grid_inter = (P,)
            inter_block_scan_fwd[grid_inter](
                BM, BF,
                BM.shape[0],
                BM.stride(), BF.stride(),
            )

            # Parallel scan epilogue to add partial sums to each block
            # Note that the first block does not need to be updated with the partial sums
            grid_epilogue = (P, num_blocks_l - 1)
            parallel_scan_epilogue_fwd[grid_epilogue](
                OM[TILE_L:], OF[TILE_L:],
                BM, BF,
                L - TILE_L,
                OM.stride(), OF.stride(), BM.stride(), BF.stride(),
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
        grid = (P, triton.cdiv(L, TILE_L))
        parallel_scan_bwd[grid](
            M, gOM, gOF,
            RM, gM, gF,
            L,
            M.stride(), gOM.stride(), gOF.stride(), RM.stride(), gM.stride(), gF.stride(),
            TILE_L=TILE_L,
        )

        BM  = RM[0::TILE_L].clone()
        gBM = gM[0::TILE_L].clone()
        gBF = gF[0::TILE_L].clone()

        # Compute partial sums
        grid_inter = (P,)
        inter_block_scan_bwd[grid_inter](
            BM, gBM, gBF,
            BM.shape[0],
            BM.stride(), gBM.stride(), gBF.stride(),
        )

        # Parallel scan epilogue to add partial sums to each block
        grid_epilogue = (P, triton.cdiv(L, TILE_L))
        parallel_scan_epilogue_bwd[grid_epilogue](
            OM, OF,
            BM, gBM, gBF,
            RM, gM, gF,
            L,
            OM.stride(), OF.stride(), 
            BM.stride(), gBM.stride(), gBF.stride(), 
            RM.stride(), gM.stride(), gF.stride(),
            TILE_L=TILE_L,
        )

        gM = gM.permute(0, 2, 3, 1)
        gF = gF.permute(0, 2, 1, 3)

        return gM.reshape(L, 4 * P), gF.reshape(L, 2 * P, 2), None