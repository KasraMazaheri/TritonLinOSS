import torch

try:
    import triton
    from .triton_parallel_scan import (
        parallel_scan_fwd,
        inter_block_scan_fwd,
        parallel_scan_epilogue_fwd,
        parallel_scan_bwd,
        inter_block_scan_bwd,
        parallel_scan_epilogue_bwd,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class ParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, F, TILE_L=None):
        """
        The forward pass is identical to your original wrapper function.
        We save the inputs and outputs for the backward pass.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is not available. Install with 'pip install damped-linoss[cuda]' "
                "or use the torch-compiled version instead."
            )
        
        if M.ndim == 1: # Unbatched
            assert F.ndim == 3
            M = M.unsqueeze(0)
            F = F.unsqueeze(0)

        B, L = F.shape[:2]
        if TILE_L is None:
            TILE_L = 128 if L <= 128 else 256 if L < 512 else 512
        elif TILE_L > 512:
            raise ValueError("TILE_L must be <= 512 for the Triton scan kernels.")
        P = F.shape[2] // 2
        assert M.shape == (B, 4 * P)
        assert F.shape == (B, L, 2 * P, 2)

        # Restructure as 2x2 matrices
        M = M.reshape(B, 2, 2, P).permute(0, 3, 1, 2)
        F = F.reshape(B, L, 2, P, 2).permute(0, 1, 3, 2, 4)

        M = M.unsqueeze(1).expand(-1, L, -1, -1, -1)

        # Allocate output tensors
        OM = torch.zeros_like(M)
        OF = torch.zeros_like(F)

        # First parallel scan
        grid = (B, P, triton.cdiv(L, TILE_L))
        parallel_scan_fwd[grid](
            M, F,
            OM, OF,
            L,
            M.stride(), F.stride(), OM.stride(), OF.stride(),
            TILE_L=TILE_L,
        )

        num_blocks_l = triton.cdiv(L, TILE_L)
        if num_blocks_l > 1:
            BM = OM[:, TILE_L - 1::TILE_L].clone()
            BF = OF[:, TILE_L - 1::TILE_L].clone()

            # Compute partial sums
            grid_inter = (B, P)
            inter_block_scan_fwd[grid_inter](
                BM, BF,
                BM.shape[1],
                BM.stride(), BF.stride(),
            )

            # Parallel scan epilogue to add partial sums to each block
            # Note that the first block does not need to be updated with the partial sums
            grid_epilogue = (B, P, num_blocks_l - 1)
            parallel_scan_epilogue_fwd[grid_epilogue](
                OM[:, TILE_L:], OF[:, TILE_L:],
                BM, BF,
                L - TILE_L,
                OM.stride(), OF.stride(), BM.stride(), BF.stride(),
                TILE_L=TILE_L,
            )
        
        # Save tensors and constants for backward pass
        ctx.save_for_backward(M, F, OM, OF)
        ctx.TILE_L = TILE_L

        OM = OM.permute(0, 1, 3, 4, 2).reshape(B, L, 4 * P)
        OF = OF.permute(0, 1, 3, 2, 4).reshape(B, L, 2 * P, 2)

        return OM, OF

    @staticmethod
    def backward(ctx, gOM, gOF):
        """
        Backward pass using the three-stage reverse scan, mirroring the forward pass structure.
        """
        M, F, OM, OF = ctx.saved_tensors
        TILE_L = ctx.TILE_L

        B, L, P = F.shape[:3]
        assert M.shape == (B, L, P, 2, 2)
        assert F.shape == (B, L, P, 2, 2)

        if gOM.ndim == 2: # Unbatched
            assert gOF.ndim == 3
            gOM = gOM.unsqueeze(0)
            gOF = gOF.unsqueeze(0)

        assert gOM.shape == (B, L, 4 * P)
        assert gOF.shape == (B, L, 2 * P, 2)

        gOM = gOM.reshape(B, L, 2, 2, P).permute(0, 1, 4, 2, 3)
        gOF = gOF.reshape(B, L, 2, P, 2).permute(0, 1, 3, 2, 4)

        # Allocate output tensors
        RM = torch.zeros_like(M)
        gM = torch.zeros_like(M)
        gF = torch.zeros_like(F)

        # First parallel scan
        grid = (B, P, triton.cdiv(L, TILE_L))
        parallel_scan_bwd[grid](
            M, gOM, gOF,
            RM, gM, gF,
            L,
            M.stride(), gOM.stride(), gOF.stride(), RM.stride(), gM.stride(), gF.stride(),
            TILE_L=TILE_L,
        )

        num_blocks_l = triton.cdiv(L, TILE_L)
        BM  = RM[:, 0::TILE_L].clone()
        gBM = gM[:, 0::TILE_L].clone()
        gBF = gF[:, 0::TILE_L].clone()

        if num_blocks_l > 1:
            # Compute partial sums
            grid_inter = (B, P)
            inter_block_scan_bwd[grid_inter](
                BM, gBM, gBF,
                BM.shape[1],
                BM.stride(), gBM.stride(), gBF.stride(),
            )

        # Parallel scan epilogue to add partial sums to each block
        grid_epilogue = (B, P, num_blocks_l)
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

        gM = gM.permute(0, 1, 3, 4, 2).reshape(B, L, 4 * P).sum(1)
        gF = gF.permute(0, 1, 3, 2, 4).reshape(B, L, 2 * P, 2)

        return gM, gF, None
