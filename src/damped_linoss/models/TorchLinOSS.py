import abc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Literal, Tuple

try:
    from damped_linoss.parallel_scan.torch_interface import (
        ParallelScanFunction,
        TRITON_AVAILABLE,
    )
except ImportError:
    TRITON_AVAILABLE = False
    ParallelScanFunction = None

from damped_linoss.parallel_scan.torch_associative_scan import associative_scan


SCAN_TYPE = Tuple[torch.Tensor, torch.Tensor]
DISCRETIZATION = Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"]
INITIALIZATION = Literal["AG", "RT"]
STABILITY = Literal["oscillatory", "stable"]


def binary_operator(q_i: SCAN_TYPE, q_j: SCAN_TYPE) -> SCAN_TYPE:
    """
    Binary operator for parallel scan of linear recurrence.

    Args:
        q_i: tuple (A_i, b_i)
             A_i shape: (..., 4 * P)
             b_i shape: (..., 2 * P, 2)
        q_j: tuple (A_j, b_j)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    iA, iB, iC, iD = torch.chunk(A_i, 4, dim=-1)
    jA, jB, jC, jD = torch.chunk(A_j, 4, dim=-1)

    A_new_part = jA * iA + jB * iC
    B_new_part = jA * iB + jB * iD
    C_new_part = jC * iA + jD * iC
    D_new_part = jC * iB + jD * iD

    # Concatenate back to shape (..., 4*P)
    A_new = torch.cat([A_new_part, B_new_part, C_new_part, D_new_part], dim=-1)

    b_i1, b_i2 = torch.chunk(b_i, 2, dim=-2)

    # unsqueeze the last dim of A parts to broadcast over the complex dimension of b.
    jA_bs = jA.unsqueeze(-1)
    jB_bs = jB.unsqueeze(-1)
    jC_bs = jC.unsqueeze(-1)
    jD_bs = jD.unsqueeze(-1)

    new_b1 = jA_bs * b_i1 + jB_bs * b_i2
    new_b2 = jC_bs * b_i1 + jD_bs * b_i2

    new_b = torch.cat([new_b1, new_b2], dim=-2)

    return A_new, new_b + b_j


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        """
        Initializes the Gated Linear Unit (GLU) module.
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        Applies the GLU activation: w1(x) * sigmoid(w2(x))
        Handles (L, H) and (B, L, H) inputs.
        """
        return self.w1(x) * torch.sigmoid(self.w2(x))


class _AbstractLinOSSLayer(nn.Module):
    def __init__(self, use_triton=None):
        super().__init__()
        self.use_triton = use_triton

    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError

    def _should_use_triton(self, tensor):
        """Determine whether to use Triton backend based on override flag and tensor device."""
        if self.use_triton is None:
            # Auto: use Triton if available and tensor is on CUDA
            use_triton = TRITON_AVAILABLE and tensor.is_cuda

            # Warn if CUDA is available but Triton is not
            if tensor.is_cuda and not TRITON_AVAILABLE:
                warnings.warn(
                    "Running on CUDA but Triton is not available. "
                    "Performance may be suboptimal. "
                    "Install Triton with 'pip install damped-linoss[cuda]' for better performance.",
                    UserWarning,
                    stacklevel=4,
                )

            return use_triton
        else:
            # Manual override
            if self.use_triton and not TRITON_AVAILABLE:
                raise RuntimeError(
                    "Triton backend requested but not available. "
                    "Install with 'pip install damped-linoss[cuda]'."
                )
            if self.use_triton and not tensor.is_cuda:
                raise RuntimeError(
                    "Triton backend requested but input tensors are not on CUDA. "
                    "Move the model and inputs to CUDA or set use_triton=False."
                )
            return self.use_triton


def _uniform_parameter(shape, half_width, *, device=None, dtype=torch.float32):
    parameter = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    init.uniform_(parameter, -half_width, half_width)
    return parameter


def _normal_parameter(shape, std, *, device=None, dtype=torch.float32):
    parameter = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    init.normal_(parameter, std=std)
    return parameter


def _mat_im(A, G, step):
    S = 1 + step * G + step**2 * A
    return 1 / S, -step * A / S, step / S, (1 + step * G) / S, step / S, step**2 / S


def _mat_imex(A, G, step):
    S = 1 + step * G
    return 1 / S, -step * A / S, step / S, 1 - step**2 * A / S, step / S, step**2 / S


def _mat_imex2(A, G, step):
    return 1 - step * G, -step * A, step * (1 - step * G), 1 - step**2 * A, step, step**2


def _mat_imex3(A, G, step):
    S = 1 + step**2 * A
    return (
        (1 - step * G) / S,
        -step * A / S,
        step * (1 - step * G) / S,
        1 / S,
        step / S,
        step**2 / S,
    )


def _mat_ex(A, G, step):
    return 1 - step * G, -step * A, step, torch.ones_like(A), step, torch.zeros_like(A)


MATRIX_FNS = {
    "IM": _mat_im,
    "IMEX": _mat_imex,
    "IMEX2": _mat_imex2,
    "IMEX3": _mat_imex3,
    "EX": _mat_ex,
}


def _rt_to_ag(discretization, trace, determinant, step):
    if discretization == "IM":
        A = (determinant - trace + 1) / (determinant * step**2)
        G = (-2 * determinant + trace) / (determinant * step)
    elif discretization == "IMEX":
        A = (determinant - trace + 1) / (determinant * step**2)
        G = (1 - determinant) / (determinant * step)
    elif discretization == "IMEX2":
        A = (determinant - trace + 1) / step**2
        G = (1 - determinant) / step
    elif discretization == "IMEX3":
        denom = determinant - trace
        A = (-determinant + trace - 1) / (step**2 * denom)
        G = (2 * determinant - trace) / (step * denom)
    elif discretization == "EX":
        A = (determinant - trace + 1) / step**2
        G = (2 - trace) / step
    else:
        raise ValueError(f"Unknown discretization {discretization!r}")
    return A, G


def _project_ag_oscillatory(discretization, A_diag, G_diag, steps, eps=0.0):
    h = steps
    h2 = torch.clamp(steps**2, min=1e-6)

    if discretization == "IM":
        A_low_1 = -G_diag / h
        A_low_2 = G_diag**2 / 4
        A_diag = torch.maximum(torch.maximum(A_diag, A_low_1), A_low_2)
    elif discretization == "IMEX":
        G_diag = F.relu(G_diag)
        A_low = (2 + h * G_diag - 2 * torch.sqrt(1 + h * G_diag)) / h2
        A_high = (2 + h * G_diag + 2 * torch.sqrt(1 + h * G_diag)) / h2
        A_diag = torch.clamp(A_diag, min=A_low * (1 + eps), max=A_high * (1 - eps))
    elif discretization == "IMEX2":
        G_diag = torch.minimum(F.relu(G_diag), (1 / h) * (1 - eps))
        A_low = (2 - h * G_diag - 2 * torch.sqrt(1 - h * G_diag)) / h2
        A_high = (2 - h * G_diag + 2 * torch.sqrt(1 - h * G_diag)) / h2
        A_diag = torch.clamp(A_diag, min=A_low * (1 + eps), max=A_high * (1 - eps))
    elif discretization == "IMEX3":
        G_diag = torch.minimum(F.relu(G_diag), (1 / h) * (1 - eps))
        A_low = G_diag**2 / torch.clamp(4 * (1 - h * G_diag), min=1e-6)
        A_diag = A_low * (1 + eps) + F.relu(A_diag - A_low * (1 + eps))
    elif discretization == "EX":
        G_diag = torch.minimum(F.relu(G_diag), (4 / h) * (1 - eps))
        A_low = 0.25 * G_diag**2
        A_high = G_diag / h
        A_diag = torch.clamp(A_diag, min=A_low * (1 + eps), max=A_high * (1 - eps))
    else:
        raise ValueError(f"Unknown discretization {discretization!r}")

    return A_diag, G_diag


def _project_ag_stability(discretization, A_diag, G_diag, steps, eps=0.0):
    h = steps
    h2 = torch.clamp(steps**2, min=1e-6)

    if discretization == "IM":
        A_low_1 = -G_diag / h
        A_low_2 = -(2 * h * G_diag + 4) / h2
        A_diag = torch.maximum(torch.maximum(torch.maximum(A_diag, A_low_1), A_low_2), torch.zeros_like(A_diag))
    elif discretization == "IMEX":
        G_diag = F.relu(G_diag)
        A_high = (4 + 2 * h * G_diag) / h2
        A_diag = torch.minimum(F.relu(A_diag), A_high * (1 - eps))
    elif discretization == "IMEX2":
        G_diag = torch.minimum(F.relu(G_diag), (2 / h) * (1 - eps))
        A_high = (4 - 2 * h * G_diag) / h2
        A_diag = torch.minimum(F.relu(A_diag), A_high * (1 - eps))
    elif discretization == "IMEX3":
        A_low_1 = (2 * h * G_diag - 4) / h2
        A_low_2 = -G_diag / h
        A_diag = torch.maximum(torch.maximum(torch.maximum(A_diag, A_low_1), A_low_2), torch.zeros_like(A_diag))
    elif discretization == "EX":
        G_diag = torch.minimum(F.relu(G_diag), (4 / h) * (1 - eps))
        A_low = F.relu((2 * h * G_diag - 4) / h2)
        A_high = G_diag / h
        A_diag = torch.clamp(A_diag, min=A_low * (1 + eps), max=A_high * (1 - eps))
    else:
        raise ValueError(f"Unknown discretization {discretization!r}")

    return A_diag, G_diag


def _lambda_sq(discretization, A, G, step):
    m11, m12, m21, m22, _, _ = MATRIX_FNS[discretization](A, G, step)
    return m11 * m22 - m12 * m21


class LinOSSSequenceMixer(_AbstractLinOSSLayer):
    """Unified Discretax-style LinOSS layer with optional Triton scan backend."""

    def __init__(
        self,
        in_features,
        state_dim=64,
        discretization: DISCRETIZATION = "IMEX",
        initialization: INITIALIZATION = "AG",
        damping=True,
        stability: STABILITY = "oscillatory",
        projection_eps=0.0,
        input_normalization=False,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi / 4,
        num_heads=1,
        use_head_gating=False,
        use_head_output_projection=False,
        A_max=1.0,
        G_max=1.0,
        use_triton=None,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__(use_triton)
        if discretization not in MATRIX_FNS:
            raise ValueError(f"Unknown discretization {discretization!r}")
        if initialization not in {"AG", "RT"}:
            raise ValueError(f"Unknown initialization {initialization!r}")
        if stability not in {"oscillatory", "stable"}:
            raise ValueError(f"Unknown stability {stability!r}")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if in_features % num_heads != 0:
            raise ValueError(f"in_features={in_features} must be divisible by num_heads={num_heads}")
        if state_dim % num_heads != 0:
            raise ValueError(f"state_dim={state_dim} must be divisible by num_heads={num_heads}")
        if input_normalization and not damping:
            raise ValueError("input_normalization requires damping=True.")

        self.in_features = in_features
        self.state_dim = state_dim
        self.discretization = discretization
        self.initialization = initialization
        self.damping = damping
        self.stability = stability
        self.projection_eps = projection_eps
        self.input_normalization = input_normalization
        self.num_heads = num_heads
        self.head_hidden_dim = in_features // num_heads
        self.head_state_dim = state_dim // num_heads
        self.use_head_gating = use_head_gating and num_heads > 1
        self.use_head_output_projection = use_head_output_projection and num_heads > 1

        factory = {"device": device, "dtype": dtype}
        A_flat, G_flat, steps_flat = self._init_recurrence_parameters(
            state_dim, discretization, initialization, damping, r_min, r_max, theta_max, A_max, G_max, **factory
        )
        self.A_diag = nn.Parameter(A_flat.reshape(num_heads, self.head_state_dim))
        self.G_diag = nn.Parameter(G_flat.reshape(num_heads, self.head_state_dim))
        self.steps = nn.Parameter(steps_flat.reshape(num_heads, self.head_state_dim))

        self.B = _uniform_parameter(
            (num_heads, self.head_state_dim, self.head_hidden_dim, 2),
            1.0 / math.sqrt(self.head_hidden_dim),
            **factory,
        )
        self.C = _uniform_parameter(
            (num_heads, self.head_hidden_dim, self.head_state_dim, 2),
            1.0 / math.sqrt(self.head_state_dim),
            **factory,
        )
        self.D = _normal_parameter((num_heads, self.head_hidden_dim), 1.0, **factory)

        gamma_log = torch.zeros(state_dim, **factory)
        if input_normalization:
            with torch.no_grad():
                steps = torch.sigmoid(steps_flat)
                projector = _project_ag_oscillatory if stability == "oscillatory" else _project_ag_stability
                A_proj, G_proj = projector(discretization, A_flat, G_flat, steps, projection_eps)
                lam_sq = _lambda_sq(discretization, A_proj, G_proj, steps)
                gamma_log = 0.5 * torch.log(torch.clamp(1.0 - lam_sq, min=1e-6))
        self.gamma_log = nn.Parameter(gamma_log.reshape(num_heads, self.head_state_dim))

        self.head_gate = nn.Linear(in_features, num_heads, device=device, dtype=dtype) if self.use_head_gating else None
        self.head_output_projection = (
            nn.Linear(in_features, in_features, device=device, dtype=dtype)
            if self.use_head_output_projection
            else None
        )

    @staticmethod
    def _init_recurrence_parameters(
        state_dim,
        discretization,
        initialization,
        damping,
        r_min,
        r_max,
        theta_max,
        A_max,
        G_max,
        *,
        device=None,
        dtype=torch.float32,
    ):
        steps = torch.empty(state_dim, device=device, dtype=dtype)
        init.normal_(steps, std=0.5)

        if not damping:
            A_diag = torch.rand(state_dim, device=device, dtype=dtype) * A_max
            return A_diag, torch.zeros_like(A_diag), steps

        if initialization == "AG":
            A_diag = torch.rand(state_dim, device=device, dtype=dtype) * A_max
            G_diag = torch.rand(state_dim, device=device, dtype=dtype) * G_max
            return A_diag, G_diag, steps

        step_sigmoid = torch.sigmoid(steps)
        mag = torch.sqrt(torch.rand(state_dim, device=device, dtype=dtype) * (r_max**2 - r_min**2) + r_min**2)
        arg = torch.rand(state_dim, device=device, dtype=dtype) * theta_max
        trace = 2 * mag * torch.cos(arg)
        determinant = mag**2
        A_diag, G_diag = _rt_to_ag(discretization, trace, determinant, step_sigmoid)
        return A_diag.real.to(dtype=dtype), G_diag.real.to(dtype=dtype), steps

    def _project_parameters(self, steps):
        G_diag = self.G_diag if self.damping else torch.zeros_like(self.G_diag)
        projector = _project_ag_oscillatory if self.stability == "oscillatory" else _project_ag_stability
        return projector(self.discretization, self.A_diag, G_diag, steps, self.projection_eps)

    def _scan(self, M_elements, F_elements, input_sequence):
        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F_elements)
            return xs

        L = F_elements.shape[1]
        M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
        _, xs = associative_scan(
            binary_operator,
            (M_expanded, F_elements),
            reverse=False,
            axis=1,
        )
        return xs

    def _apply_recurrence(self, x, steps):
        batch_size, seq_len, _ = x.shape
        x_heads = x.reshape(batch_size, seq_len, self.num_heads, self.head_hidden_dim)
        # The Triton scan is batch-parallel; each head is an independent scan.
        x_flat = x_heads.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_hidden_dim)

        A, G = self._project_parameters(steps)
        flat_shape = (batch_size * self.num_heads, self.head_state_dim)
        A_flat = A.unsqueeze(0).expand(batch_size, -1, -1).reshape(flat_shape)
        G_flat = G.unsqueeze(0).expand(batch_size, -1, -1).reshape(flat_shape)
        step_flat = steps.unsqueeze(0).expand(batch_size, -1, -1).reshape(flat_shape)
        gamma_flat = self.gamma_log.unsqueeze(0).expand(batch_size, -1, -1).reshape(flat_shape)
        B_flat = (
            self.B.unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .reshape(batch_size * self.num_heads, self.head_state_dim, self.head_hidden_dim, 2)
        )

        Bu = torch.einsum("nlh,npht->nlpt", x_flat, B_flat)
        Bu = Bu * torch.exp(gamma_flat)[:, None, :, None]

        M_11, M_12, M_21, M_22, f1, f2 = MATRIX_FNS[self.discretization](A_flat, G_flat, step_flat)
        M_elements = torch.cat([M_11, M_12, M_21, M_22], dim=-1)
        F1 = Bu * f1[:, None, :, None]
        F2 = Bu * f2[:, None, :, None]
        F_elements = torch.cat([F1, F2], dim=-2)
        xs = self._scan(M_elements, F_elements, x)
        ys = xs[..., self.head_state_dim :, :]
        return ys.reshape(batch_size, self.num_heads, seq_len, self.head_state_dim, 2).permute(0, 2, 1, 3, 4)

    def forward(self, input_sequence):
        squeeze_batch = False
        if input_sequence.dim() == 2:
            input_sequence = input_sequence.unsqueeze(0)
            squeeze_batch = True
        elif input_sequence.dim() != 3:
            raise ValueError("input_sequence must have shape (L, H) or (B, L, H)")
        if input_sequence.shape[-1] != self.in_features:
            raise ValueError(f"Expected input feature dim {self.in_features}, got {input_sequence.shape[-1]}")

        steps = torch.sigmoid(self.steps)
        ys = self._apply_recurrence(input_sequence, steps)
        x_heads = input_sequence.reshape(input_sequence.shape[0], input_sequence.shape[1], self.num_heads, self.head_hidden_dim)

        Cy_complex = torch.einsum("nlhpt,hfpt->nlhft", ys, self.C)
        head_outputs = Cy_complex[..., 0] - Cy_complex[..., 1]
        head_outputs = head_outputs + x_heads * self.D[None, None, :, :]

        if self.head_gate is not None:
            gate_weights = torch.softmax(self.head_gate(input_sequence), dim=-1)
            head_outputs = head_outputs * gate_weights[..., None]

        output = head_outputs.reshape(input_sequence.shape[0], input_sequence.shape[1], self.in_features)
        if self.head_output_projection is not None:
            output = self.head_output_projection(output)
        return output.squeeze(0) if squeeze_batch else output


class IMLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, use_triton=None):
        super().__init__(use_triton)

        # Initialize parameters
        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)  # Matches jax.random.uniform default

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B, input_sequence, step):
        """Compute the LxP output of LinOSS-IM given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
        M_11 = 1.0 - step**2.0 * A_diag * schur_comp
        M_12 = -1.0 * step * A_diag * schur_comp
        M_21 = step * schur_comp
        M_22 = schur_comp

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        M_11_b = M_11.view(view_shape)
        M_21_b = M_21.view(view_shape)
        step_b = step.view(view_shape)

        F1 = M_11_b * Bu_elements * step_b
        F2 = M_21_b * Bu_elements * step_b

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        A_diag = F.relu(self.A_diag)

        # ys is (..., L, P, 2)
        ys = self._recurrence(A_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class IMEXLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, use_triton=None):
        super().__init__(use_triton)

        # Initialize parameters
        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)  # Matches jax.random.uniform default

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B, input_sequence, step):
        """Compute the LxP output of LinOSS-IM given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        M_11 = torch.ones_like(A_diag)
        M_12 = -1.0 * step * A_diag
        M_21 = step
        M_22 = 1.0 - (step**2.0) * A_diag

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        step_b = step.view(view_shape)

        F1 = Bu_elements * step_b
        F2 = Bu_elements * (step_b**2.0)

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        A_diag = F.relu(self.A_diag)

        # ys is (..., L, P, 2)
        ys = self._recurrence(A_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class DampedLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, use_triton=None):
        super().__init__(use_triton)

        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)
        steps = torch.sigmoid(self.steps)

        mags = torch.sqrt(torch.rand(state_dim) * (r_max**2 - r_min**2) + r_min**2)
        G_diag_init = (1 - mags**2) / (steps.detach() * mags**2)
        self.G_diag = nn.Parameter(G_diag_init)

        theta = torch.rand(state_dim) * theta_max
        A_diag_init = self._map_theta_to_A(
            theta, F.relu(self.G_diag.detach()), steps.detach()
        )
        self.A_diag = nn.Parameter(A_diag_init)

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _map_theta_to_A(self, thetas, G_diag, steps):
        """Map theta angles to A diagonal values."""
        cos_theta = torch.cos(thetas)
        tan_theta = torch.tan(thetas)
        tan_theta_sq = tan_theta**2

        sqrt_term = 4 * torch.sqrt(
            steps**4 * cos_theta ** (-2) + steps**5 * G_diag * cos_theta ** (-2)
        )

        common_term = steps**2 * (
            -4
            - 2 * steps * G_diag
            - 4 * tan_theta_sq
            - 2 * steps * G_diag * tan_theta_sq
        )

        denominator = 2 * steps**4 * (1 + tan_theta_sq)

        A_plus = (sqrt_term - common_term) / denominator
        A_minus = (-sqrt_term - common_term) / denominator

        A_diag = torch.where(thetas > math.pi / 2, A_plus, A_minus)

        return A_diag

    def _recurrence(self, A_diag, G_diag, B, input_sequence, step):
        """Compute the LxP output of Damped-LinOSS given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            G_diag          (float32):      diagonal damping matrix   (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        Identity = torch.ones_like(A_diag)
        S = Identity + step * G_diag
        M_11 = 1.0 / S
        M_12 = -step / S * A_diag
        M_21 = step / S
        M_22 = Identity - step**2 / S * A_diag

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        step_b = step.view(view_shape)
        S_b = S.view(view_shape)

        F1 = step_b * (1.0 / S_b) * Bu_elements
        F2 = (step_b**2) * (1.0 / S_b) * Bu_elements

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        G_diag = F.relu(self.G_diag)
        A_boundary_low = (
            2 + steps * G_diag - 2 * torch.sqrt(1 + steps * G_diag)
        ) / steps**2
        A_boundary_high = (
            2 + steps * G_diag + 2 * torch.sqrt(1 + steps * G_diag)
        ) / steps**2
        A_diag = (
            A_boundary_low
            + F.relu(self.A_diag - A_boundary_low)
            - F.relu(self.A_diag - A_boundary_high)
        )

        # ys is (..., L, P, 2)
        ys = self._recurrence(A_diag, G_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class LinOSSBlock(nn.Module):
    def __init__(
        self,
        layer_name,
        state_dim,
        hidden_dim,
        r_min,
        r_max,
        theta_max,
        drop_rate,
        use_triton=None,
    ):
        super().__init__()
        layer_map = {
            "IM": IMLayer,
            "IMEX": IMEXLayer,
            "Damped": DampedLayer,
        }
        if layer_name not in layer_map.keys():
            raise KeyError(f"Layer name {layer_name} not defined.")

        # BatchNorm1d *requires* (N, C, L)
        # Our C (Channels) is hidden_dim
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)

        self.layer = layer_map[layer_name](
            state_dim,
            hidden_dim,
            r_min,
            r_max,
            theta_max,
            use_triton=use_triton,
        )
        self.glu = GLU(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # x shape is (L, H) or (B, L, H)
        skip = x

        # Apply BatchNorm
        # We need (N, C, L) where C = hidden_dim
        x_t = x
        if x.dim() == 2:
            x_t = x_t.unsqueeze(0)
        x_norm = self.norm(x_t.permute(0, 2, 1)).permute(0, 2, 1)
        if x.dim() == 2:
            x_norm = x_norm.squeeze(0)
        x = x_norm

        x = self.layer(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.glu(x)
        x = self.drop(x)
        x = skip + x

        return x


class LinOSSBackboneBlock(nn.Module):
    """Discretax-style residual block around the unified LinOSS sequence mixer."""

    def __init__(
        self,
        hidden_dim,
        state_dim=64,
        num_heads=1,
        use_head_gating=False,
        use_head_output_projection=False,
        discretization: DISCRETIZATION = "IMEX",
        initialization: INITIALIZATION = "AG",
        damping=True,
        stability: STABILITY = "oscillatory",
        projection_eps=0.0,
        input_normalization=False,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi / 4,
        A_max=1.0,
        G_max=1.0,
        drop_rate=0.1,
        prenorm=True,
        use_bias=True,
        use_triton=None,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, device=device, dtype=dtype)
        self.sequence_mixer = LinOSSSequenceMixer(
            in_features=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            use_head_gating=use_head_gating,
            use_head_output_projection=use_head_output_projection,
            discretization=discretization,
            initialization=initialization,
            damping=damping,
            stability=stability,
            projection_eps=projection_eps,
            input_normalization=input_normalization,
            r_min=r_min,
            r_max=r_max,
            theta_max=theta_max,
            A_max=A_max,
            G_max=G_max,
            use_triton=use_triton,
            dtype=dtype,
            device=device,
        )
        self.channel_mixer = GLU(hidden_dim, hidden_dim, bias=use_bias)
        self.channel_mixer.to(device=device, dtype=dtype)
        self.drop = nn.Dropout(p=drop_rate)
        self.prenorm = prenorm

    def _norm_sequence(self, x):
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x.squeeze(0) if squeeze_batch else x

    def forward(self, x):
        skip = x
        if self.prenorm:
            x = self._norm_sequence(x)
        x = self.sequence_mixer(x)
        x = self.drop(F.gelu(x))
        x = self.channel_mixer(x)
        x = self.drop(x)
        x = skip + x
        if not self.prenorm:
            x = self._norm_sequence(x)
        return x


class LinOSSBackbone(nn.Module):
    """Minimal multi-block PyTorch LinOSS backbone matching the Discretax model surface."""

    def __init__(
        self,
        hidden_dim,
        num_blocks=4,
        state_dim=64,
        num_heads=1,
        use_head_gating=False,
        use_head_output_projection=False,
        discretization: DISCRETIZATION = "IMEX",
        initialization: INITIALIZATION = "AG",
        damping=True,
        stability: STABILITY = "oscillatory",
        projection_eps=0.0,
        input_normalization=False,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi / 4,
        A_max=1.0,
        G_max=1.0,
        drop_rate=0.1,
        prenorm=True,
        use_bias=True,
        use_triton=None,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if state_dim % num_heads != 0:
            raise ValueError(f"state_dim={state_dim} must be divisible by num_heads={num_heads}")

        self.blocks = nn.ModuleList(
            [
                LinOSSBackboneBlock(
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    num_heads=num_heads,
                    use_head_gating=use_head_gating,
                    use_head_output_projection=use_head_output_projection,
                    discretization=discretization,
                    initialization=initialization,
                    damping=damping,
                    stability=stability,
                    projection_eps=projection_eps,
                    input_normalization=input_normalization,
                    r_min=r_min,
                    r_max=r_max,
                    theta_max=theta_max,
                    A_max=A_max,
                    G_max=G_max,
                    drop_rate=drop_rate,
                    prenorm=prenorm,
                    use_bias=use_bias,
                    use_triton=use_triton,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LinOSS(nn.Module):
    def __init__(
        self,
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification=False,
        tanh_output=False,
        output_step=1,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi,
        drop_rate=0.05,
        use_triton=None,
        **kwargs,
    ):
        super().__init__()

        self.linear_encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LinOSSBlock(
                    layer_name,
                    state_dim,
                    hidden_dim,
                    r_min,
                    r_max,
                    theta_max,
                    drop_rate,
                    use_triton=use_triton,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_decoder = nn.Linear(hidden_dim, output_dim)

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def forward(self, x):
        # x is (L, H_in) or (B, L, H_in)
        x = self.linear_encoder(x)

        for block in self.blocks:
            x = block(x)

        if self.classification:
            x = torch.mean(x, dim=-2)
            x = self.linear_decoder(x)
            x = F.softmax(x, dim=-1)
        else:
            x = x[..., self.output_step - 1 :: self.output_step, :]
            x = self.linear_decoder(x)
            if self.tanh_output:
                x = torch.tanh(x)

        return x
