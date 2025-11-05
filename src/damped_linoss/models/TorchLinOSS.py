import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from src.damped_linoss.parallel_scan.torch_interface import ParallelScanFunction


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes the Gated Linear Unit (GLU) module.
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        """
        Applies the GLU activation: w1(x) * sigmoid(w2(x))
        Handles (L, H) and (B, L, H) inputs.
        """
        return self.w1(x) * torch.sigmoid(self.w2(x))


class _AbstractLinOSSLayer(nn.Module):
    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError


class IMLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max):
        super().__init__()

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
            M_elements = M.unsqueeze(0).expand(B, -1) # (B, 4*P)
        else:
            M_elements = M # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1) # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1) # Shape for (B, L, P, 2)

        M_11_b = M_11.view(view_shape)
        M_21_b = M_21.view(view_shape)
        step_b = step.view(view_shape)

        F1 = M_11_b * Bu_elements * step_b
        F2 = M_21_b * Bu_elements * step_b
        
        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2) # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        _, xs = ParallelScanFunction.apply(M_elements, F)
        
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
    ):
        super().__init__()
        layer_map = {
            "IM": IMLayer,
            # "IMEX": IMEXLayer,
            # "Damped": DampedLayer,
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


class LinOSS(nn.Module):
    def __init__(
        self,
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification,
        tanh_output,
        output_step,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi,
        drop_rate=0.05,
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