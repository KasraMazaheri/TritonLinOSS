"""
This module provides a function to generate a model based on a model name and
hyperparameters. It supports various types of models, including Neural CDEs,
RNNs, and the S5 model.

Function:
- `create_model`: Generates and returns a model instance along with its state
                  (if applicable) based on the provided model name and hyperparameters.

Parameters for `create_model`:
- `model_name`: A string specifying the model architecture to create.
                Supported values include 'log_ncde', 'ncde', 'nrde', 'lru',
                'LinOSS', 'S5', 'rnn_linear', 'rnn_gru', 'rnn_lstm', and 'rnn_mlp'.
- `data_dim`: The input data dimension.
- `logsig_dim`: The dimension of the log-signature used in NRDE and Log-NCDE models.
- `logsig_depth`: The depth of the log-signature used in NRDE and Log-NCDE models.
- `intervals`: The intervals used in NRDE and Log-NCDE models.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in models like LRU or S5.
- `vf_depth`: The depth of the vector field network for CDE models.
- `vf_width`: The width of the vector field network for CDE models.
- `classification`: A boolean indicating whether the task is classification (True) or
                    regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `model_dim`: The state-space model dimension for SSMs and RNNs or the model dimension
                for transformers.
- `ssm_blocks`: The number of SSM blocks in S5 models.
- `solver`: The ODE solver used in CDE models, with a default of `diffrax.Heun()`.
- `stepsize_controller`: The step size controller used in CDE models.
                         Defaults to `diffrax.ConstantStepSize()`.
- `dt0`: The initial time step for the solver.
- `max_steps`: The maximum number of steps for the solver.
- `scale`: A scaling factor applied to the vf initialisation in CDE models.
- `lambd`: A regularisation parameter used in Log-NCDE models.
- `linoss_discretization` (str, optional): Discretization type for LinOSS models.
                                           "IM" or "IMEX". Defaults to "IM".
- `damping` (str, optional): If True, uses damped model version for LinOSS.
                             Defaults to False.
- `r_min` (float, optional): Minimum eigenvalue magnitude for LinOSS initialization.
                             Defaults to 0.9
- `theta_max` (float, optional): Maximum eigenvalue phase for LinOSS initialization.
                                 Defaults to 3.141.
- `key`: A JAX PRNG key for random number generation.

Returns:
- A tuple containing the created model and its state (if applicable).

Raises:
- `ValueError`: If required hyperparameters for the specified model are not provided or
                if an unknown model name is passed.
"""

import diffrax
import equinox as eqx
import jax.random as jr

from linoss.models.LogNeuralCDEs import LogNeuralCDE
from linoss.models.LRU import LRU
from linoss.models.NeuralCDEs import NeuralCDE, NeuralRDE
from linoss.models.RNN import BasicRNN, StackedRNN
from linoss.models.S5 import S5
from linoss.models.LinOSS import LinOSS
from linoss.models.Transformer import Transformer


def create_model(
    model_name,
    data_dim,
    label_dim,
    hidden_dim,
    logsig_depth=None,
    logsig_dim=None,
    intervals=None,
    num_blocks=None,
    vf_depth=None,
    vf_width=None,
    classification=True,
    linear_output=False,
    output_step=1,
    use_last_output=False,
    model_dim=None,
    ssm_blocks=None,
    solver=diffrax.Heun(),
    stepsize_controller=diffrax.ConstantStepSize(),
    dt0=1,
    max_steps=16**4,
    scale=1.0,
    lambd=0.0,
    linoss_discretization="IM",
    damping=False,
    r_min=0.9,
    theta_max=3.141,
    encoder_blocks=2,
    decoder_blocks=2,
    num_heads=4,
    stack_rnn=False,
    mlp_depth=None,
    mlp_width=None,
    *,
    key,
):
    if model_name == "log_ncde":
        if logsig_depth is None or intervals is None:
            raise ValueError("Must specify logsig_depth and intervals for a Log-NCDE.")
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a Log-NCDE.")
        return (
            LogNeuralCDE(
                vf_width,
                vf_depth,
                model_dim,
                data_dim,
                logsig_depth,
                label_dim,
                classification,
                output_step,
                intervals,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                lambd,
                key=key,
            ),
            None,
        )
    if model_name == "ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NCDE.")
        return (
            NeuralCDE(
                vf_width,
                vf_depth,
                model_dim,
                data_dim,
                label_dim,
                classification,
                output_step,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                key=key,
            ),
            None,
        )
    elif model_name == "nrde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NRDE.")
        return (
            NeuralRDE(
                vf_width,
                vf_depth,
                model_dim,
                data_dim,
                logsig_dim,
                label_dim,
                classification,
                output_step,
                intervals,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                key=key,
            ),
            None,
        )
    elif model_name == "LRU":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for LRU.")
        lru = LRU(
            num_blocks,
            data_dim,
            model_dim,
            hidden_dim,
            label_dim,
            classification,
            linear_output,
            output_step,
            r_min=r_min,
            key=key,
        )
        state = eqx.nn.State(lru)
        return lru, state
    elif model_name == "S5":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for S5.")
        if model_dim is None:
            raise ValueError("Must specify model_dim for S5.")
        if ssm_blocks is None:
            raise ValueError("Must specify ssm_blocks for S5.")
        ssm = S5(
            num_blocks,
            data_dim,
            model_dim,
            ssm_blocks,
            hidden_dim,
            label_dim,
            classification,
            linear_output,
            output_step,
            "lecun_normal",
            False,
            True,
            "zoh",
            0.001,
            0.1,
            1.0,
            key=key,
        )
        state = eqx.nn.State(ssm)
        return ssm, state
    elif model_name == "LinOSS":
        ssm = LinOSS(
            num_blocks,
            data_dim,
            model_dim,
            hidden_dim,
            label_dim,
            classification,
            linear_output,
            output_step,
            use_last_output,
            linoss_discretization,
            damping,
            r_min,
            theta_max,
            key=key,
        )
        state = eqx.nn.State(ssm)
        return ssm, state
    elif model_name in ["LSTM", "GRU", "MLP_RNN", "Linear_RNN"]:
        if stack_rnn:
            rnn = StackedRNN(
                model_name,
                num_blocks,
                data_dim,
                model_dim,
                hidden_dim,
                label_dim,
                classification,
                output_step,
                linear_output,
                mlp_depth=mlp_depth,
                mlp_width=mlp_width,
                key=key,
            )
            state = eqx.nn.State(rnn)
        else:
            rnn = BasicRNN(
                model_name,
                data_dim,
                model_dim,
                label_dim,
                classification,
                output_step,
                linear_output,
                mlp_depth=mlp_depth,
                mlp_width=mlp_width,
                key=key,
            )
            state = None
        return rnn, state
    elif model_name == "Transformer":
        transformer = Transformer(
            encoder_blocks,
            decoder_blocks,
            data_dim,
            model_dim,
            label_dim,
            num_heads,
            classification,
            linear_output,
            key=key,
        )
        return transformer, None
    else:
        raise ValueError(f"Unknown model name: {model_name}")
