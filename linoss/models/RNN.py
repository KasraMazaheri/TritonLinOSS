"""
This module implements the `RNN` class and various RNN cell classes using JAX and Equinox. The `RNN`
class is designed to handle both classification and regression tasks, and can be configured with different
types of RNN cells.

Attributes of the `RNN` class:
- `cell`: The RNN cell used within the RNN, which can be one of several types (e.g., `LinearCell`, `GRUCell`,
          `LSTMCell`, `MLPCell`).
- `output_layer`: The linear layer applied to the hidden state to produce the model's output.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

RNN Cell Classes:
- `_AbstractRNNCell`: An abstract base class for all RNN cells, defining the interface for custom RNN cells.
- `LinearCell`: A simple RNN cell that applies a linear transformation to the concatenated input and hidden state.
- `GRUCell`: An implementation of the Gated Recurrent Unit (GRU) cell.
- `LSTMCell`: An implementation of the Long Short-Term Memory (LSTM) cell.
- `MLPCell`: An RNN cell that applies a multi-layer perceptron (MLP) to the concatenated input and hidden state.

Each RNN cell class implements the following methods:
- `__init__`: Initialises the RNN cell with the specified input dimensions and hidden state size.
- `__call__`: Applies the RNN cell to the input and hidden state, returning the updated hidden state.

The `RNN` class also includes:
- A `__call__` method that processes a sequence of inputs, returning either the final output for classification or a
sequence of outputs for regression.
"""

import abc
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from linoss.models.LRU import GLU


class _AbstractRNNCell(eqx.Module):
    """Abstract RNN Cell class."""

    cell: eqx.Module

    @abc.abstractmethod
    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, input):
        raise NotImplementedError


class LinearCell(_AbstractRNNCell):
    cell: eqx.nn.Linear

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.Linear(data_dim + hidden_dim, hidden_dim, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))

        scan_fn = lambda state, input: (
            self.cell(jnp.concatenate([state, input])),
            self.cell(jnp.concatenate([state, input])),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states


class GRUCell(_AbstractRNNCell):
    cell: eqx.nn.GRUCell

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.GRUCell(data_dim, hidden_dim, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))

        scan_fn = lambda state, input: (
            self.cell(input, state),
            self.cell(input, state),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states


class LSTMCell(_AbstractRNNCell):
    cell: eqx.nn.LSTMCell

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.LSTMCell(data_dim, hidden_dim, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))
        hidden = (hidden,) * 2

        scan_fn = lambda state, input: (
            self.cell(input, state),
            self.cell(input, state),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states[0]  # LSTM specific


class MLPCell(_AbstractRNNCell):
    cell: eqx.nn.MLP

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.MLP(data_dim + hidden_dim, hidden_dim, width, depth, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))

        scan_fn = lambda state, input: (
            self.cell(jnp.concatenate([state, input])),
            self.cell(jnp.concatenate([state, input])),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states


class BasicRNN(eqx.Module):
    """
    Single-layer RNN with linear output transformation.
    Recurrent cell can be "linear", "lstm", "gru", or "mlp".
    """

    output_layer: eqx.nn.Linear
    cell: _AbstractRNNCell
    classification: bool
    linear_output: bool
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False
    output_step: int

    def __init__(
        self,
        model_name,
        data_dim,
        model_dim,
        label_dim,
        classification=True,
        output_step=1,
        linear_output=False,
        mlp_depth=None,
        mlp_width=None,
        *,
        key
    ):
        cell_key, output_key = jr.split(key, 2)
        cell_map = {
            "LSTM": LSTMCell,
            "GRU": GRUCell,
            "Linear_RNN": LinearCell,
            "MLP_RNN": MLPCell,
        }
        if model_name == "mlp" and (mlp_depth is None or mlp_width is None):
            raise ValueError("MLP type RNN requires specifying depth and width.")

        self.cell = cell_map[model_name](
            data_dim, model_dim, mlp_depth, mlp_width, key=cell_key
        )
        self.output_layer = eqx.nn.Linear(
            model_dim, label_dim, use_bias=False, key=output_key
        )
        self.classification = classification
        self.output_step = output_step
        self.linear_output = linear_output

    def __call__(self, x):
        x = self.cell(x)

        if self.classification:
            return jax.nn.softmax(self.output_layer(x[-1]), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.output_layer)(x)
            if not self.linear_output:
                x = jax.nn.tanh(x)

        return x


class RNNBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    cell: _AbstractRNNCell
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self, cell_type, rnn_dim, hidden_dim, vf_depth, vf_width, drop_rate=0.1, *, key
    ):
        cellkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False
        )
        self.cell = cell_type(hidden_dim, rnn_dim, vf_depth, vf_width, key=cellkey)
        self.glu = GLU(rnn_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.cell(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


class StackedRNN(eqx.Module):
    """
    Multi-layer deep RNN with GLU activation, skip connections, and dropout.
    Linear input/output operations.
    Recurrent cell can be "linear", "lstm", "gru", or "mlp".
    """

    input_layer: eqx.nn.Linear
    blocks: List[RNNBlock]
    output_layer: eqx.nn.Linear
    classification: bool
    linear_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        model_name,
        num_blocks,
        data_dim,
        rnn_dim,
        hidden_dim,
        label_dim,
        classification=True,
        output_step=1,
        linear_output=False,
        mlp_depth=None,
        mlp_width=None,
        *,
        key
    ):
        cell_map = {
            "LSTM": LSTMCell,
            "GRU": GRUCell,
            "Linear_RNN": LinearCell,
            "MLP_RNN": MLPCell,
        }
        cell_type = cell_map[model_name]

        input_layer_key, *block_keys, output_layer_key = jr.split(key, num_blocks + 2)
        self.input_layer = eqx.nn.Linear(data_dim, hidden_dim, key=input_layer_key)
        self.blocks = [
            RNNBlock(cell_type, rnn_dim, hidden_dim, mlp_depth, mlp_width, key=key)
            for key in block_keys
        ]
        self.output_layer = eqx.nn.Linear(
            hidden_dim, label_dim, use_bias=False, key=output_layer_key
        )
        self.classification = classification
        self.output_step = output_step
        self.linear_output = linear_output

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.input_layer)(x)

        for i, (block, key) in enumerate(zip(self.blocks, dropkeys)):
            x, state = block(x, state, key=key)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.output_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.output_layer)(x)
            if not self.linear_output:
                x = jax.nn.tanh(x)

        return x, state
