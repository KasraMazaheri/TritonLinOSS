"""
This module defines functions for creating datasets, building models,
and training them using JAX and Equinox.
The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `idx`: The run identification number
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting
                 a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'`
            for regression and `'accuracy'` for classification.
- `include_time`: A boolean indicating whether to include time as a channel
                  in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented
                  for depths 1 and 2.
- `linoss_discretization`: The discretization method (LinOSS).
                           Currently implemented for 'IM' and 'IMEX'.
- `r_min`: The minimum eigenvalue magnitude for initialization sampling (LinOSS).
- `theta_max`: The maximum eigenvalue phase for initialization sampling (LinOSS).
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function.
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.

The module also includes the following key functions:

- `calc_output`: Computes the model output, handling stateful and nondeterministic
                 models with JAX's `vmap` for batching.
- `classification_loss`: Computes the loss for classification tasks, including
                         optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional
                     regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters
               based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping,
                 and saving progress at regular intervals.
"""

import os
import shutil
import time
import warnings
from datetime import datetime

import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from linoss.data_processing.dataset import (
    StandardDataset,
    BucketedDataset,
    CoeffDataset,
    PathDataset,
)
from linoss.data_processing.create_dataset import create_dataset
from linoss.models.generate_model import create_model

# Ignore warning in loss fn calculation
warnings.simplefilter("ignore", category=jnp.ComplexWarning)

@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic, Y=None):
    if Y is not None:
        if stateful or nondeterministic:
            raise NotImplementedError
        output = jax.vmap(model)(X, Y)
        return output, state
    else:
        if stateful:
            if nondeterministic:
                output, state = jax.vmap(
                    model,
                    axis_name="batch",
                    in_axes=(0, None, None),
                    out_axes=(0, None),
                )(X, state, key)
            else:
                output, state = jax.vmap(
                    model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
                )(X, state)
        elif nondeterministic:
            output = jax.vmap(model, in_axes=(0, None))(X, key)
        else:
            output = jax.vmap(model)(X)

        return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(
    diff_model, static_model, X, y, state, key, is_transformer: bool
):
    if is_transformer:
        raise NotImplementedError

    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    norm = 0
    if model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key, is_transformer: bool):
    model = eqx.combine(diff_model, static_model)
    if is_transformer:
        pred_y, state = calc_output(
            model, X, state, key, model.stateful, model.nondeterministic, Y=y
        )
    else:
        pred_y, state = calc_output(
            model, X, state, key, model.stateful, model.nondeterministic
        )
    pred_y = jnp.squeeze(pred_y)

    norm = 0
    if model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd

    return (
        jnp.mean(jnp.mean((pred_y - y) ** 2.0, axis=1)) + norm,
        state,
    )


@eqx.filter_jit
def make_step(
    model,
    filter_spec,
    X,
    y,
    loss_fn,
    state,
    opt,
    opt_state,
    key,
    is_transformer: bool,
):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(
        diff_model, static_model, X, y, state, key, is_transformer
    )
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


def train_model(
    model_name,
    model,
    metric,
    filter_spec,
    state,
    dataset,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    key,
    results_dir,
    output_dir,
    idx,
):
    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if model_name == "Transformer":
        is_transformer = True
    else:
        is_transformer = False

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
        test_metric = jnp.nan
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
        test_metric = jnp.nan
    no_val_improvement = 0
    all_time = []
    start = time.time()

    best_model = jtu.tree_map(lambda x: x, model)
    best_state = jtu.tree_map(lambda x: x, state)

    for step, data in zip(
        range(num_steps),
        dataset.dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        model, state, opt_state, value = make_step(
            model,
            filter_spec,
            X,
            y,
            loss_fn,
            state,
            opt,
            opt_state,
            stepkey,
            is_transformer,
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            for i, data in enumerate(
                dataset.dataloaders["train"].loop_epoch(batch_size)
            ):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                    y if is_transformer else None,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                train_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = jnp.squeeze(prediction)
                train_metric = jnp.mean((prediction - y) ** 2)
            predictions = []
            labels = []
            for data in dataset.dataloaders["val"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                    y if is_transformer else None,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                val_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = jnp.squeeze(prediction)
                val_metric = jnp.mean((prediction - y) ** 2)
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train metric: {train_metric}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )
            start = time.time()
            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        break
                else:
                    no_val_improvement = 0
                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    best_model = jtu.tree_map(lambda x: x, model)
                    best_state = jtu.tree_map(lambda x: x, state)
                    val_metric_for_best_model.append(val_metric)
                    predictions = []
                    labels = []
                    for data in dataset.dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                            y if is_transformer else None,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                    else:
                        prediction = jnp.squeeze(prediction)
                        test_metric = jnp.mean((prediction - y) ** 2)
                    print(f"Test metric: {test_metric}")
                running_loss = 0.0
                all_train_metric.append(train_metric)
                all_val_metric.append(val_metric)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_metric_save = jnp.array(all_train_metric)
                all_val_metric_save = jnp.array(all_val_metric)
                all_time_save = jnp.array(all_time)
                test_metric_save = jnp.array(test_metric)
                jnp.save(output_dir + "/steps.npy", steps)
                jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
                jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
                jnp.save(output_dir + "/all_time.npy", all_time_save)
                jnp.save(output_dir + "/test_metric.npy", test_metric_save)

    print(f"Test metric: {test_metric}")
    os.makedirs(results_dir, exist_ok=True)
    f = open(results_dir + "/id_" + str(idx) + ".txt", "a")
    f.write(str(test_metric * 100.0) + "\n")
    f.close()

    return best_model, best_state


def create_dataset_model_and_train(
    seed,
    idx,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    time_duration,
    model_name,
    stepsize,
    logsig_depth,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir,
    model_args,
):
    key = jr.PRNGKey(seed)
    datasetkey, modelkey, trainkey, key = jr.split(key, 4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_str = (
        "/outputs/"
        + f"{model_name}/"
        + f"{dataset_name}/"
        + f"id_{idx}_seed_{seed}_timestamp_{timestamp}"
    )
    output_dir = output_parent_dir + output_str
    results_dir = output_parent_dir + "/results/" + model_name + "/" + dataset_name

    # Delete if it already exists
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been deleted and recreated.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    # Hardcoded properties
    if model_name in [
        "LinOSS",
        "LRU",
        "S5",
        "LSTM",
        "GRU",
        "MLP_RNN",
        "Linear_RNN",
        "Transformer",
    ]:
        # Variable sequence length
        if dataset_name in ["IMDb"]:
            dataset_type = BucketedDataset
        else:
            dataset_type = StandardDataset
    elif model_name in ["ncde"]:
        dataset_type = CoeffDataset
    elif model_name in ["nrde", "log_ncde"]:
        dataset_type = PathDataset
    else:
        raise ValueError(f"Model name {model_name} not implemented")

    dataset_args = {
        "name": dataset_name,
        "data_dir": data_dir,
        "dataset_type": dataset_type,
        "task_type": "classification" if metric == "accuracy" else "regression",
        "time_duration": time_duration if include_time else None,
        "use_presplit": use_presplit,
        "stepsize": stepsize,
        "depth": logsig_depth,
        "key": datasetkey,
    }
    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(**dataset_args)

    print(f"Creating model {model_name}")
    hyperparameters = {
        "model_name": model_name,
        "data_dim": dataset.data_dim,
        "label_dim": dataset.label_dim,
        "logsig_dim": dataset.logsig_dim if hasattr(dataset, "logsig_dim") else None,
        "intervals": dataset.intervals if hasattr(dataset, "intervals") else None,
        "classification": metric == "accuracy",
        "output_step": output_step,
        **model_args,
        "key": modelkey,
    }
    model, state = create_model(**hyperparameters)

    # n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    # print(f"Total number of parameters: {n_params}")

    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    if model_name == "nrde" or model_name == "log_ncde":
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(
                where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
            )
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))

    model, state = train_model(
        model_name,
        model,
        metric,
        filter_spec,
        state,
        dataset,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        results_dir,
        output_dir,
        idx,
    )

    return model, state, hyperparameters, dataset_args
