"""
This module defines functions for creating datasets, building models,
and training them using JAX and Equinox.

The main function, `create_dataset_model_and_train`, is designed to initialize the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:
- `run_folder`: Absolute path to training run folder.
- `hyperparameters`: Dictionary of model hyperparameters.

The module also includes the following key functions:
- `calc_output`: Computes the model output, handling stateful and nondeterministic
                 models with JAX's `vmap` for batching.
- `classification_loss`: Computes the loss for classification tasks, including
                         optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional
                     regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters
               based on the computed gradients.
- `evaluate`: Computes the classification/non-classification metric given a 
              specific dataloader split.
- `train_model`: Handles the training loop, managing metrics, early stopping,
                 and saving progress at regular intervals.
"""
import os
import time
import warnings
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import equinox as eqx

from linoss.data.create_dataset import create_dataset
from linoss.models.create_model import create_model

# Ignore warning in loss fn calculation
warnings.simplefilter("ignore", category=jnp.ComplexWarning)


def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    bsz, _, _ = X.shape
    if stateful:
        if nondeterministic:
            keys = jr.split(key, bsz)
            output, state = jax.vmap(
                model,
                axis_name="batch",
                in_axes=(0, None, 0),
                out_axes=(0, None),
            )(X, state, keys)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        keys = jr.split(key, bsz)
        output = jax.vmap(model, in_axes=(0, 0))(X, keys)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)), state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean((jnp.squeeze(pred_y) - jnp.squeeze(y)) ** 2.0), state


@eqx.filter_jit
def make_step(model, X, y, loss_fn, state, opt, opt_state, key):
    (value, state), grads = loss_fn(model, X, y, state, key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


def evaluate(inference_model, state, dataloader_iter, key):
    predictions = []
    labels = []
    for data in dataloader_iter:
        eval_key, key = jr.split(key, 2)
        X, y = data
        prediction, _ = calc_output(
            inference_model,
            X,
            state,
            eval_key,
            inference_model.stateful,
            inference_model.nondeterministic,
        )
        predictions.append(prediction)
        labels.append(y)
    prediction = jnp.vstack(predictions)
    y = jnp.vstack(labels)

    if inference_model.classification:
        metric = jnp.mean(jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1))
    else:
        metric = jnp.mean((jnp.squeeze(prediction) - jnp.squeeze(y)) ** 2.0)
    
    return metric


def train_model(
    run_folder: str,
    model,
    state,
    dataset,
    metric: str,
    num_steps: int,
    print_steps: int,
    lr: float,
    batch_size: int,
    key: jax.Array,
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

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr)
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
        # Make step
        X, y = data
        stepkey, key = jr.split(key, 2)
        model, state, opt_state, value = make_step(model, X, y, loss_fn, state, opt, opt_state, stepkey)
        running_loss += value

        # Evaluation
        if (step + 1) % print_steps == 0:
            train_key, val_key, test_key, key = jr.split(key, 4)
            inference_model = eqx.tree_inference(model, value=True)
            train_iter = dataset.dataloaders["train"].loop_epoch(batch_size)
            val_iter = dataset.dataloaders["val"].loop_epoch(batch_size)
            train_metric = evaluate(inference_model, state, train_iter, train_key)
            val_metric = evaluate(inference_model, state, val_iter, val_key)

            # Print status
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train metric: {train_metric}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )
            start = time.time()
            running_loss = 0.0

            # Improvement checking / early stopping
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
                test_iter = dataset.dataloaders["test"].loop_epoch(batch_size)
                test_metric = evaluate(inference_model, state, test_iter, test_key)
                print(f"Test metric: {test_metric}")

            # Log results
            all_time.append(total_time)
            all_train_metric.append(train_metric)
            all_val_metric.append(val_metric)

            log_data = jnp.stack([
                jnp.arange(0, step + 2, print_steps)[1:],
                jnp.array(all_time),
                jnp.array(all_train_metric[1:]),
                jnp.array(all_val_metric[1:])
            ], axis=1)
            jnp.save(run_folder+ "/log_metrics.npy", log_data)

    # Log final results
    print(f"Test metric: {test_metric}")
    f = open(run_folder + "/results.txt", "a")
    f.write(str(test_metric) + "\n")
    f.close()

    return best_model, best_state


def create_dataset_model_and_train(
    run_folder: str,
    hyperparameters: dict,
):
    seed = safe_load(hyperparameters, "seed", int)
    model_name = safe_load(hyperparameters, "model_name", str)
    dataset_name = safe_load(hyperparameters, "dataset_name", str)
    dataset_key, model_key, train_key = jr.split(jr.PRNGKey(seed), 3)

    def delete_file_if_exists(file):
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted: {file}")

    delete_file_if_exists(run_folder + "/metadata.txt")
    delete_file_if_exists(run_folder + "/results.txt")
    delete_file_if_exists(run_folder + "/log_metrics.npy")
    delete_file_if_exists(run_folder + "/model.eqx")
    delete_file_if_exists(run_folder + "/state.eqx")

    f = open(run_folder + "/metadata.txt", "a")
    f.write(f'Time of execution: {datetime.now().strftime("%Y%m%d_%H%M%S")} \n')
    f.write("log_metrics.npy columns: [step, time, train metric, val metric] \n")
    f.close()

    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(
        name=dataset_name,
        data_dir=safe_load(hyperparameters, "data_dir", str),
        classification=safe_load(hyperparameters, "classification", bool),
        time_duration=safe_load(hyperparameters, "time_duration", float) if safe_load(hyperparameters, "include_time", bool) else None,
        use_presplit=safe_load(hyperparameters, "use_presplit", bool),
        key=dataset_key
    )

    print(f"Creating model {model_name}")
    hyperparameters |= {"input_dim": dataset.data_dim, "output_dim": dataset.label_dim}
    model, state = create_model(
        hyperparameters=hyperparameters,
        key=model_key,
    )

    model, state = train_model(
        run_folder,
        model,
        state,
        dataset,
        metric=safe_load(hyperparameters, "metric", str),
        num_steps=safe_load(hyperparameters, "num_steps", int),
        print_steps=safe_load(hyperparameters, "print_steps", int),
        lr=safe_load(hyperparameters, "lr", float),
        batch_size=safe_load(hyperparameters, "batch_size", int),
        key=train_key,
    )

    return model, state
