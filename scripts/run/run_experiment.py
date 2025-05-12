"""
This script loads hyperparameters from config JSON files and trains models on
specified datasets using the `create_dataset_model_and_train` function from `train.py`.

The results and outputs are saved in linoss/ base directory.

Arguments for `run_experiments.py`:
- `model_names` (List[str]): List of models to train.
- `dataset_names` (List[str]): List of datasets to train.
- `config_folder` (str): Directory containing all configuration files,
                         relative to base directory linoss/.
- `task_id` (int, optional): Process ID -- for batching. Defaults to 0.
- `num_tasks` (int, optional): Number of processes -- for batching. Defaults to 1.
- `save_model` (bool, optional): If True, saves model .eqx files. Defaults to False.
"""

import os
import sys
import json
import diffrax
import glob
import pickle
import equinox as eqx
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from linoss.train import create_dataset_model_and_train


def parse_config(
    config_file: str,
    model_name: str,
    dataset_name: str,
) -> Tuple[List[int], Dict]:
    """
    Creates training arguments from a configuration file.

    Args:
        config_file (str): The path to the JSON configuration file.
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.

    Returns:
        List[int]: The list of run seeds.
        dict: The dictionary of run arguments.
    """
    with open(config_file, "r") as file:
        data = json.load(file)

    def safe_load(data, key, dtype=None):
        val = data.get(key, None)
        if val is not None and dtype is not None:
            val = dtype(val)
        return val

    # All arguments
    seeds = safe_load(data, "seeds")
    lr_scheduler = eval(data["lr_scheduler"])
    num_steps = safe_load(data, "num_steps")
    print_steps = safe_load(data, "print_steps")
    batch_size = safe_load(data, "batch_size", int)
    metric = safe_load(data, "metric")
    use_presplit = safe_load(data, "use_presplit", bool)
    lr = safe_load(data, "lr", float)
    include_time = safe_load(data, "include_time", bool)
    time_duration = safe_load(data, "time_duration", float)
    model_dim = safe_load(data, "model_dim", int)
    hidden_dim = safe_load(data, "hidden_dim", int)
    num_blocks = safe_load(data, "num_blocks", int)

    # Model-specific arguments
    # LinOSS type
    linoss_discretization = safe_load(data, "linoss_discretization", str)
    damping = safe_load(data, "damping", bool)
    # LinOSS + LRU initialization
    r_min = safe_load(data, "r_min", float)
    theta_max = safe_load(data, "theta_max", float)
    # S5 initialization
    ssm_blocks = safe_load(data, "ssm_blocks", int)
    # Transformer
    num_heads = safe_load(data, "num_heads", int)
    encoder_blocks = safe_load(data, "encoder_blocks", int)
    decoder_blocks = safe_load(data, "decoder_blocks", int)
    # RNNs
    stack_rnn = safe_load(data, "stack_rnn", bool)
    mlp_width = safe_load(data, "mlp_width", int)
    mlp_depth = safe_load(data, "mlp_depth", int)
    # All of the above
    linear_output = safe_load(data, "linear_output", bool)
    # Neural ODEs
    vf_depth = safe_load(data, "vf_depth", int)
    vf_width = safe_load(data, "vf_width", int)
    dt0 = safe_load(data, "dt0", float)
    scale = safe_load(data, "scale")
    lambd = safe_load(data, "lambd", float)
    logsig_depth = safe_load(data, "depth", int)
    stepsize = safe_load(data, "stepsize", int)
    solver = diffrax.Heun()
    controller = diffrax.ConstantStepSize()

    # Dataset-specific arguments
    if dataset_name == "ppg":
        output_step = safe_load(data, "output_step", int)
    else:
        output_step = 1

    # Paths
    data_dir = BASE_DIR / "data"
    output_parent_dir = BASE_DIR

    # Form model arguments
    model_args = {
        "num_blocks": num_blocks,
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
        "linear_output": linear_output,
        "linoss_discretization": linoss_discretization,
        "damping": damping,
        "r_min": r_min,
        "theta_max": theta_max,
        "ssm_blocks": ssm_blocks,
        "num_heads": num_heads,
        "encoder_blocks": encoder_blocks,
        "decoder_blocks": decoder_blocks,
        "stack_rnn": stack_rnn,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "vf_depth": vf_depth,
        "vf_width": vf_width,
        "dt0": dt0,
        "scale": scale,
        "lambd": lambd,
        "logsig_depth": logsig_depth,
        "solver": solver,
        "stepsize_controller": controller,
    }
    # Form run arguments
    run_args = {
        "data_dir": str(data_dir),
        "use_presplit": use_presplit,
        "dataset_name": dataset_name,
        "output_step": output_step,
        "metric": metric,
        "include_time": include_time,
        "time_duration": time_duration,
        "model_name": model_name,
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
        "num_steps": num_steps,
        "print_steps": print_steps,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "batch_size": batch_size,
        "output_parent_dir": str(output_parent_dir),
        "model_args": model_args,
    }

    return seeds, run_args


def run_experiments(
    model_names: List[str],
    dataset_names: List[str],
    config_folder: str,
    task_id: Optional[int] = 0,
    num_tasks: Optional[int] = 1,
    save_model: Optional[bool] = False,
) -> None:
    """
    Runs a series of training experiments.
    Iterates over config files found in
        BASE_DIR/{config_folder}/{model_name}/{dataset_name}.

    Args:
        model_names (List[str]): List of models to train.
        dataset_names (List[str]): List of datasets to train.
        config_folder (str): Relative directory within linoss/ containing config files.
        task_id (int, optional): Process ID -- for batching. Defaults to 0.
        num_tasks (int, optional): Number of processes -- for batching. Defaults to 1.
        save_model (bool, optional): If True, saves model .eqx files. Defaults to False.
    """
    # Test all models on all datasets
    for model_name in model_names:
        for dataset_name in dataset_names:

            # Batching
            run_config_folder = BASE_DIR / config_folder / model_name / dataset_name
            config_files = glob.glob(str(run_config_folder / "config_*"))
            config_files = config_files[task_id : len(config_files) : num_tasks]
            for i, config_file in enumerate(config_files):
                idx = int(config_file[-8:-5])

                # Load configuration
                print(f"Loading config {config_file}")
                seeds, run_args = parse_config(
                    str(config_file), model_name, dataset_name
                )

                # Test all seeds
                for seed in seeds:
                    print(f"Running experiment with seed: {seed}")
                    model, state, hyperparameters, dataset_args = (
                        create_dataset_model_and_train(seed=seed, idx=idx, **run_args)
                    )

                    # Save model
                    if save_model:
                        save_dir = (
                            BASE_DIR
                            / "saves"
                            / model_name
                            / dataset_name
                            / str(idx)
                            / str(seed)
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        with open(save_dir / "hyperparameters.pkl", "wb") as f:
                            pickle.dump(hyperparameters, f)
                        with open(save_dir / "dataset_args.pkl", "wb") as f:
                            pickle.dump(dataset_args, f)
                        eqx.tree_serialise_leaves(save_dir / "model.eqx", model)
                        eqx.tree_serialise_leaves(save_dir / "state.eqx", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="names of models to test",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="names of datasets to test",
    )
    parser.add_argument(
        "--config_folder",
        type=str,
        default="config/repeats",
        help="path to config folder, relative to base directory linoss/",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="batching: id number of process",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
        help="batching: total number of processes",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="whether or not to save the model",
    )
    args = parser.parse_args()

    run_experiments(
        args.model_names,
        args.dataset_names,
        args.config_folder,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        save_model=args.save_model,
    )
