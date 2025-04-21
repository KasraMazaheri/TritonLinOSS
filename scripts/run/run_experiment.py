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

    # All arguments
    seeds = data["seeds"]
    lr_scheduler = eval(data["lr_scheduler"])
    num_steps = data["num_steps"]
    print_steps = data["print_steps"]
    batch_size = data["batch_size"]
    metric = data["metric"]
    use_presplit = data["use_presplit"]
    lr = float(data["lr"])
    include_time = data["time"].lower() == "true"
    hidden_dim = int(data["hidden_dim"])
    T = float(data["T"])

    # Model-specific arguments
    if model_name == "LinOSS":
        linoss_discretization = data["linoss_discretization"]
        damping = data["damping"]
        r_min = data.get("r_min", 0)
        theta_max = data.get("theta_max", 3.1415)
    else:
        linoss_discretization = None
        damping = False
        r_min = None
        theta_max = None

    if model_name == "S5":
        ssm_blocks = int(data["ssm_blocks"])
    else:
        ssm_blocks = None

    if model_name in [
        "lru",
        "S5",
        "LinOSS",
        "rnn_lstm",
        "rnn_gru",
        "rnn_mlp",
        "rnn_linear",
    ]:
        ssm_dim = int(data["ssm_dim"])
    else:
        ssm_dim = None

    if model_name in ["log_ncde", "nrde", "ncde"]:
        vf_depth = int(data["vf_depth"])
        vf_width = int(data["vf_width"])
        dt0 = float(data["dt0"])
        scale = data["scale"]
        num_blocks = None
        if model_name in ["log_ncde", "nrde"]:
            logsig_depth = int(data["depth"])
            stepsize = int(float(data["stepsize"]))
        else:
            logsig_depth = 1
            stepsize = 1
        if model_name == "log_ncde":
            lambd = float(data["lambd"])
        else:
            lambd = None
    else:
        vf_depth = None
        vf_width = None
        logsig_depth = 1
        stepsize = 1
        lambd = None
        scale = None
        dt0 = None
        num_blocks = int(data["num_blocks"])

    if model_name == "Transformer":
        num_heads = int(data["num_heads"])
        decoder_blocks = int(data["decoder_blocks"])
        encoder_only = bool(data["encoder_only"])
    else:
        num_heads = None
        decoder_blocks = None
        encoder_only = None

    # Dataset-specific arguments
    if dataset_name == "ppg":
        output_step = int(data["output_step"])
    else:
        output_step = 1

    # Paths
    data_dir = BASE_DIR / "data"
    output_parent_dir = BASE_DIR

    # Form model arguments
    model_args = {
        "num_blocks": num_blocks,
        "hidden_dim": hidden_dim,
        "vf_depth": vf_depth,
        "vf_width": vf_width,
        "ssm_dim": ssm_dim,
        "ssm_blocks": ssm_blocks,
        "decoder_blocks": decoder_blocks,
        "num_heads": num_heads,
        "encoder_only": encoder_only,
        "dt0": dt0,
        "solver": diffrax.Heun(),
        "stepsize_controller": diffrax.ConstantStepSize(),
        "scale": scale,
        "lambd": lambd,
    }
    # Form run arguments
    run_args = {
        "data_dir": str(data_dir),
        "use_presplit": use_presplit,
        "dataset_name": dataset_name,
        "output_step": output_step,
        "metric": metric,
        "include_time": include_time,
        "T": T,
        "model_name": model_name,
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
        "linoss_discretization": linoss_discretization,
        "damping": damping,
        "r_min": r_min,
        "theta_max": theta_max,
        "model_args": model_args,
        "num_steps": num_steps,
        "print_steps": print_steps,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "batch_size": batch_size,
        "output_parent_dir": str(output_parent_dir),
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
            num_configs = len(glob.glob(str(run_config_folder / "config_*")))
            idxs = range(num_configs)[task_id:num_configs:num_tasks]
            for idx in idxs:

                # Load configuration
                config_file = run_config_folder / f"config_{idx:03}.json"
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
                            BASE_DIR / "saves" / model_name / dataset_name / str(seed)
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
