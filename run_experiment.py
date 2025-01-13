"""
This script loads hyperparameters from JSON files and trains models on specified datasets using
the `create_dataset_model_and_train` function from `train.py` or its PyTorch equivalent. The results
are saved in the output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and then calls the appropriate training function based on the
framework (PyTorch or JAX).

Arguments for `run_experiments`:
- `model_names`: List of model architectures to use.
- `dataset_names`: List of datasets to train on.
- `experiment_folder`: Directory containing JSON configuration files.
"""
import sys
import json
import diffrax
from train import create_dataset_model_and_train

from data_dir.project_dir import get_linoss_directory


def run_experiments(model_names, dataset_names, experiment_folder, task_id=0, save_params=False):

    for model_name in model_names:
        for dataset_name in dataset_names:
            config_file = experiment_folder + f"/{model_name}/{dataset_name}.json"
            with open(config_file, "r") as file:
                data = json.load(file)
            print(f"Loading config {config_file}")

            seeds = data["seeds"]
            data_dir = str(get_linoss_directory() / data["data_dir"])
            output_parent_dir = str(get_linoss_directory() / data["output_parent_dir"])
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]
            if model_name == "LinOSS":
                linoss_discretization = data["linoss_discretization"]
                damping = data["damping"]
            else:
                linoss_discretization = None
                damping = False
            use_presplit = data["use_presplit"]
            T = data["T"]
            if model_name in ["lru", "S5", "S6", "mamba", "LinOSS"]:
                dt0 = None
            else:
                dt0 = float(data["dt0"])
            scale = data["scale"]
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])
            if model_name in ["log_ncde", "nrde", "ncde"]:
                vf_depth = int(data["vf_depth"])
                vf_width = int(data["vf_width"])
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
                ssm_dim = None
                num_blocks = None
            else:
                vf_depth = None
                vf_width = None
                logsig_depth = 1
                stepsize = 1
                lambd = None
                ssm_dim = int(data["ssm_dim"])
                num_blocks = int(data["num_blocks"])
            if model_name in ["S5", "LinOSS"]:
                ssm_blocks = int(data["ssm_blocks"])
            else:
                ssm_blocks = None
            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            else:
                output_step = 1

            model_args = {
                "num_blocks": num_blocks,
                "hidden_dim": hidden_dim,
                "vf_depth": vf_depth,
                "vf_width": vf_width,
                "ssm_dim": ssm_dim,
                "ssm_blocks": ssm_blocks,
                "dt0": dt0,
                "solver": diffrax.Heun(),
                "stepsize_controller": diffrax.ConstantStepSize(),
                "scale": scale,
                "lambd": lambd,
            }
            run_args = {
                "data_dir": data_dir,
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
                "model_args": model_args,
                "num_steps": num_steps,
                "print_steps": print_steps,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "batch_size": batch_size,
                "output_parent_dir": output_parent_dir,
                "id": task_id,
                "save_params": save_params,
            }
            run_fn = create_dataset_model_and_train

            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


if __name__ == "__main__":
    task_id = int(sys.argv[1])

    model_names = ["LinOSS"]
    dataset_names = ["Heartbeat"]
    linoss_dir = get_linoss_directory()
    # experiment_folder = linoss_dir / "experiment_configs" / "batch" / f"base_config_{task_id:03}"
    experiment_folder = linoss_dir / "experiment_configs" / "repeats" / "LinOSS"

    run_experiments(model_names, dataset_names, str(experiment_folder), task_id=task_id, save_params=True)