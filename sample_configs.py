import os
import json
import itertools
import numpy.random as random
from scipy.stats import loguniform

from data_dir.project_dir import get_linoss_directory


def sample_configs(
    in_dir,
    out_dir,
    model_name,
    dataset_name,
    continuous_params={},
    discrete_params={},
    num_samples=10,
):
    for i in range(num_samples):
        # I/O
        in_filename = in_dir / model / (dataset + ".json")
        os.makedirs(out_dir / model / dataset , exist_ok=True)
        out_filename = out_dir / model / dataset / f"config_{i:03}.json"

        # Sample configurations
        discrete_samples = {key: val[1](random.choice(val[0])) for key, val in discrete_params.items()}
        continuous_samples = {
            key: val[1](loguniform.rvs(*val[0])) if key == "lr" else val[1](random.uniform(*val[0])) 
            for key, val in continuous_params.items()
        }

        # Read/write config
        with open(in_filename, "r") as in_file:
            data = json.load(in_file)

        data["dataset_name"] = dataset_name
        data["model_name"] = model_name
        data = data | discrete_samples | continuous_samples

        with open(out_filename, "w") as out_file:
            json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    # Input / output config files
    linoss_dir = get_linoss_directory()
    in_dir = linoss_dir / "experiment_configs" / "repeats" 
    out_dir = linoss_dir / "experiment_configs" / "random"

    # Models & Datasets
    models = ["LinOSS"]
    datasets = ["ppg"]

    # Enumerate hyperparameter grid (vals, dtype)
    # Continuous
    learning_rates = ([1e-5, 1e-2], float)
    # Discrete
    hidden_dims = ([16, 64, 128], int)
    state_dims = ([16, 64, 128], int) # ssm_dim=256 memory error with PPG
    blocks = ([2, 4, 6], int)
    time = ([False, True], str)
    # Constant params
    discretization = (["IMEX"], str)
    damping = ([True], bool)
    parameterization = (["complex"], str)

    # Number of configurations
    num_samples = 10

    # Write configuration files
    for dataset in datasets:
        for model in models:
            sample_configs(
                in_dir,
                out_dir,
                model,
                dataset,
                continuous_params={
                    "lr": learning_rates,
                }, 
                discrete_params={
                    "hidden_dim": hidden_dims, 
                    "ssm_dim": state_dims, 
                    "num_blocks": blocks, 
                    "time": time, 
                    "linoss_discretization": discretization, 
                    "damping": damping,
                    "parameterization": parameterization,
                },
                num_samples=num_samples,
            )