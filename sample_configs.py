import os
import json
import itertools
import numpy.random as random
from scipy.stats import loguniform
import numpy as np

from data_dir.project_dir import get_linoss_directory


def sample_configs(
    base_config,
    out_dir,
    model_name,
    dataset_name,
    continuous_params={},
    discrete_params={},
    num_samples=10,
):
    for i in range(num_samples):
        # I/O
        os.makedirs(out_dir / model / dataset , exist_ok=True)
        out_filename = out_dir / model / dataset / f"config_{i:03}.json"

        # Sample configurations
        discrete_samples = {key: val[1](random.choice(val[0])) for key, val in discrete_params.items()}
        continuous_samples = {}
        for key, val in continuous_params.items():
            if key == "lr":
                sample = val[1](loguniform.rvs(*val[0]))
            elif key == "r_min":
                sample = val[1](np.sqrt(random.uniform() * (val[0][1] ** 2 - val[0][0] ** 2) + val[0][0] ** 2))
            else:
                sample = val[1](random.uniform(*val[0]))
            continuous_samples[key] = sample

        # Read/write config
        with open(base_config, "r") as in_file:
            data = json.load(in_file)

        data["dataset_name"] = dataset_name
        data["model_name"] = model_name
        data = data | discrete_samples | continuous_samples

        with open(out_filename, "w") as out_file:
            json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    # Input / output config files
    linoss_dir = get_linoss_directory()
    out_dir = linoss_dir / "experiment_configs" / "experiment"

    # Models & Datasets
    models = ["LinOSS"]
    datasets = ["synthetic_regression"]

    # Enumerate hyperparameter grid (vals, dtype)
    # Continuous
    learning_rates = ([1e-4, 1e-2], float)
    r_min = ([0.0, 1.0], float)
    # Discrete
    hidden_dims = ([16, 64, 128], int)
    state_dims = ([16, 64, 256], int)
    blocks = ([1], int)
    time = ([True, False], str)
    # Constant params
    discretization = (["IMEX"], str)
    damping = ([True], bool)

    # Number of configurations
    num_samples = 10

    # Write configuration files
    for dataset in datasets:
        for model in models:
            base_config = linoss_dir / "experiment_configs" / "repeats" / model / dataset / "config_000.json"
            sample_configs(
                base_config,
                out_dir,
                model,
                dataset,
                continuous_params={
                    "lr": learning_rates,
                    "r_min": r_min,
                }, 
                discrete_params={
                    "hidden_dim": hidden_dims, 
                    "ssm_dim": state_dims, 
                    "num_blocks": blocks, 
                    "time": time, 
                    "linoss_discretization": discretization, 
                    "damping": damping,
                },
                num_samples=num_samples,
            )