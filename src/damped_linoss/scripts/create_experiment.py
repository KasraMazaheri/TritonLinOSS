"""
Use this script to generate many run subfolders in an experiment directory.

It's not automated very intelligently, you will need to specify which parameters should be
iterated over and which ones are static, as well as filling in all other default hyperparameter values.

An example is provided for generating D-LinOSS hyperparameter sweeps, but note that additional hyperparameters
may be necessary for other models, and some of these hyperparameters are only relevant for LinOSS. To see a list
of the model-specific hyperparameters, look in `linoss/models/generate_model.py`.
"""

import itertools
import os
from pathlib import Path

import numpy as np
import yaml

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    experiment_folder = str(BASE_DIR) + "/experiments/D-LinOSS/PPG/"
    model_name = "LinOSS"
    dataset_name = "PPG"

    # Hyperparameter sweep
    seed = [0, 1, 2, 3, 4]
    lr = [1e-3, 1e-4, 1e-5]
    state_dim = [16, 64, 256]
    hidden_dim = [16, 64, 128]
    num_blocks = [2, 4, 6]
    include_time = [False, True]

    combos = itertools.product(
        seed, lr, state_dim, hidden_dim, num_blocks, include_time
    )

    for i, (se, lr, sd, hd, nb, tm) in enumerate(combos):
        hyperparameters = {
            "seed": se,
            "model_name": "LinOSS",
            "dataset_name": "PPG",
            "data_dir": "/lustre/home/jboyer/linoss/data",
            "lr": lr,
            "num_steps": 100000,
            "print_steps": 1000,
            "batch_size": 4,
            "classification": False,
            "metric": "mse",
            "use_presplit": True,
            "include_time": tm,
            "time_duration": 1.0,
            "tanh_output": False,
            "output_step": 128,
            "layer_name": "Damped",
            "num_blocks": nb,
            "state_dim": sd,
            "hidden_dim": hd,
            "r_min": 0.9,
            "r_max": 1.0,
            "theta_max": np.pi / 2,
            "drop_rate": 0.05,
        }

        # Write config
        run_folder = experiment_folder + f"run_{i:03}/"
        os.makedirs(run_folder, exist_ok=True)
        with open(run_folder + "hyperparameters.yaml", "w") as file:
            hyperparameters = yaml.dump(hyperparameters, file)
