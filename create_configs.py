import os
import json
import itertools
import numpy as np

from data_dir.project_dir import get_linoss_directory


def create_configs(
    in_dir,
    out_dir,
    model_name,
    dataset_name,
    learning_rates,
    hidden_dims,
    state_dims,
    blocks,
    time,
    discretization,
    damping,
    r_min,
):
    combos = itertools.product(
        learning_rates, 
        hidden_dims, 
        state_dims, 
        blocks, 
        time, 
        discretization, 
        damping,
        r_min,
    )

    for i, (lr, hd, sd, nb, t, dis, dam, r) in enumerate(combos):
        # I/O
        in_filename = in_dir / model / dataset / "config_000.json"
        os.makedirs(out_dir / model / dataset , exist_ok=True)
        out_filename = out_dir / model / dataset / f"config_{i:03}.json"
        
        # Read/write config
        with open(in_filename, "r") as in_file:
            data = json.load(in_file)

        data["dataset_name"] = dataset_name
        data["model_name"] = model_name
        data["lr"] = float(lr)
        data["hidden_dim"] = int(hd)
        data["ssm_dim"] = int(sd)
        data["num_blocks"] = int(nb)
        data["time"] = str(t)
        data["linoss_discretization"] = str(dis)
        data["damping"] = bool(dam)
        data["r_min"] = float(r)

        with open(out_filename, "w") as out_file:
            json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    # Input / output config files
    linoss_dir = get_linoss_directory()
    in_dir = linoss_dir / "experiment_configs" / "repeats" 
    out_dir = linoss_dir / "experiment_configs" / "grid"

    # Models & Datasets
    models = ["LinOSS"]
    datasets = ["ppg"]

    # Enumerate hyperparameter grid
    learning_rates = [1e-3, 1e-4, 1e-5]
    hidden_dims = [16, 64, 128]
    state_dims = [16, 64, 256]
    blocks = [2, 4, 6]
    time = [True, False]
    discretization = ["IMEX"]
    damping = [True]
    r_min = [0.9]

    # Write configuration files
    for dataset in datasets:
        for model in models:
            create_configs(
                in_dir,
                out_dir,
                model,
                dataset,
                learning_rates,
                hidden_dims,
                state_dims,
                blocks,
                time,
                discretization,
                damping,
                r_min,
            )