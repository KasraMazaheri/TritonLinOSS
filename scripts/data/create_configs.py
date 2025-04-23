import os
import json
import itertools
import numpy as np
from pathlib import Path

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def create_configs(
    input_dir,
    output_dir,
    model_names,
    dataset_names,
    learning_rates,
    hidden_dims,
    state_dims,
    blocks,
    time,
    discretization,
    damping,
    r_min,
    theta_max,
):
    for mn in model_names:
        for dn in dataset_names:

            combos = itertools.product(
                learning_rates,
                hidden_dims,
                state_dims,
                blocks,
                time,
                discretization,
                damping,
                r_min,
                theta_max,
            )

            for i, (lr, hd, sd, nb, t, dis, dam, r, theta) in enumerate(combos):
                # I/O
                input_filename = input_dir / mn / dn / "config_000.json"
                os.makedirs(output_dir / mn / dn, exist_ok=True)
                output_filename = output_dir / mn / dn / f"config_{i:03}.json"

                # Read/write config
                with open(input_filename, "r") as in_file:
                    data = json.load(in_file)

                data["model_name"] = str(mn)
                data["dataset_name"] = str(dn)
                data["lr"] = float(lr)
                data["hidden_dim"] = int(hd)
                data["ssm_dim"] = int(sd)
                data["num_blocks"] = int(nb)
                data["time"] = str(t)
                data["linoss_discretization"] = str(dis)
                data["damping"] = bool(dam)
                data["r_min"] = float(r)
                data["theta_max"] = float(theta)

                with open(output_filename, "w") as out_file:
                    json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    # Script inputs
    experiment_name = "mini"
    model_names = ["LinOSS"]
    dataset_names = ["IMDb"]
    learning_rates = [1e-3, 1e-5]
    hidden_dims = [64, 128]
    ssm_dims = [64, 128]
    num_blocks = [2, 6]
    include_time = [False]
    discretization = ["IMEX"]
    damping = [True]
    r_min = [0.0, 0.9]
    theta_max = [np.pi]

    # Input / output config files
    input_dir = BASE_DIR / "config" / "repeats"
    output_dir = BASE_DIR / "config" / experiment_name

    # Write configuration files
    create_configs(
        input_dir,
        output_dir,
        model_names,
        dataset_names,
        learning_rates,
        hidden_dims,
        ssm_dims,
        num_blocks,
        include_time,
        discretization,
        damping,
        r_min,
        theta_max,
    )
