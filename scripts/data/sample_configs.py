import os
import json
import itertools
import numpy as np
from pathlib import Path
from scipy.stats import loguniform

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def sample_configs(
    input_dir,
    output_dir,
    model_names,
    dataset_names,
    learning_rates,
    model_dims,
    hidden_dims,
    num_blocks,
    num_samples=100,
):
    for mn in model_names:
        for dn in dataset_names:
            for i in range(num_samples):
                # I/O
                input_filename = input_dir / mn / dn / "config_000.json"
                os.makedirs(output_dir / mn / dn, exist_ok=True)
                output_filename = output_dir / mn / dn / f"config_{i:03}.json"

                # Read/write config
                with open(input_filename, "r") as in_file:
                    data = json.load(in_file)

                data["model_name"] = str(mn)
                data["dataset_name"] = str(dn)

                # OVERWRITTEN!
                data["seeds"] = [2345]

                data["lr"] = loguniform.rvs(min(learning_rates), max(learning_rates))
                data["hidden_dim"] = int(
                    loguniform.rvs(min(hidden_dims), max(hidden_dims))
                )
                data["model_dim"] = int(
                    loguniform.rvs(min(model_dims), max(model_dims))
                )
                data["num_blocks"] = round(
                    np.random.uniform(min(num_blocks), max(num_blocks))
                )

                with open(output_filename, "w") as out_file:
                    json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    # Input / output config files
    experiment_name = "random"
    input_dir = BASE_DIR / "config" / "human"
    output_dir = BASE_DIR / "config" / experiment_name

    # Script inputs
    num_samples = 64
    model_names = ["GRU"]
    dataset_names = ["Stirring"]
    learning_rates = [5e-3, 5e-5]
    model_dims = [32, 1024]
    hidden_dims = [32, 1024]
    num_blocks = [2, 6]

    # Write configuration files
    sample_configs(
        input_dir,
        output_dir,
        model_names,
        dataset_names,
        learning_rates,
        model_dims,
        hidden_dims,
        num_blocks,
        num_samples=num_samples,
    )
