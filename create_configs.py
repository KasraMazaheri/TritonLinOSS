import os
import json
import itertools

from data_dir.project_dir import get_linoss_directory


def create_configs(
    in_dir,
    out_dir,
    model_name,
    dataset_name,
    params={},
):
    labels = params.keys()
    combos = itertools.product(params.values())
    for i, combo in enumerate(combos):
        # I/O
        in_filename = in_dir / model / (dataset + ".json")
        os.makedirs(out_dir / model / dataset , exist_ok=True)
        out_filename = out_dir / model / dataset / f"config_{i:03}.json"
        
        # Read/write config
        with open(in_filename, "r") as in_file:
            data = json.load(in_file)

        data["dataset_name"] = dataset_name
        data["model_name"] = model_name
        sample_data = {labels[i]: combo[i] for i in range(len(combo))}
        data = data | sample_data

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
    learning_rates = [1e-5, 1e-2]
    hidden_dims = [16, 64, 128]
    state_dims = [16, 64, 128, 256]
    blocks = [2, 4, 6]
    time = [False, True]
    discretization = ["IMEX"]
    damping = [True]
    parameterization = ["stable"]

    # Write configuration files
    for dataset in datasets:
        for model in models:
            create_configs(
                in_dir,
                out_dir,
                model,
                dataset,
                params={
                    "lr": learning_rates,
                    "hidden_dim": hidden_dims, 
                    "ssm_dim": state_dims, 
                    "num_blocks": blocks, 
                    "time": time, 
                    "linoss_discretization": discretization, 
                    "damping": damping,
                    "parameterization": parameterization,
                },
            )