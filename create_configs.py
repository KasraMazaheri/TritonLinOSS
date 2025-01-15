import os
import json
import itertools

from data_dir.project_dir import get_linoss_directory


def write_config(
    in_filename,
    out_filename,
    model_name,
    dataset_name,
    learning_rate,
    hidden_dim,
    state_dim,
    blocks,
    time,
    discretization,
    damping,
):
    with open(in_filename, 'r') as in_file:
        data = json.load(in_file)

    data['dataset_name'] = dataset_name
    data['model_name'] = model_name
    data['linoss_discretization'] = str(discretization)
    data['damping'] = bool(damping)
    data['lr'] = str(learning_rate)
    data['hidden_dim'] = str(hidden_dim)
    data['ssm_dim'] = str(state_dim)
    data['num_blocks'] = str(blocks)
    data['time'] = str(time)
    data['parameterization'] = str(parameterization)

    with open(out_filename, 'w') as out_file:
        json.dump(data, out_file, indent=4)


if __name__ == '__main__':
    # Input / output config files
    linoss_dir = get_linoss_directory()
    in_dir = linoss_dir / 'experiment_configs' / 'repeats' 
    out_dir = linoss_dir / 'experiment_configs' / 'grid'

    # Models & Datasets
    models = ["LinOSS"]
    datasets = ["ppg"]

    # Enumerate hyperparameter grid
    learning_rates = [1e-3, 1e-4, 1e-5]
    hidden_dims = [16, 64, 128]
    state_dims = [16, 64, 256]
    blocks = [2, 4, 6]
    time = [False, True]
    discretization = ['IMEX']
    damping = [True]
    parameterization = ['stable']

    # Write configuration files
    for dataset in datasets:
        for model in models:
            params = itertools.product(learning_rates, hidden_dims, state_dims, blocks, time, discretization, damping, parameterization)
            for i, combo in enumerate(params):
                in_filename = in_dir / model / (dataset + '.json')
                os.makedirs(out_dir / f'config_{i:03}' / model, exist_ok=True)
                out_filename = out_dir / f'config_{i:03}' / model / (dataset + '.json')
                write_config(
                    in_filename,
                    out_filename,
                    model,
                    dataset,
                    *combo,
                )