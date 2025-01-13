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
):
    with open(in_filename, 'r') as in_file:
        data = json.load(in_file)

    data['dataset_name'] = dataset_name
    if model_name == 'LinOSS_IM':
        data['model_name'] = 'LinOSS'
        data['linoss_discretization'] = 'IM'
        data['damping'] = False
    elif model_name == 'LinOSS_IMEX':
        data['model_name'] = 'LinOSS'
        data['linoss_discretization'] = 'IMEX'
        data['damping'] = False
    elif model_name == 'LinOSS_IMEX_damped':
        data['model_name'] = 'LinOSS'
        data['linoss_discretization'] = 'IMEX'
        data['damping'] = True
    else:
        data['model_name'] = model_name
    data['lr'] = str(learning_rate)
    data['hidden_dim'] = str(hidden_dim)
    data['ssm_dim'] = str(state_dim)
    data['num_blocks'] = str(blocks)
    data['time'] = str(time)

    with open(out_filename, 'w') as out_file:
        json.dump(data, out_file, indent=4)


if __name__ == '__main__':
    # Input / output config files
    linoss_dir = get_linoss_directory()
    in_dir = linoss_dir / 'experiment_configs' / 'repeats' 
    out_dir = linoss_dir / 'experiment_configs' / 'batch'

    # Models & Datasets
    models = ["LinOSS_IMEX_damped", "LinOSS_IMEX", "S5", "lru"]
    datasets = ["EigenWorms", "SelfRegulationSCP1", "SelfRegulationSCP2", "EthanolConcentration", "Heartbeat", "MotorImagery"]

    # Enumerate hyperparameter grid
    learning_rates = [1e-3, 1e-4, 1e-5]
    hidden_dims = [16, 64, 128]
    state_dims = [16, 64, 256]
    blocks = [2, 4, 6]
    time = [False, True]

    # Write configuration files
    for dataset in datasets:
        for model in models:
            params = itertools.product(learning_rates, hidden_dims, state_dims, blocks, time)
            for i, combo in enumerate(params):
                os.makedirs(out_dir / f'base_config_{i:03}' / model, exist_ok=True)
                in_filename = in_dir / 'LinOSS' / (dataset + '.json')
                out_filename = out_dir / f'base_config_{i:03}' / model / (dataset + '.json')
                write_config(
                    in_filename,
                    out_filename,
                    model,
                    dataset,
                    *combo,
                )