"""
Use this script to generate many run subfolders in an experiment directory.

It's not automated very intelligently, you will need to specify which parameters should be
iterated over and which ones are static, as well as filling in all other default hyperparameter values.

An example is provided for generating D-LinOSS hyperparameter sweeps, but note that additional hyperparameters
may be necessary for other models, and some of these hyperparameters are only relevant for LinOSS. To see a list
of the model-specific hyperparameters, look in `linoss/models/generate_model.py`.
"""
import os
import yaml
import itertools
import numpy as np


def create_grid_experiment(experiment_folder, model_name, dataset_name):
    # Hyperparameter sweep
    seed = [0, 1, 2, 3, 4]
    lr = [1e-3, 1e-4, 1e-5]
    state_dim = [16, 64, 256]
    hidden_dim = [16, 64, 128]
    num_blocks = [2, 4, 6]
    include_time = [False, True]

    combos = itertools.product(seed, lr, state_dim, hidden_dim, num_blocks, include_time)

    for i, (se, lr, sd, hd, nb, tm) in enumerate(combos):
        hyperparameters = {
            "seed": se,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "data_dir": "/lustre/home/jboyer/damped-linoss/data",
            "lr": lr,
            "num_steps": 100000,
            "print_steps": 1000,
            "batch_size": 16,
            "classification": True,
            "use_presplit": True,
            "include_time": tm,
            "time_duration": 1.0,
            "tanh_output": False,
            "output_step": 1,
            "layer_name": "Damped",
            "num_blocks": nb,
            "state_dim": sd,
            "hidden_dim": hd,
            "r_min": 0.9,
            "r_max": 1.0,
            "theta_max": np.pi,
            "drop_rate": 0.1
        }

        # Write config
        run_folder = experiment_folder + f"run_{i:03}/"
        os.makedirs(run_folder, exist_ok=True)
        with open(run_folder + "hyperparameters.yaml", "w") as file:
            hyperparameters = yaml.dump(hyperparameters, file)


def create_random_experiment(experiment_folder, model_name, dataset_name):
    # Hyperparameter sweep
    num_runs = 150
    learning_rate = [5e-3, 5e-5]
    state_dim = [16, 256]
    hidden_dim = [16, 256]
    num_blocks = [2, 6]
    include_time = [False, True]
    batch_size = [4, 64]
    A_max = [1.0, 10.0]
    G_max = [1.0, 10.0]
    # dt_std = [0.0, 1.0]
    # drop_rate = [0.0, 0.1]
    weight_decay = [0.0, 0.05]
    cosine_annealing = [False, True]

    for i in range(num_runs):
        se = int(np.random.randint(0, num_runs))
        lr = float(np.exp(np.random.uniform(np.log(learning_rate[0]), np.log(learning_rate[1]))))
        tm = bool(np.random.choice(include_time))
        nb = int(np.random.uniform(*num_blocks))
        sd = int(np.exp(np.random.uniform(np.log(state_dim[0]), np.log(state_dim[1]))))
        hd = int(np.exp(np.random.uniform(np.log(hidden_dim[0]), np.log(hidden_dim[1]))))
        bs = int(np.random.uniform(*batch_size))
        am = float(np.random.uniform(*A_max))
        gm = float(np.random.uniform(*G_max))
        # ds = float(np.random.uniform(*dt_std))
        # dr = float(np.random.uniform(*drop_rate))
        wd = float(np.random.uniform(*weight_decay))
        cs = bool(np.random.choice(cosine_annealing))

        hyperparameters = {
            "seed": se,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "data_dir": "/lustre/home/jboyer/damped-linoss/data",
            "classification": True,
            "use_presplit": True,
            "output_step": 1,
            "num_steps": 100000,
            "print_steps": 1000,
            "batch_size": bs,
            "lr": lr,
            "weight_decay": wd,
            "cosine_annealing": cs,
            "include_time": tm,
            "time_duration": 1.0,
            "tanh_output": False,
            "layer_name": "DampedIMEX2",
            "num_blocks": nb,
            "state_dim": sd,
            "hidden_dim": hd,
            "A_min": 0.0,
            "A_max": am,
            "G_min": 0.0,
            "G_max": gm,
            "dt_std": 0.5,
            "drop_rate": 0.1,
        }

        # Write config
        run_folder = experiment_folder + f"run_{i:03}/"
        os.makedirs(run_folder, exist_ok=True)
        with open(run_folder + "hyperparameters.yaml", "w") as file:
            hyperparameters = yaml.dump(hyperparameters, file)


if __name__ == "__main__":
    model_name = "LinOSS"
    dataset_name = "SequentialCifar10"
    experiment_folder = f"experiments/D-LinOSS-IMEX2/{dataset_name}/"

    # create_grid_experiment(experiment_folder, model_name, dataset_name)
    create_random_experiment(experiment_folder, model_name, dataset_name)
                                                                           