"""
This script serves as a main entrypoint to training models, executing (optionally in batched fashion)
a series of training runs as specified by an experiment folder.

One training run within an experiment corresponds to one folder in your file system, containing a
`hyperparameters.yaml` file and storing all outputs, logs, and model checkpoints for that particular run.

Begin by generating a set of empty run folders containing the desired hyperparameter spread using
`create_experiment.py`. Then run this script with the --experiment_folder flag, which will iterate
through subfolders and execute corresponding training runs.

Experiment outputs can be postprocessed with `postprocess_results.py`.

Batching splits the experiment workload over $num_tasks$ different processes, assuming this script is
launched $num_tasks$ different times, independently, with varying values of $task_id$.
"""

import argparse
import os

import equinox as eqx
import yaml

from damped_linoss.train import create_dataset_model_and_train


def run_experiments(
    experiment_folder: str,
    task_id: int = 0,
    num_tasks: int = 1,
):
    """
    Runs a series of training experiments.
    Iterates over all run subfolders in run_folder/

    Args:
        experiment_folder (str): Absolute path to parent directory containing all run subfolders.
        task_id (int): Process ID -- for batching. Defaults to 0.
        num_tasks (int): Number of processes -- for batching. Defaults to 1.
    """
    # Load all run subfolders
    run_folders = sorted([f.name for f in os.scandir(experiment_folder) if f.is_dir()])

    # Batching
    num_runs = len(run_folders)
    idxs = range(num_runs)[task_id:num_runs:num_tasks]
    for idx in idxs:
        run_folder = experiment_folder + "/" + run_folders[idx]

        # Load hyperparameters
        with open(run_folder + "/hyperparameters.yaml", "r") as file:
            hyperparameters = yaml.safe_load(file)

        # Train model
        print(f"Running experiment {idx}")
        model, state = create_dataset_model_and_train(run_folder, hyperparameters)

        # Save model
        eqx.tree_serialise_leaves(run_folder + "/model.eqx", model)
        eqx.tree_serialise_leaves(run_folder + "/state.eqx", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Absolute path to parent directory containing all subfolders to be run.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="batching: id number of process",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
        help="batching: total number of processes",
    )
    args = parser.parse_args()

    run_experiments(
        args.experiment_folder,
        args.task_id,
        args.num_tasks,
    )
