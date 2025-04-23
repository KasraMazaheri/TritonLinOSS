"""
This script loads a pre-saved model in .eqx format and ,,,

Arguments for `analyze_parameters.py`:
`save_model_folder` (str): Relative directory to linoss/ storing model save.
"""

import os
import pickle
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import equinox as eqx
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt

from linoss.models.generate_model import create_model
from linoss.data_processing.create_dataset import create_dataset

BASE_DIR = Path(__file__).resolve().parent.parent


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def analyze_parameters(
    save_model_folder: str,
):
    """
    Loads a pre-saved model and ,,,

    Args:
        save_model_folder (str): Relative directory to linoss/ storing model save.
    """
    save_dir = BASE_DIR / save_model_folder
    params_dir = str(save_dir / "params")
    figures_dir = BASE_DIR / "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load hyperparameters and create empty model
    with open(save_dir / "hyperparameters.pkl", "rb") as f:
        hyperparameters = pickle.load(f)
    empty_model, empty_state = create_model(**hyperparameters)

    # Save model parameters, if requested
    empty_model.save_params(params_dir)

    # Load model
    loaded_model = eqx.tree_deserialise_leaves(save_dir / "model.eqx", empty_model)
    # loaded_state = eqx.tree_deserialise_leaves(save_dir / "state.eqx", empty_state)
    inference_model = eqx.tree_inference(loaded_model, True)  # Inference mode

    # Save model parameters, if requested
    inference_model.save_params(params_dir)

    fig, ax = plt.subplots(1, 1)

    block_dirs = glob.glob(params_dir + "/block_*")
    print(block_dirs)
    for i, block_dir in enumerate(block_dirs):
        print(f"Reporting block #{i}")
        a = np.load(f"{block_dir}/M.npy")
        val, vec = np.linalg.eig(a)
        print(
            "Average eigenvalue magnitude: ",
            np.mean(np.abs(val)),
            ", std dev: ",
            np.std(np.abs(val)),
        )
        print(
            "Average eigenvalue argument (abs): ",
            np.mean(np.abs(np.angle(val))),
            ", std dev: ",
            np.std(np.abs(np.angle(val))),
        )

        ax.scatter(val.real, val.imag)

        pairs = [
            [vec[:, i] for i in range(vec.shape[1]) if np.abs(vec[j, i]) > 1e-6]
            for j in range(vec.shape[0])
        ]
        cosines = [np.vdot(pair[0], pair[1]).real for pair in pairs if len(pair) == 2]
        print("Average eigenvector pair angle: ", np.mean(np.arccos(cosines)))

        steps = np.load(f"{block_dir}/steps.npy")
        print("Avg Delta t: ", np.mean(steps), ", Std dev: ", np.std(steps))

    ax.plot(
        np.cos(np.linspace(0, 2 * np.pi, num=100)),
        np.sin(np.linspace(0, 2 * np.pi, num=100)),
        color="black",
        linewidth=1,
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid()
    ax.set_aspect("equal")
    ax.set_title("Learned Eigenvalues for LinOSS-IM - SCP2")
    fig.savefig(figures_dir / "eigen.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_folder",
        type=str,
        required=True,
        help="path to model save folder, relative to base directory linoss/",
    )
    args = parser.parse_args()

    analyze_parameters(
        args.model_save_folder,
    )
