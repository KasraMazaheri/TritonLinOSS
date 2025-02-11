"""
This script loads a pre-saved model in .eqx format and computes inference outputs
for specified data. Also provides the option to save model parameter matrices
or hidden states during inference.

Arguments for `run_inference.py`:
`save_model_folder` (str): Relative directory to linoss/ storing model save.
`save_parameters` (bool, optional): If True, saves internal model parameters.
                                    Defaults to False.
`save_states` (bool, optional): If True, saves ssm states during inference
                                (this is data-intensive). Defaults to False.
"""

import os
import pickle
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Optional

from linoss.models.generate_model import create_model
from linoss.data.datasets import create_dataset

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def run_inference(
    save_model_folder: str,
    save_parameters: Optional[bool] = False,
    save_states: Optional[bool] = False,
):
    """
    Loads a pre-saved model and computes inference outputs on entire dataset.

    Args:
        save_model_folder (str): Relative directory to linoss/ storing model save.
        save_parameters (bool, optional): If True, saves internal model parameters.
                                          Defaults to False.
        save_states (bool, optional): If True, saves ssm states during inference
                                      (this is data-intensive). Defaults to False.
    """
    save_dir = BASE_DIR / save_model_folder
    output_dir = save_dir / "inference"
    os.makedirs(output_dir, exist_ok=True)
    params_dir = save_dir / "params" if save_parameters else None
    states_dir = save_dir / "states" if save_states else None

    # Load hyperparameters and create empty model
    with open(save_dir / "hyperparameters.pkl", "rb") as f:
        hyperparameters = pickle.load(f)
    empty_model, empty_state = create_model(**hyperparameters)

    # Load model
    loaded_model = eqx.tree_deserialise_leaves(save_dir / "model.eqx", empty_model)
    loaded_state = eqx.tree_deserialise_leaves(save_dir / "state.eqx", empty_state)
    inference_model = eqx.tree_inference(loaded_model, True)  # Inference mode

    # Save model parameters, if requested
    if save_parameters:
        inference_model.save_params(str(params_dir))

    # Load dataset arguments and create dataset
    with open(save_dir / "dataset_args.pkl", "rb") as f:
        dataset_args = pickle.load(f)
    dataset = create_dataset(**dataset_args)

    # Run inference by split
    for split, dataloader in dataset.raw_dataloaders.items():
        print(f"Running inference on split {split}")
        inputs = jnp.array(dataloader.data)
        outputs = []
        for x in tqdm(inputs):
            output, _ = inference_model(
                x, loaded_state, hyperparameters["key"], save_dir=str(states_dir)
            )
            outputs.append(output)
        outputs = jnp.stack(outputs)

        # Save inputs and outputs
        print(f"Saving inference inputs and outputs to {output_dir}")
        save_pickle(output_dir / f"inputs_{split}.pkl", inputs)
        save_pickle(output_dir / f"outputs_{split}.pkl", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_folder",
        type=str,
        required=True,
        help="path to model save folder",
    )
    parser.add_argument(
        "--save_parameters",
        action="store_true",
        help="whether or not to save model parameters",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="whether or not to save internal ssm states during inference",
    )
    args = parser.parse_args()

    run_inference(
        args.model_save_folder,
        save_parameters=args.save_parameters,
        save_states=args.save_states,
    )
