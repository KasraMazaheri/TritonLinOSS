"""
This script loads a pre-saved model in .eqx format and computes inference outputs for specified data.
Also provides the option to save model parameter matrices or hidden states during inference.

Arguments:
- save_dir (str): Directory to saved model
- output_dir (str): Directory to save inference outputs
- save_params (bool): (Optional, default=False) If True, saves model parameter matrices to save_dir
- save_states (bool): (Optional, default=False) If True, saves inference hidden states to save_dir
"""
import os
import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from typing import List

from data_dir.project_dir import get_linoss_directory
from models.generate_model import create_model
from data_dir.datasets import create_dataset


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename: str, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def inference(
    save_dir: str,
    output_dir: str,
    save_params: bool = False,
    save_states: bool = False,
):
    save_states_dir = save_dir + "/states/" if save_states else None
    os.makedirs(output_dir, exist_ok=True)

    # Load hyperparameters and create empty model
    with open(save_dir + "/hyperparameters.pkl", "rb") as f:
        hyperparameters = pickle.load(f)
    empty_model, empty_state = create_model(**hyperparameters)

    # Load model
    loaded_model = eqx.tree_deserialise_leaves(save_dir + "/model.eqx", empty_model)
    loaded_state = eqx.tree_deserialise_leaves(save_dir + "/state.eqx", empty_state)
    inference_model = eqx.tree_inference(loaded_model, True) # Inference mode

    # Save model parameters, if requested
    if save_params:
        inference_model.save_params(save_dir)

    # Load dataset arguments and create dataset
    with open(save_dir + "/dataset_args.pkl", "rb") as f:
        dataset_args = pickle.load(f)
    dataset = create_dataset(**dataset_args)

    # Run inference by split
    for split, dataloader in dataset.raw_dataloaders.items():
        print(f"Running inference on split {split}")
        inputs = jnp.array(dataloader.data)
        outputs = []
        for x in tqdm(inputs):
            output, _ = inference_model(x, loaded_state, hyperparameters['key'], save_dir=save_states_dir)
            outputs.append(output)
        outputs = jnp.stack(outputs)
        
        # Save inputs and outputs
        save_pickle(output_dir + f"/inputs_{split}.pkl", inputs)
        save_pickle(output_dir + f"/outputs_{split}.pkl", outputs)


if __name__ == "__main__":
    save_dir = get_linoss_directory() / "saves" / "lru" / "pouring" / "3456"
    output_dir = save_dir / "inference"

    inference(
        str(save_dir),
        str(output_dir),
        save_params=True,
        save_states=True,
    )