"""
This module implements the `DatasetLoader` class for interfacing with raw data
and creating instances of `Dataset`.

It supports multiple different subclasses, each defined for a different dataset.

1. `UEALoader`: Loads datasets belonging to the UEA multivariate time series
    classification benchmarks.
2. `PPGLoader`: Loads the PPG-DaLiA dataset.
3. `SE3Loader`: Loads (experimental, private) datasets on learning trajectories
    expressed in 3D translation / rotations.
4. `ToyLoader`: Loads the toy dataset from Log-NCDEs.
5. `SyntheticLoader`: Loads the synthetic dataset from in D-LinOSS.
6. `Cifar10Loader`: Loads the Cifar10 dataset per Long Range Arena benchmark usage.
7. `IMDbLoader`: Loads the IMDb dataset per Long Range Arena benchmark usage.

To include support for a new dataset, create a new subclass of `DatasetLoader`
and, at the minimum, implement the private method `_load_and_process_data` to
load data and labels from file and return a tuple of split data. This split
can be aribtrary if not using the `use_presplit` flag. Then adjust the dictionary
in `create_dataset`.

Additionally, the function `create_dataset` serves as an entrypoint to dataset
loading functionality.
"""

import os
from typing import Tuple, Literal, Union
from pathlib import Path
from abc import ABC, abstractmethod
from collections import Counter
import string

import pickle
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.nn
import torchvision
import tensorflow as tf

from linoss.data_processing.dataset import (
    StandardDataset,
    BucketedDataset,
    CoeffDataset,
    PathDataset,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# =============================================
# SECTION: Utility functions
# =============================================


def get_subfolders(folder):
    if os.path.exists(folder):
        return [f.name for f in os.scandir(folder) if f.is_dir()]
    return []


def split(data, bounds: list):
    assert all([b < 1 for b in bounds])
    n = len(data)
    bounds = [0] + [int(n * b) for b in bounds] + [n]
    split_data = [data[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return tuple(split_data)


# =============================================
# SECTION: DatasetLoader Base Definition
# =============================================


class DatasetLoader(ABC):
    """
    Base class for loading/processing datasets
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        dataset_type: Union[
            type[StandardDataset],
            type[BucketedDataset],
            type[CoeffDataset],
            type[PathDataset],
        ],
        task_type: Literal["classification", "regression"],
    ):
        if dataset_type not in [
            StandardDataset,
            BucketedDataset,
            CoeffDataset,
            PathDataset,
        ]:
            raise ValueError(
                f"Argument dataset_type cannot be {dataset_type}."
                + "See `dataset.py` for valid Dataset classes."
            )
        if task_type not in ["classification", "regression"]:
            raise ValueError(
                "`task_type` should be either 'classification' or 'regression'"
                + f", not {task_type}"
            )
        self.name = name
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.task_type = task_type

    @abstractmethod
    def _load_and_process_data(self) -> Tuple:
        """
        Loads data and labels from respective file locations
        and performs any necessary processing actions.

        Returns:
            (tuple): (train_data, val_data, test_data)
            (tuple): (train_labels, val_labels, test_labels)
        """
        pass

    def _shuffle(
        self,
        data: Tuple,
        labels: Tuple,
        key: jax.Array,
        val_proportion: float = 0.15,
        test_proportion: float = 0.15,
    ) -> Tuple[Tuple, Tuple]:
        """
        Shuffles data ordering and re-splits based on proportion kwargs.

        Args:
            data (tuple): (train_data, val_data, test_data)
            labels (tuple): (train_labels, val_labels, test_labels)
            key (jax.Array): Randomization key, from jax.random.key().
            val_proportion (float): The proportion of the dataset for to validation.
            test_proportion (float): The proportion of the dataset for to test.

        Returns:
            (tuple): (train_data, val_data, test_data),
                     (train_labels, val_labels, test_labels)
        """
        train_data, val_data, test_data = data
        train_labels, val_labels, test_labels = labels

        permutation_key, key = jr.split(key)
        idxs = jr.permutation(
            permutation_key, len(train_data) + len(val_data) + len(test_data)
        )
        if isinstance(train_data, jnp.ndarray) or isinstance(train_data, np.ndarray):
            full_data = jnp.concatenate((train_data, val_data, test_data), axis=0)
            shuffled_data = full_data[idxs]
        else:
            full_data = train_data + val_data + test_data
            shuffled_data = [full_data[i] for i in idxs.tolist()]
        if isinstance(train_labels, jnp.ndarray) or isinstance(
            train_labels, np.ndarray
        ):
            full_labels = jnp.concatenate(
                (train_labels, val_labels, test_labels), axis=0
            )
            shuffled_labels = full_labels[idxs]
        else:
            full_labels = train_labels + val_labels + test_labels
            shuffled_labels = [full_labels[i] for i in idxs.tolist()]

        bounds = [1.0 - val_proportion - test_proportion, 1.0 - test_proportion]
        data = split(shuffled_data, bounds)
        labels = split(shuffled_labels, bounds)

        return data, labels

    def _append_time(self, data, time_duration):
        """
        Appends a linearly interpolated time vector to start of arrays.

        Args:
            data (tuple): (train_data, val_data, test_data)
            time_duration (float): Time vector interpolated from 0 to this value.

        Returns:
            (tuple): (train_data, val_data, test_data)
        """
        train_data, val_data, test_data = data

        if isinstance(train_data, list):
            raise NotImplementedError(
                "Including time vector for variable length sequences not implemented."
            )
        else:
            num_timesteps = train_data.shape[1]
            time = jnp.linspace(0, time_duration, num=num_timesteps, endpoint=False)
            train_time = jnp.repeat(time[np.newaxis, ...], len(train_data), axis=0)[
                ..., np.newaxis
            ]
            train_data = jnp.concatenate((train_time, train_data), axis=2)
            val_time = jnp.repeat(time[np.newaxis, ...], len(val_data), axis=0)[
                ..., np.newaxis
            ]
            val_data = jnp.concatenate((val_time, val_data), axis=2)
            test_time = jnp.repeat(time[np.newaxis, ...], len(test_data), axis=0)[
                ..., np.newaxis
            ]
            test_data = jnp.concatenate((test_time, test_data), axis=2)

        return (train_data, val_data, test_data)

    def _calculate_dimension(self, data, labels):
        """Calculate dataset dimensions. Overwrite if needed."""
        train_data, _, _ = data
        train_labels, _, _ = labels

        data_dim = train_data[0].shape[1] if train_data[0].ndim == 2 else 1

        # 1D sample could mean (n,1) or (1,n)
        if train_labels[0].ndim == 1:
            if self.task_type == "regression":
                label_dim = 1
            else:
                label_dim = len(train_labels[0])
        else:
            label_dim = train_labels[0].shape[-1]

        return data_dim, label_dim

    def data_out_func(self, batch):
        """Runtime post-batching operation. Overwrite if neeeded."""
        return batch

    def create_dataset(
        self,
        use_presplit,
        time_duration,
        key,
        stepsize=None,
        depth=None,
        in_memory=False,
    ):
        data, labels = self._load_and_process_data()

        if not use_presplit:
            shuffle_key, key = jr.split(key)
            data, labels = self._shuffle(data, labels, shuffle_key)

        if time_duration is not None:
            data = self._append_time(data, time_duration)

        data_dim, label_dim = self._calculate_dimension(data, labels)

        return self.dataset_type(
            self.name,
            data,
            labels,
            data_dim,
            label_dim,
            in_memory,
            self.data_out_func,
            time_duration=time_duration,
            stepsize=stepsize,
            depth=depth,
        )


# =============================================
# SECTION: Dataset-specific DatasetLoaders
# =============================================


class UEALoader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + f"/processed/UEA/{self.name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(self.data_dir + f"/processed/UEA/{self.name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

        bounds = [0.7, 0.85]
        split_data = split(data, bounds)
        split_labels = split(onehot_labels, bounds)

        return split_data, split_labels


class PPGLoader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + "/processed/PPG/ppg/X_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(self.data_dir + "/processed/PPG/ppg/y_train.pkl", "rb") as f:
            train_labels = pickle.load(f)
        with open(self.data_dir + "/processed/PPG/ppg/X_val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(self.data_dir + "/processed/PPG/ppg/y_val.pkl", "rb") as f:
            val_labels = pickle.load(f)
        with open(self.data_dir + "/processed/PPG/ppg/X_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open(self.data_dir + "/processed/PPG/ppg/y_test.pkl", "rb") as f:
            test_labels = pickle.load(f)

        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        return data, labels


class MocapLoader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + "/processed/Mocap/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(self.data_dir + "/processed/Mocap/labels.pkl", "rb") as f:
            labels = pickle.load(f)

        label_map = {"jump": 0, "run": 1, "walk": 2}
        labels = jnp.array([label_map[la] for la in labels])
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

        # Pad data
        max_len = np.max([len(d) for d in data])
        padded_seqs = []
        for seq in data:
            num_dim = seq.shape[1]
            padded_seq = np.pad(
                seq.reshape(-1, num_dim),
                pad_width=((0, max_len - len(seq)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_seqs.append(padded_seq)
        data = jnp.asarray(np.array(padded_seqs))

        bounds = [0.7, 0.85]
        split_data = split(data, bounds)
        split_labels = split(onehot_labels, bounds)

        return split_data, split_labels


class SE3Loader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + f"/processed/SE3/{self.name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(self.data_dir + f"/processed/SE3/{self.name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)

        bounds = [0.7, 0.85]
        split_data = split(data, bounds)
        split_labels = split(labels, bounds)

        return split_data, split_labels


class ToyLoader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + "/processed/toy/signature/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(self.data_dir + "/processed/toy/signature/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        if self.name == "signature1":
            labels = ((jnp.sign(labels[0][:, 2]) + 1) / 2).astype(int)
        elif self.name == "signature2":
            labels = ((jnp.sign(labels[1][:, 2, 5]) + 1) / 2).astype(int)
        elif self.name == "signature3":
            labels = ((jnp.sign(labels[2][:, 2, 5, 0]) + 1) / 2).astype(int)
        elif self.name == "signature4":
            labels = ((jnp.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(int)

        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

        bounds = [0.7, 0.85]
        split_data = split(data, bounds)
        split_labels = split(onehot_labels, bounds)

        return split_data, split_labels


class SyntheticLoader(DatasetLoader):
    def _load_and_process_data(self):
        with open(self.data_dir + f"/processed/{self.name}/X_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(self.data_dir + f"/processed/{self.name}/y_train.pkl", "rb") as f:
            train_labels = pickle.load(f)
        with open(self.data_dir + f"/processed/{self.name}/X_val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(self.data_dir + f"/processed/{self.name}/y_val.pkl", "rb") as f:
            val_labels = pickle.load(f)
        with open(self.data_dir + f"/processed/{self.name}/X_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open(self.data_dir + f"/processed/{self.name}/y_test.pkl", "rb") as f:
            test_labels = pickle.load(f)

        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        return data, labels


class Cifar10Loader(DatasetLoader):
    def _load_and_process_data(self):
        # Load CIFAR-10
        download_dir = BASE_DIR / "data" / "raw" / "cifar"
        dataset_train = torchvision.datasets.CIFAR10(
            download_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.Grayscale(),
        )
        dataset_test = torchvision.datasets.CIFAR10(
            download_dir,
            train=False,
            transform=torchvision.transforms.Grayscale(),
        )
        data_dim = 1  # One grayscale channel
        num_classes = 10

        # CIFAR-10 grayscale normalization (from S5)
        mean = 122.6 / 255.0
        std = 61.0 / 255.0

        # Convert to numpy arrays first (need to do this for tensorflow datasets)
        train_data = []
        train_labels = []
        for image, label in dataset_train:
            train_data.append(np.array(image))
            train_labels.append(np.array(label))
        train_data = jnp.array(train_data).reshape(-1, 32 * 32, data_dim)
        train_data = (train_data / 255 - mean) / std
        train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)

        test_data = []
        test_labels = []
        for image, label in dataset_test:
            test_data.append(np.array(image))
            test_labels.append(np.array(label))
        test_data = jnp.array(test_data).reshape(-1, 32 * 32, data_dim)
        test_data = (test_data / 255 - mean) / std
        test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

        bounds = [0.9]  # From S5
        (train_data, val_data) = split(train_data, bounds)
        (train_labels, val_labels) = split(train_labels, bounds)
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        return data, labels


class NoisyCifar10Loader(DatasetLoader):
    def _load_and_process_data(self):
        # Load CIFAR-10
        download_dir = BASE_DIR / "data" / "raw" / "cifar"
        dataset_train = torchvision.datasets.CIFAR10(
            download_dir,
            train=True,
            download=True,
        )
        dataset_test = torchvision.datasets.CIFAR10(
            download_dir,
            train=False,
        )
        num_classes = 10

        # Convert to numpy arrays first (need to do this for tensorflow datasets)
        train_data = []
        train_labels = []
        for image, label in dataset_train:
            train_data.append(np.array(image))
            train_labels.append(np.array(label))
        train_data = jnp.array(train_data)
        train_labels = jnp.array(train_labels)
        test_data = []
        test_labels = []
        for image, label in dataset_test:
            test_data.append(np.array(image))
            test_labels.append(np.array(label))
        test_data = jnp.array(test_data)
        test_labels = jnp.array(test_labels)

        # Normalize by channel
        mean = np.mean(train_data, axis=[0, 1, 2])
        std = np.std(train_data, axis=[0, 1, 2])
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        # Flatten channels
        train_data = jnp.array(train_data).reshape(-1, 32, 96)
        test_data = jnp.array(test_data).reshape(-1, 32, 96)

        # One-hot labels
        train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)
        test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

        # Split data
        bounds = [0.9]  # From S5
        (train_data, val_data) = split(train_data, bounds)
        (train_labels, val_labels) = split(train_labels, bounds)
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        # out_dir = BASE_DIR / "data" / "processed" / "NoisyCifar10"
        # os.makedirs(out_dir, exist_ok=True)
        # with open(out_dir / "X_train.pkl", "wb") as f:
        #     pickle.dump(train_data, f)
        # with open(out_dir / "y_train.pkl", "wb") as f:
        #     pickle.dump(train_labels, f)
        # with open(out_dir / "X_val.pkl", "wb") as f:
        #     pickle.dump(val_data, f)
        # with open(out_dir / "y_val.pkl", "wb") as f:
        #     pickle.dump(val_labels, f)
        # with open(out_dir / "X_test.pkl", "wb") as f:
        #     pickle.dump(test_data, f)
        # with open(out_dir / "y_test.pkl", "wb") as f:
        #     pickle.dump(test_labels, f)

        return data, labels

    def data_out_func(self, batch):
        """Noisify during runtime"""
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=(batch.shape[0], 968, batch.shape[-1]))
        noisy_batch = jnp.concatenate([batch, noise], axis=1)
        return noisy_batch


class IMDbLoader(DatasetLoader):
    start_char = 1
    oov_char = 2
    end_char = 3
    index_from = 4
    max_length = 4096  # per S5
    min_freq = 15  # per S5
    num_class = 2  # positive, negative

    def _load_and_process_data(self):
        imdb_data_dir = BASE_DIR / "data" / "raw" / "imdb"
        if imdb_data_dir.exists():
            # TODO load from cache
            raise NotImplementedError()
            return

        # File is cached at ~/.keras/dataset/, copied to data/raw/imdb
        # TODO copy/save to dataset_path
        (train_data, train_labels), (test_data, test_labels) = (
            tf.keras.datasets.imdb.load_data(
                start_char=self.start_char,
                oov_char=self.oov_char,
                index_from=self.index_from,
                seed=42,
            )
        )

        # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/get_word_index
        idx_counts = Counter()
        for idxs in train_data:
            idx_counts.update(idxs)
        word_index = tf.keras.datasets.imdb.get_word_index()
        inverted_word_index = {
            i + self.index_from: word
            for (word, i) in word_index.items()
            if idx_counts[i + self.index_from] >= self.min_freq
        }

        char_vocab = list(string.printable)
        self.num_chars = len(char_vocab) + 4  # bos, eos, unk, pad
        char_index = {char: i + self.index_from for i, char in enumerate(char_vocab)}
        # inverted_char_index = {i: char for char, i in char_index.items()}

        # From word tokens to char tokens
        def convert_tokens(data):
            out = []
            for x in data:
                tokens = []  # Already begins with start token

                for i, idx in enumerate(x):
                    # Add a space between words
                    if i > 1:
                        tokens.append(char_index[" "])

                    # Process word
                    if idx in inverted_word_index:
                        chars = list(inverted_word_index[idx])
                        if all([c in char_index for c in chars]):
                            tokens += [char_index[c] for c in chars]
                        else:
                            tokens.append(self.oov_char)
                    elif idx in [0, 1, 2, 3]:
                        tokens.append(idx)
                    else:
                        tokens.append(self.oov_char)

                # Truncate sequence
                if len(tokens) > self.max_length + 1:
                    tokens = tokens[: self.max_length + 1]

                tokens.append(self.end_char)
                out.append(np.array(tokens, dtype=int))

            return out

        train_data = convert_tokens(train_data)
        test_data = convert_tokens(test_data)

        train_labels = jax.nn.one_hot(
            np.array(train_labels), num_classes=self.num_class
        )
        test_labels = jax.nn.one_hot(np.array(test_labels), num_classes=self.num_class)

        # No validation provided: train = test = 25000
        # Use last 10000 of training set for validation
        bounds = [0.6]
        (train_data, val_data) = split(train_data, bounds)
        (train_labels, val_labels) = split(train_labels, bounds)
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        return data, labels

    def _calculate_dimension(self, data, labels):
        data_dim = self.num_chars
        label_dim = self.num_class

        return data_dim, label_dim

    def data_out_func(self, batch):
        """One-hot during runtime (dataset too large)"""
        batch_one_hot = jax.nn.one_hot(
            batch.reshape((len(batch), -1)), num_classes=self.num_chars
        )
        return batch_one_hot


class MNISTLoader(DatasetLoader):
    def _load_and_process_data(self):
        download_dir = BASE_DIR / "data" / "raw" / "mnist"
        dataset_train = torchvision.datasets.MNIST(
            download_dir,
            train=True,
            download=True,
        )
        dataset_test = torchvision.datasets.MNIST(
            download_dir,
            train=False,
        )
        data_dim = 28
        num_classes = 10

        train_data = []
        train_labels = []
        for image, label in dataset_train:
            train_data.append(np.array(image))
            train_labels.append(np.array(label))
        train_data = jnp.array(train_data).reshape(-1, 28, data_dim)
        train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)

        # Normalize
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean) / std

        test_data = []
        test_labels = []
        for image, label in dataset_test:
            test_data.append(np.array(image))
            test_labels.append(np.array(label))
        test_data = jnp.array(test_data).reshape(-1, 28, data_dim)
        test_data = (test_data - mean) / std
        test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

        bounds = [0.9] 
        (train_data, val_data) = split(train_data, bounds)
        (train_labels, val_labels) = split(train_labels, bounds)
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)

        return data, labels


# =============================================
# SECTION: Entrypoint function
# =============================================


def create_dataset(
    name: str,
    data_dir: str,
    dataset_type: Union[
        type[StandardDataset],
        type[BucketedDataset],
        type[CoeffDataset],
        type[PathDataset],
    ],
    task_type: Literal["classification", "regression"],
    time_duration: float,
    use_presplit: bool,
    stepsize,
    depth,
    *,
    key,
):
    dataset_loaders = (
        {
            "Mocap": MocapLoader,
            "SyntheticRegression": SyntheticLoader,
            "ppg": PPGLoader,
            "Cifar10": Cifar10Loader,
            "NoisyCifar10": NoisyCifar10Loader,
            "IMDb": IMDbLoader,
            "MNIST": MNISTLoader,
        }
        | {name: UEALoader for name in get_subfolders(data_dir + "/processed/UEA")}
        | {name: ToyLoader for name in get_subfolders(data_dir + "/processed/toy")}
        | {name: SE3Loader for name in get_subfolders(data_dir + "/processed/SE3")}
    )

    if name in dataset_loaders:
        return dataset_loaders[name](
            name, data_dir, dataset_type, task_type
        ).create_dataset(
            use_presplit,
            time_duration,
            key,
            stepsize,
            depth,
        )
    else:
        raise ValueError(f"Dataset {name} not found")
