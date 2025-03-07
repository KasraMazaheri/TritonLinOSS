"""
This module implements `Dataset` classes, which process data from its original
structure into any special encoding required by certain models.
Also, stores `Dataloader` instances and dataset metadata that are accessed
during training.

It supports four different subclasses, tailored for use
with different types of models available in this repository.

1. `StandardDataset`: Time series data with fixed-length sequences.
    Used by RNNs, SSMs, and the like.

2. `BucketedDataset`: Time series data with variable-length sequences.
    Contains extra functionality for batching and processing during training time.
    Used by RNNs, SSMs, and the like.

3. `CoeffDataset`: Used by NCDEs.

4. `PathDataset`: Used by NRDEs and Log-NCDEs
"""

from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp

from linoss.data_processing.dataloader import (
    BaseDataloader,
    StandardDataloader,
    BucketedDataloader,
    CoeffDataloader,
    PathDataloader,
)
from linoss.data_processing.generate_coeffs import batch_calc_coeffs
from linoss.data_processing.generate_paths import batch_calc_paths


class BaseDataset(ABC):
    def __init__(
        self,
        name,
        data,
        labels,
        data_dim,
        label_dim,
        in_memory,
        data_out_func,
        **kwargs,
    ):
        self.name = name
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.in_memory = in_memory
        self.data_out_func = data_out_func

        self._generate(data, labels, **kwargs)

    @abstractmethod
    def _generate(self, data, labels, **kwargs):
        """Generates dataloaders and any additional properties."""
        pass


class StandardDataset(BaseDataset):
    def _generate(self, data, labels, **kwargs):
        (train_data, val_data, test_data) = data
        (train_labels, val_labels, test_labels) = labels

        train_loader = StandardDataloader(
            train_data,
            train_labels,
            self.in_memory,
            self.data_out_func,
        )
        val_loader = StandardDataloader(
            val_data,
            val_labels,
            self.in_memory,
            self.data_out_func,
        )
        test_loader = StandardDataloader(
            test_data,
            test_labels,
            self.in_memory,
            self.data_out_func,
        )

        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }


class BucketedDataset(BaseDataset):
    def _generate(self, data, labels, **kwargs):
        (train_data, val_data, test_data) = data
        (train_labels, val_labels, test_labels) = labels

        train_loader = BucketedDataloader(
            train_data,
            train_labels,
            self.in_memory,
            self.data_out_func,
        )
        val_loader = BucketedDataloader(
            val_data, val_labels, self.in_memory, self.data_out_func
        )
        test_loader = BucketedDataloader(
            test_data, test_labels, self.in_memory, self.data_out_func
        )

        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }


class CoeffDataset(BaseDataset):
    def _generate(self, data, labels, **kwargs):
        if "time_duration" not in kwargs:
            raise ValueError("`CoeffDataset` requires time_duration argument.")
        else:
            time_duration = kwargs["time_duration"]

        (train_data, val_data, test_data) = data
        (train_labels, val_labels, test_labels) = labels

        num_timesteps = train_data.shape[1]
        time = jnp.linspace(0, time_duration, num=num_timesteps, endpoint=False)
        train_time = jnp.repeat(time, len(train_data), axis=0)[..., np.newaxis]
        val_time = jnp.repeat(time, len(val_data), axis=0)[..., np.newaxis]
        test_time = jnp.repeat(time, len(test_data), axis=0)[..., np.newaxis]

        train_coeffs = batch_calc_coeffs(train_data, True, time_duration)
        val_coeffs = batch_calc_coeffs(val_data, True, time_duration)
        test_coeffs = batch_calc_coeffs(test_data, True, time_duration)

        train_coeff_data = (
            train_time,
            train_coeffs,
            train_data[:, 0, :],
        )
        val_coeff_data = (
            val_time,
            val_coeffs,
            val_data[:, 0, :],
        )
        test_coeff_data = (
            test_time,
            test_coeffs,
            test_data[:, 0, :],
        )

        train_loader = CoeffDataloader(train_coeff_data, train_labels, self.in_memory)
        val_loader = CoeffDataloader(val_coeff_data, val_labels, self.in_memory)
        test_loader = CoeffDataloader(test_coeff_data, test_labels, self.in_memory)

        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }


class PathDataset(BaseDataset):
    def _generate(self, data, labels, **kwargs):
        if "time_duration" not in kwargs:
            raise ValueError("`PathDataset` requires time_duration argument.")
        else:
            time_duration = kwargs["time_duration"]

        if "stepsize" not in kwargs:
            raise ValueError("`PathDataset` requires stepsize argument.")
        else:
            stepsize = kwargs["stepsize"]

        if "depth" not in kwargs:
            raise ValueError("`PathDataset` requires depth argument.")
        else:
            depth = kwargs["depth"]

        (train_data, val_data, test_data) = data
        (train_labels, val_labels, test_labels) = labels

        num_timesteps = train_data.shape[1]
        time = jnp.linspace(0, time_duration, num=num_timesteps, endpoint=False)
        train_time = jnp.repeat(time, len(train_data), axis=0)[..., np.newaxis]
        val_time = jnp.repeat(time, len(val_data), axis=0)[..., np.newaxis]
        test_time = jnp.repeat(time, len(test_data), axis=0)[..., np.newaxis]

        intervals = jnp.arange(0, num_timesteps, stepsize)
        intervals = jnp.concatenate((intervals, jnp.array([num_timesteps])))
        self.intervals = intervals * (time_duration / num_timesteps)

        train_path = batch_calc_paths(train_data, stepsize, depth)
        val_path = batch_calc_paths(val_data, stepsize, depth)
        test_path = batch_calc_paths(test_data, stepsize, depth)

        train_path_data = (
            train_time,
            train_path,
            train_data[:, 0, :],
        )
        val_path_data = (
            val_time,
            val_path,
            val_data[:, 0, :],
        )
        test_path_data = (
            test_time,
            test_path,
            test_data[:, 0, :],
        )

        self.logsig_dim = train_path.shape[-1]

        train_loader = PathDataloader(train_path_data, train_labels, self.in_memory)
        val_loader = PathDataloader(val_path_data, val_labels, self.in_memory)
        test_loader = PathDataloader(test_path_data, test_labels, self.in_memory)

        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
