<h1 align='center'> TritonLinOSS <br></h1>

This repository implements a torch-based implementation of Damped Linear Oscillatory State-Space Models (D-LinOSS) using efficient Triton kernels as its backend. This codebase is adapted from the <a href="https://github.com/jaredbmit/damped-linoss">Learning to Dissipate Energy in Oscillatory State-Space Models</a> repository.

---

This repository is implemented in python 3.10 and uses Jax and PyTorch as the machine learning framework, with Triton as the backend for the custom parallel scan used in the PyTorch implementation.

## PyTorch Implementation

The PyTorch-based implementation of D-LinOSS can be found at `src/damped_linoss/models/TorchLinOSS.py`. This implementation is functionally equivalent to `src/damped_linoss/models/LinOSS.py` as is verified by the test suits.

## Install

```bash
pip install "damped-linoss[cuda] @ git+https://github.com/KasraMazaheri/TritonLinOSS"
```

or using `uv`:

```bash
uv add git+https://github.com/KasraMazaheri/TritonLinOSS --extra cuda
```

## Local installation

### Step 1: Clone the repository

```bash
git clone https://github.com/KasraMazaheri/TritonLinOSS
cd TritonLinOSS
```

### Step 2: install dependencies

#### Option 1: With CUDA/Triton support

```bash
pip install -e ".[cuda]"
```

#### Option 2: CPU-only or without Triton

If CUDA is not available (works with torch.compile). We don't recommend using this for training/production as it is significantly slower. It is intended to be used mainly for testing and development.

```bash
pip install -e .
```

#### Option 3: Development Installation (with tests, requires CUDA)

```bash
pip install -e ".[dev]"
```

#### Option 4: using `uv`

Configuring the TritonLinOSS environment using `uv` (recommended):

```bash
cd TritonLinOSS/
uv sync
```

add flags `--extra cuda` or `--extra dev` to include CUDA or `dev` dependencies respectively.

### original JAX implementation

```bash
pip install -e ".[jax]"
```

---

<!-- ## Data

The folder `scripts` contains the scripts for downloading data, preprocessing the data, and creating dataloaders and datasets. Raw data should be downloaded into the `data/raw` folder. Processed data should be saved into the `data/processed` folder in the following format: 
```
processed/{collection}/{dataset_name}/data.pkl, 
processed/{collection}/{dataset_name}/labels.pkl,
processed/{collection}/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by running `scripts/download_uea.py` and preprocessed by running `scripts/process_uea.py`.

### The PPG-DaLiA Dataset

The PPG-DaLiA dataset is a multivariate time series regression dataset, where the aim is to predict a person’s heart rate using data collected from a wrist-worn device. The dataset can be downloaded from the <a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">UCI Machine Learning Repository</a>. The data should be unzipped and saved in the `data/raw` folder in the following format `PPG_FieldStudy/S{i}/S{i}.pkl`. The data can be preprocessed by running the `scripts/process_ppg.py` script.

### The Weather Dataset

(TODO: explain downloading procedure)

---

## Experiments

The code for training and evaluating the models is contained in `linoss/train.py`. Experiments can be run using the `run_experiment.py` script. This script requires you to specify a folder containing hyperparameter spreads for a given experiment. These experiment folders can be generated using `create_experiment.py`.

To view the outputs of an experiment:
```
uv run process_results.py <experiment_folder>
``` -->