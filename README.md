<h1 align='center'> TritonLinOSS <br></h1>

This repository implements a torch-based implementation of Damped Linear Oscillatory State-Space Models (D-LinOSS) using efficient Triton kernels as its backend. This codebase is adapted from the <a href="https://github.com/jaredbmit/damped-linoss">Learning to Dissipate Energy in Oscillatory State-Space Models</a> repository.

---

This repository is implemented in python 3.10 and uses Jax and PyTorch as the machine learning framework, with Triton as the backend for the custom parallel scan used in the PyTorch implementation.

## PyTorch Implementation

The PyTorch-based implementation of D-LinOSS can be found at `src/damped_linoss/models/TorchLinOSS.py`. This implementation is functionally equivalent to `src/damped_linoss/models/LinOSS.py` as is verified by the test suits.

The current PyTorch surface also includes the Discretax-style LinOSS layer and
backbone:

```python
import torch
from damped_linoss import LinOSSBackbone, LinOSSSequenceMixer

x = torch.randn(8, 256, 64, device="cuda")

layer = LinOSSSequenceMixer(
    in_features=64,
    state_dim=128,
    num_heads=2,
    discretization="IMEX3",  # IM, IMEX, IMEX2, IMEX3, or EX
    initialization="AG",     # AG or RT
    damping=True,
    stability="oscillatory", # oscillatory or stable
    use_triton=True,
).cuda()
y = layer(x)

model = LinOSSBackbone(
    hidden_dim=64,
    num_blocks=4,
    state_dim=128,
    num_heads=2,
    drop_rate=0.1,
    use_triton=True,
).cuda()
y = model(x)
```

`LinOSSSequenceMixer` is the minimal core layer intended for reuse in other
Torch models. It supports batched `(B, L, H)` and unbatched `(L, H)` inputs,
the newer `IMEX2`, `IMEX3`, and `EX` discretizations, stable or oscillatory
projection, optional LRU-style input normalization, and multi-head gating or
output projection. `LinOSSBackbone` stacks this layer in residual GLU blocks
without adding task-specific encoders, decoders, or training code.

## Benchmark Snapshot

The numbers below were measured on an NVIDIA H200 with `torch==2.12.0+cu126`
and the default Triton scan tile heuristic. Timings are milliseconds per call
after warmup. They are intended as a quick sanity check, not a full tuning
study.

| Workload | Backend | Forward | Forward + backward |
| --- | ---: | ---: | ---: |
| Core layer, B=8 L=256 H=64 P=128 heads=1 | Torch native scan | 3.81 | 12.12 |
| Core layer, B=8 L=256 H=64 P=128 heads=1 | Torch + Triton scan | 0.65 | 2.95 |
| Core layer, B=8 L=256 H=64 P=128 heads=2 | Torch + Triton scan | 0.68 | 3.10 |
| Core layer, B=4 L=512 H=128 P=256 heads=4 | Torch + Triton scan | 0.70 | 3.07 |
| 2-block backbone, B=8 L=256 H=64 P=128 heads=2 | Torch native scan | 8.06 | 25.11 |
| 2-block backbone, B=8 L=256 H=64 P=128 heads=2 | Torch + Triton scan | 1.87 | 6.67 |

For the comparable single-head core layer, the Triton scan path was about
5.8x faster than the native Torch scan in forward and about 4.1x faster for
forward plus backward. A Discretax/JAX GPU reference remains faster for the
single-head core layer in this snapshot, so the Torch path should be viewed as
deployable and substantially accelerated, with more kernel-level optimization
still available.

## Installation

### Option 1: With CUDA/Triton support

```bash
pip install -e ".[cuda]"
```

### Option 2: CPU-only or without Triton

If CUDA is not available (works with torch.compile):

```bash
pip install -e .
```

### Development Installation (with tests)

```bash
pip install -e ".[dev]"
```

Configuring the TritonLinOSS environment:

```bash
cd TritonLinOSS/
uv sync
```

This will create a virtual environment in `TritonLinOSS/.venv`.

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
