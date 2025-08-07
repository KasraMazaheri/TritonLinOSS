<h1 align='center'> Learning to Dissipate Energy in Oscillatory State-Space Models <br></h1>

This repository implements Damped Linear Oscillatory State-Space Models (D-LinOSS), an expressive and efficient extension to the original LinOSS model. This codebase is adapted from the <a href="https://github.com/tk-rusch/linoss">linoss</a> and <a href="https://github.com/Benjamin-Walker/log-neural-cdes">log-neural-cdes</a> repositories.

---

## Requirements

This repository is implemented in python 3.10 and uses Jax as the machine learning framework.

### Environment

The code for preprocessing the datasets, training LinOSS, S5, LRU, NCDE, NRDE, and Log-NCDE uses the following packages:
- `jax` and `jaxlib` for automatic differentiation.
- `equinox` for constructing neural networks.
- `optax` for neural network optimisers.
- `sktime` for handling time series data in ARFF format.
- `matplotlib` for plotting.

```
conda create -n linoss python=3.10
conda activate linoss
conda install sktime matplotlib -c conda-forge
pip install -U "jax[cuda12]" equinox==0.13.0 optax==0.2.5
```

Jax and jaxlib version 0.6.2 were used at the time of experimentation.

If running `scripts/data/process_uea.py` throws this error: No module named 'packaging'
Then run: `pip install packaging`

---

## Data

The folder `scripts/data` contains the scripts for downloading data, preprocessing the data, and creating dataloaders and datasets. Raw data should be downloaded into the `data/raw` folder. Processed data should be saved into the `data/processed` folder in the following format: 
```
processed/{collection}/{dataset_name}/data.pkl, 
processed/{collection}/{dataset_name}/labels.pkl,
processed/{collection}/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by running `scripts/data/download_uea.py` and preprocessed by running `scripts/data/process_uea.py`.

### The PPG-DaLiA Dataset

The PPG-DaLiA dataset is a multivariate time series regression dataset, where the aim is to predict a person’s heart rate using data collected from a wrist-worn device. The dataset can be downloaded from the <a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">UCI Machine Learning Repository</a>. The data should be unzipped and saved in the `data/raw` folder in the following format `PPG_FieldStudy/S{i}/S{i}.pkl`. The data can be preprocessed by running the `scripts/data/process_ppg.py` script.

### The Weather Dataset

(TODO: explain downloading procedure)

---

## Experiments

The code for training and evaluating the models is contained in `linoss/train.py`. Experiments can be run using the `scripts/run_experiment.py` script. This script requires you to specify a folder containing hyperparameter spreads for a given experiment. These experiment folders can be generated using `scripts/create_experiment.py`.

---

## Reproducing the Results

(TODO: re-running experiments with new codebase)

```
python scripts/run_experiments --experiment_folder experiments/D-LinOSS/PPG
```

(TODO: explain post-processing)

---