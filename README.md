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
- `diffrax` for differential equation solvers.
- `signax` for calculating the signature.
- `sktime` for handling time series data in ARFF format.
- `tqdm` for progress bars.
- `matplotlib` for plotting.

```
conda create -n LinOSS python=3.10
conda activate LinOSS
conda install sktime=0.30.1 tqdm=4.66.4 matplotlib=3.8.4 -c conda-forge
# Substitue for correct Jax pip install: https://jax.readthedocs.io/en/latest/installation.html
pip install -U "jax[cuda12]" "jaxlib[cuda12]" equinox==0.11.8 optax==0.2.2 diffrax==0.5.1 signax==0.1.1
```

Jax and jaxlib version 0.4.34 were used at the time of experimentation.

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

---

## Experiments

The code for training and evaluating the models is contained in `train.py`. Experiments can be run using the `scripts/run/run_experiment.py` script. This script requires you to specify the names of the models you want to train, the names of the datasets you want to train on, and a directory which contains configuration files. The configuration files should be organised as `config/{collection_name}/{model_name}/{dataset_name}/config_{idx:03d}.json` and contain the
following fields:
- `seeds`: A list of seeds to use for training.
- `lr_scheduler`: A function which takes the learning rate and returns the new learning rate.
- `num_steps`: The number of steps to train for.
- `print_steps`: The number of steps between printing the loss.
- `batch_size`: The batch size.
- `metric`: The metric to use for evaluation.
- `classification`: Whether the task is a classification task.
- `lr`: The initial learning rate.
- `include_time`: Whether to include time as a channel.
- `time_duration`: Duration of time when included as a channel.

Any further specific model parameters, such as:
- `linoss_discretization`: (Only for LinOSS) Discretization scheme. Choices are ['IM','IMEX']
- `damping`: (Only for LinOSS) Whether or not to include damping.

See `config/repeats` for examples.

---

## Reproducing the Results

The configuration files for all the experiments with fixed hyperparameters can be found in the `config/repeats` folder and `scripts/run/run_experiment.py` can be configured to run the repeat experiments for the UEA and PPG datasets:

```
python scripts/run/run_experiments --model_name LinOSS --dataset_names EigenWorms SelfRegulationSCP1 SelfRegulationSCP2 EthanolConcentration Heartbeat MotorImagery ppg --config_folder config/repeats
```

The `results` folder contains final scores from the UEA, PPG, and Weather experiments.

The `outputs` folder contains output files from the UEA, PPG, and Weather experiments. 

---