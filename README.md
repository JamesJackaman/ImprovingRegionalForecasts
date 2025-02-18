# Improving regional weather forecasts with neural interpolation

This repository is intended to supplement the proceedings article "Improving regional weather forecasts with neural interpolation" and contains the implementation required to reproduce the numerical experiments presented within.

The data used here is generated via a finite element discretisation using [Firedrake](https://www.firedrakeproject.org/). The dataset used can be found at the DOI [10.5281/zenodo.14803077](https://doi.org/10.5281/zenodo.14803077), or a similar dataset can be generated (or modified) using the Python scripts in [generate_data](generate_data), as described [here](#data-generation). 

## Installation

This repository depends on [Python 3](https://www.python.org/downloads/) (v3.11.6), [PyTorch](https://pytorch.org) (v2.1.2), [NumPy](https://numpy.org) (v1.24.0) and [SciPy](https://scipy.org) (v1.11.4). The recommended installation procedure is to create a Python 3.11(.6) virtual environment and pip install the requirements with

```cmd
pip3 install -r requirements.txt
```

To generate data, one must install Firedrake. Up-to-date installation instructions can be found [here](https://www.firedrakeproject.org/download.html). 

## Code structure

`main.py`: This is the main script in this repository and, by default, trains a variety of CNNs with a varying amount of $L_2$-like regularisation (as shown in the results of the paper). It is possible to vary the model parameters by changing parameter values in an instance of the `super_parameters` class, a required input to the function `model_run()`.

`dataset.py`: Loads and preprocesses the data. The function `all_data()` will load both the model input and output (which have the same shape, as the input data has been interpolated onto the fine mesh by the data generation routine). The desired amount of time levels can be specified by the parameter `stack_size`. The function returns a training and testing dataset, with 70% of the data used for training. 

`getData.py`: Checks if the dataset exists locally and downloads it if it doesn't. This process can be slow, so downloading immediately is recommended. This is run automatically by `dataset.py`. 

## Data generation

`swe.py`: Generates one simulation of the shallow water equations in 1D with random parameters initialised from the `generator` class. The simulation runs over both a coarse and fine spatial grid. The coarse solution is interpolated onto the finer grid, so the data sizes are the same. One should modify the `generator` class in this file to change the randomised initial data.

`call_swe.py`: A wrapper for `swe.py` generating one instance of random data for an arbitrary function call `i`. The data will be saved to a temporary file `tmp/swe{i}.pickle`. This function should be called with `python3 call_swe.py --iteration {i}`. 

`generate_swe.py`: A script that generates `1000` runs of `swe.py` and saves them as `data_swe.pickle` in the `data_generation` subdirectory. This file must be moved to the root directory to be used by default.

