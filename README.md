# Improving regional weather forecasts with neural interpolation

This repository is intended to supplement the proceedings article "Improving regional weather forecasts with neural interpolation" and contains the implementation required to reproduce the numerical experiments presented within.

The data used here is generated via a finite element discretisation using [Firedrake](https://www.firedrakeproject.org/). The dataset used can be found at the DOI [10.5281/zenodo.14803077](https://doi.org/10.5281/zenodo.14803077), or a similar dataset can be generated (or modified) using the Python scripts in [generate_data](generate_data), as described [here](#data-generation). 

## Installation

This repository depends on [Python 3](https://www.python.org/downloads/) (v3.11.6), [PyTorch](https://pytorch.org) (v2.1.2), [NumPy](https://numpy.org) (v1.24.0), [SciPy](https://scipy.org) (v1.11.4) and [Matplotlib](https://matplotlib.org) (v3.10.0). Drawing the figure also requires a LaTeX installation. The recommended installation procedure is to create a Python 3.11(.6) virtual environment and pip install the requirements with

```cmd
pip3 install -r requirements.txt
```

To generate data, one must install Firedrake. Up-to-date installation instructions can be found [here](https://www.firedrakeproject.org/download.html). 

## Code structure

`main.py`: This is the main script in this repository and, by default, trains every network in the results of the paper (Table 1, Figure 3 and the further experiments in Appendix B) for three random seeds. Each trained model and its errors on the validation data are saved in `results/`. Finished runs are skipped, so `main.py` can be restarted after an interruption. It is possible to choose the experiments in the list `experiments` and the seeds with `--seeds`, and to vary the other model parameters by changing parameter values in an instance of the `super_parameters` class, a required input to the function `model_run()`.

`plot.py`: Prints Table 1 and the other results quoted in the paper, and draws Figure 3 as `results/Figure3.pdf`, from the results saved by `main.py`.

`dataset.py`: Loads and preprocesses the data. The function `all_data()` will load both the model input and output (which have the same shape, as the input data has been interpolated onto the fine mesh by the data generation routine). The desired amount of time levels can be specified by the parameter `stacksize`. The function returns a training and testing dataset, with 70% of the data used for training.

`getData.py`: Checks if the dataset exists locally and downloads it if it doesn't. This process can be slow, so downloading immediately is recommended. This is run automatically by `dataset.py`. 

## Get started (tl;dr)

### Step 1 

Download the data by running `python3 getData.py` (or downloading the data from [https://doi.org/10.5281/zenodo.14803077](https://doi.org/10.5281/zenodo.14803077)).

### Step 2

Install `torch`, `numpy`, `scipy` and `matplotlib` in a Python3 virtual environment.

### Step 3

Run `main.py` to replicate the results presented in the paper. The three seeds can also be run in parallel (`python3 main.py --seeds 1`, `--seeds 2` and `--seeds 7`).

### Step 4

Once all runs have finished, run `plot.py` to print the results and draw the figure.

## Data generation

`swe.py`: Generates one simulation of the shallow water equations in 1D with random parameters initialised from the `generator` class. The simulation runs over both a coarse and fine spatial grid. The coarse solution is interpolated onto the finer grid, so the data sizes are the same. One should modify the `generator` class in this file to change the randomised initial data.

`call_swe.py`: A wrapper for `swe.py` generating one instance of random data for an arbitrary function call `i`. The data will be saved to a temporary file `tmp/swe{i}.pickle`. This function should be called with `python3 call_swe.py {i}`.

`generate_swe.py`: A script that generates `1000` runs of `swe.py` and saves them as `data_swe.pickle` in the `generate_data` subdirectory. This file must be moved to the root directory to be used by default.

