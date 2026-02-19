# Improving EV Aggregate Flexibility with End-to-End Learning

Apoorva Thanvantri, Christopher Yeh, Nicolas Christianson, Adam Wierman
<br>**California Institute of Technology**, Department of Computing and Mathematical Sciences


This repo contains code for the following paper: 

**Improving EV Aggregate Flexibility with End-to-End Learning**
<br>A. Thanvantri, C. Yeh, N. Christianson, A. Wierman

## Installation Instructions

Running code from this repo requires: 
- python 3.11
- cuopt 25.08
- cvxpy 1.7
- ecos 2.0
- matplotlib 3.10
- numpy 2.2
- pandas 2.3
- scikit-learn 1.7
- scipy 1.16
- wandb 1.21
- tdqm 4.67
- cvxpylayers 0.1.9
- torch 2.8.0

We recommend using the [conda](https://docs.conda.io/) package manager.

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).

2. Install the packages from the `env.yml` file:
    ```bash
    conda env update --file env.yml --prune
    ```

3. Activate the conda environment
    ```bash
    conda activate evflex
    ```

## Experiment: Peak Power Minimization

1. We use load profiles (in the `load_data` folder) from Pecan Street Dataport which consist of the electricity consumption for 25 households over a 18 hour time horizon for a 6-month span in 2019. 

2. Run `ppm_experiment.py` to train the ICNN and save the checkpoints. 

3. Run `generate_comparison.py` to compare the ICNN models to other baselines.

4. Run `generate_slices.ipynb` to plot a 2D slice of an ICNN, True Minkowski Sum, and General Affine Approximation (from Taha et al. 2024)

## Experiment: Cvxpylayer vs. PGD Experiment

1. Run `cvxpylayer_vs_pgd.py` to generate the timing comparison of using cvxpylayers vs the PGD approach.
 
2. Run `plot_cvxpylayer_vs_pgd.py` to plot how the timing trend evolves as the dimension of the problem increases. 
