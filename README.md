# PyJWAS: Python Whole-genome Analysis Software

PyJWAS is a Python library for performing whole-genome analyses, including genomic prediction and genome-wide association studies (GWAS), using Bayesian multiple regression methods. This project is a Python translation and reimagining of the Julia library [JWAS.jl](https://github.com/reworkhow/JWAS.jl).

## Overview

The goal of PyJWAS is to provide a robust, efficient, and user-friendly toolkit for researchers and practitioners in animal and plant breeding, quantitative genetics, and related fields. It aims to implement a range of Bayesian models commonly used in genomic analyses, such as:

*   BayesA, BayesB, BayesC, BayesCπ
*   Bayesian Lasso
*   RR-BLUP (Ridge Regression BLUP) / GBLUP
*   And potentially other advanced models like single-step (ssGBLUP, ssBR) and multi-trait models.

## Features (Planned/In-Progress)

*   **Model Building**: Flexible model definition using equation strings.
*   **Data Input**: Support for pedigree files, genotype data from various formats.
*   **MCMC Engine**: Efficient Markov Chain Monte Carlo engine for Bayesian inference.
*   **Variety of Bayesian Methods**: Implementation of common Bayesian alphabet methods.
*   **Pedigree Processing**: Tools for handling pedigree information and calculating relationship matrices (e.g., A-inverse).
*   **Genotype Data Processing**: Quality control, centering, and formatting of marker data.
*   **Output**: Comprehensive output of posterior means, variances, MCMC samples, and relevant statistics.

## Installation (Conceptual)

Once packaged, PyJWAS would typically be installed using pip:

```bash
pip install pyjwas
```

## Basic Usage (Conceptual)

```python
# Conceptual example
from pyjwas.core import build_model, MCMCInfo
from pyjwas.pedigree import read_pedigree
from pyjwas.genotypes_io import get_genotypes_data
from pyjwas.mcmc import run_mcmc
import pandas as pd

# 1. Load data
# phenotype_df = pd.read_csv("phenotypes.csv")
# pedigree_data = read_pedigree("pedigree.csv")
# genotype_data = get_genotypes_data("genotypes.csv", method="BayesC")

# 2. Build the model
# model_eq = "y = intercept + fixed_effect + animal + snp_effects"
# mme = build_model(model_eq, R_value=10.0) # Initial residual variance
# # Set random effects, covariates, associate pedigree, genotypes etc.
# # mme.add_random_effect("animal", pedigree_data, G_animal_prior)
# # mme.add_genotype_effects(genotype_data)
# # set_covariate(mme, "fixed_effect")

# 3. Set MCMC parameters
# mme.mcmc_info = MCMCInfo(chain_length=50000, burnin=10000)

# 4. Run MCMC (assuming mme object is fully built with MME components)
# results = run_mcmc(mme, phenotype_df)

# 5. Analyze results
# print(results["mean_solutions"])
# print(results["mean_residual_variance"])

# For a runnable example with simulated data for an RR-BLUP model,
# please see `examples/run_rrblup_simulation.py`.
```

## Development Status

PyJWAS is currently under active development. Core data structures, model building logic, some utilities, and an initial MCMC engine (supporting RR-BLUP for single traits) have been implemented. Pedigree and genotype data handling modules are also in progress.

## License

This project is licensed under the GPL-2.0 License - see the [LICENSE](LICENSE) file for details. This matches the license of the original JWAS.jl library.

## Contributing

Contributions are welcome. Please refer to (conceptual) contributing guidelines.

## Acknowledgements

This project is inspired by and aims to translate the functionalities of JWAS.jl, developed by Hao Cheng, Tianjing Zhao, Rohan Fernando, and Dorian Garrick.
```
