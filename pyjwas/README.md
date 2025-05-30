# pyjwas: Python-based Joint Whole-genome Association Studies

## Description

`pyjwas` is a Python library designed to facilitate Whole-Genome Association Studies (GWAS) and related genomic analyses. It provides tools for data handling, model definition, MCMC-based GBLUP (Genomic Best Linear Unbiased Prediction) analysis, and results processing.

## Status

**Current Version:** 0.1.0-alpha

This library is currently in an alpha stage. Core features for GBLUP analysis are implemented, along with basic data handling and model parsing. However, it is still under active development, and APIs might change. More features, an expanded range of statistical models, and further optimizations are planned.

## Current Features

*   **Data Handling (`pyjwas.data`)**:
    *   Loading of phenotype data from CSV files.
    *   Loading of genotype data from CSV files (numeric coding, e.g., 0, 1, 2).
    *   Calculation of Genomic Relationship Matrix (GRM) using VanRaden's method.
    *   Basic design matrix construction (X and Z).
*   **Model Definition (`pyjwas.model`, `pyjwas.parser`)**:
    *   Parsing of model equations (e.g., "y = intercept + fixed_effect + random_effect").
    *   Definition of fixed and random effects for models.
*   **MCMC Sampling (`pyjwas.mcmc`)**:
    *   Base MCMC sampler class for extensibility.
    *   GBLUP sampler implementing Gibbs sampling for:
        *   Fixed effects (β).
        *   Genomic breeding values (g).
        *   Genetic variance (σ²g).
        *   Residual variance (σ²e).
    *   Posterior summary generation (mean, std, credible intervals).
    *   EBV summary generation.
*   **Results Handling (`pyjwas.results`)**:
    *   Saving MCMC samples to CSV files.
    *   Saving summary DataFrames (like EBVs) to CSV files.

## Installation

Currently, `pyjwas` is not packaged for PyPI. To use it, clone the repository and install the dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace with actual URL when available
    cd pyjwas
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    If you plan to run tests or contribute, also install development dependencies:
    ```bash
    pip install -r requirements-dev.txt
    ```

3.  **Set up PYTHONPATH (if not installing as a package):**
    To import `pyjwas` from scripts outside the `pyjwas` root directory, add the parent directory of `pyjwas` (the one containing the `pyjwas` folder itself) to your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/parent_directory_of_pyjwas
    ```

## Quick Start: GBLUP Example

Here's a basic example of how to run a GBLUP analysis using `pyjwas`:

```python
import pandas as pd
import numpy as np
from pyjwas.data import PhenotypeData, GenotypeData
from pyjwas.model import build_model
from pyjwas.mcmc import run_mcmc
from pyjwas.results import save_summary_df, save_mcmc_samples

# --- 1. Prepare Data (Example: Create dummy CSV files) ---
# Phenotype data (ID, phenotype_y, fixed_effect_x)
pheno_content = """ID,y,X1
id1,10.5,1
id2,12.1,0
id3,9.8,1
id4,11.5,0
id5,10.2,1
"""
with open("dummy_pheno.csv", "w") as f:
    f.write(pheno_content)

# Genotype data (ID, marker1, marker2, ...)
geno_content = """ID,m1,m2,m3
id1,0,1,2
id2,1,2,1
id3,2,1,0
id4,0,0,1
id5,1,1,1
"""
with open("dummy_geno.csv", "w") as f:
    f.write(geno_content)

# --- 2. Load Data ---
pheno_data = PhenotypeData("dummy_pheno.csv")
geno_data = GenotypeData("dummy_geno.csv") # GRM will be calculated automatically if needed

# --- 3. Define the Model ---
# Model: y = intercept + X1 + g (where g is the genomic random effect)
model_def = build_model("y = intercept + X1")
model_def.set_covariate("X1") # Specify X1 as a fixed effect covariate

# --- 4. Run MCMC (GBLUP) ---
# run_mcmc now returns the sampler instance
sampler = run_mcmc(
    model_definition=model_def,
    phenotype_data=pheno_data,
    genotype_data=geno_data,
    n_iterations=2000,  # Recommended: 10,000+ for real analysis
    burn_in=500,        # Recommended: 1,000+ for real analysis
    method="GBLUP"
)

# --- 5. Get Results ---
# Access samples directly from the sampler instance
mcmc_samples = sampler.samples 

# Get posterior summaries
beta_summary = sampler.get_posterior_summary("beta")
g_summary = sampler.get_posterior_summary("g")
variance_summary = sampler.get_posterior_summary("variance_components")

print("\\n--- Posterior Summaries ---")
if beta_summary:
    print("Beta (Fixed Effects) Mean:", beta_summary['mean'])
if variance_summary:
    print("Genetic Variance Mean:", variance_summary['genetic']['mean'])
    print("Residual Variance Mean:", variance_summary['residual']['mean'])

# Get EBV summary DataFrame
ebv_df = sampler.get_ebv_summary()
if ebv_df is not None:
    print("\\n--- EBV Summary (First 5) ---")
    print(ebv_df.head())

# --- 6. Save Results ---
if ebv_df is not None:
    save_summary_df(ebv_df, "output/ebv_summary.csv")

# Save all MCMC samples (beta, g, variance_components)
# This creates files like: output/samples_beta.csv, output/samples_g.csv, etc.
if mcmc_samples:
    save_mcmc_samples(mcmc_samples, "output/mcmc_run1_samples")

print("\\nQuick start example finished. Check the 'output' directory for results.")
```

## Directory Structure

```
pyjwas/
├── pyjwas/                 # Main library source code
│   ├── __init__.py
│   ├── data.py
│   ├── mcmc.py
│   ├── model.py
│   ├── parser.py
│   └── results.py
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── data/               # Test data files
│   │   ├── test_genotypes.csv
│   │   └── test_phenotypes.csv
│   ├── test_data.py
│   ├── test_mcmc_gblup.py
│   └── test_model_parser.py
├── README.md               # This file
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development and testing dependencies
└── LICENSE                 # MIT License file
```

## Contributing

Contributions are welcome! Please follow the standard fork-pull request workflow. Before submitting a PR, ensure tests pass and consider adding new tests for your features. (Further details to be added: coding style, issue tracker, etc.)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
