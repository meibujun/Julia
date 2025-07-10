# DynamicEpistasisGBLUP.jl

**A cutting-edge Julia package for genomic prediction with dynamic orthogonal epistasis, GPU acceleration, and advanced quantitative genetic models.**

## Overview

`DynamicEpistasisGBLUP.jl` implements state-of-the-art methods for genomic prediction in livestock breeding and quantitative genetics research. It focuses on incorporating dynamic orthogonal epistatic effects into the Genomic Best Linear Unbiased Prediction (GBLUP) framework. The package is designed for high performance, leveraging Julia's capabilities for scientific computing and GPU acceleration via CUDA.jl and KernelAbstractions.jl.

The methodologies are inspired by and aim to implement solutions similar to those proposed in advanced research for handling complex genetic architectures, including the dynamic nature of genetic effects across generations and the importance of non-additive (epistatic) variance.

## Features (Implemented and In-Progress)

*   **Core GBLUP Framework:**
    *   Efficient computation of additive Genomic Relationship Matrices (GRMs) using VanRaden's method.
    *   Orthogonal decomposition of genetic effects.
    *   Restricted Maximum Likelihood (REML) estimation of variance components (e.g., AI-REML).
*   **Epistasis Modeling:**
    *   Computation of pairwise epistatic GRMs (e.g., using Hadamard products or symmetric polynomial approaches).
    *   Inclusion of epistatic effects in the GBLUP model.
*   **Dynamic Modeling:**
    *   Support for updating allele frequencies and GRMs across generations in simulated selection scenarios.
*   **Population Simulation:**
    *   Flexible simulation of base populations with user-defined genetic architectures (additive QTLs, epistatic QTL pairs).
    *   Simulation of selection and reproduction over multiple generations.
*   **GPU Acceleration:**
    *   Many core computations (GRMs, kernels for linear algebra) are implemented using CUDA.jl and KernelAbstractions.jl for significant speedups.
*   **Advanced Algorithms (some are stubs/in-progress):**
    *   Sparse epistasis detection methods (e.g., screening, Elastic Net, Group LASSO).
    *   Walsh-Hadamard Transform for specific epistasis models.
    *   NOIA framework for orthogonal parameterization.
    *   Symmetric polynomial methods for efficient GRM computation.
    *   Augmented AI-REML concepts for variance component estimation.
    *   Tensor Core optimization sketches for NVIDIA GPUs.
    *   Distributed computing framework stubs for handling massive datasets.
*   **Supporting Modules:**
    *   Multivariate trait analysis extensions.
    *   Breeding program optimization tools.
    *   Validation and benchmarking utilities.
    *   Visualization capabilities.

## Installation

This package is currently under development. Once registered, it can be installed using Julia's package manager:
```julia
import Pkg
Pkg.add("DynamicEpistasisGBLUP")
```
For the current development version, you might need to clone the repository and use `Pkg.dev()`.

## Basic Usage (Conceptual)

```julia
using DynamicEpistasisGBLUP
using Distributions, CUDA

# Ensure CUDA is functional for GPU features
if CUDA.functional()
    CUDA.allowscalar(false)
end

# 1. Simulate a base population
base_pop = simulate_population(
    n_individuals=500, n_snps=5000,
    h2_narrow=0.3, h2_broad=0.45
)

# 2. Fit a GBLUP model (e.g., including epistasis)
# This function would perform GRM calculation and REML internally on GPU by default
fitted_model, gebvs = orthogonal_epistasis_gblup(base_pop, include_epistasis=true)

println("Estimated Variance Components:")
println("  σ²_a: ", fitted_model.variance.σ²_a)
println("  σ²_aa: ", fitted_model.variance.σ²_aa)
println("  σ²_e: ", fitted_model.variance.σ²_e)
println("  h²: ", fitted_model.variance.h²)
println("  H²: ", fitted_model.variance.H²)

# 3. Perform cross-validation (example)
# cv_results = cross_validation(base_pop, n_folds=3, include_epistasis=true)
# println("Cross-validation results:")
# println(cv_results)

# 4. Run a multi-generation selection simulation (example)
# populations_history, models_history, gains_history = simulate_long_term_genetic_gain(
#     base_pop,
#     num_generations_to_simulate = 5
# )
# println("Genetic gain over generations: ", gains_history)

# 5. Use advanced features (e.g., sparse epistasis - conceptual)
# using DynamicEpistasisGBLUP.SparseEpistasis
# if CUDA.functional()
#   selected_interactions, _ = detect_sparse_interactions_screening(
#       base_pop.genotypes.data,
#       CuArray(base_pop.phenotypes.values)
#   )
#   # sparse_model = fit_elastic_net_epistasis(...)
# end

```

## Modules Structure

The package is organized into several modules/files within the `src/` directory:

*   `types.jl`: Core data structures.
*   `utils.jl`: Utility functions.
*   `gpu_kernels.jl`: Low-level GPU kernels (using KernelAbstractions.jl).
*   `simulation.jl`: Population simulation.
*   `grm_computation.jl`: Additive and epistatic GRM calculations.
*   `epistasis_core.jl`: Core logic for fitting GBLUP with epistasis.
*   `reml.jl`: REML algorithms for variance component estimation.
*   `prediction.jl`: Genomic prediction for new individuals, cross-validation.
*   And various advanced modules: `walsh_hadamard.jl`, `noia_framework.jl`, `symmetric_polynomials.jl`, `augmented_aireml.jl`, `sparse_epistasis.jl`, `distributed_computing.jl`, `gpu_optimization.jl`, `multivariate_extension.jl`, `breeding_optimization.jl`, `visualization.jl`.

## Contribution and Development

This package aims for high standards in both algorithmic implementation and software engineering. Contributions and suggestions are welcome. Please refer to the developmental roadmap and coding guidelines (to be established).

## License

This package is licensed under the MIT License. See the `LICENSE` file for details (currently not created, but assumed MIT as per `Project.toml`).
