"""
# Models Module

Statistical models for genomic prediction and analysis.

Includes:
- Genomic Relationship Matrix (GRM) computation
- GBLUP (Genomic Best Linear Unbiased Prediction)
- BayesR and other Bayesian models
- Deep Learning models

## Examples

```julia
# Compute GRM
G = compute_grm(geno; method=:vanraden)

# Train GBLUP model
model = GBLUPModel()
fit!(model, geno, pheno; G=G)

# Predict breeding values
gebv = predict(model, geno)
```
"""
module Models

using LinearAlgebra
using Statistics
using SparseArrays
using Printf
using Random

using ..Core
using ..Data

# Include abstract model interface first
include("abstract_model.jl")

# Include submodules
include("grm.jl")
include("gblup.jl")
include("crossvalidation.jl")
include("grm_parallel.jl")
include("bayesr.jl")
include("bayescpi.jl")
include("rkhs.jl")
include("deepgblup.jl")

# Export Abstract Model Interface
export AbstractGenomicModel, ModelType, ModelStatus
export LINEAR_MODEL, BAYESIAN_MODEL, KERNEL_MODEL, DEEP_LEARNING, ENSEMBLE_MODEL
export NOT_FITTED, FITTING, FITTED, FAILED
export fit!, predict, model_name, model_type
export is_fitted, score, get_hyperparameters, set_hyperparameters!
export feature_importance, get_fitted_values, get_variance_components, heritability
export compare_models, select_best_model
export EnsembleModel, print_model_summary

# Export GRM functions
export compute_grm, compute_grm_vanraden, compute_grm_additive
export center_genotypes, scale_genotypes

# Export Parallel GRM functions
export compute_grm_parallel, compute_grm_vanraden_parallel, compute_grm_additive_parallel
export compute_symmetric_product_parallel, benchmark_threading

# Export GBLUP
export GBLUPModel, fit!, predict
export GBLUPResult

# Export Cross-validation
export CVResult
export kfold_cv, loo_cv, random_cv
export create_folds

# Export BayesR
export BayesRModel, BayesRResult

# Export BayesCπ
export BayesCpiResults, fit_bayescpi, predict_bayescpi

# Export RKHS
export KernelFunction, LinearKernel, GaussianKernel, PolynomialKernel, ExponentialKernel
export RKHSResults, fit_rkhs, predict_rkhs, cross_validate_bandwidth

# Export Deep GBLUP
export DeepGBLUP, DeepGBLUPResults, train_deepgblup!, predict_deepgblup

end # module Models
