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

# Include submodules
include("grm.jl")
include("gblup.jl")
include("crossvalidation.jl")
include("grm_parallel.jl")

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

end # module Models
