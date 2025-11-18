"""
# Population Structure Module

Analysis of population structure and stratification in genomic data.

Includes:
- Principal Component Analysis (PCA)
- Population clustering
- Outlier detection
- Stratification correction

## Examples

```julia
# Perform PCA
pca_result = pca(geno; n_pcs=10, ld_prune=true)

# Detect outliers
outliers = detect_outliers_pca(pca_result; threshold=6.0)

# Cluster samples
clusters = cluster_samples(pca_result; n_clusters=3)
```
"""
module PopulationStructure

using LinearAlgebra
using Statistics
using Printf
using Random

using ..Core
using ..Data
using ..QC

# Include modules
include("pca.jl")

# Export PCA
export PCAResult
export pca, detect_outliers_pca, cluster_samples

end # module PopulationStructure
