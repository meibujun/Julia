# GenomicPro.jl

`GenomicPro.jl` is a comprehensive software package for genomic prediction in Julia. It provides a suite of tools for quality control, variance component estimation, and breeding value prediction using various models, including GBLUP, SSGBLUP, and advanced Bayesian and deep learning methods.

## Installation

```julia
using Pkg
Pkg.add("GenomicPro")
```

## Basic Usage

```julia
using GenomicPro

# Load data
geno = read_genotypes("genotypes.vcf")
pheno = read_phenotypes("phenotypes.csv", :ID, [:MilkYield])

# Compute GRM
G = compute_grm(geno)
y = pheno.table[:, :MilkYield]

# Estimate variance components
vc = estimate_variance_components(G, y)
λ = vc.residual_variance / vc.genetic_variance

# Solve for breeding values
results = solve_gblup(G, y, λ)
println("Breeding values: ", results.breeding_values)
```
