# OrthogonalGenomics.jl

`OrthogonalGenomics.jl` is a comprehensive Julia package for genomic prediction in livestock, implementing the Generalized Dynamic Orthogonal Epistasis model. It provides tools for:

-   **Genomic Relationship Matrix (GRM) Construction:** Including additive, dominance, and epistatic (AA, AD, DD) GRMs.
-   **Variance Component Estimation:** Using REML, AI-REML, and EM algorithms.
-   **Genomic Prediction:** With OG-BLUP, reaction norm models, and other advanced methods.
-   **Breeding Program Optimization:** Tools for optimal mating and selection strategies.
-   **Population Simulation:** Flexible simulation of complex genetic architectures and breeding scenarios.

This package is designed for researchers and breeders who need a powerful, flexible, and high-performance tool for modern genomic analysis.

## Installation

```julia
using Pkg
Pkg.add("OrthogonalGenomics")
```

## Quick Start

```julia
using OrthogonalGenomics

# Simulate a population
population = simulate_population(n_individuals=500, n_markers=1000, n_qtl=50)

# Fit the OG-BLUP model
model = fit_ogblup(population)

# Predict breeding values
gebv = predict_gebv(model, population)

# Optimize matings
mating_plan = optimal_mating(model, population)

println("Mean GEBV: ", mean(gebv))
println("Top 5 matings: ", mating_plan.matings[1:5])
```
