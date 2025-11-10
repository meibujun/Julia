# User Guide

This guide provides a comprehensive overview of how to use `GenomicPro.jl` for genomic prediction analysis.

## Installation

```julia
using Pkg
Pkg.add("GenomicPro")
```

## Basic Workflow

1.  **Load Data**: Load your genotype, phenotype, and pedigree data.
2.  **Quality Control**: Apply filters to remove low-quality data.
3.  **Run Analysis**: Choose a model (GBLUP, SSGBLUP, etc.) and run the analysis.
4.  **Inspect Results**: Examine the breeding values and other model outputs.
