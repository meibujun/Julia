"""
    Data

Data structures for genomic data.

This module provides memory-efficient and performant data structures for:
- Genotypes (2-bit encoding, sparse matrices, memory-mapped files)
- Phenotypes (multi-trait support)
- Pedigrees (relationship matrices)

# Key Features

- **CompactGenotypes**: 2-bit encoding saves 96.8% memory
- **Lazy computation**: Allele frequencies computed on-demand and cached
- **Validation**: Built-in data quality checks
- **Flexibility**: Multiple storage formats

# Exports

- `CompactGenotypes`: Main genotype data structure
"""
module Data

using LinearAlgebra
using Statistics
using SparseArrays
using Printf

using ..Core

# Export types
export CompactGenotypes

# Include submodules
include("genotypes.jl")

end # module Data
