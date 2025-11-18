"""
    Core

Core types, interfaces, exceptions and utilities for GenomicPro2.

This module defines the foundational abstractions that all other modules build upon.
"""
module Core

using LinearAlgebra
using Statistics

# Export types
export AbstractGenomicData, AbstractGenotypeData, AbstractPhenotypeData, AbstractPedigreeData
export ValidationResult
export GenomicProException, DataValidationError, DimensionMismatchError, ConvergenceError

# Export interface functions
export n_samples, n_markers, sample_ids, marker_ids, allele_frequencies
export validate

# Include submodules
include("types.jl")
include("interfaces.jl")
include("exceptions.jl")
include("validation.jl")

end # module Core
