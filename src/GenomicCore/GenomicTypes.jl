module GenomicTypes

export AbstractGenomicModel, AbstractGenomicData, AbstractGenotypeData
export AbstractSolver, AbstractAlgorithm

"""
Abstract base type for all genomic models (Linear, Bayesian, Deep Learning).
"""
abstract type AbstractGenomicModel end

"""
Abstract base type for all genomic data containers.
"""
abstract type AbstractGenomicData end

"""
Abstract base type for genotype data (In-memory, Memory-mapped, Sparse).
"""
abstract type AbstractGenotypeData <: AbstractGenomicData end

"""
Abstract base type for solvers (PCG, MCMC, Newton-Raphson).
"""
abstract type AbstractSolver end

"""
Abstract base type for algorithms/methods (GWAS, GBLUP, BayesR).
"""
abstract type AbstractAlgorithm end

end # module Types
