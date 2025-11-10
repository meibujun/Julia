module GenomicPro

using LinearAlgebra
using SparseArrays
using Statistics
using DataFrames
using CSV
# using CUDA # Commented out for now to avoid dependency issues if not available
using Lux
using Optimisers
using Zygote
using Random
using ArgParse
# using Distributions # Commented out to reduce dependencies for now

# Abstract types
include("GenomicProCore/types.jl")

# Core Data Structures and I/O
include("GenomicProCore/twobit.jl")
include("GenomicProCore/dosage_matrix.jl")
include("GenomicProCore/phenotype.jl")
include("GenomicProCore/pedigree.jl")

# QC and Preprocessing
include("GenomicProCore/filters.jl")
include("GenomicProCore/validators.jl")

# Prediction pipelines
include("GenomicProCore/grm.jl")
include("GenomicProCore/solvers.jl")
include("GenomicProCore/variance_components.jl")
include("GenomicProCore/ssgblup.jl")

# GPU Acceleration
# include("GenomicProCore/grm_kernels.jl")
# include("GenomicProCore/pcg_gpu.jl")

# Bayesian Methods
include("GenomicProCore/bayesian_framework.jl")
include("GenomicProCore/bayesr_mcmc.jl")
include("GenomicProCore/bayesrc.jl")
include("GenomicProCore/mcmc_diagnostics.jl")

# Deep Learning
include("GenomicProCore/deep_learning.jl")

# Multi-omics
# include("GenomicProMultiOmics/GenomicProMultiOmics.jl")

# Production features
include("model_zoo.jl")
include("cli.jl")

# Exports
export AbstractGenomicData, AbstractGenotypeData, AbstractPhenotypeData, AbstractPedigreeData
export TwoBitGenotypes, DosageMatrix, PhenotypeData, PedigreeData
export read_phenotypes, read_pedigree
export AbstractQCFilter, MissingRateFilter, MAFFilter, HWEFilter
export QCPipeline, apply_qc, apply_filter
export AbstractValidator, MendelianConsistencyValidator, PopulationStratificationDetector, validate
export compute_grm
export solve_gblup, solve_ssgblup
export estimate_variance_components
export BayesRModel, BayesRCModel, run_bayesr_mcmc, run_bayesrc_mcmc
export DeepGBLUPModel, train_deep_gblup!, predict_deep_gblup
export load_model
export main

end # module
