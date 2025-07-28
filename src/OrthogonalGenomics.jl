# OrthogonalGenomics.jl - Main Module

module OrthogonalGenomics

# Standard library imports
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Distributed

# External dependencies
using Distributions
using StatsBase
using DataFrames
using CSV
using Optim
using ProgressMeter
using JuMP
using GLPK

# Core modules
include("core/types.jl")
include("core/constants.jl")
include("core/encoding.jl")
include("core/validation.jl")

# Genomics modules
include("genomics/grm_construction.jl")
include("genomics/variance_components.jl")
include("genomics/qc.jl")

# Model modules
include("models/ogblup.jl")
include("models/reaction_norm.jl")

# Breeding modules
include("breeding/mating.jl")

# Simulation modules
include("simulation/population.jl")

# I/O modules
include("io/plink.jl")
include("io/vcf.jl")

# Utility modules
include("utils/statistics.jl")
include("utils/matrix_ops.jl")
include("utils/helpers.jl")

# Visualization modules
include("visualization/plots.jl")
include("visualization/reports.jl")


# Export main functions and types
using .CoreTypes
export GenotypeData, Population, GRMSet, OGBLUP, SimulationParameters, MatingPlan

using .Encoding
export noia_encode

using .GRMConstruction
export compute_grm_set

using .VarianceComponentEstimation
export estimate_variance_components

using .OGBLUPModel
export fit_ogblup, predict_gebv

using .PopulationSimulation
export simulate_population

using .QualityControl
export quality_control

using .MatingOptimization
export optimal_mating, MinimizeInbreeding, MaximizeGeneticGain


function __init__()
    @info "OrthogonalGenomics.jl loaded. Welcome!"
end

end # module
