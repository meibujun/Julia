module DynamicEpistasisGBLUP

# Import necessary packages
using CUDA
using CSV
using DataFrames
using Distributions
using JWAS
using LinearAlgebra
using Plots
using Random
using StatsBase
using XSim

# Export functions that will be part of the public API
export simulate_population, run_gblup, run_epistasis_gblup, evaluate_models

# Include other source files
include("simulation.jl")
include("gblup.jl")
include("gpu_acceleration.jl")
include("main.jl")

end # module
