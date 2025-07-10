"""
DynamicEpistasisGBLUP.jl - A cutting-edge Julia package for genomic prediction with dynamic orthogonal epistasis

This package implements the state-of-the-art Dynamic Orthogonal Epistasis framework for genomic prediction
in livestock breeding, with specific optimizations for sheep populations. It combines GPU acceleration,
advanced algorithms, and Julia's performance capabilities to handle 100K+ SNPs with 10K+ individuals.

Author: Advanced AI Implementation (via AI Agent)
License: MIT
Julia: 1.9+
"""
module DynamicEpistasisGBLUP

# Core dependencies
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using ProgressMeter
using DataStructures

# GPU and parallel computing
using CUDA
using KernelAbstractions
# using Distributed # This will be used within the DistributedComputing module
using SharedArrays

# Specialized genetic analysis packages
# using MixedModels # Consider if this is a true dependency or if implementing parts
# using StatsModels
# using GLM
# using Optim

# Data handling
using DataFrames
using CSV
# using JLD2 # For saving/loading, can be conditional or specific module
# using BSON

# Performance optimization
using LoopVectorization
using StaticArrays
# using StructArrays # Useful, check if used or can be introduced
# using FLoops # For multi-threading, can be conditional

# Type definitions for maximum performance
const Float = Float32  # Use Float32 for GPU efficiency
const GeneticValue = Float64  # High precision for genetic values

# Export main API
export
    # Data structures
    GenotypeMatrix, PhenotypeData, PopulationData,
    GeneticArchitecture, VarianceComponents, OrthogonalGBLUP, # QTLEffects, EpistaticEffects were not defined as structs

    # Core algorithms
    compute_grm!, compute_epistatic_grm!,
    orthogonal_epistasis_gblup, # dynamic_reml is likely internal to REML fitting

    # Simulation
    simulate_population, simulate_selection,

    # Utilities
    update_allele_frequencies!, # genomic_prediction was a function in prediction.jl
    benchmark_grm_computation, # from utils.jl in original, moved to top level exports
    run_demo # From demo script, useful to export for users

# Include submodules
include("types.jl")
include("utils.jl") # General utilities
include("gpu_kernels.jl")

include("simulation.jl")
include("grm_computation.jl")

include("epistasis_core.jl") # Renamed from epistasis.jl to avoid conflict with SparseEpistasis module name
include("reml.jl")
include("prediction.jl")

# Advanced modules
include("walsh_hadamard.jl")
include("noia_framework.jl")
include("symmetric_polynomials.jl")
include("augmented_aireml.jl")
include("sparse_epistasis.jl") # This is a module itself

# Application modules
include("multivariate_extension.jl")
include("breeding_optimization.jl")

# Optional/Utility modules
include("distributed_computing.jl")
include("gpu_optimization.jl")
include("validation.jl")
include("visualization.jl")


# Demo function (if included directly in package)
# The demo script logic will be primarily in the main module for now, or moved to examples/
function run_demo()
    println("=== Dynamic Orthogonal Epistasis GBLUP Demo ===\n")

    # Set parameters
    n_individuals = 100 # Reduced for quick demo
    n_snps = 500    # Reduced for quick demo
    n_generations = 2 # Reduced for quick demo

    # 1. Simulate base population
    println("1. Simulating Mongolian sheep population...")
    base_population = simulate_population(
        n_individuals = n_individuals,
        n_snps = n_snps,
        n_qtl_additive = 10, # Reduced
        n_qtl_epistatic = 10, # Reduced
        h2_narrow = 0.30,
        h2_broad = 0.40
    )
    println("   Population size: $n_individuals")
    println("   Number of SNPs: $n_snps")
    println("   Narrow-sense h²: 0.30")
    println("   Broad-sense h²: 0.40")

    # 2. Benchmark GRM computation (on smaller scale for demo)
    println("\n2. Benchmarking GRM computation (demo scale)...")
    if CUDA.functional()
        benchmark_grm_computation(50, 200) # Even smaller for benchmark demo
    else
        println("   Skipping GPU benchmark as CUDA is not functional.")
    end

    # 3. Run selection simulation
    println("\n3. Simulating $n_generations generations of selection...")
    populations, models = simulate_selection(
        base_population,
        n_generations = n_generations,
        selection_intensity = 0.20,
        update_model_frequency = 1
    )
    println("   Simulation complete. Generated $(length(populations)-1) new generations.")

    # 4. Cross-validation comparison (simplified for demo)
    println("\n4. Running simplified cross-validation...")
    using .Validation # Make sure Validation module is accessible

    # Use only the base population for a quick CV demo
    cv_results_df = cross_validation(
        populations[1], # Base population
        n_folds = 2, # Reduced folds
        include_epistasis = true
    )
    println("   Cross-validation results for base population:")
    println(cv_results_df)


    # 5. Analyze results (basic summary for demo)
    println("\n5. Analyzing results (demo summary)...")
    # For a real analysis, you'd use the `compare_models` function from Validation
    # This is a placeholder for the demo
    if !isempty(models)
        final_model = models[end]
        println("   Final Model Variance Components:")
        println("     σ²_a: ", final_model.variance.σ²_a)
        println("     σ²_aa: ", final_model.variance.σ²_aa)
        println("     σ²_e: ", final_model.variance.σ²_e)
        println("     h²: ", final_model.variance.h²)
        println("     H²: ", final_model.variance.H²)
    end


    # 6. Create visualization (simple plot for demo)
    println("\n6. Creating accuracy plot (demo)...")
    # A more complex plot would use `plot_generation_accuracy` from Visualization module
    # This requires more structured CV results across generations.
    # For now, just indicate where it would go or make a placeholder.
    if CUDA.functional() && !isempty(cv_results_df.accuracy)
        using Plots
        if Plots.backend() != Plots.GRBackend()
             try Plots.gr() catch; println("GR backend for Plots.jl not available.") end
        end
        if Plots.backend() == Plots.GRBackend()
            acc_plot = plot(1:nrow(cv_results_df), cv_results_df.accuracy, title="CV Accuracy (Base Pop)", marker=:circle, xlabel="Fold", ylabel="Accuracy")
            try
                savefig(acc_plot, "demo_accuracy_base_pop.png")
                println("   Saved demo_accuracy_base_pop.png")
            catch e
                println("   Could not save demo plot: $e")
            end
        else
            println("   Skipping demo plot generation as GR backend is not active.")
        end

    else
        println("   Skipping accuracy plot (CUDA not functional or no CV results).")
    end

    println("\n=== Demo completed successfully! ===")

    return populations, models, cv_results_df # Return DataFrame for consistency
end


# Re-export from submodules if needed, or ensure they export their own APIs
# For example, if SparseEpistasis.detect_sparse_interactions should be part of the main API:
# export detect_sparse_interactions # This would require `using .SparseEpistasis: detect_sparse_interactions` or similar

end # module DynamicEpistasisGBLUP
