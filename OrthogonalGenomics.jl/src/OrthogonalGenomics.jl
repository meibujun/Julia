module OrthogonalGenomics

# Standard library imports
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Distributed
using SharedArrays
using Printf
using Dates
using Logging

# External dependencies
using Distributions
using StatsBase
using DataFrames
using CSV
using Optim
using ProgressMeter
using MLBase
using BSON
using JLD2
using CovarianceEstimation
using BlockDiagonals
using IterativeSolvers
using Preconditioners

# Set up logging
const logger = ConsoleLogger(stderr, Logging.Info)
global_logger(logger)

# Version information
const PACKAGE_VERSION = v"1.0.0"
const MIN_JULIA_VERSION = v"1.6.0"

# Verify Julia version
if VERSION < MIN_JULIA_VERSION
    error("OrthogonalGenomics requires Julia $MIN_JULIA_VERSION or later")
end

# Include core modules
include("core/types.jl")
include("core/encoding.jl")
include("core/validation.jl")
include("core/constants.jl")

# Include genomics modules
include("genomics/grm_construction.jl")
include("genomics/variance_components.jl")
include("genomics/qc.jl")

# Include model modules
include("models/ogblup.jl")
include("models/reaction_norm.jl")

# Include breeding modules
include("breeding/mating.jl")

# Include simulation modules
include("simulation/population.jl")

# Package-level configuration
struct Config
    num_threads::Int
    use_gpu::Bool
    precision::Type{<:AbstractFloat}
    temp_dir::String
    log_level::LogLevel
    parallel_backend::Symbol  # :threads, :distributed, :cuda
end

const DEFAULT_CONFIG = Config(
    Threads.nthreads(),
    false,
    Float64,
    tempdir(),
    Logging.Info,
    :threads
)

# Global configuration
const CONFIG = Ref(DEFAULT_CONFIG)

"""
    set_config!(; kwargs...)

Set global configuration options for OrthogonalGenomics.

# Keywords
- `num_threads::Int`: Number of threads to use
- `use_gpu::Bool`: Enable GPU acceleration
- `precision::Type`: Floating point precision (Float32/Float64)
- `temp_dir::String`: Temporary directory for intermediate files
- `log_level::LogLevel`: Logging verbosity
- `parallel_backend::Symbol`: Parallel backend (:threads, :distributed, :cuda)
"""
function set_config!(; kwargs...)
    current = CONFIG[]
    new_config = Config(
        get(kwargs, :num_threads, current.num_threads),
        get(kwargs, :use_gpu, current.use_gpu),
        get(kwargs, :precision, current.precision),
        get(kwargs, :temp_dir, current.temp_dir),
        get(kwargs, :log_level, current.log_level),
        get(kwargs, :parallel_backend, current.parallel_backend)
    )
    CONFIG[] = new_config
    global_logger(ConsoleLogger(stderr, new_config.log_level))
    return new_config
end

# High-level API functions

"""
    genomic_evaluation(genotypes, phenotypes; kwargs...)

Perform complete genomic evaluation with automatic model selection.

# Arguments
- `genotypes`: Genotype matrix or GenotypeData object
- `phenotypes`: Phenotype vector/matrix

# Keywords
- `model::Symbol = :auto`: Model type (:ogblup, :additive, :bayesian, :auto)
- `traits::Vector{Symbol}`: Trait names for multi-trait analysis
- `environments::Matrix`: Environmental covariates for G×E
- `include_dominance::Bool = true`: Include dominance effects
- `include_epistasis::Bool = true`: Include epistatic effects
- `cross_validation::Bool = true`: Perform cross-validation
- `cv_folds::Int = 5`: Number of CV folds
- `report::Bool = true`: Generate evaluation report

# Returns
- `results::EvaluationResults`: Complete evaluation results
"""
function genomic_evaluation(genotypes, phenotypes;
                           model::Symbol = :auto,
                           traits::Vector{Symbol} = Symbol[],
                           environments::Union{Matrix, Nothing} = nothing,
                           include_dominance::Bool = true,
                           include_epistasis::Bool = true,
                           cross_validation::Bool = true,
                           cv_folds::Int = 5,
                           report::Bool = true)

    @info "Starting genomic evaluation pipeline"

    # Input validation and conversion
    geno_data = prepare_genotype_data(genotypes)
    pheno_data = prepare_phenotype_data(phenotypes, traits)

    # Quality control
    @info "Performing quality control"
    geno_data, pheno_data = quality_control(geno_data, pheno_data)

    # Population structure assessment
    pop_structure = assess_population_structure(geno_data)

    # Model selection if auto
    if model == :auto
        model = select_optimal_model(geno_data, pheno_data, pop_structure)
        @info "Automatically selected model: $model"
    end

    # Build population object
    population = Population(
        geno_data,
        pheno_data.values,
        isnothing(environments) ? zeros(size(pheno_data.values, 1), 0) : environments,
        0,  # Current generation
        nothing,  # No pedigree
        nothing,  # No true values
        Dict(:structure => pop_structure)
    )

    # Fit selected model
    fitted_model = fit_model(population, model;
                            include_dominance = include_dominance,
                            include_epistasis = include_epistasis)

    # Cross-validation if requested
    cv_results = nothing
    if cross_validation
        @info "Performing $cv_folds-fold cross-validation"
        cv_results = cross_validate(population, typeof(fitted_model);
                                   k_folds = cv_folds)
    end

    # Generate comprehensive results
    results = EvaluationResults(
        model = fitted_model,
        predictions = predict_gebv(fitted_model, population),
        variance_components = fitted_model.var_comp,
        accuracy = cv_results,
        metadata = Dict(
            :date => now(),
            :package_version => PACKAGE_VERSION,
            :model_type => model,
            :n_individuals => geno_data.n_individuals,
            :n_markers => geno_data.n_markers
        )
    )

    # Generate report if requested
    if report
        generate_evaluation_report(results, "genomic_evaluation_report.html")
    end

    return results
end

"""
    breeding_optimization(population, selection_target; kwargs...)

Optimize breeding program with advanced genetic models.

# Arguments
- `population::Population`: Current breeding population
- `selection_target::SelectionTarget`: Breeding objectives

# Keywords
- `n_generations::Int = 10`: Number of generations to simulate
- `mating_strategy::Symbol = :optimal`: Mating strategy
- `maintain_diversity::Bool = true`: Diversity constraints
- `use_genomic_mating::Bool = true`: Use genomic information for mating

# Returns
- `program::BreedingProgram`: Optimized breeding program
"""
function breeding_optimization(population::Population,
                             selection_target::SelectionTarget;
                             n_generations::Int = 10,
                             mating_strategy::Symbol = :optimal,
                             maintain_diversity::Bool = true,
                             use_genomic_mating::Bool = true)

    @info "Optimizing breeding program for $n_generations generations"

    # Fit comprehensive genetic model
    model = fit_ogblup(population;
                      include_dominance = true,
                      include_epistasis = true)

    # Initialize breeding program
    program = BreedingProgram(
        base_population = population,
        genetic_model = model,
        selection_target = selection_target,
        constraints = maintain_diversity ? DiversityConstraints() : NoConstraints()
    )

    # Optimize selection and mating for each generation
    for gen in 1:n_generations
        @info "Planning generation $gen"

        # Selection decisions
        selected = optimize_selection(program, population, model)

        # Mating decisions
        if use_genomic_mating
            mating_plan = optimize_genomic_mating(
                model, population, selected;
                strategy = mating_strategy,
                constraints = program.constraints
            )
        else
            mating_plan = random_mating(selected)
        end

        # Predict next generation
        next_gen = predict_offspring(population, mating_plan, model)

        # Update program
        update_breeding_program!(program, selected, mating_plan, next_gen)

        # Simulate actual next generation if needed
        population = simulate_next_generation(population, mating_plan)
    end

    return program
end


# Package exports
export
    # Main API functions
    genomic_evaluation, breeding_optimization,

    # Core types
    GenotypeData, Population, GeneticValues, Pedigree, GRMSet,
    VarianceComponents, ConvergenceInfo, ModelInfo,

    # Models
    OGBLUP, ReactionNormGBLUP,
    fit_ogblup, fit_reaction_norm,

    # Prediction
    predict_gebv, predict_phenotype, predict_offspring,

    # Variance components
    estimate_variance_components, estimate_heritability,

    # Cross-validation
    cross_validate, CrossValidationResult,

    # Breeding functions
    optimal_mating, estimate_heterosis, optimize_selection,
    MatingPlan,

    # Simulation
    simulate_population, simulate_selection,
    SimulationParameters,

    # I/O
    read_plink, read_vcf, write_results,

    # Utilities
    compute_grm, quality_control, set_config!


# Package initialization
function __init__()
    # Set up parallel workers if available
    if Threads.nthreads() > 1
        @info "OrthogonalGenomics initialized with $(Threads.nthreads()) threads"
    end

    # Check for GPU availability
    gpu_available = false # check_gpu_availability()
    if gpu_available
        @info "GPU acceleration available"
    end

    # Set random seed for reproducibility
    Random.seed!(42)

    # Print welcome message
    printstyled("OrthogonalGenomics.jl v$PACKAGE_VERSION loaded\n", color=:green)
    println("Advanced genomic prediction with dynamic orthogonal epistasis")
end

end # module
