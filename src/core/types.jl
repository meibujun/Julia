# OrthogonalGenomics.jl - Core Type Definitions
# International-standard type system for genomic prediction

using DataFrames
using Distributions
using SparseArrays

# --- Core Data Structures ---

struct GenotypeData{T<:Real, M<:AbstractMatrix{T}}
    genotypes::M
    allele_freq::Vector{Float64}
    maf_filter::Float64
    call_rate::Vector{Float64}
    n_individuals::Int
    n_markers::Int
    marker_info::DataFrame
    sample_info::DataFrame
    ploidy::Int
end

struct Population{G<:GenotypeData}
    genotype_data::G
    phenotypes::Matrix{Float64}
    environments::Matrix{Float64}
    generation::Int
    pedigree::Union{Any, Nothing} # Replace Any with Pedigree later
    true_genetic_values::Union{Any, Nothing} # Replace Any with GeneticValues later
    metadata::Dict{Symbol, Any}
end

struct GRMSet{T<:AbstractMatrix{Float64}}
    G::T
    D::T
    G_AA::T
    G_AD::T
    G_DD::T
    eigendecomp::Union{Any, Nothing} # Replace Any with EigenDecomposition later
    trace_ratios::NamedTuple
    generation::Int
    allele_freq_used::Vector{Float64}
end

# --- Model & Results Structures ---

struct VarianceComponents
    estimates::NamedTuple
    standard_errors::NamedTuple
    h²::Float64
    H²::Float64
    h²_se::Float64
    H²_se::Float64
    log_likelihood::Float64
    convergence::Any # Replace Any with ConvergenceInfo later
    proportion_variance::NamedTuple
end

abstract type PredictionModel end
abstract type LinearModel <: PredictionModel end

struct OGBLUP{G<:GRMSet, V<:VarianceComponents} <: LinearModel
    grms::G
    var_comp::V
    fixed_effects::Matrix{Float64}
    fixed_coef::Matrix{Float64}
    random_effects::Dict{Symbol, Matrix{Float64}}
    fitted_values::Matrix{Float64}
    residuals::Matrix{Float64}
    model_info::Any # Replace Any with ModelInfo later
end

struct ReactionNormGBLUP{T<:AbstractMatrix{Float64}} <: LinearModel
    G::T
    fixed_effects::Matrix{Float64}
    fixed_coef::Vector{Float64}
    intercepts::Matrix{Float64}
    slopes::Matrix{Float64}
    var_intercept::Vector{Float64}
    var_slope::Vector{Float64}
    cov_int_slope::Vector{Float64}
    residual_var::Vector{Float64}
    environments_range::Tuple{Float64, Float64}
    model_info::Any # Replace Any with ModelInfo later
end


# --- Simulation Sub-types ---

abstract type GeneticArchitecture end
struct ComplexArchitecture <: GeneticArchitecture
    n_qtl::Int
    qtl_distribution::Symbol
    h2_additive::Float64
    h2_dominance::Float64
    h2_epistasis_aa::Float64
    n_epistatic_pairs::Int
end

abstract type MatingSystem end
struct RandomMating <: MatingSystem end

abstract type SelectionScheme end
struct TruncationSelection <: SelectionScheme
    proportion::Float64
end

abstract type EnvironmentalModel end
struct NoEnvironment <: EnvironmentalModel end
struct GxEModel <: EnvironmentalModel
    gxe_variance::Float64
    n_environments::Int
end

@kwdef struct SimulationParameters
    n_individuals::Int = 1000
    n_generations::Int = 10
    n_chromosomes::Int = 30
    chromosome_lengths::Vector{Float64} = fill(100.0, 30)
    n_markers::Int = 50000
    maf_distribution::Distribution = Beta(0.4, 0.4)
    architecture::GeneticArchitecture = ComplexArchitecture(n_qtl=100, qtl_distribution=:normal, h2_additive=0.3, h2_dominance=0.05, h2_epistasis_aa=0.1, n_epistatic_pairs=50)
    mating_system::MatingSystem = RandomMating()
    selection_scheme::SelectionScheme = TruncationSelection(0.2)
    environmental_model::EnvironmentalModel = NoEnvironment()
    recombination_model::Symbol = :haldane
    mutation_rate::Float64 = 1e-8
end

# Placeholder for types to be fully defined later
struct Pedigree end
struct GeneticValues end
struct EigenDecomposition end
struct ConvergenceInfo end
struct ModelInfo end

export GenotypeData, Population, GRMSet, VarianceComponents, PredictionModel,
       LinearModel, OGBLUP, ReactionNormGBLUP, SimulationParameters,
       GeneticArchitecture, ComplexArchitecture, MatingSystem, RandomMating,
       SelectionScheme, TruncationSelection, EnvironmentalModel, NoEnvironment, GxEModel
