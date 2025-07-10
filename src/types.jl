# ===== src/types.jl =====
"""
    DynamicEpistasisGBLUP.Types

This file defines core data structures used throughout the DynamicEpistasisGBLUP package.
These structures are designed to be type-stable and optimized for genomic computations,
including GPU acceleration where appropriate.
"""

# Ensure main module types are accessible if this is not a submodule itself
# using ..DynamicEpistasisGBLUP: Float, GeneticValue # If these are defined in the main module and needed here directly
# For now, assuming Float and GeneticValue are defined in the main module and this file is `include`d.

using CUDA, SparseArrays, DataFrames # Dependencies for this file

# Genotype representation using bit-packed storage (Example - not fully implemented in provided code)
# struct CompressedGenotype{N}
#     data::NTuple{N, UInt8}  # 2-bit encoding per SNP
# end

# GPU-optimized genotype matrix
"""
    GenotypeMatrix{T<:AbstractFloat}

A mutable struct to store genotype data, optimized for GPU computations.

# Fields
- `data::CuArray{T, 2}`: GPU array storing genotype values (e.g., 0, 1, 2 for allele counts, or standardized values). Dimensions are Individuals × SNPs.
- `missing_mask::CuSparseMatrixCSR{Bool, Int32}`: Sparse matrix indicating missing genotypes. `true` if data at `(i,j)` is missing. Uses CSR format for efficient row-wise operations if needed. `Int32` for indices is common for CUDA sparse matrices.
- `allele_freq::CuVector{T}`: GPU vector storing allele frequencies for each SNP. These are typically reference allele frequencies (p).
- `n_individuals::Int32`: Number of individuals (rows in `data`).
- `n_snps::Int32`: Number of SNPs (columns in `data`).
- `ploidy::Int8`: Ploidy of the organism (e.g., 2 for diploid).
"""
mutable struct GenotypeMatrix{T<:AbstractFloat}
    data::CuArray{T, 2}  # Individuals × SNPs
    missing_mask::CuSparseMatrixCSR{Bool, Int32} # Explicit Int32 for CuSparseMatrixCSR indices
    allele_freq::CuVector{T}
    n_individuals::Int32
    n_snps::Int32
    ploidy::Int8
end

# Phenotype data structure
"""
    PhenotypeData{T<:AbstractFloat}

Stores phenotype information for a set of individuals.

# Fields
- `values::Vector{T}`: Vector of phenotypic values for the trait(s) of interest. For multiple traits, this might be a matrix or a vector of vectors, though current usage implies a single vector for one trait analysis at a time by GBLUP core.
- `trait_names::Vector{Symbol}`: Names of the traits corresponding to the columns in `values` (if `values` becomes a matrix for multiple traits) or a single trait name.
- `fixed_effects::Union{Nothing, DataFrame}`: Optional DataFrame containing fixed effect covariates for each individual. Each row corresponds to an individual, columns are different fixed effects.
- `random_effects::Union{Nothing, DataFrame}`: Optional DataFrame for additional known random effects (e.g., contemporary groups not captured by pedigree/GRM).
"""
struct PhenotypeData{T<:AbstractFloat}
    values::Vector{T}
    trait_names::Vector{Symbol}
    fixed_effects::Union{Nothing, DataFrame}
    random_effects::Union{Nothing, DataFrame}
end

# Population data container
"""
    PopulationData{T<:AbstractFloat}

A container struct holding all relevant data for a population at a specific point in time (e.g., a generation).

# Fields
- `genotypes::GenotypeMatrix{T}`: Genotype data for the population.
- `phenotypes::PhenotypeData{T}`: Phenotype data for the population.
- `pedigree::Union{Nothing, SparseMatrixCSC{T, Int}}`: Optional pedigree information, typically as a sparse matrix representing the numerator relationship matrix (A). `Int` for indices.
- `generation::Int32`: Generation number or identifier for this population data.
- `metadata::Dict{Symbol, Any}`: Dictionary for storing any additional metadata, such as simulation parameters, true genetic values, or population descriptors.
"""
mutable struct PopulationData{T<:AbstractFloat}
    genotypes::GenotypeMatrix{T}
    phenotypes::PhenotypeData{T}
    pedigree::Union{Nothing, SparseMatrixCSC{T, Int}} # Explicit Int for SparseMatrixCSC indices
    generation::Int32
    metadata::Dict{Symbol, Any}
end

# Genetic architecture for simulation
"""
    GeneticArchitecture{T<:AbstractFloat}

Defines the true genetic architecture of traits for simulation purposes.
Specifies which SNPs are QTLs, their effect sizes, and interaction patterns.

# Fields
- `n_qtl_additive::Int32`: Number of QTLs with additive effects.
- `additive_qtl_actual_indices::Vector{Int32}`: Vector of actual column indices (in the genotype matrix) for SNPs that have additive effects. Length must match `n_qtl_additive`.
- `additive_effects::Vector{T}`: Vector of additive effect sizes, corresponding one-to-one with `additive_qtl_actual_indices`.
- `n_epistatic_pairs::Int32`: Number of locus pairs exhibiting epistatic interactions.
- `epistatic_pairs_actual_indices::Vector{Tuple{Int32, Int32}}`: Vector of tuples, where each tuple `(snp_idx1, snp_idx2)` contains the actual column indices of an interacting SNP pair. Length must match `n_epistatic_pairs`.
- `epistatic_effects::Vector{T}`: Vector of epistatic interaction effect sizes, corresponding one-to-one with `epistatic_pairs_actual_indices`.
- `h2_narrow_target::T`: The target narrow-sense heritability (due to additive effects) used for setting up the simulation.
- `h2_broad_target::T`: The target broad-sense heritability (due to all genetic effects) used for simulation setup.
"""
struct GeneticArchitecture{T<:AbstractFloat}
    # Additive effects components
    n_qtl_additive::Int32                   # Number of QTLs with additive effects
    additive_qtl_actual_indices::Vector{Int32} # Actual column indices of these additive QTLs in genotype matrix
    additive_effects::Vector{T}             # Additive effects corresponding to additive_qtl_actual_indices

    # Epistatic effects components
    n_epistatic_pairs::Int32                # Number of epistatic locus PAIRS
    epistatic_pairs_actual_indices::Vector{Tuple{Int32, Int32}} # Pairs of actual column SNP indices for epistatic interactions
    epistatic_effects::Vector{T}            # Epistatic effects corresponding to epistatic_pairs_actual_indices

    # Optional: Combined sorted list of all unique SNPs involved in any effect (A or AA)
    # This was `qtl_positions` before. Can be derived if needed, or stored if frequently used.
    # all_unique_qtl_indices::Vector{Int32}

    # Simulation target parameters
    h2_narrow_target::T       # Target narrow-sense heritability for simulation setup
    h2_broad_target::T        # Target broad-sense heritability for simulation setup

    # Constructor to ensure consistency (example)
    function GeneticArchitecture(
        n_add::Int32, add_indices::Vector{Int32}, add_effects::Vector{T},
        n_epi_pairs::Int32, epi_pairs_indices::Vector{Tuple{Int32,Int32}}, epi_effects::Vector{T},
        h2n::T, h2b::T
    ) where T <: AbstractFloat
        if length(add_indices) != n_add || length(add_effects) != n_add
            error("Mismatch in additive QTL counts, indices, and effects lengths.")
        end
        if length(epi_pairs_indices) != n_epi_pairs || length(epi_effects) != n_epi_pairs
            error("Mismatch in epistatic pair counts, indices, and effects lengths.")
        end
        new{T}(n_add, add_indices, add_effects, n_epi_pairs, epi_pairs_indices, epi_effects, h2n, h2b)
    end
end

# Variance components structure
mutable struct VarianceComponents{T<:AbstractFloat}
    σ²_a::T      # Additive genetic variance
    σ²_aa::T     # Epistatic genetic variance
    σ²_e::T      # Residual variance
    σ²_p::T      # Phenotypic variance
    h²::T        # Narrow-sense heritability
    H²::T        # Broad-sense heritability

    # Default constructor
    VarianceComponents{T}(σ²_a, σ²_aa, σ²_e, σ²_p, h², H²) where T = new(σ²_a, σ²_aa, σ²_e, σ²_p, h², H²)
    # Constructor for initialization
    function VarianceComponents{T}(; σ²_a::T=T(0.0), σ²_aa::T=T(0.0), σ²_e::T=T(0.0), σ²_p::T=T(0.0), h²::T=T(0.0), H²::T=T(0.0)) where T <: AbstractFloat # Added T bound
        new{T}(σ²_a, σ²_aa, σ²_e, σ²_p, h², H²)
    end
end


# GBLUP model structure
"""
    OrthogonalGBLUP{T<:AbstractFloat}

Represents a fitted GBLUP model, potentially including orthogonal epistatic effects.
Stores the GRMs used, estimated variance components, and fixed effect estimates.

# Fields
- `G::CuArray{T,2}`: The additive Genomic Relationship Matrix (GRM) used for model fitting. Stored on GPU.
- `G_aa::Union{Nothing, CuArray{T,2}}`: The epistatic GRM (e.g., additive-by-additive). `Nothing` if epistasis was not included in the model. Stored on GPU if present.
- `variance::VarianceComponents{T}`: Struct holding the estimated variance components (σ²_a, σ²_aa, σ²_e, σ²_p, h², H²).
- `fixed_effects::Union{Nothing, Matrix{T}}`: Estimated coefficients for fixed effects included in the model (e.g., overall mean/intercept). Typically a column vector (Matrix with one column). Stored on CPU.
- `generation::Int32`: Identifier for the generation or dataset on which this model was trained.
"""
struct OrthogonalGBLUP{T<:AbstractFloat}
    G::CuArray{T, 2}       # Additive GRM
    G_aa::Union{Nothing, CuArray{T, 2}}    # Epistatic GRM, can be Nothing if not included
    variance::VarianceComponents{T}
    fixed_effects::Union{Nothing, Matrix{T}} # Store fixed effect estimates if any
    # population_allele_freq::Union{Nothing, CuVector{T}} # Consider adding this: allele freqs of training pop
    generation::Int32 # Generation this model was trained on/for
end

# If QTLEffects and EpistaticEffects were meant to be structs:
# struct QTLEffects{T<:AbstractFloat}
#   positions::Vector{Int32}
#   effects::Vector{T}
# end

# struct EpistaticEffects{T<:AbstractFloat}
#   pairs::Vector{Tuple{Int32, Int32}}
#   effects::Vector{T}
# end

# Export relevant types from this file if it were a module
# export GenotypeMatrix, PhenotypeData, PopulationData, GeneticArchitecture, VarianceComponents, OrthogonalGBLUP
