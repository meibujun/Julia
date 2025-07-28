# Input validation and data integrity checks

module Validation

using Statistics
using LinearAlgebra
using DataFrames
using ..Constants
using ..CoreTypes

export validate_genotypes, validate_phenotypes, validate_grm,
       check_data_integrity, validate_model_inputs

function validate_genotypes(genotypes::AbstractMatrix;
                          ploidy::Int=2,
                          allow_missing::Bool=true)

    n, m = size(genotypes)

    if n == 0 || m == 0
        throw(ArgumentError("Genotype matrix cannot be empty"))
    end

    valid_values = Set(0:ploidy)
    if allow_missing
        push!(valid_values, MISSING_GENOTYPE)
    end

    for g in genotypes
        if !(g in valid_values)
            throw(ArgumentError("Invalid genotype value found: \$g"))
        end
    end

    return true
end

function validate_phenotypes(phenotypes::AbstractVecOrMat)
    if isempty(phenotypes)
        throw(ArgumentError("Phenotype data cannot be empty"))
    end
    if any(isinf.(phenotypes)) || any(isnan.(phenotypes))
        @warn "Phenotypes contain Inf or NaN values."
    end
    return true
end

function check_data_integrity(geno_data::GenotypeData, phenotypes::AbstractMatrix)
    if geno_data.n_individuals != size(phenotypes, 1)
        throw(DimensionMismatch("Genotype and phenotype data have different numbers of individuals."))
    end
    return true
end

function validate_model_inputs(; population::Population, kwargs...)
    validate_genotypes(population.genotype_data.genotypes)
    validate_phenotypes(population.phenotypes)
    check_data_integrity(population.genotype_data, population.phenotypes)
    return true
end
