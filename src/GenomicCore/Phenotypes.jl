module Phenotypes

import ..GenomicTypes
using DataFrames
using Statistics

export PhenotypeData, standardize!, get_trait, get_covariates

"""
    PhenotypeData

Container for phenotypes and covariates.
"""
struct PhenotypeData
    data::DataFrame
    trait_names::Vector{String}
    covariate_names::Vector{String}
    sample_id_col::String
end

"""
    get_trait(pheno::PhenotypeData, trait::String)

Extract a trait vector. Returns a Vector{Float64} with missing values handled (e.g., skipped or imputed, currently just returns raw vector with Missing type allowed, or we can enforce Float64 and error on missing).
For GWAS, we usually want complete cases or imputed.
Here we return `Vector{Union{Float64, Missing}}`.
"""
function get_trait(pheno::PhenotypeData, trait::String)
    if trait ∉ pheno.trait_names
        error("Trait $trait not found.")
    end
    return pheno.data[!, trait]
end

"""
    get_covariates(pheno::PhenotypeData)

Extract covariate matrix. Returns Matrix{Float64}.
Rows match the DataFrame.
"""
function get_covariates(pheno::PhenotypeData)
    if isempty(pheno.covariate_names)
        return Matrix{Float64}(undef, nrow(pheno.data), 0)
    end
    
    # Select columns
    df_cov = select(pheno.data, pheno.covariate_names)
    
    # Convert to Matrix
    # TODO: Handle categorical covariates (one-hot encoding)
    # For now, assume numeric
    return Matrix{Float64}(df_cov)
end

"""
    standardize!(pheno::PhenotypeData)

Standardize traits (mean=0, std=1).
"""
function standardize!(pheno::PhenotypeData)
    for trait in pheno.trait_names
        y = pheno.data[!, trait]
        # Handle missing values
        # We only standardize non-missing values
        valid_idx = .!ismissing.(y)
        if any(valid_idx)
            vals = collect(skipmissing(y))
            μ = mean(vals)
            σ = std(vals)
            if σ > 0
                pheno.data[valid_idx, trait] .= (y[valid_idx] .- μ) ./ σ
            end
        end
    end
end

end # module Phenotypes
