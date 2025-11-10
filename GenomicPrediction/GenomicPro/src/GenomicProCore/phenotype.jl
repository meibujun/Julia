# src/GenomicProData/phenotype.jl

using DataFrames

"""
    PhenotypeData <: AbstractPhenotypeData

Concrete implementation for storing and managing phenotypic trait data.

This structure uses a DataFrame internally to provide a flexible and powerful
way to handle phenotypic measurements, covariates, and metadata. It is designed
to integrate seamlessly with the Julia data ecosystem.

# Fields
- `table::DataFrame`: A DataFrame where rows represent individuals and columns
  represent traits, covariates, or identifiers.
- `sample_id_col::Symbol`: The name of the column in `table` that contains
  unique sample identifiers.
- `trait_cols::Vector{Symbol}`: A vector of column names corresponding to
  phenotypic traits.

# Examples
```julia
using DataFrames

# Create a DataFrame with phenotype data
df = DataFrame(
    AnimalID = ["ID001", "ID002", "ID003"],
    MilkYield = [30.5, 28.2, 33.0],
    FatPercent = [3.8, 4.1, 3.5],
    Age = [3, 4, 3]
)

# Create a PhenotypeData object
pheno_data = PhenotypeData(df, :AnimalID, [:MilkYield, :FatPercent])

# Get basic information
n_samples, n_traits = size(pheno_data)
trait_names = get_traits(pheno_data)
sample_ids = get_sample_ids(pheno_data)
```
"""
struct PhenotypeData <: AbstractPhenotypeData
    table::DataFrame
    sample_id_col::Symbol
    trait_cols::Vector{Symbol}

    function PhenotypeData(table::DataFrame, sample_id_col::Symbol, trait_cols::Vector{Symbol})
        # Validation
        @assert sample_id_col in names(table) "Sample ID column not found in DataFrame."
        @assert all(trait_col -> trait_col in names(table), trait_cols) "One or more trait columns not found in DataFrame."
        new(table, sample_id_col, trait_cols)
    end
end

# Core interface implementation
Base.size(pd::PhenotypeData) = (nrow(pd.table), length(pd.trait_cols))
get_sample_ids(pd::PhenotypeData) = pd.table[:, pd.sample_id_col]
get_traits(pd::PhenotypeData) = pd.trait_cols

function get_phenotype(pd::PhenotypeData, sample_id, trait::Symbol)
    row = pd.table[pd.table[:, pd.sample_id_col] .== sample_id, :]
    if nrow(row) == 0
        error("Sample ID '$sample_id' not found.")
    end
    return row[1, trait]
end

using CSV

"""
    read_phenotypes(filepath::String, sample_id_col::Symbol, trait_cols::Vector{Symbol})

Read phenotype data from a CSV file.

# Arguments
- `filepath::String`: Path to the CSV file.
- `sample_id_col::Symbol`: The name of the column containing sample IDs.
- `trait_cols::Vector{Symbol}`: A vector of column names for the traits.

# Returns
- `PhenotypeData`: A `PhenotypeData` object containing the data from the CSV file.
"""
function read_phenotypes(filepath::String, sample_id_col::Symbol, trait_cols::Vector{Symbol})
    df = CSV.read(filepath, DataFrame)
    return PhenotypeData(df, sample_id_col, trait_cols)
end
