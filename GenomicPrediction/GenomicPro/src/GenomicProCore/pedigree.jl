# src/GenomicProData/pedigree.jl

using DataFrames

"""
    PedigreeData <: AbstractPedigreeData

Concrete implementation for storing and managing pedigree information.

This structure uses a DataFrame to store the pedigree table, which includes
columns for the individual, sire, and dam.

# Fields
- `table::DataFrame`: A DataFrame with columns for individual, sire, and dam IDs.
- `id_col::Symbol`: The name of the column for the individual's ID.
- `sire_col::Symbol`: The name of the column for the sire's ID.
- `dam_col::Symbol`: The name of the column for the dam's ID.

# Examples
```julia
using DataFrames

# Create a DataFrame with pedigree data
df = DataFrame(
    Animal = ["ID003", "ID004", "ID005"],
    Sire = ["ID001", "ID001", "ID002"],
    Dam = ["ID002", "ID003", "ID004"]
)

# Create a PedigreeData object
ped_data = PedigreeData(df, :Animal, :Sire, :Dam)
```
"""
struct PedigreeData <: AbstractPedigreeData
    table::DataFrame
    id_col::Symbol
    sire_col::Symbol
    dam_col::Symbol

    function PedigreeData(table::DataFrame, id_col::Symbol, sire_col::Symbol, dam_col::Symbol)
        @assert id_col in names(table) "Individual ID column not found."
        @assert sire_col in names(table) "Sire ID column not found."
        @assert dam_col in names(table) "Dam ID column not found."
        new(table, id_col, sire_col, dam_col)
    end
end

# Core interface implementation
Base.size(pd::PedigreeData) = (nrow(pd.table), 3)
get_sample_ids(pd::PedigreeData) = pd.table[:, pd.id_col]

using CSV, SparseArrays

"""
    read_pedigree(filepath::String, id_col::Symbol, sire_col::Symbol, dam_col::Symbol)

Read pedigree data from a CSV file.
...
"""
function read_pedigree(filepath::String, id_col::Symbol, sire_col::Symbol, dam_col::Symbol)
    df = CSV.read(filepath, DataFrame)
    return PedigreeData(df, id_col, sire_col, dam_col)
end

"""
    compute_A_inverse(ped::PedigreeData)

Compute the inverse of the numerator relationship matrix (A) directly from pedigree.
This method is much more efficient than forming A and then inverting it.

# Returns
- `SparseMatrixCSC{Float64,Int}`: The sparse inverse of A.
"""
function compute_A_inverse(ped::PedigreeData)
    # This is a simplified implementation of Henderson's method.
    n = nrow(ped.table)
    id_map = Dict(ped.table[i, ped.id_col] => i for i in 1:n)

    A_inv = spzeros(n, n)

    for i in 1:n
        sire = ped.table[i, ped.sire_col]
        dam = ped.table[i, ped.dam_col]

        sire_idx = get(id_map, sire, 0)
        dam_idx = get(id_map, dam, 0)

        # Contribution from individual
        A_inv[i, i] += 1.0

        # Contribution from parents
        if sire_idx != 0
            A_inv[i, sire_idx] -= 0.5
            A_inv[sire_idx, i] -= 0.5
            A_inv[sire_idx, sire_idx] += 0.25
        end

        if dam_idx != 0
            A_inv[i, dam_idx] -= 0.5
            A_inv[dam_idx, i] -= 0.5
            A_inv[dam_idx, dam_idx] += 0.25
        end

        if sire_idx != 0 && dam_idx != 0
            A_inv[sire_idx, dam_idx] += 0.25
            A_inv[dam_idx, sire_idx] += 0.25
        end
    end

    return A_inv
end
