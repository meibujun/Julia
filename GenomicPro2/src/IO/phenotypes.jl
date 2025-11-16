"""
Phenotype data I/O operations.

Supports CSV format with flexible column specifications.
"""

"""
    PhenotypeData

Container for phenotype information.

# Fields
- `sample_ids::Vector{String}`: Individual IDs
- `trait_names::Vector{String}`: Trait names
- `values::Matrix{Float64}`: Phenotype values (samples × traits)
- `covariates::Dict{String, Vector}`: Optional covariates
- `metadata::Dict{Symbol, Any}`: Additional metadata
"""
mutable struct PhenotypeData
    sample_ids::Vector{String}
    trait_names::Vector{String}
    values::Matrix{Float64}
    covariates::Dict{String, Vector}
    metadata::Dict{Symbol, Any}

    function PhenotypeData(sample_ids::Vector{String},
                          trait_names::Vector{String},
                          values::Matrix{Float64};
                          covariates::Dict{String, Vector} = Dict{String, Vector}(),
                          metadata::Dict{Symbol, Any} = Dict{Symbol, Any}())
        n_samples = length(sample_ids)
        n_traits = length(trait_names)

        if size(values) != (n_samples, n_traits)
            throw(DimensionMismatchError(
                "values",
                (n_samples, n_traits),
                size(values)
            ))
        end

        # Validate covariates
        for (name, vals) in covariates
            if length(vals) != n_samples
                throw(DimensionMismatchError(
                    "covariate $name",
                    n_samples,
                    length(vals)
                ))
            end
        end

        return new(sample_ids, trait_names, values, covariates, metadata)
    end
end

# Interface implementations
n_samples(pheno::PhenotypeData) = length(pheno.sample_ids)
n_traits(pheno::PhenotypeData) = length(pheno.trait_names)
sample_ids(pheno::PhenotypeData) = pheno.sample_ids

"""
    Base.getindex(pheno::PhenotypeData, i::Int, j::Int)

Get phenotype value for sample i, trait j.
"""
Base.getindex(pheno::PhenotypeData, i::Int, j::Int) = pheno.values[i, j]

"""
    Base.getindex(pheno::PhenotypeData, ::Colon, j::Int)

Get all phenotype values for trait j.
"""
Base.getindex(pheno::PhenotypeData, ::Colon, j::Int) = pheno.values[:, j]

"""
    Base.show(io::IO, pheno::PhenotypeData)

Display summary of phenotype data.
"""
function Base.show(io::IO, pheno::PhenotypeData)
    println(io, "PhenotypeData:")
    println(io, "  Samples: $(n_samples(pheno))")
    println(io, "  Traits: $(n_traits(pheno))")
    println(io, "  Trait names: ", join(pheno.trait_names, ", "))

    if !isempty(pheno.covariates)
        println(io, "  Covariates: ", join(keys(pheno.covariates), ", "))
    end

    # Show missing data summary
    n_missing = sum(isnan, pheno.values)
    missing_pct = 100 * n_missing / length(pheno.values)
    @printf(io, "  Missing: %d (%.2f%%)\n", n_missing, missing_pct)
end

"""
    read_phenotypes(file::String; id_col="ID", trait_cols=nothing, covariate_cols=String[],
                   delimiter=',', header=true, missing_values=["NA", ".", "-9"])

Read phenotype data from a delimited text file (CSV/TSV).

# Arguments
- `file::String`: Path to phenotype file
- `id_col::String`: Column name for sample IDs (default: "ID")
- `trait_cols`: Column name(s) for traits. If `nothing`, uses all numeric columns except ID and covariates
- `covariate_cols::Vector{String}`: Column names for covariates (default: empty)
- `delimiter::Char`: Field delimiter (default: ',')
- `header::Bool`: Whether file has header row (default: true)
- `missing_values::Vector{String}`: Values to treat as missing (default: ["NA", ".", "-9"])

# Returns
- `PhenotypeData`: Phenotype data structure

# Examples
```julia
# Read simple phenotype file
pheno = read_phenotypes("phenotypes.csv", id_col="IID", trait_cols="Yield")

# Read with multiple traits
pheno = read_phenotypes("phenotypes.csv", trait_cols=["Yield", "Height", "Weight"])

# Read with covariates
pheno = read_phenotypes("phenotypes.csv",
                       trait_cols="Yield",
                       covariate_cols=["Sex", "Location"])
```
"""
function read_phenotypes(file::String;
                        id_col::String = "ID",
                        trait_cols = nothing,
                        covariate_cols::Vector{String} = String[],
                        delimiter::Char = ',',
                        header::Bool = true,
                        missing_values::Vector{String} = ["NA", ".", "-9", ""])
    if !isfile(file)
        throw(FileFormatError("Phenotype file not found: $file", :phenotype, file))
    end

    # Read file
    lines = readlines(file)
    if isempty(lines)
        throw(FileFormatError("Phenotype file is empty", :phenotype, file))
    end

    # Parse header
    if !header
        throw(ArgumentError("Non-header phenotype files not yet supported"))
    end

    header_line = strip(lines[1])
    col_names = String[strip(x) for x in split(header_line, delimiter)]

    # Find ID column
    id_col_idx = findfirst(==(id_col), col_names)
    if id_col_idx === nothing
        throw(FileFormatError(
            "ID column '$id_col' not found in file. Available columns: $(join(col_names, ", "))",
            :phenotype,
            file
        ))
    end

    # Parse data lines
    sample_ids = String[]
    data_dict = Dict{String, Vector{Any}}()
    for col in col_names
        data_dict[col] = Any[]
    end

    for (line_num, line) in enumerate(lines[2:end])
        if isempty(strip(line))
            continue
        end

        parts = String[strip(x) for x in split(line, delimiter)]
        if length(parts) != length(col_names)
            @warn "Line $(line_num + 1) has $(length(parts)) columns, expected $(length(col_names)). Skipping."
            continue
        end

        # Store sample ID
        push!(sample_ids, parts[id_col_idx])

        # Store all columns
        for (i, col) in enumerate(col_names)
            val = parts[i]
            # Convert to float if possible
            if val in missing_values
                push!(data_dict[col], NaN)
            else
                try
                    push!(data_dict[col], parse(Float64, val))
                catch
                    # Keep as string
                    push!(data_dict[col], val)
                end
            end
        end
    end

    n_samples = length(sample_ids)

    # Determine trait columns
    if trait_cols === nothing
        # Use all numeric columns except ID and covariates
        excluded_cols = Set([id_col, covariate_cols...])
        trait_cols = String[]
        for col in col_names
            if col ∉ excluded_cols && all(x -> isa(x, Float64), data_dict[col])
                push!(trait_cols, col)
            end
        end
    elseif isa(trait_cols, String)
        trait_cols = [trait_cols]
    end

    if isempty(trait_cols)
        throw(FileFormatError("No trait columns found in file", :phenotype, file))
    end

    # Extract trait values
    n_traits = length(trait_cols)
    trait_values = Matrix{Float64}(undef, n_samples, n_traits)

    for (j, trait) in enumerate(trait_cols)
        if !haskey(data_dict, trait)
            throw(FileFormatError(
                "Trait column '$trait' not found in file. Available: $(join(col_names, ", "))",
                :phenotype,
                file
            ))
        end
        trait_values[:, j] = Float64.(data_dict[trait])
    end

    # Extract covariates
    covariates = Dict{String, Vector}()
    for cov in covariate_cols
        if !haskey(data_dict, cov)
            throw(FileFormatError(
                "Covariate column '$cov' not found in file. Available: $(join(col_names, ", "))",
                :phenotype,
                file
            ))
        end
        covariates[cov] = data_dict[cov]
    end

    # Create PhenotypeData
    pheno = PhenotypeData(
        sample_ids,
        trait_cols,
        trait_values;
        covariates = covariates,
        metadata = Dict(:source_file => file)
    )

    return pheno
end

"""
    write_phenotypes(file::String, pheno::PhenotypeData; delimiter=',', include_covariates=true)

Write phenotype data to a delimited text file.

# Arguments
- `file::String`: Output file path
- `pheno::PhenotypeData`: Phenotype data to write
- `delimiter::Char`: Field delimiter (default: ',')
- `include_covariates::Bool`: Whether to include covariate columns (default: true)

# Examples
```julia
write_phenotypes("output_phenotypes.csv", pheno)
write_phenotypes("output_phenotypes.tsv", pheno, delimiter='\\t')
```
"""
function write_phenotypes(file::String, pheno::PhenotypeData;
                         delimiter::Char = ',',
                         include_covariates::Bool = true)
    open(file, "w") do io
        # Write header
        header_cols = ["ID", pheno.trait_names...]
        if include_covariates && !isempty(pheno.covariates)
            append!(header_cols, collect(keys(pheno.covariates)))
        end
        println(io, join(header_cols, delimiter))

        # Write data
        for i in 1:n_samples(pheno)
            row = [pheno.sample_ids[i]]

            # Add trait values
            for j in 1:n_traits(pheno)
                val = pheno.values[i, j]
                if isnan(val)
                    push!(row, "NA")
                else
                    push!(row, string(val))
                end
            end

            # Add covariates
            if include_covariates && !isempty(pheno.covariates)
                for cov_name in keys(pheno.covariates)
                    push!(row, string(pheno.covariates[cov_name][i]))
                end
            end

            println(io, join(row, delimiter))
        end
    end
end

"""
    merge_genotype_phenotype(geno::CompactGenotypes, pheno::PhenotypeData; match_on=:sample_id)

Merge genotype and phenotype data, keeping only samples present in both datasets.

Returns: (geno_matched, pheno_matched, common_ids)
"""
function merge_genotype_phenotype(geno::CompactGenotypes, pheno::PhenotypeData;
                                 match_on::Symbol = :sample_id)
    geno_ids = Set(sample_ids(geno))
    pheno_ids = Set(sample_ids(pheno))

    # Find common samples
    common_ids = sort(collect(intersect(geno_ids, pheno_ids)))

    if isempty(common_ids)
        throw(DataValidationError(
            "No common samples found between genotype and phenotype data",
            :sample_ids,
            (length(geno_ids), length(pheno_ids))
        ))
    end

    # Get indices for common samples
    geno_indices = [findfirst(==(id), sample_ids(geno)) for id in common_ids]
    pheno_indices = [findfirst(==(id), sample_ids(pheno)) for id in common_ids]

    # Subset genotype data
    geno_matched = subset_samples(geno, geno_indices)

    # Subset phenotype data
    pheno_matched = PhenotypeData(
        pheno.sample_ids[pheno_indices],
        pheno.trait_names,
        pheno.values[pheno_indices, :];
        covariates = Dict(k => v[pheno_indices] for (k, v) in pheno.covariates),
        metadata = copy(pheno.metadata)
    )

    @info "Merged data" n_common=length(common_ids) n_geno_only=length(geno_ids)-length(common_ids) n_pheno_only=length(pheno_ids)-length(common_ids)

    return geno_matched, pheno_matched, common_ids
end
