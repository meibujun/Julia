module PedigreeTools

export load_pedigree, convert_pedigree!, restore_pedigree, validate_pedigree, extract_lineage, plot_pedigree, PedigreeIDMapper

using DataFrames
using CSV
using XLSX
using Dates
using Graphs
using GraphPlot
using Statistics
using Distributed
# CUDA is optional; load conditionally to avoid errors if not installed
try
    using CUDA
catch
    @warn "CUDA.jl not installed or CUDA not available; GPU features disabled"
end

# ==================== Data Loading ====================

"""
    load_pedigree(path::AbstractString; format::Symbol = :auto)

Load pedigree data from CSV, TSV/TXT, or Excel file into a DataFrame.
The format is inferred from file extension when `format = :auto`.
"""
function load_pedigree(path::AbstractString; format::Symbol = :auto)
    fmt = format
    if fmt == :auto
        ext = lowercase(splitext(path)[2])
        fmt = ext in [".csv"] ? :csv :
              ext in [".txt", ".tsv"] ? :txt :
              ext in [".xlsx", ".xls"] ? :excel :
              error("Unsupported file extension: $ext")
    end
    if fmt == :csv
        return CSV.read(path, DataFrame)
    elseif fmt == :txt
        return CSV.read(path, DataFrame; delim='\t')
    elseif fmt == :excel
        data = XLSX.readtable(path, 1)  # first sheet
        return DataFrame(data...)
    else
        error("Unknown format $fmt")
    end
end

# ==================== Core Data Structure ====================

mutable struct PedigreeIDMapper
    to_int::Dict{Any,Int}
    to_orig::Dict{Int,Any}
    missing_val::Int
    id_type::Symbol
end

function PedigreeIDMapper(missing_val::Int=0)
    PedigreeIDMapper(Dict{Any,Int}(), Dict{Int,Any}(), missing_val, :unknown)
end

# ==================== Utility Functions ====================

function standardize_column_names!(df::DataFrame)
    original = names(df)
    clean = Symbol[]
    for n in original
        s = replace(strip(string(n)), r"\s+" => "_")
        push!(clean, Symbol(s))
    end
    if original != clean
        rename!(df, Dict(zip(original, clean)))
    end
    return df
end

function detect_column(df::DataFrame, patterns::Vector{Symbol})
    cols = names(df)
    for p in patterns
        p = Symbol(p)
        if p in cols
            return p
        end
        pl = lowercase(string(p))
        for c in cols
            if lowercase(string(c)) == pl
                return c
            end
        end
        for c in cols
            if occursin(pl, lowercase(string(c)))
                return c
            end
        end
    end
    return nothing
end

"""
    clean_additional_columns!(df::DataFrame)

Attempt to standardize auxiliary columns like sex, birthdate, and breed.
Sex is normalized to "M", "F", or missing. Birthdates are parsed to `Date`.
Breed names are uppercased.
"""
function clean_additional_columns!(df::DataFrame)
    # Sex
    sex_col = detect_column(df, [:sex, :Sex, :gender, :性别])
    if !isnothing(sex_col)
        df[!, sex_col] = Vector{Union{String,Missing}}(df[!, sex_col])
        for i in 1:nrow(df)
            v = df[i, sex_col]
            if ismissing(v)
                df[i, sex_col] = missing
            else
                s = uppercase(strip(string(v)))
                df[i, sex_col] = s in ["M","MALE","♂"] ? "M" :
                                 s in ["F","FEMALE","♀"] ? "F" : missing
            end
        end
    end
    # Birthdate
    birth_col = detect_column(df, [:birthdate, :BirthDate, :dob, :出生日期])
    if !isnothing(birth_col)
        df[!, birth_col] = Vector{Union{Date,Missing}}(undef, nrow(df))
        for i in 1:nrow(df)
            v = df[i, birth_col]
            df[i, birth_col] = try
                ismissing(v) ? missing : Date(v)
            catch
                try
                    Date(string(v))
                catch
                    missing
                end
            end
        end
    end
    # Breed
    breed_col = detect_column(df, [:breed, :Breed, :品种])
    if !isnothing(breed_col)
        df[!, breed_col] = [ismissing(v) ? missing : uppercase(strip(string(v))) for v in df[!, breed_col]]
    end
end

# ==================== ID Mapping ====================

function analyze_id_type(ids)
    u = unique(filter(!ismissing, ids))
    if isempty(u); return :empty; end
    all_numeric = true; has_strings = false
    for id in u
        if isa(id, AbstractString)
            has_strings = true
            if isnothing(tryparse(Int, id))
                all_numeric = false
                break
            end
        elseif !isa(id, Integer)
            all_numeric = false
        end
    end
    if all_numeric && !has_strings
        :numeric
    elseif all_numeric && has_strings
        :string_numeric
    else
        :mixed
    end
end

function create_id_mappings!(mapper::PedigreeIDMapper, ids)
    u = unique(filter(!ismissing, ids))
    if isempty(u)
        mapper.id_type = :empty
        return
    end
    mapper.id_type = analyze_id_type(u)
    used = Set{Int}()
    for id in u
        val = if isa(id, Integer)
            id
        elseif isa(id, AbstractString)
            p = tryparse(Int, id)
            if !isnothing(p) && !(p in used)
                p
            else
                nothing
            end
        else
            nothing
        end
        if isnothing(val) || val in used
            val = 1
            while val in used
                val += 1
            end
        end
        mapper.to_int[id] = val
        mapper.to_orig[val] = id
        push!(used, val)
    end
end

function map_forward(mapper::PedigreeIDMapper, values; parallel::Symbol=:none)
    n = length(values)
    res = Vector{Int}(undef, n)
    if parallel == :threads
        Threads.@threads for i in 1:n
            v = values[i]
            res[i] = _map_one(mapper, v)
        end
    elseif parallel == :processes && nprocs() > 1
        res = collect(pmap(v -> _map_one(mapper, v), values))
    else
        for i in 1:n
            res[i] = _map_one(mapper, values[i])
        end
    end
    res
end

function _map_one(mapper::PedigreeIDMapper, v)
    if ismissing(v)
        mapper.missing_val
    elseif v == 0 || v == "0"
        mapper.missing_val
    elseif haskey(mapper.to_int, v)
        mapper.to_int[v]
    else
        mapper.missing_val
    end
end

function apply_forward_mapping!(df::DataFrame, col::Symbol, mapper::PedigreeIDMapper; parallel::Symbol=:none)
    if col in names(df)
        df[!, col] = map_forward(mapper, df[!, col]; parallel)
    end
end

function apply_reverse_mapping(df::DataFrame, col::Symbol, mapper::PedigreeIDMapper)
    if !(col in names(df)); return nothing; end
    n = nrow(df)
    if mapper.id_type == :numeric
        out = Vector{Union{Int,Missing}}(undef, n)
        for i in 1:n
            v = df[i,col]
            out[i] = v == mapper.missing_val ? missing : get(mapper.to_orig, v, v)
        end
    else
        out = Vector{Union{String,Missing}}(undef, n)
        for i in 1:n
            v = df[i,col]
            if v == mapper.missing_val
                out[i] = missing
            else
                out[i] = string(get(mapper.to_orig, v, v))
            end
        end
    end
    out
end

# ==================== Main API ====================

"""
    convert_pedigree!(df::DataFrame; missing_value::Int=0, parallel::Symbol=:none, verbose::Bool=false)

Convert pedigree IDs to integers, cleaning auxiliary columns.
Supports optional parallel mapping using threads or processes.
"""
function convert_pedigree!(df::DataFrame; missing_value::Int=0, parallel::Symbol=:none, verbose::Bool=false)
    standardize_column_names!(df)
    clean_additional_columns!(df)
    id_col = detect_column(df, [:ID,:id,:animal_id,:个体])
    id_col === nothing && error("Cannot find ID column")
    sire_col = detect_column(df, [:sire,:Sire,:father,:父亲])
    dam_col = detect_column(df, [:dam,:Dam,:mother,:母亲])
    mapper = PedigreeIDMapper(missing_value)
    create_id_mappings!(mapper, df[!, id_col])
    apply_forward_mapping!(df, id_col, mapper; parallel)
    if !isnothing(sire_col); apply_forward_mapping!(df, sire_col, mapper; parallel); end
    if !isnothing(dam_col); apply_forward_mapping!(df, dam_col, mapper; parallel); end
    if verbose
        println("Converted $(nrow(df)) records. ID type: $(mapper.id_type)")
    end
    return df, mapper
end

"""
    restore_pedigree(df::DataFrame, mapper::PedigreeIDMapper)

Return a new DataFrame with original IDs restored.
"""
function restore_pedigree(df::DataFrame, mapper::PedigreeIDMapper)
    out = DataFrame()
    for col in names(df)
        if col in [:ID,:sire,:dam]
            val = apply_reverse_mapping(df, col, mapper)
            if !isnothing(val); out[!,col] = val; end
        else
            out[!,col] = copy(df[!,col])
        end
    end
    out
end

# ==================== Validation and Analysis ====================

"""
    validate_pedigree(df::DataFrame)

Return a dictionary with validation results: cycles, isolated nodes, and missing ancestors.
"""
function validate_pedigree(df::DataFrame)
    id_col = detect_column(df, [:ID,:id])
    sire_col = detect_column(df, [:sire])
    dam_col = detect_column(df, [:dam])
    ids = Set(df[!, id_col])
    g = DiGraph()
    add_vertices!(g, length(ids))
    id_to_idx = Dict(id => i for (i,id) in enumerate(ids))
    missing_parents = Set{Any}()
    for row in eachrow(df)
        child = row[id_col]; cidx = id_to_idx[child]
        for (pcol) in (sire_col, dam_col)
            if pcol !== nothing
                p = row[pcol]
                if ismissing(p) || p == 0
                    continue
                elseif p in ids
                    add_edge!(g, cidx, id_to_idx[p])
                else
                    push!(missing_parents, p)
                end
            end
        end
    end
    has_cycles = !isdag(g)
    isolated = [k for (k,v) in id_to_idx if degree(g, v) == 0]
    return Dict(:has_cycles=>has_cycles, :isolated=>isolated, :missing_ancestors=>collect(missing_parents))
end

"""
    extract_lineage(df::DataFrame, id; generations::Int=3)

Return a DataFrame with ancestors of `id` up to `generations`.
"""
function extract_lineage(df::DataFrame, id; generations::Int=3)
    id_col = detect_column(df, [:ID,:id])
    sire_col = detect_column(df, [:sire])
    dam_col = detect_column(df, [:dam])
    lookup = Dict(row[id_col] => row for row in eachrow(df))
    result = DataFrame(ID=String[], generation=Int[])
    function recurse(current, gen)
        gen>generations && return
        if haskey(lookup, current)
            push!(result, (string(current), gen))
            row = lookup[current]
            if !isnothing(sire_col) && !ismissing(row[sire_col]) && row[sire_col] != 0
                recurse(row[sire_col], gen+1)
            end
            if !isnothing(dam_col) && !ismissing(row[dam_col]) && row[dam_col] != 0
                recurse(row[dam_col], gen+1)
            end
        end
    end
    recurse(id, 0)
    return result
end

"""
    plot_pedigree(df::DataFrame; id=nothing, generations=2)

Visualize the pedigree graph. When `id` is provided, show its ancestors up to `generations`.
"""
function plot_pedigree(df::DataFrame; id=nothing, generations=2)
    subdf = df
    if id !== nothing
        ids = extract_lineage(df, id; generations=generations).ID
        subdf = df[in.(df[!, detect_column(df, [:ID])], Ref(ids)), :]
    end
    id_col = detect_column(subdf, [:ID])
    sire_col = detect_column(subdf, [:sire])
    dam_col = detect_column(subdf, [:dam])
    ids = Set(subdf[!, id_col])
    g = DiGraph(length(ids))
    id_to_idx = Dict(id => i for (i,id) in enumerate(ids))
    for row in eachrow(subdf)
        c = id_to_idx[row[id_col]]
        for pcol in (sire_col, dam_col)
            if pcol !== nothing
                p = row[pcol]
                if p in ids
                    add_edge!(g, c, id_to_idx[p])
                end
            end
        end
    end
    labels = collect(string.(ids))
    gplot(g, nodelabel=labels)
end

end # module
