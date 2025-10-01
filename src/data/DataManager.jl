module DataManager

using CSV
using DataFrames
using LinearAlgebra
using Statistics
using StatsBase
using Tables
using Random

export DataRepository, load_phenotypes, load_pedigree, load_genotypes, load_environment,
       load_multiomics!, integrate_data!, validate_data, compute_relationship_matrix

const DEFAULT_ID = :animal

mutable struct DataRepository
    phenotypes::DataFrame
    pedigrees::DataFrame
    genotypes::DataFrame
    environments::DataFrame
    omics::Dict{Symbol,DataFrame}
    cache::Dict{Symbol,Any}

    function DataRepository(; phenotypes = DataFrame(), pedigrees = DataFrame(),
            genotypes = DataFrame(), environments = DataFrame(),
            omics = Dict{Symbol,DataFrame}(), cache = Dict{Symbol,Any}())
        return new(phenotypes, pedigrees, genotypes, environments, omics, cache)
    end
end

function _normalize_id_column!(df::DataFrame; id_col::Symbol = DEFAULT_ID, source::AbstractString = "")
    haskey(df, id_col) || throw(ArgumentError("$(source) data must contain a column named $(id_col)"))
    df[!, id_col] = convert(Vector{Union{Missing,String}}, string.(df[!, id_col]))
    return df
end

function _load_table(path_or_io; kwargs...)
    table = DataFrame(CSV.File(path_or_io; kwargs...))
    return table
end

function load_phenotypes(path_or_io; id_col::Symbol = DEFAULT_ID, trait_cols::Vector{Symbol} = Symbol[],
        fixed_cols::Vector{Symbol} = Symbol[], kwargs...)
    df = _load_table(path_or_io; kwargs...)
    _normalize_id_column!(df; id_col, source = "Phenotype")
    if !isempty(trait_cols)
        missing_cols = setdiff(trait_cols, Symbol.(names(df)))
        !isempty(missing_cols) && throw(ArgumentError("Missing trait columns: $(missing_cols)"))
    end
    df
end

function load_pedigree(path_or_io; id_col::Symbol = DEFAULT_ID, sire_col::Symbol = :sire,
        dam_col::Symbol = :dam, kwargs...)
    df = _load_table(path_or_io; kwargs...)
    for col in (id_col, sire_col, dam_col)
        haskey(df, col) || throw(ArgumentError("Pedigree data must contain column $(col)"))
        df[!, col] = convert(Vector{Union{Missing,String}}, string.(df[!, col]))
    end
    rename!(df, Dict(id_col => DEFAULT_ID, sire_col => :sire, dam_col => :dam))
    df
end

function load_genotypes(path_or_io; id_col::Symbol = DEFAULT_ID, kwargs...)
    df = _load_table(path_or_io; kwargs...)
    _normalize_id_column!(df; id_col, source = "Genotype")
    rename!(df, Dict(id_col => DEFAULT_ID))
    for name in names(df)
        name == DEFAULT_ID && continue
        df[!, name] = coalesce.(df[!, name], 0)
        df[!, name] = Float64.(df[!, name])
    end
    df
end

function load_environment(path_or_io; id_col::Symbol = DEFAULT_ID, kwargs...)
    df = _load_table(path_or_io; kwargs...)
    _normalize_id_column!(df; id_col, source = "Environment")
    rename!(df, Dict(id_col => DEFAULT_ID))
    df
end

function load_multiomics!(repo::DataRepository, name::Symbol, path_or_io; id_col::Symbol = DEFAULT_ID, kwargs...)
    df = _load_table(path_or_io; kwargs...)
    _normalize_id_column!(df; id_col, source = string(name))
    rename!(df, Dict(id_col => DEFAULT_ID))
    repo.omics[name] = df
    repo.cache = Dict{Symbol,Any}()
    return repo
end

function integrate_data!(repo::DataRepository; how::Symbol = :left)
    joined = repo.phenotypes
    for (label, table) in pairs(repo.omics)
        joined = _generic_join(joined, table; how, suffix = "_" * String(label))
    end
    if !isempty(repo.environments)
        joined = _generic_join(joined, repo.environments; how, suffix = "_env")
    end
    repo.phenotypes = joined
    repo.cache = Dict{Symbol,Any}()
    return repo
end

function _generic_join(left::DataFrame, right::DataFrame; how::Symbol, suffix::AbstractString)
    isempty(right) && return left
    makeunique = true
    joiner = how == :inner ? innerjoin : leftjoin
    joined = joiner(left, right; on = DEFAULT_ID, makeunique)
    rename!(joined, n -> occursin("_right", String(n)) ? Symbol(replace(String(n), "_right" => suffix)) : n)
    return joined
end

function validate_data(repo::DataRepository)
    report = Dict{Symbol,Any}()
    ph = repo.phenotypes
    report[:phenotype_rows] = nrow(ph)
    report[:phenotype_missing_ids] = count(ismissing, ph[!, DEFAULT_ID])
    report[:duplicate_animals] = nrow(ph) - length(unique(skipmissing(ph[!, DEFAULT_ID])))
    if !isempty(repo.genotypes)
        geno_ids = Set(skipmissing(repo.genotypes[!, DEFAULT_ID]))
        ph_ids = Set(skipmissing(ph[!, DEFAULT_ID]))
        report[:genotyped_not_phenotyped] = length(setdiff(geno_ids, ph_ids))
        report[:phenotyped_not_genotyped] = length(setdiff(ph_ids, geno_ids))
    end
    if !isempty(repo.pedigrees)
        ped_ids = Set(skipmissing(repo.pedigrees[!, DEFAULT_ID]))
        report[:pedigree_missing] = length(setdiff(ped_ids, Set(skipmissing(ph[!, DEFAULT_ID]))))
        report[:pedigree_cycles] = _detect_pedigree_cycles(repo.pedigrees)
    end
    return report
end

function _detect_pedigree_cycles(pedigree::DataFrame)
    parents = Dict{String,Tuple{Union{String,Nothing},Union{String,Nothing}}}()
    for row in eachrow(pedigree)
        parents[String(row[:animal])] = (
            ismissing(row[:sire]) ? nothing : String(row[:sire]),
            ismissing(row[:dam]) ? nothing : String(row[:dam])
        )
    end
    visited = Dict{String,Symbol}()
    function dfs(node::String)
        status = get(visited, node, :unseen)
        status == :active && return true
        status == :done && return false
        visited[node] = :active
        sire, dam = get(parents, node, (nothing, nothing))
        if sire !== nothing && dfs(sire)
            return true
        end
        if dam !== nothing && dfs(dam)
            return true
        end
        visited[node] = :done
        return false
    end
    for node in keys(parents)
        dfs(node) && return true
    end
    return false
end

function compute_relationship_matrix(repo::DataRepository; type::Symbol = :genomic,
        ids = nothing, method::Symbol = :vanraden, regularisation::Float64 = 1e-6)
    ids === nothing && (ids = collect(skipmissing(repo.phenotypes[!, DEFAULT_ID])))
    ids = String.(ids)
    if type == :genomic
        return _compute_genomic(repo, ids; method, regularisation)
    elseif type == :pedigree
        return _compute_pedigree(repo, ids; regularisation)
    elseif type == :single_step
        return _compute_single_step(repo, ids; method, regularisation)
    else
        throw(ArgumentError("Unknown relationship matrix type $(type)"))
    end
end

function _compute_genomic(repo::DataRepository, ids::Vector{String}; method::Symbol, regularisation::Float64)
    isempty(repo.genotypes) && throw(ArgumentError("Genotype table is empty"))
    geno = repo.genotypes
    markers = setdiff(names(geno), [DEFAULT_ID])
    isempty(markers) && throw(ArgumentError("No genotype markers present"))
    idx_map = Dict{String,Int}()
    for (i, row) in enumerate(eachrow(geno))
        id = row[DEFAULT_ID]
        ismissing(id) && continue
        idx_map[String(id)] = i
    end
    missing = String[]
    M = zeros(Float64, length(ids), length(markers))
    for (i, id) in enumerate(ids)
        pos = get(idx_map, id, 0)
        pos == 0 && push!(missing, id)
        pos == 0 && continue
        for (j, marker) in enumerate(markers)
            M[i, j] = Float64(geno[pos, marker])
        end
    end
    !isempty(missing) && throw(ArgumentError("Missing genotype records for: $(join(missing, ", "))"))
    p = mean.(eachcol(M) ./ 2)
    W = similar(M)
    for j in axes(M, 2)
        W[:, j] .= M[:, j] .- 2p[j]
    end
    denom = 2 * sum(p .* (1 .- p))
    denom ≈ 0 && (denom = size(M, 2))
    G = (W * transpose(W)) ./ denom
    G += I * regularisation
    return Symmetric(G), ids
end

function _compute_pedigree(repo::DataRepository, ids::Vector{String}; regularisation::Float64)
    isempty(repo.pedigrees) && throw(ArgumentError("Pedigree table is empty"))
    pedigree = repo.pedigrees
    pedmap = Dict{String,Tuple{Union{String,Nothing},Union{String,Nothing}}}()
    for row in eachrow(pedigree)
        pedmap[String(row[:animal])] = (
            ismissing(row[:sire]) ? nothing : String(row[:sire]),
            ismissing(row[:dam]) ? nothing : String(row[:dam])
        )
    end
    ancestors = Set{String}(ids)
    queue = collect(ids)
    while !isempty(queue)
        current = pop!(queue)
        sire, dam = get(pedmap, current, (nothing, nothing))
        if sire !== nothing && !(sire in ancestors)
            push!(ancestors, sire); push!(queue, sire)
        end
        if dam !== nothing && !(dam in ancestors)
            push!(ancestors, dam); push!(queue, dam)
        end
    end
    animals = sort!(collect(ancestors); by = x -> (_pedigree_depth(x, pedmap), x))
    index = Dict(animal => i for (i, animal) in enumerate(animals))
    n = length(animals)
    A = zeros(Float64, n, n)
    for (animal, i) in pairs(index)
        sire, dam = get(pedmap, animal, (nothing, nothing))
        sidx = sire === nothing ? 0 : get(index, sire, 0)
        didx = dam === nothing ? 0 : get(index, dam, 0)
        for j in 1:i-1
            val = 0.5 * ((sidx == 0 ? 0.0 : A[sidx, j]) + (didx == 0 ? 0.0 : A[didx, j]))
            A[i, j] = val
            A[j, i] = val
        end
        if sidx == 0 && didx == 0
            A[i, i] = 1.0
        else
            cross = 0.0
            if sidx != 0 && didx != 0
                cross = A[sidx, didx]
            end
            A[i, i] = 1.0 + 0.5 * cross
        end
    end
    selector = [index[id] for id in ids]
    sub = A[selector, selector]
    sub += I * regularisation
    return Symmetric(sub), ids
end

function _pedigree_depth(id::String, pedmap::Dict{String,Tuple{Union{String,Nothing},Union{String,Nothing}}},
        memo::Dict{String,Int} = Dict{String,Int}(), stack::Set{String} = Set{String}())
    if haskey(memo, id)
        return memo[id]
    end
    if id in stack
        return 0
    end
    push!(stack, id)
    sire, dam = get(pedmap, id, (nothing, nothing))
    depth_sire = sire === nothing ? 0 : _pedigree_depth(sire, pedmap, memo, stack)
    depth_dam = dam === nothing ? 0 : _pedigree_depth(dam, pedmap, memo, stack)
    depth = max(depth_sire, depth_dam) + (sire === nothing && dam === nothing ? 0 : 1)
    memo[id] = depth
    delete!(stack, id)
    return depth
end

function _compute_single_step(repo::DataRepository, ids::Vector{String}; method::Symbol, regularisation::Float64)
    A_full, all_ids = _compute_full_pedigree(repo)
    selector = [findfirst(==(id), all_ids) for id in ids]
    A = A_full[selector, selector]
    geno_ids = collect(skipmissing(repo.genotypes[!, DEFAULT_ID]))
    isempty(geno_ids) && throw(ArgumentError("Cannot compute single-step matrix without genotypes"))
    G, g_ids = _compute_genomic(repo, String.(geno_ids); method, regularisation)
    idx_map = Dict(id => findfirst(==(id), all_ids) for id in all_ids)
    genotyped_indices = [idx_map[id] for id in g_ids]
    A22 = A_full[genotyped_indices, genotyped_indices]
    A_inv = inv(Matrix(A_full))
    A22_inv = inv(Matrix(Symmetric(A22)))
    G_inv = inv(Matrix(Symmetric(G)))
    H_inv = copy(A_inv)
    for (i_local, i_global) in enumerate(genotyped_indices)
        for (j_local, j_global) in enumerate(genotyped_indices)
            H_inv[i_global, j_global] += G_inv[i_local, j_local] - A22_inv[i_local, j_local]
        end
    end
    H = Symmetric(inv(H_inv))
    return H[selector, selector], ids
end

function _compute_full_pedigree(repo::DataRepository)
    isempty(repo.pedigrees) && throw(ArgumentError("Pedigree table is empty"))
    ped = repo.pedigrees
    ids = String.(unique(skipmissing(ped[!, DEFAULT_ID])))
    parents = union(ids, String.(skipmissing(ped[!, :sire])), String.(skipmissing(ped[!, :dam])))
    ids = sort!(collect(parents); by = identity)
    matrix, _ = _compute_pedigree(repo, ids; regularisation = 0.0)
    return matrix, ids
end

end
