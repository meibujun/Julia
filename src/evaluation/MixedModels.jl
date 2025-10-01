module MixedModels

using LinearAlgebra
using DataFrames
using Statistics
using Random

import ..DataManager: DataRepository, compute_relationship_matrix
import ..ModelSpec: ModelSpec, RandomEffectSpec, find_trait

export run_evaluation, GeneticEvalResult

struct GeneticEvalResult
    trait::Symbol
    method::Symbol
    fixed_effects::DataFrame
    breeding_values::DataFrame
    variance_components::Dict{Symbol,Float64}
    converged::Bool
end

function Base.show(io::IO, result::GeneticEvalResult)
    println(io, "Genetic evaluation result for trait $(result.trait) via $(result.method)")
    println(io, "Variance components: $(result.variance_components)")
    println(io, "Top breeding values:")
    show(io, first(result.breeding_values, min(5, nrow(result.breeding_values))))
end

function run_evaluation(model::ModelSpec, repo::DataRepository; trait::Union{Symbol,AbstractString} = model.traits[1].name,
        method::Symbol = :BLUP, h2::Float64 = 0.3, residual_var = nothing, regularisation::Float64 = 1e-6)
    trait_symbol = Symbol(trait)
    trait_spec = find_trait(model, trait_symbol)
    df = repo.phenotypes
    haskey(df, trait_symbol) || throw(ArgumentError("Trait column $(trait_symbol) not found in phenotype table"))
    y = Float64.(coalesce.(df[!, trait_symbol], mean(skipmissing(df[!, trait_symbol]))))
    X, xnames = _build_fixed_matrix(df, model.fixed_effects)
    Z, animal_ids = _build_random_matrix(df, model.random_effects)
    if isempty(animal_ids)
        animal_ids = String.(coalesce.(df[!, :animal], ""))
        Z = Matrix{Float64}(I, length(y), length(y))
    end
    if method == :BLUP
        rel_type = :pedigree
    elseif method == :GBLUP
        rel_type = :genomic
    elseif method == :SSGBLUP
        rel_type = :single_step
    else
        throw(ArgumentError("Unknown evaluation method $(method)"))
    end
    relationship, rel_ids = compute_relationship_matrix(repo; type = rel_type, ids = animal_ids, regularisation)
    perm = [_find_index(id, rel_ids) for id in animal_ids]
    any(==(0), perm) && throw(ArgumentError("Relationship matrix missing some animal IDs"))
    relationship = Matrix(relationship)[perm, perm]
    σg, σe = _variance_components(y; h2, residual_var)
    λ = σe / max(σg, 1e-8)
    XtX = X' * X
    XtZ = X' * Z
    ZtX = Z' * X
    ZtZ = Z' * Z
    K_inv = inv(relationship)
    p = size(X, 2)
    q = size(Z, 2)
    C = zeros(Float64, p + q, p + q)
    C[1:p, 1:p] .= XtX
    C[1:p, p+1:end] .= XtZ
    C[p+1:end, 1:p] .= ZtX
    C[p+1:end, p+1:end] .= ZtZ .+ λ .* K_inv
    rhs = vcat(X' * y, Z' * y)
    sol = C \ rhs
    β = sol[1:p]
    u = sol[p+1:end]
    pev_block = diag(inv(C)[p+1:end, p+1:end]) .* σe
    reliability = 1 .- clamp.(pev_block ./ max(σg, eps()), 0, 1)
    fixed_df = DataFrame(term = xnames, estimate = β)
    bv_df = DataFrame(animal = animal_ids, breeding_value = u, reliability = reliability)
    variance = Dict(:genetic => σg, :residual => σe)
    return GeneticEvalResult(trait_symbol, method, fixed_df, bv_df, variance, true)
end

function _build_fixed_matrix(df::DataFrame, effects::Vector{Symbol})
    n = nrow(df)
    columns = Vector{Vector{Float64}}()
    names = Symbol[]
    push!(columns, ones(Float64, n))
    push!(names, :Intercept)
    for effect in effects
        haskey(df, effect) || throw(ArgumentError("Missing fixed effect column $(effect)"))
        col = df[!, effect]
        if eltype(col) <: Number
            vec = Float64.(coalesce.(col, mean(skipmissing(col))))
            push!(columns, vec)
            push!(names, effect)
        else
            levels = unique(skipmissing(col))
            length(levels) <= 1 && continue
            base = first(levels)
            for level in levels[2:end]
                vec = [(!ismissing(v) && v == level) ? 1.0 : 0.0 for v in col]
                push!(columns, vec)
                push!(names, Symbol(string(effect, "__", level)))
            end
        end
    end
    X = hcat(columns...)
    return X, names
end

function _build_random_matrix(df::DataFrame, random_effects::Vector{RandomEffectSpec})
    n = nrow(df)
    if isempty(random_effects)
        return zeros(Float64, n, 0), String[]
    end
    effect = random_effects[1]
    factor = effect.factor
    haskey(df, factor) || throw(ArgumentError("Random effect factor $(factor) missing from phenotype table"))
    ids = String.(coalesce.(df[!, factor], ""))
    uniq = unique(ids)
    mapping = Dict(id => i for (i, id) in enumerate(uniq))
    Z = zeros(Float64, n, length(uniq))
    for i in 1:n
        Z[i, mapping[ids[i]]] = 1.0
    end
    return Z, uniq
end

function _find_index(id::String, ids::Vector{String})
    pos = findfirst(==(id), ids)
    pos === nothing && return 0
    return pos
end

function _variance_components(y::Vector{Float64}; h2::Float64, residual_var)
    total_var = var(y)
    if residual_var === nothing
        σg = max(total_var * h2, 1e-6)
        σe = max(total_var - σg, 1e-6)
    else
        σe = residual_var
        denom = max(1 - h2, 1e-6)
        σg = σe * h2 / denom
    end
    return σg, σe
end

end
