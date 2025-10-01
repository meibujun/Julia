module ModelSpec

using DataFrames
using StatsBase

export TraitSpec, RandomEffectSpec, ModelSpec, define_model, describe, find_trait

struct TraitSpec
    name::Symbol
    distribution::Symbol
    link::Symbol
    options::Dict{Symbol,Any}
end

struct RandomEffectSpec
    factor::Symbol
    kind::Symbol
    covariance::Symbol
    options::Dict{Symbol,Any}
end

struct ModelSpec
    traits::Vector{TraitSpec}
    fixed_effects::Vector{Symbol}
    random_effects::Vector{RandomEffectSpec}
    options::Dict{Symbol,Any}
end

function ModelSpec(traits::Vector{TraitSpec}; fixed_effects = Symbol[],
        random_effects = RandomEffectSpec[], options = Dict{Symbol,Any}())
    return new(traits, Symbol.(fixed_effects), random_effects, options)
end

function define_model(; traits::Union{Vector{Symbol},Vector{AbstractString}},
        fixed::Union{Vector{Symbol},Vector{AbstractString}} = Symbol[],
        random = Vector{Tuple{<:AbstractString,Symbol}}(), trait_types = Dict{Symbol,Symbol}(),
        links = Dict{Symbol,Symbol}(), options = Dict{Symbol,Any}())
    trait_specs = TraitSpec[]
    for t in traits
        name = Symbol(t)
        dist = get(trait_types, name, :gaussian)
        link = get(links, name, dist == :binary ? :logit : :identity)
        push!(trait_specs, TraitSpec(name, dist, link, Dict{Symbol,Any}()))
    end
    random_specs = RandomEffectSpec[]
    for entry in random
        if entry isa RandomEffectSpec
            push!(random_specs, entry)
        else
            factor, kind = entry
            push!(random_specs, RandomEffectSpec(Symbol(factor), kind, :identity, Dict{Symbol,Any}()))
        end
    end
    return ModelSpec(trait_specs; fixed_effects = Symbol.(fixed), random_effects = random_specs, options)
end

function describe(model::ModelSpec)
    lines = String[]
    push!(lines, "Traits: " * join(string.(t.name for t in model.traits), ", "))
    push!(lines, "Fixed effects: " * (isempty(model.fixed_effects) ? "(none)" : join(string.(model.fixed_effects), ", ")))
    if isempty(model.random_effects)
        push!(lines, "Random effects: (none)")
    else
        for (i, re) in enumerate(model.random_effects)
            push!(lines, "Random $(i): factor=$(re.factor), kind=$(re.kind), covariance=$(re.covariance)")
        end
    end
    return join(lines, "\n")
end

function find_trait(model::ModelSpec, name::Symbol)
    for trait in model.traits
        trait.name == name && return trait
    end
    throw(ArgumentError("Trait $(name) not defined in model"))
end

end
