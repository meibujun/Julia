module ModelSpec

using DataFrames
using StatsBase

export TraitSpec, RandomEffectSpec, ModelSpec, define_model, describe, find_trait

"""
    TraitSpec

用于描述单个性状的统计特性，包括分布类型与链接函数。
"""
struct TraitSpec
    name::Symbol
    distribution::Symbol
    link::Symbol
    options::Dict{Symbol,Any}
end

"""
    RandomEffectSpec

封装随机效应因子的配置，例如动物加性、母系等。
"""
struct RandomEffectSpec
    factor::Symbol
    kind::Symbol
    covariance::Symbol
    options::Dict{Symbol,Any}
end

"""
    ModelSpec

组合性状、固定效应与随机效应配置的核心结构。
"""
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

"""
    define_model(; traits, fixed = Symbol[], random = Tuple[], trait_types = Dict(),
        links = Dict(), options = Dict())

高层接口：依据用户输入快速构建 `ModelSpec`。
"""
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

"""
    describe(model)

生成包含模型结构摘要的多行字符串，便于日志或界面展示。
"""
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

"""
    find_trait(model, name)

从模型中检索指定性状，若不存在则抛出异常。
"""
function find_trait(model::ModelSpec, name::Symbol)
    for trait in model.traits
        trait.name == name && return trait
    end
    throw(ArgumentError("Trait $(name) not defined in model"))
end

end
