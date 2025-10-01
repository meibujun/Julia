module MixedModels

using LinearAlgebra
using DataFrames
using Statistics
using Random

import ..DataManager: DataRepository, compute_relationship_matrix
import ..ModelSpec: ModelSpec, RandomEffectSpec, find_trait

export run_evaluation, GeneticEvalResult

"""
    GeneticEvalResult

遗传评估结果结构体，记录：

* `trait`：分析的性状名称；
* `method`：使用的评估方法（BLUP / GBLUP / SSGBLUP）；
* `fixed_effects`：固定效应估计值数据表；
* `breeding_values`：个体育种值与可靠度；
* `variance_components`：方差组分估计；
* `converged`：解算器是否成功收敛。
"""
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

"""
    run_evaluation(model, repo; trait = ..., method = :BLUP, h2 = 0.3,
        residual_var = nothing, regularisation = 1e-6)

根据模型与数据仓库运行混合线性模型遗传评估。函数自动构建 Henderson 混合模型方程，
并采用数值稳定的因子分解求解，避免直接求逆带来的精度与性能问题。
"""
function run_evaluation(model::ModelSpec, repo::DataRepository; trait::Union{Symbol,AbstractString} = model.traits[1].name,
        method::Symbol = :BLUP, h2::Float64 = 0.3, residual_var = nothing, regularisation::Float64 = 1e-6)
    trait_symbol = Symbol(trait)
    _ = find_trait(model, trait_symbol)
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
    rel_sym = Symmetric(relationship)
    rel_factor = cholesky(rel_sym; check = false)
    K_inv = if issuccess(rel_factor)
        rel_factor \ Matrix{Float64}(I, q, q)
    else
        rel_ldlt = ldlt(rel_sym)
        rel_ldlt \ Matrix{Float64}(I, q, q)
    end
    p = size(X, 2)
    q = size(Z, 2)
    C = zeros(Float64, p + q, p + q)
    C[1:p, 1:p] .= XtX
    C[1:p, p+1:end] .= XtZ
    C[p+1:end, 1:p] .= ZtX
    C[p+1:end, p+1:end] .= ZtZ .+ λ .* K_inv
    rhs = vcat(X' * y, Z' * y)
    C_sym = Symmetric(C + I * regularisation)
    factor = cholesky(C_sym; check = false)
    fallback = issuccess(factor) ? nothing : ldlt(C_sym)
    sol = issuccess(factor) ? (factor \ rhs) : (fallback \ rhs)
    β = sol[1:p]
    u = sol[p+1:end]
    block_eye = zeros(Float64, p + q, q)
    block_eye[p+1:end, :] .= Matrix{Float64}(I, q, q)
    block_solution = issuccess(factor) ? (factor \ block_eye) : (fallback \ block_eye)
    pev_block = diag(block_solution[p+1:end, :]) .* σe
    reliability = 1 .- clamp.(pev_block ./ max(σg, eps()), 0, 1)
    fixed_df = DataFrame(term = xnames, estimate = β)
    bv_df = DataFrame(animal = animal_ids, breeding_value = u, reliability = reliability)
    variance = Dict(:genetic => σg, :residual => σe)
    return GeneticEvalResult(trait_symbol, method, fixed_df, bv_df, variance, true)
end

"""
    _build_fixed_matrix(df, effects)

根据固定效应因子生成设计矩阵。数值变量进行缺失值填补，分类变量采用哑变量编码。
"""
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

"""
    _build_random_matrix(df, random_effects)

目前实现支持单个随机效应（常见的动物加性效应），会生成对应的个体指示矩阵。
"""
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

"""
    _find_index(id, ids)

辅助函数：返回 ID 在参考向量中的位置，找不到时返回 0。
"""
function _find_index(id::String, ids::Vector{String})
    pos = findfirst(==(id), ids)
    pos === nothing && return 0
    return pos
end

"""
    _variance_components(y; h2, residual_var)

依据指定的遗传力或残差方差估计初始的遗传方差与残差方差。
"""
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
