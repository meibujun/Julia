# ============================================================================
# BLUP评估模块 - 混合模型方程求解器
# AnimalBreeding.jl
# ============================================================================

"""
    build_design_matrices(phenotypes, model, animal_map) -> (Matrix, Dict, Vector)

根据模型定义和表型数据构建设计矩阵。

# 返回
- `X::Matrix{Float64}`: 固定效应设计矩阵。
- `Z_dict::Dict{String, SparseMatrixCSC{Float64, Int}}`: 随机效应设计矩阵的字典。
- `y::Vector{Float64}`: 观测值向量。
"""
function build_design_matrices(phenotypes::DataFrame, model::ModelSpec, animal_map::Dict)
    @info "构建设计矩阵..."

    if isempty(model.traits)
        error("模型未指定任何性状。")
    end

    trait = model.traits[1]
    if !(trait in names(phenotypes))
        error("表型数据中不存在性状列 '$trait'。")
    end

    required_cols = Set{String}([trait, "animal"])
    for effect_name in model.fixed_effects
        push!(required_cols, effect_name)
    end

    for col in required_cols
        if !(col in names(phenotypes))
            error("表型数据缺少构建设计矩阵所需的列 '$col'。")
        end
    end

    mask = .!ismissing.(phenotypes[!, trait])
    for col in required_cols
        mask .&= .!ismissing.(phenotypes[!, col])
    end

    kept = count(mask)
    dropped = nrow(phenotypes) - kept
    if kept == 0
        error("构建设计矩阵所需的记录全部缺失，无法继续。")
    elseif dropped > 0
        @warn "因缺失值移除了 $dropped 条记录以构建设计矩阵。"
    end

    pheno_clean = phenotypes[mask, :]
    y = Float64.(pheno_clean[!, trait])
    n_obs = length(y)

    X_cols = [ones(Float64, n_obs)]
    for effect_name in model.fixed_effects
        column = pheno_clean[!, effect_name]
        value_type = eltype(skipmissing(column))
        if value_type <: Number
            push!(X_cols, Float64.(column))
        else
            levels = unique(column)
            if length(levels) > 1
                for level in levels[2:end]
                    indicators = column .== level
                    push!(X_cols, Float64.(indicators))
                end
            end
        end
    end
    X = hcat(X_cols...)

    Z_dict = Dict{String, SparseMatrixCSC{Float64, Int}}()
    n_total_animals = length(animal_map)

    for effect in model.random_effects
        if effect.type == :additive
            I_idx = Int[]
            J_idx = Int[]
            V_val = Float64[]
            for (row_idx, animal_id) in enumerate(pheno_clean.animal)
                col_idx = get(animal_map, animal_id, 0)
                if col_idx == 0
                    continue
                end
                push!(I_idx, row_idx)
                push!(J_idx, col_idx)
                push!(V_val, 1.0)
            end
            Z = sparse(I_idx, J_idx, V_val, n_obs, n_total_animals)
            Z_dict[effect.name] = Z
        else
            warn_msg = string("暂未实现随机效应类型 ", effect.type, "，将被忽略。")
            @warn warn_msg
        end
    end

    if isempty(Z_dict)
        @info "模型中未定义随机效应，仅构建固定效应矩阵。"
    else
        dims = first(values(Z_dict)) |> size
        finish_msg = string(
            "设计矩阵构建完成: X",
            size(X),
            ", Z",
            dims,
            ", y",
            length(y),
        )
        @info finish_msg
    end

    return X, Z_dict, y
end

const _RandomEffectBlock = NamedTuple{(:effect, :range), Tuple{RandomEffect, UnitRange{Int}}}

function _included_random_effects(model::ModelSpec,
                                  Z_dict::Dict{String,SparseMatrixCSC{Float64, Int}})
    included = RandomEffect[]
    skipped = String[]
    for effect in model.random_effects
        if haskey(Z_dict, effect.name)
            push!(included, effect)
        else
            push!(skipped, effect.name)
        end
    end
    if !isempty(skipped)
        @warn "以下随机效应缺少设计矩阵，将在MME中忽略: " * join(skipped, ", ")
    end
    return included
end

function _random_effect_blocks(n_fixed::Int,
                               effects::Vector{RandomEffect},
                               Z_dict::Dict{String,SparseMatrixCSC{Float64, Int}})
    blocks = _RandomEffectBlock[]
    current = n_fixed
    for effect in effects
        dim = size(Z_dict[effect.name], 2)
        range = current + 1:current + dim
        push!(blocks, (effect=effect, range=range))
        current += dim
    end
    return blocks
end

function _relationship_inverse_matrix(dm::DataManager)
    if !isnothing(dm.H_inv_matrix)
        return Matrix(dm.H_inv_matrix)
    elseif !isnothing(dm.A_inv_matrix)
        return Matrix(dm.A_inv_matrix)
    else
        error("模型需要A⁻¹或H⁻¹，但未在DataManager中计算。")
    end
end

"""
    setup_mme(X, Z_dict, y, dm, model, variances) -> (SparseMatrixCSC, Vector)

构建稀疏的混合模型方程 (MME) 系统。
"""
function setup_mme(X::Matrix{Float64}, Z_dict::Dict{String,SparseMatrixCSC{Float64,Int}},
                   y::Vector{Float64}, dm::DataManager, model::ModelSpec,
                   variances::Dict{String,Float64})
    @info "构建混合模型方程 (MME)..."

    n_fixed = size(X, 2)
    included_effects = _included_random_effects(model, Z_dict)
    n_random = sum(size(Z_dict[effect.name], 2) for effect in included_effects)
    n_total = n_fixed + n_random

    C = spzeros(n_total, n_total)
    rhs = zeros(Float64, n_total)

    sigma2_e = variances["residual"]
    R_inv_val = 1.0 / sigma2_e

    C[1:n_fixed, 1:n_fixed] = (X' * X) .* R_inv_val
    rhs[1:n_fixed] = (X' * y) .* R_inv_val

    relationship_inv = nothing
    blocks = _random_effect_blocks(n_fixed, included_effects, Z_dict)
    for block in blocks
        effect = block.effect
        range = block.range
        Z = Z_dict[effect.name]

        XtZ = X' * Z
        C[1:n_fixed, range] = XtZ .* R_inv_val
        C[range, 1:n_fixed] = XtZ' .* R_inv_val

        C[range, range] = (Z' * Z) .* R_inv_val

        if effect.type == :additive
            if !haskey(variances, effect.name)
                error("未提供随机效应 $(effect.name) 的方差估计，无法构建MME。")
            end
            if isnothing(relationship_inv)
                relationship_inv = _relationship_inverse_matrix(dm)
            end
            lambda = sigma2_e / variances[effect.name]
            C[range, range] += lambda .* relationship_inv
        end

        rhs[range] = (Z' * y) .* R_inv_val
    end

    return C, rhs
end

"""
    solve_mme(C::SparseMatrixCSC, rhs::Vector{Float64}) -> Vector{Float64}

使用高效的稀疏求解器求解 MME 系统。
"""
function solve_mme(C::SparseMatrixCSC, rhs::Vector{Float64})
    @info "求解MME (维度: $(length(rhs)))..."
    solution = C \ rhs
    @info "MME求解完成。"
    return solution
end
