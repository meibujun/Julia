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
    pheno_clean = phenotypes[.!ismissing.(phenotypes[!, trait]), :]
    y = Float64.(pheno_clean[!, trait])
    n_obs = length(y)

    X_cols = Vector{Vector{Float64}}()
    push!(X_cols, ones(Float64, n_obs))
    for effect_name in model.fixed_effects
        column = pheno_clean[!, effect_name]
        value_type = eltype(skipmissing(column))
        if value_type <: Number
            push!(X_cols, Float64.(column))
        else
            levels = unique(column)
            if length(levels) > 1
                ref_level = levels[1]
                for level in levels[2:end]
                    push!(X_cols, Float64.(column .== level))
                end
                @debug "固定效应 $(effect_name) 使用基准水平 $(ref_level)" ref_level
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
            @warn "暂未实现随机效应类型 $(effect.type)，将被忽略。"
        end
    end

    if isempty(Z_dict)
        @info "模型中未定义随机效应，仅构建固定效应矩阵。"
    else
        dims = first(values(Z_dict)) |> size
        @info "设计矩阵构建完成: X$(size(X)), Z$(dims), y$(length(y))"
    end

    return X, Z_dict, y
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
    n_random = sum(size(Z, 2) for Z in values(Z_dict))
    n_total = n_fixed + n_random

    C = spzeros(n_total, n_total)
    rhs = zeros(Float64, n_total)

    sigma2_e = variances["residual"]
    R_inv_val = 1.0 / sigma2_e

    C[1:n_fixed, 1:n_fixed] = (X' * X) .* R_inv_val
    rhs[1:n_fixed] = (X' * y) .* R_inv_val

    current_pos = n_fixed
    relationship_inv_cache = nothing

    for effect in model.random_effects
        name = effect.name
        Z = Z_dict[name]
        dim = size(Z, 2)
        start_idx = current_pos + 1
        end_idx = current_pos + dim

        XtZ = X' * Z
        C[1:n_fixed, start_idx:end_idx] = XtZ .* R_inv_val
        C[start_idx:end_idx, 1:n_fixed] = XtZ' .* R_inv_val

        C[start_idx:end_idx, start_idx:end_idx] = (Z' * Z) .* R_inv_val

        if effect.type == :additive
            lambda = sigma2_e / variances[name]
            if relationship_inv_cache === nothing
                relationship_inv_cache = _relationship_inverse(dm)
            end
            penalty = relationship_inv_cache
            if size(penalty, 1) != dim
                error("关系矩阵维度 ($(size(penalty,1))) 与随机效应 '$name' 维度 ($dim) 不一致。")
            end
            C[start_idx:end_idx, start_idx:end_idx] += lambda .* penalty
        end

        rhs[start_idx:end_idx] = (Z' * y) .* R_inv_val
        current_pos += dim
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

function _relationship_inverse(dm::DataManager)
    if !isnothing(dm.H_inv_matrix)
        return SparseMatrixCSC(dm.H_inv_matrix)
    elseif !isnothing(dm.A_inv_matrix)
        return SparseMatrixCSC(dm.A_inv_matrix)
    else
        error("模型需要A⁻¹或H⁻¹，但未在DataManager中计算。")
    end
end

function _random_effect_ranges(model::ModelSpec, Z_dict::Dict{String,SparseMatrixCSC{Float64,Int}}, n_fixed::Int)
    ranges = Dict{String, UnitRange{Int}}()
    current_pos = n_fixed
    for effect in model.random_effects
        name = effect.name
        Z = Z_dict[name]
        dim = size(Z, 2)
        ranges[name] = current_pos + 1:current_pos + dim
        current_pos += dim
    end
    return ranges
end

function _accumulate_random_offsets!(buffer::Vector{Float64}, model::ModelSpec,
                                     Z_dict::Dict{String,SparseMatrixCSC{Float64,Int}},
                                     solutions::Vector{Float64}, ranges::Dict{String,UnitRange{Int}})
    fill!(buffer, 0.0)
    for effect in model.random_effects
        name = effect.name
        if !haskey(ranges, name)
            continue
        end
        rng = ranges[name]
        buffer .+= Z_dict[name] * solutions[rng]
    end
    return buffer
end
