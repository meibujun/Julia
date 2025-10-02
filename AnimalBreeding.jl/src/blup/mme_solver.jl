# ============================================================================
# BLUP评估模块 - 混合模型方程求解器
# AnimalBreeding.jl
# ============================================================================

"""
    build_design_matrices(phenotypes, model, animal_map) -> (Matrix, Dict, Vector)

根据模型定义和表型数据构建设计矩阵。

# 参数
- `phenotypes::DataFrame`: 表型数据。
- `model::ModelSpec`: 模型规格。
- `animal_map::Dict`: 全局动物ID到索引的映射。

# 返回
- `X::Matrix`: 固定效应设计矩阵。
- `Z_dict::Dict`: 随机效应设计矩阵的字典，键为效应名。
- `y::Vector`: 观测值向量。
"""
function build_design_matrices(phenotypes::DataFrame, model::ModelSpec, animal_map::Dict)
    @info "构建设计矩阵..."

    trait = model.traits[1] # 简化为单性状
    pheno_clean = phenotypes[.!ismissing.(phenotypes[!, trait]), :]
    y = Vector{Float64}(pheno_clean[!, trait])
    n_obs = length(y)

    # --- 构建固定效应设计矩阵 X ---
    X_cols = [ones(n_obs)] # 截距项
    for effect_name in model.fixed_effects
        levels = unique(pheno_clean[!, effect_name])
        if length(levels) > 1
            for level in levels[2:end] # k-1 哑变量
                push!(X_cols, pheno_clean[!, effect_name] .== level)
            end
        end
    end
    X = hcat(X_cols...)

    # --- 构建随机效应设计矩阵 Z ---
    Z_dict = Dict{String, SparseMatrixCSC{Float64, Int}}()
    n_total_animals = length(animal_map)

    for effect in model.random_effects
        if effect.type == :additive
            I = 1:n_obs
            # 使用全局animal_map来对齐
            J = [get(animal_map, id, 0) for id in pheno_clean.animal]
            V = ones(n_obs)

            valid_indices = J .> 0
            Z = sparse(I[valid_indices], J[valid_indices], V[valid_indices], n_obs, n_total_animals)
            Z_dict[effect.name] = Z
        end
    end

    @info "设计矩阵构建完成: X($(size(X))), Z($(values(Z_dict) |> first |> size)), y($(length(y)))"
    return X, Z_dict, y
end

"""
    setup_mme(X, Z_dict, y, dm, model, variances) -> (SparseMatrixCSC, Vector)

构建稀疏的混合模型方程 (MME) 系统。
"""
function setup_mme(X::Matrix, Z_dict::Dict, y::Vector, dm::DataManager, model::ModelSpec, variances::Dict)
    @info "构建混合模型方程 (MME)..."

    n_fixed = size(X, 2)
    n_total = n_fixed + sum(size(Z, 2) for Z in values(Z_dict))

    C = spzeros(n_total, n_total)
    rhs = zeros(n_total)

    sigma2_e = variances["residual"]
    R_inv_val = 1.0 / sigma2_e

    # 固定效应部分
    C[1:n_fixed, 1:n_fixed] = (X' * X) .* R_inv_val
    rhs[1:n_fixed] = (X' * y) .* R_inv_val

    current_pos = n_fixed
    for effect in model.random_effects
        name = effect.name
        Z = Z_dict[name]
        dim = size(Z, 2)
        start_idx = current_pos + 1
        end_idx = current_pos + dim

        # 交叉部分
        XtZ = X' * Z
        C[1:n_fixed, start_idx:end_idx] = XtZ .* R_inv_val
        C[start_idx:end_idx, 1:n_fixed] = XtZ' .* R_inv_val

        # 随机效应自身部分
        C[start_idx:end_idx, start_idx:end_idx] = (Z' * Z) .* R_inv_val

        # 添加关系矩阵的逆
        if effect.type == :additive
            lambda = sigma2_e / variances[name]
            # 根据模型选择A⁻¹或H⁻¹
            # 简化：假设dm中已存好正确的矩阵
            if !isnothing(dm.H_inv_matrix)
                C[start_idx:end_idx, start_idx:end_idx] += lambda .* dm.H_inv_matrix
            elseif !isnothing(dm.A_inv_matrix)
                C[start_idx:end_idx, start_idx:end_idx] += lambda .* dm.A_inv_matrix
            else
                error("模型需要A⁻¹或H⁻¹，但未在DataManager中计算。")
            end
        end

        rhs[start_idx:end_idx] = (Z' * y) .* R_inv_val
        current_pos += dim
    end

    return C, rhs
end

"""
    solve_mme(C::SparseMatrixCSC, rhs::Vector) -> Vector

使用高效的稀疏求解器求解 MME 系统。
"""
function solve_mme(C::SparseMatrixCSC, rhs::Vector)
    @info "求解MME (维度: $(length(rhs)))..."
    # Julia的 `\` 运算符对稀疏矩阵有高度优化的实现
    solution = C \ rhs
    @info "MME求解完成。"
    return solution
end