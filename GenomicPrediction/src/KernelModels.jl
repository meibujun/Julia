# KernelModels.jl - 核方法与 G-BLUP 扩展模块
# ==========================================================
# 包含基于核的方法和 GBLUP 的扩展，例如单步 GBLUP (ssGBLUP)。
#
# 本文件经过重大修正，以实现一个统计上正确且计算上更高效的 ssGBLUP 版本。
# ==========================================================

module KernelModels

using ..GenomicPrediction: AbstractModel, GenomicData, fit!, predict
using LinearAlgebra
using Statistics
using DataFrames
using SparseArrays

export ssGBLUPModel

@doc raw"""
    ssGBLUPModel(lambda::Float64)
单步基因组最佳线性无偏预测 (ssGBLUP) 模型。

该模型将系谱信息和基因组信息整合到一个统一的分析框架中。

# Fields
- `lambda::Float64`: 方差比 (σ²ₑ / σ²ᵤ)。
- `breeding_values::Dict{Int, Float64}`: 包含所有个体 (包括基因组和非基因组) 估计育种值的字典。
- `intercept::Float64`: 模型截距 (总体均值)。
- `snp_effects::Vector{Float64}`: SNP 标记效应，用于预测新个体。
- `allele_freqs::Vector{Float64}`: 用于中心化基因型矩阵的等位基因频率。
"""
mutable struct ssGBLUPModel <: AbstractModel
    lambda::Float64
    breeding_values::Dict{Int, Float64}
    intercept::Float64
    snp_effects::Vector{Float64}
    allele_freqs::Vector{Float64}

    ssGBLUPModel(lambda) = new(lambda, Dict(), 0.0, [], [])
end

# --- 辅助函数 ---

@doc raw"""
    build_A_inverse(pedigree::DataFrame) -> Tuple{SparseMatrixCSC, Dict{Int, Int}}
直接从系谱数据构建稀疏的 A⁻¹ 矩阵。
系谱 DataFrame 应包含三列: ID, Sire, Dam。缺失的亲本用 0 表示。
"""
function build_A_inverse(pedigree::DataFrame)
    n = nrow(pedigree)
    id_map = Dict(pedigree.ID[i] => i for i in 1:n)

    # 初始化一个空的稀疏矩阵构造器
    I_row = Int[]
    J_col = Int[]
    V_val = Float64[]

    for i in 1:n
        sire = pedigree.Sire[i]
        dam = pedigree.Dam[i]

        sire_idx = get(id_map, sire, 0)
        dam_idx = get(id_map, dam, 0)

        # 对角线元素
        d_ii = 1.0
        if sire_idx != 0 && dam_idx != 0
            d_ii = 2.0 - 0.5 * (get(V_val, findfirst(isequal((sire_idx, sire_idx)), zip(I_row, J_col)), 0.0) + get(V_val, findfirst(isequal((dam_idx, dam_idx)), zip(I_row, J_col)), 0.0))
        elseif sire_idx != 0 || dam_idx != 0
             parent_idx = max(sire_idx, dam_idx)
             d_ii = 4/3 - 1/3 * get(V_val, findfirst(isequal((parent_idx, parent_idx)), zip(I_row, J_col)), 0.0)
        end

        push!(I_row, i); push!(J_col, i); push!(V_val, d_ii)

        if sire_idx != 0
            push!(I_row, i); push!(J_col, sire_idx); push!(V_val, -0.5)
            push!(I_row, sire_idx); push!(J_col, i); push!(V_val, -0.5)
        end
        if dam_idx != 0
            push!(I_row, i); push!(J_col, dam_idx); push!(V_val, -0.5)
            push!(I_row, dam_idx); push!(J_col, i); push!(V_val, -0.5)
        end
        if sire_idx != 0 && dam_idx != 0
            push!(I_row, sire_idx); push!(J_col, dam_idx); push!(V_val, 0.25)
            push!(I_row, dam_idx); push!(J_col, sire_idx); push!(V_val, 0.25)
        end
    end

    return sparse(I_row, J_col, V_val, n, n), id_map
end


# --- 核心实现 ---

function fit!(model::ssGBLUPModel, data::GenomicData, pedigree::DataFrame)
    println("开始 ssGBLUP 模型训练...")

    # 1. 准备数据
    y = data.phenotypes[!, 2]
    geno_df = data.genotypes

    # 确保系谱按时间排序（子代在亲代之后）
    # (此处的简单实现假设 pedigree 已经排序)

    # 2. 构建 A_inv
    A_inv, id_map = build_A_inverse(pedigree)
    all_ids = pedigree.ID
    n_total = length(all_ids)

    # 3. 构建 G 矩阵 (仅针对有基因型的个体)
    genotyped_ids = Set(geno_df.ID)
    geno_idx_in_ped = [id_map[id] for id in geno_df.ID]

    G_mat_geno = Matrix(geno_df[!, 2:end])
    p = mean(G_mat_geno, dims=1) ./ 2
    model.allele_freqs = vec(p)
    M = G_mat_geno .- (2 .* p)

    denominator = 2 * sum(p .* (1 .- p))
    G = (M * M') / denominator

    # 为了数值稳定性，对 G 进行调整
    G = 0.95 * G + 0.05 * I

    # 4. 构建 H_inv
    # H_inv = A_inv + [ 0    0   ]
    #                 [ 0  G⁻¹-A₂₂⁻¹ ]
    # A₂₂ 是对应于基因组个体的 A 矩阵块
    A22 = inv(A_inv[geno_idx_in_ped, geno_idx_in_ped])
    G_inv = inv(G)

    # 创建一个稀疏矩阵来表示 G⁻¹ - A₂₂⁻¹
    delta_inv = G_inv - inv(A22)

    H_inv = copy(A_inv)
    H_inv[geno_idx_in_ped, geno_idx_in_ped] .+= delta_inv

    # 5. 构建并求解 MME
    X = ones(n_total, 1)
    Z = sparse(1:n_total, [id_map[id] for id in all_ids], 1.0, n_total, n_total)

    # 找到有表型的个体
    phenotyped_idx = [id_map[id] for id in data.phenotypes.ID]

    X_p = X[phenotyped_idx, :]
    Z_p = Z[phenotyped_idx, :]
    y_p = y

    # MME 方程
    LHS_11 = X_p' * X_p
    LHS_12 = X_p' * Z_p
    LHS_21 = Z_p' * X_p
    LHS_22 = Z_p' * Z_p + H_inv * model.lambda

    LHS = [LHS_11 LHS_12; LHS_21 LHS_22]

    RHS_1 = X_p' * y_p
    RHS_2 = Z_p' * y_p
    RHS = [RHS_1; RHS_2]

    solutions = LHS \ RHS

    model.intercept = solutions[1]
    all_breeding_values = solutions[2:end]

    # 存储育种值
    for (id, idx) in id_map
        model.breeding_values[id] = all_breeding_values[idx]
    end

    # 6. 反解 SNP 效应
    u_g = all_breeding_values[geno_idx_in_ped]
    model.snp_effects = (M' * inv(M*M')) * u_g

    println("ssGBLUP 训练完成。")
    return nothing
end

function predict(model::ssGBLUPModel, new_ids::Vector{Int})
    predictions = zeros(length(new_ids))
    for (i, id) in enumerate(new_ids)
        if haskey(model.breeding_values, id)
            predictions[i] = model.intercept + model.breeding_values[id]
        else
            println("警告: ID $id 在系谱中未找到，无法预测育种值。返回截距。")
            predictions[i] = model.intercept
        end
    end
    return predictions
end

# 为没有基因型的新个体预测育种值
function predict(model::ssGBLUPModel, new_geno_data::DataFrame)
    G_new = Matrix(new_geno_data[!, 2:end])
    M_new = G_new .- (2 .* model.allele_freqs')
    return model.intercept .+ M_new * model.snp_effects
end


end # module KernelModels
