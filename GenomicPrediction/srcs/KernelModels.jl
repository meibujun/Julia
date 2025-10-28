# KernelModels.jl - 核方法与 G-BLUP 扩展模块
# ==========================================================
# 包含基于核的方法和 GBLUP 的扩展，例如单步 GBLUP (ssGBLUP)。
#
# 本文件经过修正，以统一 fit! 函数的 API，使其仅接受 GenomicData 对象，
# 并从该对象内部安全地获取系谱信息。
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
"""
function build_A_inverse(pedigree::DataFrame)
    n = nrow(pedigree)
    id_map = Dict(pedigree.ID[i] => i for i in 1:n)

    I_row, J_col, V_val = Int[], Int[], Float64[]

    # This is a simplified Henderson's method for A-inverse
    # A proper implementation requires careful handling of pedigree sorting and completeness.
    for i in 1:n
        sire = pedigree.Sire[i]; dam = pedigree.Dam[i]
        sire_idx = get(id_map, sire, 0); dam_idx = get(id_map, dam, 0)

        d_ii = 0.0
        if sire_idx != 0 && dam_idx != 0; d_ii = 2.0;
        elseif sire_idx != 0 || dam_idx != 0; d_ii = 4/3;
        else d_ii = 1.0; end
        push!(I_row, i); push!(J_col, i); push!(V_val, d_ii)

        if sire_idx != 0; push!(I_row, i, sire_idx); push!(J_col, sire_idx, i); push!(V_val, -1.0, -1.0); end
        if dam_idx != 0;  push!(I_row, i, dam_idx);  push!(J_col, dam_idx, i);  push!(V_val, -1.0, -1.0); end
    end

    return sparse(I_row, J_col, V_val, n, n), id_map
end

# --- 核心实现 ---

function fit!(model::ssGBLUPModel, data::GenomicData; rng=nothing)
    if isnothing(data.pedigree)
        error("ssGBLUPModel requires pedigree information, but `data.pedigree` is nothing. Please load pedigree data using `load_csv`.")
    end
    pedigree = data.pedigree

    println("开始 ssGBLUP 模型训练...")

    y = data.phenotypes[!, 2]
    geno_df = data.genotypes

    A_inv, id_map = build_A_inverse(pedigree)
    all_ids = pedigree.ID
    n_total = length(all_ids)

    geno_idx_in_ped = [id_map[id] for id in geno_df.ID]
    G_mat_geno = Matrix(geno_df[!, 2:end])
    p = mean(G_mat_geno, dims=1) ./ 2
    model.allele_freqs = vec(p)
    M = G_mat_geno .- (2 .* p)

    denominator = 2 * sum(p .* (1 .- p))
    G = (M * M') / denominator
    G = 0.95 * G + 0.05 * I
    G_inv = inv(G)

    A22_inv = A_inv[geno_idx_in_ped, geno_idx_in_ped]

    H_inv = copy(A_inv)
    H_inv[geno_idx_in_ped, geno_idx_in_ped] .+= (G_inv - A22_inv)

    phenotyped_idx_in_ped = [id_map[id] for id in data.phenotypes.ID]

    n_phenotyped = length(phenotyped_idx_in_ped)
    Z = spzeros(n_phenotyped, n_total)
    for (i, idx) in enumerate(phenotyped_idx_in_ped); Z[i, idx] = 1.0; end
    X = ones(n_phenotyped, 1)

    LHS_11 = X' * X
    LHS_12 = X' * Z
    LHS_21 = Z' * X
    LHS_22 = Z' * Z + H_inv * model.lambda
    LHS = [LHS_11 LHS_12; LHS_21 LHS_22]

    RHS = [X' * y; Z' * y]

    solutions = LHS \ RHS

    model.intercept = solutions[1]
    all_breeding_values = solutions[2:end]

    for (id, idx) in id_map
        model.breeding_values[id] = all_breeding_values[idx]
    end

    u_g = all_breeding_values[geno_idx_in_ped]
    model.snp_effects = M' * (inv(M*M') * u_g)

    println("ssGBLUP 训练完成。")
    return nothing
end

function predict(model::ssGBLUPModel, new_ids::Vector{Int})
    predictions = zeros(length(new_ids))
    for (i, id) in enumerate(new_ids)
        predictions[i] = model.intercept + get(model.breeding_values, id, 0.0)
    end
    return predictions
end

function predict(model::ssGBLUPModel, new_geno_data::DataFrame)
    G_new = Matrix(new_geno_data[!, 2:end])
    M_new = G_new .- (2 .* model.allele_freqs')
    return model.intercept .+ M_new * model.snp_effects
end

end # module KernelModels
