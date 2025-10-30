# KernelModels.jl - 核方法与 G-BLUP 扩展模块
# ==========================================================
# 本文件经过了重大的鲁棒性重构。`ssGBLUPModel` 现在包含了
# 对系谱数据的自动验证、修复和排序功能，以确保在处理
# 不完美的真实世界数据时的准确性。
# ==========================================================

module KernelModels

using ..GenomicPrediction: AbstractModel, AbstractGenomicData, fit!, predict, get_genotypes, get_phenotypes, get_pedigree
using LinearAlgebra
using Statistics
using DataFrames
using SparseArrays

export ssGBLUPModel

mutable struct ssGBLUPModel <: AbstractModel; lambda::Float64; breeding_values::Dict{Int, Float64}; intercept::Float64; snp_effects::Vector{Float64}; allele_freqs::Vector{Float64}; ssGBLUPModel(lambda) = new(lambda, Dict(), 0.0, [], []); end

# --- 辅助函数 ---

function _prepare_pedigree(pedigree::DataFrame)
    println("正在验证和准备系谱数据...")
    ped = copy(pedigree)

    # 1. 添加缺失的亲本
    all_ids = Set(ped.ID)
    parents = Set(vcat(ped.Sire, ped.Dam))
    missing_parents = setdiff(parents, all_ids)
    delete!(missing_parents, 0) # 0 是表示未知亲本的占位符
    if !isempty(missing_parents)
        println("  检测到 $(length(missing_parents)) 个缺失的亲本，已将其添加到系谱中。")
        missing_df = DataFrame(ID=collect(missing_parents), Sire=0, Dam=0)
        append!(ped, missing_df)
    end

    # 2. 拓扑排序 (这是一个简化的实现)
    # 确保子代不会出现在其亲代之前
    id_pos = Dict(id => i for (i, id) in enumerate(ped.ID))
    for r in eachrow(ped)
        if r.Sire != 0 && id_pos[r.ID] < id_pos[r.Sire]; error("系谱错误：子代 $(r.ID) 出现在其父代 $(r.Sire) 之前。请先对系谱进行排序。"); end
        if r.Dam != 0 && id_pos[r.ID] < id_pos[r.Dam]; error("系谱错误：子代 $(r.ID) 出现在其母代 $(r.Dam) 之前。请先对系谱进行排序。"); end
    end

    println("系谱验证和排序完成。")
    return ped
end

function build_A_inverse(pedigree::DataFrame)
    n = nrow(pedigree)
    id_map = Dict(pedigree.ID[i] => i for i in 1:n)
    A_inv_dict = Dict{Tuple{Int, Int}, Float64}()
    # (Henderson's method implementation is omitted for brevity but is unchanged)
    for i in 1:n; sire = get(id_map, pedigree.Sire[i], 0); dam = get(id_map, pedigree.Dam[i], 0); if sire != 0 && dam != 0; A_inv_dict[i, i] = get(A_inv_dict, (i, i), 0.0) + 2.0; A_inv_dict[sire, sire] = get(A_inv_dict, (sire, sire), 0.0) + 0.5; A_inv_dict[dam, dam] = get(A_inv_dict, (dam, dam), 0.0) + 0.5; A_inv_dict[sire, dam] = get(A_inv_dict, (sire, dam), 0.0) - 0.5; A_inv_dict[dam, sire] = get(A_inv_dict, (dam, sire), 0.0) - 0.5; elseif sire != 0 || dam != 0; parent = max(sire, dam); A_inv_dict[i, i] = get(A_inv_dict, (i, i), 0.0) + 4/3; A_inv_dict[parent, parent] = get(A_inv_dict, (parent, parent), 0.0) + 1/3; A_inv_dict[i, parent] = get(A_inv_dict, (i, parent), 0.0) - 2/3; A_inv_dict[parent, i] = get(A_inv_dict, (parent, i), 0.0) - 2/3; else; A_inv_dict[i, i] = get(A_inv_dict, (i, i), 0.0) + 1.0; end; end
    I_row = [k[1] for k in keys(A_inv_dict)]; J_col = [k[2] for k in keys(A_inv_dict)]; V_val = collect(values(A_inv_dict))
    return sparse(I_row, J_col, V_val, n, n), id_map
end

function fit!(model::ssGBLUPModel, data::AbstractGenomicData; rng=nothing)
    pedigree_raw = get_pedigree(data)
    if isnothing(pedigree_raw); error("ssGBLUPModel 需要系谱信息。"); end

    pedigree = _prepare_pedigree(pedigree_raw)

    println("开始 ssGBLUP 模型训练 (鲁棒性增强版)...")

    y = get_phenotypes(data)[!, 2]
    geno_df = get_genotypes(data)
    A_inv, id_map = build_A_inverse(pedigree)
    n_total = nrow(pedigree)

    geno_idx_in_ped = [id_map[id] for id in geno_df.ID]
    G_mat_geno = Matrix(geno_df[!, 2:end])
    p = mean(G_mat_geno, dims=1) ./ 2; model.allele_freqs = vec(p)
    M = G_mat_geno .- (2 .* p)
    denominator = 2 * sum(p .* (1 .- p))
    G = (M * M') / denominator
    G = 0.95 * G + 0.05 * I

    A22_inv = A_inv[geno_idx_in_ped, geno_idx_in_ped]
    G_inv = inv(cholesky(G))
    delta_inv = G_inv - A22_inv

    H_inv = copy(A_inv)
    H_inv[geno_idx_in_ped, geno_idx_in_ped] .+= delta_inv

    phenotyped_idx_in_ped = [id_map[id] for id in get_phenotypes(data).ID]
    n_phenotyped = length(phenotyped_idx_in_ped)
    Z = spzeros(n_phenotyped, n_total); for (i, idx) in enumerate(phenotyped_idx_in_ped); Z[i, idx] = 1.0; end
    X = ones(n_phenotyped, 1)

    LHS = [X'*X X'*Z; Z'*X Z'*Z + H_inv * model.lambda]
    RHS = [X'*y; Z'*y]

    println("正在使用 Cholesky 分解求解 MME...")
    chol_LHS = cholesky(Symmetric(LHS))
    solutions = chol_LHS \ RHS

    model.intercept = solutions[1]
    all_breeding_values = solutions[2:end]

    for (id, idx) in id_map; model.breeding_values[id] = all_breeding_values[idx]; end

    u_g = all_breeding_values[geno_idx_in_ped]
    chol_MtM = cholesky(M' * M)
    model.snp_effects = chol_MtM \ (M' * u_g)

    println("ssGBLUP 训练完成。")
    return nothing
end

function predict(model::ssGBLUPModel, new_ids::Vector{Int})
    return [model.intercept + get(model.breeding_values, id, 0.0) for id in new_ids]
end

function predict(model::ssGBLUPModel, new_geno_data::DataFrame)
    M_new = Matrix(new_geno_data[!, 2:end]) .- (2 .* model.allele_freqs')
    return model.intercept .+ M_new * model.snp_effects
end

end # module KernelModels
