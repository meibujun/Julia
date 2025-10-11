# ============================================================================
# 关系矩阵模块 - 单步关系矩阵 (H)
# AnimalBreeding.jl
# ============================================================================

"""
    compute_H_matrix_inv(dm::DataManager; blending_factor::Float64=0.05,
                         ridge::Float64=1e-8) -> SparseMatrixCSC{Float64,Int}

计算单步关系矩阵H的逆矩阵 (H⁻¹)，用于ssGBLUP评估。

# 参数
- `dm::DataManager`: 包含已计算好的A, A⁻¹ 和 G 矩阵的数据管理器。
- `blending_factor::Float64`: 对G矩阵进行A₂₂混合时的权重，范围建议在0.01-0.10。
- `ridge::Float64`: 为提高数值稳定性而加在G对角线上的岭回归项。

# 返回
- `SparseMatrixCSC{Float64,Int}`: 稀疏的H⁻¹矩阵。
"""
function compute_H_matrix_inv(dm::DataManager; blending_factor::Float64=0.05,
                               ridge::Float64=1e-8)
    @info "计算单步关系矩阵的逆 (H-inverse)..."

    if isnothing(dm.A_matrix) || isnothing(dm.A_inv_matrix)
        error("计算H⁻¹前，必须先在DataManager中准备好A及A⁻¹矩阵。")
    end
    if isnothing(dm.genotypes) || isnothing(dm.G_matrix)
        error("计算H⁻¹前，必须先加载基因型并计算G矩阵。")
    end

    if isempty(dm.animal_map)
        update_animal_map_from_pedigree!(dm)
    end

    id_col_name = names(dm.genotypes)[1]
    geno_ids = collect(dm.genotypes[!, id_col_name])
    genotyped_idx = Int[]
    for id in geno_ids
        idx = get(dm.animal_map, id, 0)
        if idx > 0
            push!(genotyped_idx, idx)
        else
            @warn "基因型中的个体 $(id) 未在谱系中找到，将被忽略。"
        end
    end

    n_total = size(dm.A_inv_matrix, 1)
    n_genotyped = length(genotyped_idx)
    if n_genotyped == 0
        @warn "没有基因型个体匹配到谱系，返回原始 A⁻¹。"
        return dm.A_inv_matrix
    end

    @info "  总个体数: $(n_total), 基因分型个体数: $(n_genotyped)"

    A22 = dm.A_matrix[genotyped_idx, genotyped_idx]
    A22_inv = inv(Matrix(A22))

    G_adj = (1 - blending_factor) * dm.G_matrix + blending_factor * A22
    if ridge > 0
        G_adj += ridge * I
    end
    G_inv = inv(Matrix(G_adj))

    diff_matrix = G_inv - A22_inv

    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]
    for (local_i, global_i) in enumerate(genotyped_idx)
        for (local_j, global_j) in enumerate(genotyped_idx)
            val = diff_matrix[local_i, local_j]
            if abs(val) > 1e-12
                push!(I_idx, global_i)
                push!(J_idx, global_j)
                push!(V_val, val)
            end
        end
    end

    Delta = sparse(I_idx, J_idx, V_val, n_total, n_total)
    H_inv = dm.A_inv_matrix + Delta

    @info "H-inverse 矩阵计算完成。"
    return H_inv
end
