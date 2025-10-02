# ============================================================================
# 关系矩阵模块 - 单步关系矩阵 (H)
# AnimalBreeding.jl
# ============================================================================

"""
    compute_H_matrix_inv(dm::DataManager; blending_factor::Float64=0.05) -> SparseMatrixCSC

[已修复] 计算单步关系矩阵H的逆矩阵 (H⁻¹)，用于ssGBLUP评估。

# 理论
H⁻¹ 矩阵通过整合谱系关系矩阵的逆 (A⁻¹) 和基因组关系矩阵的逆 (G⁻¹) 来构建，
从而能够同时对有基因型和无基因型的个体进行评估。

# 公式
H⁻¹ = A⁻¹ + [  0      0   ]
            [  0   G_adj⁻¹ - A₂₂⁻¹ ]

其中:
- `A⁻¹`: 全体动物的谱系关系矩阵的逆。
- `A₂₂`: 已基因分型动物对应的谱系关系子矩阵。
- `G_adj`: 为了数值稳定性和尺度一致性，经过调整的基因组关系矩阵。
- `G_adj⁻¹` 和 `A₂₂⁻¹` 分别是它们俩的逆。

# 参数
- `dm::DataManager`: 包含已计算好的A, A⁻¹ 和 G 矩阵的数据管理器。
- `blending_factor::Float64`: 一个小的混合因子，用于对G矩阵进行调整，以提高其数值稳定性并确保可逆。 G_adj = (1-w)G + wA₂₂。

# 返回
- `SparseMatrixCSC`: 稀疏的H⁻¹矩阵。
"""
function compute_H_matrix_inv(dm::DataManager; blending_factor::Float64=0.05)
    @info "计算单步关系矩阵的逆 (H-inverse)..."

    # 1. 检查所需矩阵是否存在
    if isnothing(dm.A_matrix) || isnothing(dm.A_inv_matrix) || isnothing(dm.genotypes) || isnothing(dm.G_matrix)
        error("计算H⁻¹前，必须先在DataManager中准备好A, A⁻¹ 和 G 矩阵。")
    end

    # 2. 识别基因分型个体及其在A矩阵中的索引
    id_col_name = names(dm.genotypes)[1]
    geno_ids = Set(dm.genotypes[!, id_col_name])

    # 使用 DataManager 中的全局 animal_map 来获取所有动物的索引
    # 假设 dm.pedigree.animal 包含了所有动物且已排序
    genotyped_idx = findall(animal -> animal in geno_ids, dm.pedigree.animal)

    n_total = size(dm.A_inv_matrix, 1)
    n_genotyped = length(genotyped_idx)

    if n_genotyped == 0
        @warn "基因型文件中没有在谱系中找到的动物，返回原始 A⁻¹。"
        return dm.A_inv_matrix
    end

    @info "  总个体数: $n_total, 基因分型个体数: $n_genotyped"

    # 3. [已修复] 从A矩阵中正确提取 A₂₂ 子矩阵，然后求逆
    @info "  步骤 1/4: 提取 A₂₂ 并计算其逆矩阵..."
    A₂₂ = dm.A_matrix[genotyped_idx, genotyped_idx]
    A₂₂_inv = inv(Matrix(A₂₂)) # A22 通常不大，可以直接求逆

    # 4. 调整并求逆 G 矩阵
    @info "  步骤 2/4: 调整并计算 G 逆矩阵..."
    # G_adj = (1-w)G + wA₂₂
    G_adj = (1 - blending_factor) * dm.G_matrix + blending_factor * A₂₂
    # 为保证数值稳定性，可以对G_adj对角线加一个微小值
    G_inv = inv(G_adj + I * 1e-8)

    # 5. 计算差值矩阵 G⁻¹ - A₂₂⁻¹
    @info "  步骤 3/4: 计算 G⁻¹ 和 A₂₂⁻¹ 的差值..."
    diff_matrix = G_inv - A₂₂_inv

    # 6. 构建 H⁻¹
    @info "  步骤 4/4: 将差值矩阵加到 A⁻¹ 中以构建 H⁻¹..."
    H_inv = copy(dm.A_inv_matrix) # 从A⁻¹开始

    # 将差值矩阵加到H⁻¹的相应位置
    H_inv[genotyped_idx, genotyped_idx] .+= diff_matrix

    @info "H-inverse 矩阵计算完成。"
    return H_inv
end