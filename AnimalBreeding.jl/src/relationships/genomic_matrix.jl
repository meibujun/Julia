# ============================================================================
# 关系矩阵模块 - 基因组关系矩阵 (G)
# AnimalBreeding.jl
# ============================================================================

"""
    compute_G_matrix(genotypes::DataFrame; method::Symbol=:VanRaden1) -> Matrix{Float64}

根据基因型数据计算基因组关系矩阵 (G矩阵)。

# 支持的方法
- `:VanRaden1`: VanRaden的第一种方法，这是最常用的方法。

# 参数
- `genotypes::DataFrame`: 基因型数据框，第一列为动物ID，其余列为SNP标记。
- `method::Symbol`: 计算G矩阵所使用的方法。

# 返回
- `Matrix{Float64}`: n×n的基因组关系矩阵。

# 参考文献
VanRaden, P. M. (2008). Efficient methods to compute genomic predictions. Journal of Dairy Science, 91(11), 4414-4423.
"""
function compute_G_matrix(genotypes::DataFrame; method::Symbol=:VanRaden1)
    @info "计算基因组关系矩阵 G (方法: $method)..."

    # 提取基因型矩阵 (排除ID列)
    id_col_name = names(genotypes)[1]
    marker_cols = names(genotypes)[2:end]
    n = nrow(genotypes)
    m = length(marker_cols)

    M = Matrix{Float64}(undef, n, m)
    for (j, col_name) in enumerate(marker_cols)
        col_data = genotypes[!, col_name]
        numeric_col = Vector{Float64}(undef, n)
        missing_mask = falses(n)
        for i in 1:n
            val = col_data[i]
            if ismissing(val)
                missing_mask[i] = true
            else
                numeric_col[i] = Float64(val)
            end
        end
        if all(missing_mask)
            error("标记 '$col_name' 全部缺失，无法计算G矩阵。")
        end
        mean_val = mean(numeric_col[.!missing_mask])
        numeric_col[missing_mask] .= mean_val
        M[:, j] = numeric_col
    end

    if method == :VanRaden1
        # --- VanRaden 方法一 ---
        # G = ZZ' / (2 * Σ(pᵢ * (1 - pᵢ)))
        # Z = M - P, 其中P是基于等位基因频率计算的期望基因型矩阵

        # 1. 计算每个标记的等位基因频率 pᵢ
        # 假设基因型编码为 0, 1, 2
        p = vec(mean(M, dims=1)) / 2.0

        # 2. 构建期望基因型矩阵 P
        # P的每一行都是 2p'
        P = repeat(2 .* p', n, 1)

        # 3. 中心化基因型矩阵 Z = M - P
        Z = M .- P

        # 4. 计算分母 (缩放因子)
        denominator = 2 * sum(p .* (1 .- p))
        if denominator == 0
            error("无法计算G矩阵：所有标记的等位基因频率为0或1，导致分母为0。")
        end

        # 5. 计算G矩阵
        G = (Z * Z') / denominator

    else
        error("不支持的G矩阵计算方法: $method。目前仅支持 `:VanRaden1`。")
    end

    @info "G矩阵计算完成，维度: $(size(G))。"
    return G
end
