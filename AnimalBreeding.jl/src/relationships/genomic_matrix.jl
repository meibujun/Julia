# ============================================================================
# 关系矩阵模块 - 基因组关系矩阵 (G)
# AnimalBreeding.jl
# ============================================================================

"""
    compute_G_matrix(genotypes::DataFrame; method::Symbol=:VanRaden1,
                     impute::Symbol=:mean, scaling::Bool=true) -> Matrix{Float64}

根据基因型数据计算基因组关系矩阵 (G矩阵)。

# 支持的方法
- `:VanRaden1`: VanRaden的第一种方法，这是最常用的方法。

# 额外特性
- 支持对缺失基因型进行均值或中位数插补。
- 可选的标准化缩放步骤，确保对角线均值接近1。
"""
function compute_G_matrix(genotypes::DataFrame; method::Symbol=:VanRaden1,
                          impute::Symbol=:mean, scaling::Bool=true)
    @info "计算基因组关系矩阵 G (方法: $method)..."

    id_col_name = names(genotypes)[1]
    marker_df = genotypes[:, Not(Symbol(id_col_name))]
    M = Matrix{Float64}(undef, size(marker_df, 1), size(marker_df, 2))

    # 处理缺失值并转换为Float64矩阵
    for j in 1:size(marker_df, 2)
        column = marker_df[!, j]
        if any(ismissing, column)
            replace_value = if impute == :median
                median(skipmissing(column))
            else
                mean(skipmissing(column))
            end
            column = collect(ifelse.(ismissing.(column), replace_value, column))
        end
        M[:, j] = Float64.(column)
    end

    n, m = size(M)

    if method == :VanRaden1
        p = vec(mean(M, dims=1)) / 2.0
        P = repeat(2 .* p', n, 1)
        Z = M .- P

        denominator = 2 * sum(p .* (1 .- p))
        if denominator <= eps()
            error("无法计算G矩阵：所有标记的等位基因频率为0或1，导致分母为0。")
        end

        G = (Z * Z') / denominator
    else
        error("不支持的G矩阵计算方法: $(method)。目前仅支持 `:VanRaden1`。")
    end

    if scaling
        diag_mean = mean(diag(G))
        if diag_mean > 0
            G ./= diag_mean
        end
    end

    @info "G矩阵计算完成，维度: $(size(G))。"
    return G
end
