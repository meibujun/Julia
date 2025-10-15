module QualityControl

using DataFrames
using Statistics
using StatsBase
using LinearAlgebra
using Random

"""
    qc_filter_variants!(G, variants; maf_threshold = 0.01, missing_rate = 0.05)

根据设定阈值筛选变异，返回过滤后的基因型矩阵、变异表与布尔保留向量。
"""
function qc_filter_variants!(G::AbstractMatrix, variants::DataFrame; maf_threshold::Float64 = 0.01, missing_rate::Float64 = 0.05)
    keep = trues(size(G, 2))
    for j in axes(G, 2)
        column = view(G, :, j)
        miss = count(isnan, column) / length(column)
        if miss > missing_rate
            keep[j] = false
            continue
        end
        valid = column[.!isnan.(column)]
        if isempty(valid)
            keep[j] = false
            continue
        end
        maf = mean(valid) / 2
        if maf < maf_threshold
            keep[j] = false
        end
    end
    filtered_G = G[:, keep]
    filtered_variants = variants[keep, :]
    return filtered_G, filtered_variants, keep
end

"""
    qc_filter_samples!(G, samples; missing_rate = 0.05)

过滤缺失率过高的个体，返回过滤后的基因型矩阵、样本表与布尔向量。
"""
function qc_filter_samples!(G::AbstractMatrix, samples::DataFrame; missing_rate::Float64 = 0.05)
    keep = trues(size(G, 1))
    for i in axes(G, 1)
        row = view(G, i, :)
        miss = count(isnan, row) / length(row)
        if miss > missing_rate
            keep[i] = false
        end
    end
    filtered_G = G[keep, :]
    filtered_samples = samples[keep, :]
    return filtered_G, filtered_samples, keep
end

"""
    fold_partition(n, k; seed = 2025)

生成 k 折交叉验证索引，用于折叠法检验。
"""
function fold_partition(n::Int, k::Int; seed::Int = 2025)
    rng = MersenneTwister(seed)
    idx = collect(1:n)
    shuffle!(rng, idx)
    folds = [Int[] for _ in 1:k]
    for (i, id) in enumerate(idx)
        push!(folds[mod1(i, k)], id)
    end
    return folds
end

end # module
