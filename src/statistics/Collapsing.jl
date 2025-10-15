module Collapsing

using LinearAlgebra
using StatsBase
using DataFrames
using Statistics
using Random
using Distributions
using ..Threading
using ..MathUtils
using ..QualityControl

"""
    collapsing_test(G, y; method = :burden, weights = nothing, folds = 5)

实现稀有变异折叠法，支持 Burden、CMC 和 SKAT 权重方案，并通过折叠交叉验证稳定 p 值估计。
返回包含统计量、p 值与权重信息的数据框。
"""
function collapsing_test(G::AbstractMatrix, y::AbstractVector; method::Symbol=:burden, weights=nothing, folds::Int=5)
    n, p = size(G)
    weights = isnothing(weights) ? ones(Float64, p) : weights
    folds_idx = QualityControl.fold_partition(n, folds)
    stats = Float64[]
    ps = Float64[]
    for fold in folds_idx
        train = setdiff(1:n, fold)
        test = fold
        β = fit_burden(G[train, :], y[train]; method, weights)
        ŷ = G[test, :] * β
        resid = y[test] .- ŷ
        stat = sum(resid .^ 2)
        push!(stats, stat)
        push!(ps, approximate_pvalue(stat, length(test)))
    end
    return DataFrame(method = method, mean_statistic = mean(stats), mean_pvalue = mean(ps))
end

"""
    fit_burden(G, y; method, weights)

根据指定折叠法计算权重并估计效应向量。
"""
function fit_burden(G::AbstractMatrix, y::AbstractVector; method::Symbol, weights)
    w = method == :cmc ? cmc_weights(G, weights) : method == :skat ? skat_weights(G, weights) : weights
    scaled = G .* w'
    β = pinv(scaled' * scaled + I * 1e-6) * (scaled' * y)
    return β
end

"""
    cmc_weights(G, base)

组合多位点载荷，提升稀有变异贡献。
"""
function cmc_weights(G::AbstractMatrix, base)
    maf = vec(mean(G, dims = 1) ./ 2)
    return base .* (1 .- maf)
end

"""
    skat_weights(G, base)

采用 Beta 分布型权重近似 SKAT 权重策略。
"""
function skat_weights(G::AbstractMatrix, base)
    maf = vec(mean(G, dims = 1) ./ 2)
    w = @. (maf^(-0.5)) * base
    return clamp.(w, 0.1, 10.0)
end

"""
    approximate_pvalue(stat, df)

基于卡方分布近似计算 p 值，适用于折叠残差统计。
"""
function approximate_pvalue(stat::Real, df::Int)
    return 1 - cdf(Chisq(df), stat)
end

end # module
