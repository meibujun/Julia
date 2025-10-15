module EpistasisModels

using LinearAlgebra
using Statistics
using DataFrames
using Random
using Distributions
using StatsBase
using ..MathUtils

"""
    list_epistasis_methods()

列出当前支持的 4 类上位性检测框架，涵盖 GLM 交互、方差组分、互信息与梯度提升方法。
"""
list_epistasis_methods() = (:glm, :variance_component, :mutual_information, :gradient_boosting)

"""
    epistasis_scan(G, y; methods=list_epistasis_methods(), covariates=nothing,
                   max_pairs=500, rng=MersenneTwister(2025))

对给定基因型矩阵执行多方法上位性扫描。函数会智能抽样候选 SNP 对，
对每种方法单独计算统计量与 p 值，并整合为统一结果表。适用于 90K 级别 SNP
在 HPC 环境中的批量分析。
"""
function epistasis_scan(
    G::AbstractMatrix,
    y::AbstractVector;
    methods = list_epistasis_methods(),
    covariates::Union{Nothing,AbstractMatrix} = nothing,
    max_pairs::Integer = 500,
    rng::AbstractRNG = MersenneTwister(2025)
)
    p = size(G, 2)
    pairs = sample_pairs(p, max_pairs; rng)
    results = DataFrame(
        method = String[],
        snp1 = Int[],
        snp2 = Int[],
        statistic = Float64[],
        pvalue = Float64[]
    )
    for (i, j) in pairs
        for method in methods
            stat, pval = dispatch_method(method, view(G, :, i), view(G, :, j), y; covariates, rng)
            push!(results, (String(method), i, j, stat, pval))
        end
    end
    sort!(results, [:method, :pvalue])
    return results
end

# ------------------------ 方法派发 ------------------------

function dispatch_method(method::Symbol, s1, s2, y; covariates, rng)
    if method === :glm
        return glm_method(s1, s2, y; covariates)
    elseif method === :variance_component
        return variance_component_method(s1, s2, y; covariates)
    elseif method === :mutual_information
        return mutual_information_method(s1, s2, y)
    elseif method === :gradient_boosting
        return gradient_boosting_method(s1, s2, y; covariates, rng)
    else
        error("未知的上位性方法: $method")
    end
end

# ------------------------ 候选对抽样 ------------------------

function sample_pairs(p::Int, max_pairs::Int; rng::AbstractRNG)
    max_pairs = min(max_pairs, p * (p - 1) ÷ 2)
    seen = Set{Tuple{Int,Int}}()
    pairs = Vector{Tuple{Int,Int}}()
    while length(pairs) < max_pairs
        i = rand(rng, 1:p)
        j = rand(rng, 1:p)
        i == j && continue
        a, b = i < j ? (i, j) : (j, i)
        if !(a, b) in seen
            push!(pairs, (a, b))
            push!(seen, (a, b))
        end
    end
    return pairs
end

# ------------------------ 具体方法 ------------------------

function glm_method(s1, s2, y; covariates)
    n = length(y)
    X = ones(Float64, n, 1)
    if !isnothing(covariates)
        X = hcat(X, covariates)
    end
    X = hcat(X, s1, s2, s1 .* s2)
    β = pinv(X' * X + I * 1e-8) * (X' * y)
    resid = y - X * β
    σ2 = sum(resid .^ 2) / max(n - size(X, 2), 1)
    vcov = σ2 * pinv(X' * X + I * 1e-8)
    se = sqrt(abs(vcov[end, end]))
    t = β[end] / (se + eps())
    p = 2 * (1 - cdf(TDist(max(n - size(X, 2), 1)), abs(t)))
    return abs(t), clamp(p, 0.0, 1.0)
end

function variance_component_method(s1, s2, y; covariates)
    X = isnothing(covariates) ? ones(Float64, length(y), 1) : hcat(ones(Float64, length(y), 1), covariates)
    β = pinv(X' * X + I * 1e-8) * (X' * y)
    resid = y - X * β
    Z = hcat(s1, s2, s1 .* s2)
    K = Z * Z'
    MathUtils.symmetrize!(K)
    λ = eigen(Symmetric(K)).values
    λ = λ[λ .> 1e-8]
    Q = resid' * K * resid
    stat = Q / (sum(resid .^ 2) / length(resid))
    p = 1 - cdf(Chisq(length(λ)), stat)
    return stat, clamp(p, 0.0, 1.0)
end

function mutual_information_method(s1, s2, y)
    bins = 3
    obs = map(x -> clamp(round(Int, x), 0, 2), s1 .+ s2 .* 3)
    ycat = cut_y(y, bins)
    joint = countmap(zip(obs, ycat))
    total = length(y)
    px = countmap(obs)
    py = countmap(ycat)
    mi = 0.0
    for ((x, yy), freq) in joint
        px_prob = px[x] / total
        py_prob = py[yy] / total
        pij = freq / total
        mi += pij * log(pij / (px_prob * py_prob) + eps())
    end
    stat = mi
    p = exp(-2 * total * mi)
    return stat, clamp(p, 0.0, 1.0)
end

function gradient_boosting_method(s1, s2, y; covariates, rng)
    features = hcat(s1, s2, s1 .* s2)
    if !isnothing(covariates)
        features = hcat(covariates, features)
    end
    preds = zeros(Float64, length(y))
    ν = 0.1
    for _ in 1:25
        resid = y - preds
        β = pinv(features' * features + I * 1e-6) * (features' * resid)
        preds .+= ν .* (features * β)
    end
    mse = mean((y - preds) .^ 2)
    stat = var(y) - mse
    σ = std(y) / sqrt(length(y))
    z = stat / (σ + eps())
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return stat, clamp(p, 0.0, 1.0)
end

function cut_y(y, bins::Int)
    edges = range(extrema(y)...; length = bins + 1)
    labels = Vector{Int}(undef, length(y))
    for (idx, val) in enumerate(y)
        bin = searchsortedfirst(edges, val) - 1
        labels[idx] = clamp(bin, 1, bins)
    end
    return labels
end

end # module
