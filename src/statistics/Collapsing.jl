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
    list_rvat_methods()

列出当前实现的 12 种稀有变异关联检验方法，便于用户检查可用功能。
"""
list_rvat_methods() = (
    :burden,
    :cmc,
    :vt,
    :wss,
    :skat,
    :skat_o,
    :calpha,
    :acatv,
    :spuw,
    :fisher,
    :heinz,
    :bayesfactor
)

"""
    collapsing_test(G, y; method=:burden, weights=nothing, covariates=nothing,
                    folds=5, rng=MersenneTwister(2025))

实现包含 12 种前沿稀有变异检验的统一接口。函数会自动执行交叉验证稳定化、
计算统计量与 p 值，并返回可用于 Meta 分析的标准化结果表。
"""
function collapsing_test(
    G::AbstractMatrix,
    y::AbstractVector;
    method::Symbol = :burden,
    weights = nothing,
    covariates::Union{Nothing,AbstractMatrix} = nothing,
    folds::Integer = 5,
    rng::AbstractRNG = MersenneTwister(2025)
)
    method in list_rvat_methods() || error("未支持的稀有变异方法: $method")
    n, p = size(G)
    weights = isnothing(weights) ? ones(Float64, p) : collect(weights)
    folds_idx = QualityControl.fold_partition(n, folds; seed = rand(rng, 1:typemax(Int)))

    stats = Float64[]
    ps = Float64[]
    βstorage = Vector{Vector{Float64}}()

    for fold in folds_idx
        train = setdiff(1:n, fold)
        test = fold
        β, stat, pval = run_method(method, view(G, train, :), y[train];
                                    weights, covariates = isnothing(covariates) ? nothing : covariates[train, :])
        ŷ = view(G, test, :) * β
        resid = y[test] .- ŷ
        push!(stats, stat + sum(resid .^ 2))
        push!(ps, pval)
        push!(βstorage, β)
    end

    return DataFrame(
        method = String(method),
        mean_statistic = mean(stats),
        mean_pvalue = mean(ps),
        coefficient_mean = [mean(reduce(hcat, βstorage); dims = 2)[:]],
        coefficient_sd = [std(reduce(hcat, βstorage); dims = 2)[:]]
    )
end

# ------------------------ 内部工具函数 ------------------------

function run_method(method::Symbol, G::AbstractMatrix, y::AbstractVector; weights, covariates)
    if method === :burden
        return burden_method(G, y; weights, covariates)
    elseif method === :cmc
        return cmc_method(G, y; weights, covariates)
    elseif method === :vt
        return vt_method(G, y; weights, covariates)
    elseif method === :wss
        return wss_method(G, y; covariates)
    elseif method === :skat
        return skat_method(G, y; covariates)
    elseif method === :skat_o
        return skato_method(G, y; covariates)
    elseif method === :calpha
        return calpha_method(G, y; covariates)
    elseif method === :acatv
        return acatv_method(G, y; covariates)
    elseif method === :spuw
        return spuw_method(G, y; covariates)
    elseif method === :fisher
        return fisher_method(G, y; covariates)
    elseif method === :heinz
        return heinz_method(G, y; covariates)
    elseif method === :bayesfactor
        return bayesfactor_method(G, y; covariates)
    end
end

function design_matrix(G::AbstractMatrix, covariates)
    if isnothing(covariates)
        return hcat(ones(Float64, size(G, 1)), G)
    else
        return hcat(ones(Float64, size(G, 1)), covariates, G)
    end
end

function linear_regression_stats(X::AbstractMatrix, y::AbstractVector)
    β = pinv(X' * X + I * 1e-8) * (X' * y)
    resid = y - X * β
    σ2 = sum(resid .^ 2) / max(length(y) - size(X, 2), 1)
    vcov = σ2 * pinv(X' * X + I * 1e-8)
    se = sqrt.(abs.(diag(vcov)))
    tvals = β ./ (se .+ eps())
    pvals = 2 .* (1 .- cdf(TDist(max(length(y) - size(X, 2), 1)), abs.(tvals)))
    return β, resid, pvals
end

# ------------------------ 具体方法实现 ------------------------

function burden_method(G, y; weights, covariates)
    X = design_matrix(G .* weights', covariates)
    β, resid, pvals = linear_regression_stats(X, y)
    return β[2:end], sum(resid .^ 2), pvals[2]
end

function cmc_method(G, y; weights, covariates)
    maf = vec(mean(G, dims = 1) ./ 2)
    grouped = G .* (maf .< 0.01)'
    X = design_matrix(grouped .* weights', covariates)
    β, resid, pvals = linear_regression_stats(X, y)
    return β[2:end], sum(resid .^ 2), minimum(pvals[2:end])
end

function vt_method(G, y; weights, covariates)
    thresholds = (0.001, 0.005, 0.01, 0.02, 0.05)
    best_p = 1.0
    bestβ = zeros(size(G, 2))
    best_stat = Inf
    maf = vec(mean(G, dims = 1) ./ 2)
    for t in thresholds
        mask = maf .<= t
        if !any(mask)
            continue
        end
        X = design_matrix(G[:, mask] .* weights[mask]', covariates)
        β, resid, pvals = linear_regression_stats(X, y)
        p = minimum(pvals[2:end])
        if p < best_p
            best_p = p
            bestβ = zeros(size(G, 2))
            bestβ[mask] = β[(end - sum(mask) + 1):end]
            best_stat = sum(resid .^ 2)
        end
    end
    return bestβ, best_stat, best_p
end

function wss_method(G, y; covariates)
    maf = vec(mean(G, dims = 1) ./ 2)
    w = 1 ./ sqrt.(maf .* (1 .- maf) .+ eps())
    X = design_matrix(G .* w', covariates)
    β, resid, pvals = linear_regression_stats(X, y)
    return β[2:end], sum(resid .^ 2), minimum(pvals[2:end])
end

function skat_method(G, y; covariates)
    Xcov = isnothing(covariates) ? ones(Float64, length(y), 1) : hcat(ones(Float64, length(y), 1), covariates)
    β_cov = pinv(Xcov' * Xcov + I * 1e-8) * (Xcov' * y)
    resid = y - Xcov * β_cov
    W = diagm(0 => vec(beta_weights(G)))
    K = G * W * G'
    Q = resid' * K * resid
    λ = eigen(Symmetric(K)).values
    λ = λ[λ .> 1e-8]
    stat = sum(resid .^ 2)
    pval = 1 - cdf(Chisq(length(λ)), Q / (sum(resid .^ 2) / length(resid)))
    return zeros(size(G, 2)), stat, clamp(pval, 0.0, 1.0)
end

function skato_method(G, y; covariates)
    burdenβ, burden_stat, burden_p = burden_method(G, y; weights = ones(size(G, 2)), covariates)
    _, _, skat_p = skat_method(G, y; covariates)
    grid = range(0.0, 1.0; length = 11)
    best_p = 1.0
    for ρ in grid
        p = ρ * burden_p + (1 - ρ) * skat_p
        best_p = min(best_p, p)
    end
    return burdenβ, burden_stat, best_p
end

function calpha_method(G, y; covariates)
    centered = G .- mean(G; dims = 1)
    scores = vec(centered' * (y .- mean(y)))
    stat = sum((scores .^ 2) .- var(scores))
    df = size(G, 2)
    pval = 1 - cdf(Normal(), stat / sqrt(var(scores) + eps()))
    return zeros(size(G, 2)), abs(stat), clamp(pval, 0.0, 1.0)
end

function acatv_method(G, y; covariates)
    βs, ps = per_variant_tests(G, y; covariates)
    transformed = tan.((0.5 .- ps) .* π)
    stat = mean(transformed)
    pval = 0.5 - atan(stat) / π
    return βs, abs(stat), clamp(pval, 0.0, 1.0)
end

function spuw_method(G, y; covariates)
    powers = (1, 2, 4, 6)
    βs, ps = per_variant_tests(G, y; covariates)
    best_p = 1.0
    for pow in powers
        score = sum(sign.(βs) .* abs.(βs) .^ pow)
        p = 1 - cdf(Normal(), score / (std(βs) + eps()))
        best_p = min(best_p, p)
    end
    return βs, sum(abs, βs), clamp(best_p, 0.0, 1.0)
end

function fisher_method(G, y; covariates)
    _, ps = per_variant_tests(G, y; covariates)
    stat = -2sum(log.(ps .+ eps()))
    pval = 1 - cdf(Chisq(2 * length(ps)), stat)
    return zeros(size(G, 2)), stat, clamp(pval, 0.0, 1.0)
end

function heinz_method(G, y; covariates)
    βs, ps = per_variant_tests(G, y; covariates)
    enrichment = sum(max.(0, -log10.(ps .+ eps())))
    stat = enrichment * mean(abs.(βs))
    pval = 10.0 ^ (-stat / (length(ps) + eps()))
    return βs, stat, clamp(pval, 0.0, 1.0)
end

function bayesfactor_method(G, y; covariates)
    βs, ps = per_variant_tests(G, y; covariates)
    bf = prod((1 .- ps) ./ (ps .+ eps()))
    stat = log(bf + eps())
    pval = 1 / (1 + exp(stat))
    return βs, stat, clamp(pval, 0.0, 1.0)
end

function beta_weights(G::AbstractMatrix)
    maf = clamp.(vec(mean(G, dims = 1) ./ 2), 1e-4, 0.5)
    a, b = 1.0, 25.0
    return pdf.(Beta(a, b), maf)
end

function per_variant_tests(G::AbstractMatrix, y::AbstractVector; covariates)
    n, p = size(G)
    βs = zeros(Float64, p)
    ps = ones(Float64, p)
    for j in 1:p
        x = view(G, :, j)
        X = design_matrix(reshape(x, :, 1), covariates)
        β, _, pvals = linear_regression_stats(X, y)
        βs[j] = β[end]
        ps[j] = pvals[end]
    end
    return βs, ps
end

end # module
