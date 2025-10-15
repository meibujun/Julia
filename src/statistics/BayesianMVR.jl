module BayesianMVR

using LinearAlgebra
using Random
using Distributions
using StatsBase
using DataFrames
using SparseArrays
using Statistics
using ..MathUtils

"""
    bayesian_mvr!(G, y, X; priors = Dict(:σ2 => 1.0, :τ2 => 1.0), iterations = 2000, burnin = 500)

基于 Gibbs 采样的贝叶斯多元回归，用于稀有变异上位性效应估计。
支持固定协变量矩阵 X（可为空）。
返回包含后验均值、方差以及马尔科夫链诊断指标的数据框。
"""
function bayesian_mvr!(G::AbstractMatrix, y::AbstractVector, X::Union{AbstractMatrix,Nothing};
                        priors = Dict(:σ2 => 1.0, :τ2 => 1.0), iterations::Int = 2000, burnin::Int = 500)
    n, p = size(G)
    Xmat = isnothing(X) ? ones(n, 1) : X
    β = zeros(Float64, p)
    γ = zeros(Float64, size(Xmat, 2))
    σ2 = priors[:σ2]
    τ2 = priors[:τ2]
    draws = zeros(Float64, iterations - burnin, p)
    for iter in 1:iterations
        Vβ = inv((G' * G) / σ2 + I / τ2)
        mβ = Vβ * (G' * (y - Xmat * γ)) / σ2
        β = rand(MvNormal(mβ, Symmetric((Vβ + Vβ') / 2)))
        Vγ = inv((Xmat' * Xmat) / σ2)
        mγ = Vγ * (Xmat' * (y - G * β)) / σ2
        γ = rand(MvNormal(mγ, Symmetric((Vγ + Vγ') / 2)))
        resid = y - G * β - Xmat * γ
        shape = (n + p) / 2
        scale = 2 / (resid' * resid + β' * β / τ2)
        σ2 = 1 / rand(Gamma(shape, scale))
        if iter > burnin
            draws[iter - burnin, :] .= β
        end
    end
    post_mean = vec(mean(draws, dims = 1))
    post_sd = vec(std(draws, dims = 1))
    ess = map(j -> effective_sample_size(draws[:, j]), 1:p)
    return DataFrame(marker = 1:p, posterior_mean = post_mean, posterior_sd = post_sd, ess = ess)
end

function effective_sample_size(chain::AbstractVector)
    n = length(chain)
    μ = mean(chain)
    γ = 0.0
    for lag in 1:min(100, n - 1)
        autocov = dot(chain[1:end-lag] .- μ, chain[1+lag:end] .- μ) / (n - lag)
        if autocov < 0
            break
        end
        γ += 2 * autocov
    end
    var_chain = var(chain)
    return n * var_chain / (var_chain + γ)
end

end # module
