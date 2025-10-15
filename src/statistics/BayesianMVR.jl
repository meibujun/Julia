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
    return summarize_posterior(draws)
end

function effective_sample_size(chain::AbstractVector)
    n = length(chain)
    μ = mean(chain)
    γsum = 0.0
    for lag in 1:min(100, n - 1)
        autocov = dot(chain[1:end-lag] .- μ, chain[1+lag:end] .- μ) / (n - lag)
        if autocov < 0
            break
        end
        γsum += 2 * autocov
    end
    var_chain = var(chain)
    return n * var_chain / (var_chain + γsum + eps())
end

"""
    bayesian_blasso!(G, y; λ = 1.0, iterations = 3000, burnin = 1000)

实现贝叶斯套索 (Bayesian LASSO)，利用正态-指数层级模型完成稀有变异效应的
自动收缩，兼顾稳健性与稀疏性。
"""
function bayesian_blasso!(G::AbstractMatrix, y::AbstractVector;
                          λ::Float64 = 1.0, iterations::Int = 3000, burnin::Int = 1000)
    n, p = size(G)
    β = zeros(Float64, p)
    τ = ones(Float64, p)
    σ2 = 1.0
    draws = zeros(Float64, iterations - burnin, p)
    XtX = G' * G
    Xty = G' * y
    for iter in 1:iterations
        Λ = diagm(0 => 1 ./ τ)
        Vβ = inv(XtX / σ2 + Λ)
        mβ = Vβ * (Xty / σ2)
        β = rand(MvNormal(mβ, Symmetric((Vβ + Vβ') / 2)))
        resid = y - G * β
        σ2 = 1 / rand(Gamma((n + p) / 2, 2 / (resid' * resid + β' * Λ * β)))
        for j in 1:p
            τ[j] = 1 / rand(Gamma(1.0, 2 / (λ^2 * abs(β[j]) + eps())))
        end
        if iter > burnin
            draws[iter - burnin, :] .= β
        end
    end
    return summarize_posterior(draws)
end

"""
    bayesian_bayesb!(G, y; π = 0.05, iterations = 4000, burnin = 1500)

实现贝叶斯 BayesB 模型，通过零效应与常态效应混合先验实现稀疏化。
"""
function bayesian_bayesb!(G::AbstractMatrix, y::AbstractVector;
                          π::Float64 = 0.05, iterations::Int = 4000, burnin::Int = 1500)
    n, p = size(G)
    δ = trues(p)
    β = zeros(Float64, p)
    σ2 = 1.0
    τ2 = ones(Float64, p)
    draws = zeros(Float64, iterations - burnin, p)
    XtX = G' * G
    Xty = G' * y
    for iter in 1:iterations
        for j in 1:p
            λ = τ2[j]
            Sj = XtX[j, j]
            rj = Xty[j] - sum(XtX[j, k] * β[k] for k in 1:p if k != j)
            varj = σ2 / (Sj + σ2 / λ)
            meanj = varj * rj / σ2
            prob_active = (1 - π) * pdf(Normal(meanj, sqrt(varj)), 0.0)
            prob_zero = π * pdf(Normal(0, sqrt(σ2 + λ)), 0.0)
            δ[j] = prob_active > prob_zero
            if δ[j]
                β[j] = rand(Normal(meanj, sqrt(varj)))
            else
                β[j] = 0.0
            end
            τ2[j] = 1 / rand(Gamma(1.0, 2 / (β[j]^2 + eps())))
        end
        resid = y - G * β
        σ2 = 1 / rand(Gamma((n + sum(δ)) / 2, 2 / (resid' * resid + sum(β[δ].^2 ./ τ2[δ] .+ eps()))))
        if iter > burnin
            draws[iter - burnin, :] .= β
        end
    end
    return summarize_posterior(draws)
end

function summarize_posterior(draws::AbstractMatrix)
    post_mean = vec(mean(draws, dims = 1))
    post_sd = vec(std(draws, dims = 1))
    ess = map(j -> effective_sample_size(view(draws, :, j)), 1:size(draws, 2))
    return DataFrame(marker = 1:size(draws, 2), posterior_mean = post_mean, posterior_sd = post_sd, ess = ess)
end

end # module
