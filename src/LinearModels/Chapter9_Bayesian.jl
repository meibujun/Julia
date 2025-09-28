module LinearModelsChapter9

using LinearAlgebra
using Statistics
using Distributions
using Random

export conjugate_posterior, gibbs_sampler_linear_model, posterior_predictive

"""
    conjugate_posterior(X, y, μ0, Λ0, α0, β0)

根据章节九的正态-逆伽马共轭分析，计算参数后验分布的闭式表达。
返回 `(μn, Λn, αn, βn)`，分别对应后验均值、精度矩阵与逆伽马参数。
"""
function conjugate_posterior(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, μ0::AbstractVector{<:Real}, Λ0::AbstractMatrix{<:Real}, α0::Real, β0::Real)
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    μ0vec = Vector{Float64}(μ0)
    Λ0mat = Matrix{Float64}(Λ0)
    Λn = Λ0mat + Xmat' * Xmat
    μn = Λn \ (Λ0mat * μ0vec + Xmat' * yvec)
    αn = α0 + length(yvec) / 2
    residual = yvec - Xmat * μn
    βn = β0 + 0.5 * (dot(residual, residual) + (μn - μ0vec)' * Λ0mat * (μn - μ0vec))
    return (μn = μn, Λn = Λn, αn = αn, βn = βn)
end

"""
    gibbs_sampler_linear_model(X, y; n_samples = 2000, burnin = 500, μ0 = nothing, Λ0 = nothing, α0 = 2.0, β0 = 1.0, rng = Random.default_rng())

执行章节九所述的Gibbs采样器，生成参数 `(β, σ²)` 的后验样本，用于演示贝叶斯线性模型推断。
返回一个字典，包含 `β_samples` 与 `σ2_samples`。
"""
function gibbs_sampler_linear_model(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; n_samples::Int = 2000, burnin::Int = 500, μ0 = nothing, Λ0 = nothing, α0::Real = 2.0, β0::Real = 1.0, rng::AbstractRNG = Random.default_rng())
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    p = size(Xmat, 2)
    μ0vec = μ0 === nothing ? zeros(p) : Vector{Float64}(μ0)
    Λ0mat = Λ0 === nothing ? Matrix(I, p, p) : Matrix{Float64}(Λ0)
    β_samples = zeros(n_samples, p)
    σ2_samples = zeros(n_samples)
    β_curr = zeros(p)
    σ2_curr = 1.0
    XtX = Xmat' * Xmat
    Xty = Xmat' * yvec
    for iter in 1:(n_samples + burnin)
        Λn = Λ0mat + XtX
        μn = Λn \ (Λ0mat * μ0vec + Xty)
        covβ = σ2_curr .* inv(Λn)
        β_curr = rand(rng, MvNormal(μn, covβ))
        resid = yvec - Xmat * β_curr
        αn = α0 + length(yvec) / 2
        βn = β0 + 0.5 * dot(resid, resid)
        σ2_curr = 1 / rand(rng, Gamma(αn, 1 / βn))
        if iter > burnin
            idx = iter - burnin
            β_samples[idx, :] .= β_curr
            σ2_samples[idx] = σ2_curr
        end
    end
    return Dict(:β_samples => β_samples, :σ2_samples => σ2_samples)
end

"""
    posterior_predictive(X_new, draws; rng = Random.default_rng())

利用Gibbs采样得到的参数抽样计算后验预测分布的均值与区间。
`draws` 为 `gibbs_sampler_linear_model` 的返回结果，返回 `(mean, lower, upper)`。
"""
function posterior_predictive(X_new::AbstractMatrix{<:Real}, draws::Dict{Symbol, Any}; rng::AbstractRNG = Random.default_rng())
    Xmat = Matrix{Float64}(X_new)
    β_samples = draws[:β_samples]
    σ2_samples = draws[:σ2_samples]
    n_draws = size(β_samples, 1)
    n_obs = size(Xmat, 1)
    predictive_draws = zeros(n_obs, n_draws)
    for i in 1:n_draws
        mean_vec = Xmat * β_samples[i, :]
        predictive_draws[:, i] .= mean_vec .+ sqrt(σ2_samples[i]) .* randn(rng, n_obs)
    end
    mean_pred = mean(predictive_draws, dims = 2)
    lower = mapslices(v -> quantile(v, 0.025), predictive_draws; dims = 2)
    upper = mapslices(v -> quantile(v, 0.975), predictive_draws; dims = 2)
    return (mean = vec(mean_pred), lower = vec(lower), upper = vec(upper))
end

end # module
