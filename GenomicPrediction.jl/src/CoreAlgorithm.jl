#=###############################################################################
# 核心算法模块
# 实现经典线性模型、稀疏正则化模型以及贝叶斯方法, 并提供统一的 `fit!` / `predict` 接口。
###############################################################################=#

using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distributions
using Base.Threads

"""
    abstract type PredictionModel end

抽象类型, 所有可训练模型均需继承, 以统一 `fit!` / `predict` 接口。
"""
abstract type PredictionModel end

# -------------------------- 公共辅助函数 --------------------------

"""
    _select_observations(X, y) -> (X_obs, y_obs, mask)

剔除 `y` 中的 NaN 观测, 返回筛选后的矩阵、向量及布尔掩码。
"""
function _select_observations(X::AbstractMatrix, y::AbstractVector)
    mask = .!isnan.(y)
    X_obs = @views X[mask, :]
    y_obs = y[mask]
    return X_obs, y_obs, mask
end

"""
    _standardize_design(X) -> (Z, means, stds)

对设计矩阵按列中心化并标准化, 避免共线问题, 同时返回均值与标准差供预测时复用。
"""
function _standardize_design(X::AbstractMatrix)
    Z = Array{Float64}(undef, size(X))
    p = size(X, 2)
    means = zeros(p)
    stds = ones(p)
    Threads.@threads for j in 1:p
        column = @views X[:, j]
        mask = .!isnan.(column)
        μ = any(mask) ? mean(column[mask]) : 0.0
        σ = any(mask) ? std(column[mask]; corrected=false) : 1.0
        σ = σ == 0.0 ? 1.0 : σ
        means[j] = μ
        stds[j] = σ
        zcol = similar(column, Float64)
        zcol[mask] = (column[mask] .- μ) ./ σ
        zcol[.!mask] .= 0.0
        Z[:, j] = zcol
    end
    return Z, means, stds
end

"""
    _ridge_solver(Z, y, λ) -> Vector{Float64}

高效求解带 L2 正则的正规方程。针对 p≫n 场景使用对偶形式, 避免大矩阵求逆。
"""
function _ridge_solver(Z::AbstractMatrix, y::AbstractVector, λ::Real)
    n, p = size(Z)
    λ = float(λ)
    if p <= n
        K = Symmetric(Z' * Z + λ * I)
        return K \ (Z' * y)
    else
        G = Symmetric(Z * Z' + λ * I)
        dual = G \ y
        return Z' * dual
    end
end

"""
    _soft_threshold(x, γ)

LASSO/Elastic Net 坐标下降使用的软阈值算子。
"""
_soft_threshold(x, γ) = sign(x) * max(abs(x) - γ, 0.0)

# -------------------------- 线性模型 --------------------------

Base.@kwdef mutable struct GBLUPModel <: PredictionModel
    λ::Float64 = 1.0
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

Base.@kwdef mutable struct RidgeRegressionModel <: PredictionModel
    λ::Float64 = 1.0
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

Base.@kwdef mutable struct LassoModel <: PredictionModel
    λ::Float64 = 0.1
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

Base.@kwdef mutable struct ElasticNetModel <: PredictionModel
    λ1::Float64 = 0.05
    λ2::Float64 = 0.05
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

Base.@kwdef mutable struct BayesAModel <: PredictionModel
    ν::Float64 = 4.0
    S::Float64 = 1.0
    iterations::Int = 1000
    burn_in::Int = 200
    thinning::Int = 2
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

Base.@kwdef mutable struct BayesBModel <: PredictionModel
    π::Float64 = 0.5
    ν::Float64 = 4.0
    S::Float64 = 1.0
    iterations::Int = 1000
    burn_in::Int = 200
    thinning::Int = 2
    coefficients::Vector{Float64} = Float64[]
    intercept::Float64 = 0.0
    feature_means::Vector{Float64} = Float64[]
    feature_stds::Vector{Float64} = Float64[]
    fitted::Bool = false
end

# -------------------------- 训练接口实现 --------------------------

"""
    fit!(model::RidgeRegressionModel, X, y; λ=model.λ)

对输入矩阵执行岭回归拟合, 自动处理缺失表型并保存特征缩放参数。
"""
function fit!(model::RidgeRegressionModel, X::AbstractMatrix, y::AbstractVector; λ::Real = model.λ)
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    β = _ridge_solver(Z, y_center, λ)
    model.coefficients = β
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.λ = float(λ)
    model.fitted = true
    return model
end

"""
    fit!(model::GBLUPModel, X, y; λ=model.λ)

GBLUP 采用与岭回归相同的对偶形式, 但默认正则项按标记数量缩放以匹配基因组选择中的遗传方差假设。
"""
function fit!(model::GBLUPModel, X::AbstractMatrix, y::AbstractVector; λ::Real = model.λ)
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    scale = size(Z, 2)
    β = _ridge_solver(Z, y_center, λ / max(scale, 1))
    model.coefficients = β
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.λ = float(λ)
    model.fitted = true
    return model
end

"""
    fit!(model::LassoModel, X, y; λ=model.λ, maxiter=20_000, tol=1e-5)

使用坐标下降求解 LASSO, 支持提前停止以提升效率。
"""
function fit!(model::LassoModel, X::AbstractMatrix, y::AbstractVector;
              λ::Real = model.λ, maxiter::Integer = 20_000, tol::Real = 1e-5)
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    n, p = size(Z)
    β = zeros(p)
    update = true
    iter = 0
    y_pred = Z * β
    while update && iter < maxiter
        update = false
        iter += 1
        for j in 1:p
            residual = y_center .- y_pred + β[j] .* Z[:, j]
            ρ = dot(Z[:, j], residual) / n
            newβ = _soft_threshold(ρ, λ)
            if abs(newβ - β[j]) > tol
                y_pred .+= (newβ - β[j]) .* Z[:, j]
                β[j] = newβ
                update = true
            end
        end
    end
    model.coefficients = β
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.λ = float(λ)
    model.fitted = true
    return model
end

"""
    fit!(model::ElasticNetModel, X, y; λ1=model.λ1, λ2=model.λ2,
         maxiter=20_000, tol=1e-5)

Elastic Net 坐标下降, 兼顾 L1 稀疏性与 L2 稳定性。
"""
function fit!(model::ElasticNetModel, X::AbstractMatrix, y::AbstractVector;
              λ1::Real = model.λ1, λ2::Real = model.λ2,
              maxiter::Integer = 20_000, tol::Real = 1e-5)
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    n, p = size(Z)
    β = zeros(p)
    iter = 0
    update = true
    y_pred = Z * β
    while update && iter < maxiter
        iter += 1
        update = false
        for j in 1:p
            residual = y_center .- y_pred + β[j] .* Z[:, j]
            ρ = dot(Z[:, j], residual) / n
            newβ = _soft_threshold(ρ, λ1) / (1 + λ2)
            if abs(newβ - β[j]) > tol
                y_pred .+= (newβ - β[j]) .* Z[:, j]
                β[j] = newβ
                update = true
            end
        end
    end
    model.coefficients = β
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.λ1 = float(λ1)
    model.λ2 = float(λ2)
    model.fitted = true
    return model
end

# -------------------------- 贝叶斯模型 --------------------------

"""
    _bayesian_linear_update!(β, τ2, σ2, Z, y, rng)

BayesA/B 共用的系数采样逻辑, 使用逐列 Gibbs 更新。
"""
function _bayesian_linear_update!(β::Vector{Float64}, τ2::Vector{Float64}, σ2::Float64,
                                  Z::AbstractMatrix, y::AbstractVector, rng::AbstractRNG)
    n, p = size(Z)
    for j in 1:p
        residual = y .- (Z * β) .+ Z[:, j] .* β[j]
        v_j = 1 / ((sum(abs2, Z[:, j]) / σ2) + 1 / τ2[j])
        m_j = v_j * (dot(Z[:, j], residual) / σ2)
        β[j] = randn(rng) * sqrt(v_j) + m_j
    end
    return β
end

"""
    fit!(model::BayesAModel, X, y; rng=MersenneTwister(42))

实现简化版 BayesA: 每个标记具有独立的尺度参数, 使用 Gibbs 采样估计后验均值。
"""
function fit!(model::BayesAModel, X::AbstractMatrix, y::AbstractVector;
              rng::AbstractRNG = MersenneTwister(42))
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    n, p = size(Z)

    β = zeros(p)
    τ2 = ones(p)
    σ2 = var(y_center)

    samples = zeros(p)
    saved = 0
    for iter in 1:model.iterations
        β = _bayesian_linear_update!(β, τ2, σ2, Z, y_center, rng)
        for j in 1:p
            scale = (β[j]^2 + model.S) / 2
            τ2[j] = rand(rng, InverseGamma((model.ν + 1) / 2, scale))
        end
        rss = sum(abs2, y_center .- Z * β)
        σ2 = rand(rng, InverseGamma((n + p) / 2, (rss + model.S) / 2))
        if iter > model.burn_in && iter % model.thinning == 0
            saved += 1
            samples .+= β
        end
    end
    saved == 0 && (samples .= β; saved = 1)
    model.coefficients = samples ./ saved
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.fitted = true
    return model
end

"""
    fit!(model::BayesBModel, X, y; rng=MersenneTwister(42))

BayesB 在 BayesA 基础上引入稀疏先验, 对每个标记采样是否激活。
"""
function fit!(model::BayesBModel, X::AbstractMatrix, y::AbstractVector;
              rng::AbstractRNG = MersenneTwister(42))
    X_obs, y_obs, _ = _select_observations(X, y)
    Z, means, stds = _standardize_design(X_obs)
    y_mean = mean(y_obs)
    y_center = y_obs .- y_mean
    n, p = size(Z)

    β = zeros(p)
    τ2 = ones(p)
    γ = trues(p)  # 指示变量
    σ2 = var(y_center)

    samples = zeros(p)
    saved = 0
    logitπ = log(model.π / (1 - model.π))
    for iter in 1:model.iterations
        for j in 1:p
            residual = y_center .- (Z * β) .+ Z[:, j] .* β[j]
            sj = sum(abs2, Z[:, j])
            mj = dot(Z[:, j], residual)
            log_odds = logitπ + 0.5 * (mj^2 / (σ2 * (sj + σ2 / τ2[j])) - log(1 + sj * τ2[j] / σ2))
            γ[j] = rand(rng) < (1 / (1 + exp(-log_odds)))
            if γ[j]
                v_j = 1 / ((sj / σ2) + 1 / τ2[j])
                m_j = v_j * (mj / σ2)
                β[j] = randn(rng) * sqrt(v_j) + m_j
            else
                β[j] = 0.0
            end
        end
        active = findall(γ)
        for j in active
            scale = (β[j]^2 + model.S) / 2
            τ2[j] = rand(rng, InverseGamma((model.ν + 1) / 2, scale))
        end
        rss = sum(abs2, y_center .- Z * β)
        σ2 = rand(rng, InverseGamma((n + max(length(active), 1)) / 2, (rss + model.S) / 2))
        if iter > model.burn_in && iter % model.thinning == 0
            saved += 1
            samples .+= β
        end
    end
    saved == 0 && (samples .= β; saved = 1)
    model.coefficients = samples ./ saved
    model.intercept = y_mean
    model.feature_means = means
    model.feature_stds = stds
    model.fitted = true
    return model
end

# -------------------------- 数据集适配器 --------------------------

"""
    fit!(model::PredictionModel, dataset::GenomicDataset; kwargs...)

直接使用 `GenomicDataset` 训练模型, 自动提取矩阵与表型。
"""
function fit!(model::PredictionModel, dataset::GenomicDataset; kwargs...)
    return fit!(model, dataset.genotype, dataset.phenotype; kwargs...)
end

# -------------------------- 预测接口 --------------------------

"""
    _transform_features(X, means, stds) -> Matrix

按照训练时的均值和标准差对新数据执行变换。
"""
function _transform_features(X::AbstractMatrix, means::Vector{Float64}, stds::Vector{Float64})
    size(X, 2) == length(means) || throw(DimensionMismatch("特征数量与训练时不一致"))
    Z = Array{Float64}(undef, size(X))
    Threads.@threads for j in 1:size(X, 2)
        column = X[:, j]
        transformed = (column .- means[j]) ./ stds[j]
        idx = findall(isnan, transformed)
        !isempty(idx) && (transformed[idx] .= 0.0)
        Z[:, j] = transformed
    end
    return Z
end

"""
    predict(model::PredictionModel, X) -> Vector

根据训练好的模型输出预测结果。
"""
function predict(model::PredictionModel, X::AbstractMatrix)
    model.fitted || throw(ArgumentError("模型尚未训练"))
    Z = _transform_features(X, model.feature_means, model.feature_stds)
    return Z * model.coefficients .+ model.intercept
end

function predict(model::PredictionModel, dataset::GenomicDataset)
    return predict(model, dataset.genotype)
end

