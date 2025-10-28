# CoreAlgorithm.jl - 核心算法模块
# ==========================================================
# 本文件经过重大修正，以实现 LASSO 和 ElasticNet 模型的完整功能，
# 替换了之前的占位符代码。使用了坐标下降算法。
# ==========================================================

module CoreAlgorithm

# --- 1. 导入依赖 ---
using ..GenomicPrediction: AbstractModel, GenomicData, fit!, predict
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Random
using StatsBase: Weights, sample

# --- 2. 模块接口 ---
export GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel

# --- 3. GBLUP 模型实现 (无变动) ---
@doc raw"""
    GBLUPModel(lambda::Float64)
"""
mutable struct GBLUPModel <: AbstractModel
    lambda::Float64; effects::Vector{Float64}; intercept::Float64; allele_freqs::Vector{Float64}
    GBLUPModel(lambda) = new(lambda, [], 0.0, [])
end
function fit!(model::GBLUPModel, data::GenomicData; rng=nothing)
    y = data.phenotypes[!, 2]; G = Matrix(data.genotypes[!, 2:end]); n, m = size(G)
    p = mean(G, dims=1) ./ 2; model.allele_freqs = vec(p); M = G .- (2 .* p)
    denominator = 2 * sum(p .* (1 .- p)); K = (M * M') / denominator
    X = ones(n, 1); Z = Matrix{Float64}(I, n, n)
    LHS = Z' * Z * K + I * model.lambda; RHS = Z' * (y .- mean(y))
    u = LHS \ RHS
    model.effects = (M' / (M * M')) * u; model.intercept = mean(y)
    println("GBLUP 模型训练完成。"); return nothing
end
function predict(model::GBLUPModel, new_data::DataFrame)
    M_new = Matrix(new_data[!, 2:end]) .- (2 .* model.allele_freqs')
    return model.intercept .+ M_new * model.effects
end

# --- 贝叶斯模型 (无变动) ---
abstract type BayesianModel <: AbstractModel end
# (BayesA, B, C, R and gibbs_sampler implementations are omitted for brevity, but are unchanged)
# ... [Bayesian models code as before] ...

# --- 10. LASSO 和 ElasticNet 实现 ---

# 坐标下降法的通用求解器
function coordinate_descent!(beta::Vector{Float64}, intercept::Float64, G::Matrix, y_centered::Vector, lambda1::Float64, lambda2::Float64, max_iters::Int)
    n, m = size(G)
    G_col_sq_sum = vec(sum(G.^2, dims=1))

    println("开始坐标下降 (最大迭代次数: $max_iters)...")
    for iter in 1:max_iters
        max_change = 0.0
        for j in 1:m
            old_beta_j = beta[j]

            # 计算不含第 j 个特征的预测残差
            r = y_centered - (G * beta - G[:, j] * beta[j])
            rho_j = dot(G[:, j], r)

            # 应用软阈值 (L1 惩罚) 和 L2 惩罚
            if rho_j > lambda1
                beta[j] = (rho_j - lambda1) / (G_col_sq_sum[j] + lambda2)
            elseif rho_j < -lambda1
                beta[j] = (rho_j + lambda1) / (G_col_sq_sum[j] + lambda2)
            else
                beta[j] = 0.0
            end

            max_change = max(max_change, abs(beta[j] - old_beta_j))
        end

        # 收敛检查
        if max_change < 1e-4; println("在第 $iter 次迭代收敛。"); break; end
        if iter == max_iters; println("达到最大迭代次数。"); end
    end
end

@doc raw"""
    LASSOModel(; lambda=0.1, max_iters=100)
"""
mutable struct LASSOModel <: AbstractModel
    lambda::Float64
    max_iters::Int
    effects::Vector{Float64}
    intercept::Float64

    LASSOModel(; lambda=0.1, max_iters=100) = new(lambda, max_iters, [], 0.0)
end

function fit!(model::LASSOModel, data::GenomicData; rng=nothing)
    y = data.phenotypes[!, 2]
    G = Matrix(data.genotypes[!, 2:end])

    # 中心化数据
    model.intercept = mean(y)
    y_centered = y .- model.intercept

    # 初始化效应
    model.effects = zeros(size(G, 2))

    # L1 惩罚是 lambda * n, L2 惩罚是 0
    lambda1 = model.lambda * size(G, 1)

    coordinate_descent!(model.effects, model.intercept, G, y_centered, lambda1, 0.0, model.max_iters)
end

@doc raw"""
    ElasticNetModel(; lambda=0.1, alpha=0.5, max_iters=100)
"""
mutable struct ElasticNetModel <: AbstractModel
    lambda::Float64
    alpha::Float64 # L1 和 L2 惩罚的混合比例
    max_iters::Int
    effects::Vector{Float64}
    intercept::Float64

    ElasticNetModel(; lambda=0.1, alpha=0.5, max_iters=100) = new(lambda, alpha, max_iters, [], 0.0)
end

function fit!(model::ElasticNetModel, data::GenomicData; rng=nothing)
    y = data.phenotypes[!, 2]
    G = Matrix(data.genotypes[!, 2:end])

    model.intercept = mean(y)
    y_centered = y .- model.intercept
    model.effects = zeros(size(G, 2))

    # 分解 lambda 到 L1 和 L2
    # lambda1 (L1) = n * lambda * alpha
    # lambda2 (L2) = n * lambda * (1-alpha)
    n = size(G, 1)
    lambda1 = n * model.lambda * model.alpha
    lambda2 = n * model.lambda * (1 - model.alpha)

    coordinate_descent!(model.effects, model.intercept, G, y_centered, lambda1, lambda2, model.max_iters)
end

function predict(model::Union{LASSOModel, ElasticNetModel}, new_data::DataFrame)
    G_new = Matrix(new_data[!, 2:end])
    return model.intercept .+ G_new * model.effects
end


# --- Re-add Bayesian Models for completeness of the file ---
# (This section is unchanged from the previous correct version)

abstract type BayesianModel <: AbstractModel end

function gibbs_sampler!(model::BayesianModel, y::Vector, G::Matrix; rng::AbstractRNG)
    n, m = size(G)
    beta = zeros(m); mu = mean(y); var_e = var(y) * 0.5; var_g = init_genetic_variance(model, var(y))
    println("开始 $(typeof(model)) MCMC 抽样 (迭代次数: $(model.iterations))...")
    for iter in 1:model.iterations
        e = y .- mu .- G * beta; mu = rand(rng, Normal(mean(y .- G * beta), sqrt(var_e / n))); e = y .- mu .- G * beta
        for j in 1:m
            e .+= G[:, j] * beta[j]; var_beta_j = sample_marker_variance(model, j, beta, var_g, rng)
            lhs = dot(G[:, j], G[:, j]) / var_e + 1.0 / var_beta_j; rhs = dot(G[:, j], e) / var_e
            beta[j] = rand(rng, Normal(rhs / lhs, sqrt(1.0 / lhs))); e .-= G[:, j] * beta[j]
        end
        var_g = sample_genetic_variance(model, beta, var_g, rng)
        shape_e = model.nu_e / 2 + n / 2; scale_e = (dot(e, e) + model.nu_e * model.s2_e) / 2
        var_e = rand(rng, InverseGamma(shape_e, scale_e))
        if iter > model.burn_in && (iter - model.burn_in) % model.thin == 0
            push!(model.beta_samples, copy(beta))
        end
        if iter % 100 == 0; println("迭代: $iter / $(model.iterations)..."); end
    end
    model.effects = mean(model.beta_samples); model.intercept = mu; println("MCMC 抽样完成。")
end

mutable struct BayesAModel <: BayesianModel
    nu_beta::Float64; s2_beta::Float64; nu_e::Float64; s2_e::Float64; iterations::Int; burn_in::Int; thin::Int
    effects::Vector{Float64}; intercept::Float64; beta_samples::Vector{Vector{Float64}}; marker_vars::Vector{Float64}
    BayesAModel(; nu_beta=5.0, s2_beta=0.01, nu_e=5.0, s2_e=0.1, iterations=2000, burn_in=500, thin=5) = new(nu_beta, s2_beta, nu_e, s2_e, iterations, burn_in, thin, [], 0.0, [], [])
end
init_genetic_variance(model::BayesAModel, v) = model.s2_beta
sample_marker_variance(model::BayesAModel, j::Int, beta, var_g, rng) = rand(rng, InverseGamma(model.nu_beta / 2 + 0.5, (beta[j]^2 + model.nu_beta * model.s2_beta) / 2))
sample_genetic_variance(model::BayesAModel, beta, var_g, rng) = var_g

mutable struct BayesBModel <: BayesianModel
    pi::Float64; nu_beta::Float64; s2_beta::Float64; nu_e::Float64; s2_e::Float64; iterations::Int; burn_in::Int; thin::Int
    effects::Vector{Float64}; intercept::Float64; beta_samples::Vector{Vector{Float64}}
    BayesBModel(; pi=0.95, nu_beta=5.0, s2_beta=0.01, nu_e=5.0, s2_e=0.1, iterations=2000, burn_in=500, thin=5) = new(pi, nu_beta, s2_beta, nu_e, s2_e, iterations, burn_in, thin, [], 0.0, [])
end
init_genetic_variance(model::BayesBModel, v) = model.s2_beta
sample_marker_variance(model::BayesBModel, j::Int, beta, var_g, rng) = rand(rng) < model.pi ? 1e-12 : rand(rng, InverseGamma(model.nu_beta / 2 + 0.5, (beta[j]^2 + model.nu_beta * model.s2_beta) / 2))
sample_genetic_variance(model::BayesBModel, beta, var_g, rng) = var_g

mutable struct BayesCModel <: BayesianModel
    pi::Float64; nu_g::Float64; s2_g::Float64; nu_e::Float64; s2_e::Float64; iterations::Int; burn_in::Int; thin::Int
    effects::Vector{Float64}; intercept::Float64; beta_samples::Vector{Vector{Float64}}; marker_in_model::Vector{Bool}
    BayesCModel(; pi=0.95, nu_g=5.0, s2_g=0.01, nu_e=5.0, s2_e=0.1, iterations=2000, burn_in=500, thin=5) = new(pi, nu_g, s2_g, nu_e, s2_e, iterations, burn_in, thin, [], 0.0, [], [])
end
init_genetic_variance(model::BayesCModel, v) = model.s2_g
function sample_marker_variance(model::BayesCModel, j::Int, beta, var_g, rng)
    log_p1 = log(1.0 - model.pi) + logpdf(Normal(0, sqrt(var_g)), beta[j]); log_p0 = log(model.pi) + logpdf(Normal(0, sqrt(var_g * 1e-6)), beta[j])
    prob_in = exp(log_p1) / (exp(log_p1) + exp(log_p0))
    model.marker_in_model[j] = rand(rng) < prob_in
    return model.marker_in_model[j] ? var_g : 1e-12
end
function sample_genetic_variance(model::BayesCModel, beta, var_g, rng)
    beta_in_model = beta[model.marker_in_model]; num_in_model = length(beta_in_model)
    shape = model.nu_g / 2 + num_in_model / 2; scale = (dot(beta_in_model, beta_in_model) + model.nu_g * model.s2_g) / 2
    return rand(rng, InverseGamma(shape, scale))
end

mutable struct BayesRModel <: BayesianModel
    mixture_pis::Vector{Float64}; mixture_vars::Vector{Float64}; nu_e::Float64; s2_e::Float64; iterations::Int; burn_in::Int; thin::Int
    effects::Vector{Float64}; intercept::Float64; beta_samples::Vector{Vector{Float64}}
    BayesRModel(; mixture_pis=[0.5, 0.2, 0.2, 0.1], mixture_vars=[0.0, 0.0001, 0.001, 0.01], nu_e=5.0, s2_e=0.1, iterations=2000, burn_in=500, thin=5) = new(mixture_pis, mixture_vars, nu_e, s2_e, iterations, burn_in, thin, [], 0.0, [])
end
init_genetic_variance(model::BayesRModel, v) = v * 0.5
function sample_marker_variance(model::BayesRModel, j::Int, beta, var_g, rng)
    log_probs = log.(model.mixture_pis)
    for k in 1:length(model.mixture_vars); var_k = model.mixture_vars[k] * var_g + 1e-12; log_probs[k] += logpdf(Normal(0, sqrt(var_k)), beta[j]); end
    probs = exp.(log_probs .- maximum(log_probs)); probs ./= sum(probs)
    component = sample(rng, 1:length(probs), Weights(probs))
    return model.mixture_vars[component] * var_g + 1e-12
end
sample_genetic_variance(model::BayesRModel, beta, var_g, rng) = var_g

function fit!(model::BayesianModel, data::GenomicData; rng = Random.GLOBAL_RNG)
    y = data.phenotypes[!, 2]; G = Matrix(data.genotypes[!, 2:end])
    if typeof(model) == BayesAModel; model.marker_vars = ones(size(G, 2)); end
    if typeof(model) == BayesCModel; model.marker_in_model = trues(size(G, 2)); end
    gibbs_sampler!(model, y, G; rng=rng)
end
function predict(model::BayesianModel, new_data::DataFrame)
    return model.intercept .+ Matrix(new_data[!, 2:end]) * model.effects
end


end # module CoreAlgorithm
