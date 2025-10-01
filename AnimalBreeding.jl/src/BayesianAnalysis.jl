# 贝叶斯分析模块
# 实现BayesA, BayesB, BayesC, Bayesian LASSO等贝叶斯方法
# 作者：AnimalBreeding.jl 开发团队
# 版本：1.0.0

"""
    BayesianAnalysis 贝叶斯分析模块

    提供基因组选择中的贝叶斯方法实现：
    - BayesA：每个标记具有独立的方差（t分布先验）
    - BayesB：标记效应的两点混合先验（spike and slab）
    - BayesC：共同方差的标记效应混合模型
    - Bayesian LASSO：双指数（Laplace）先验
    - MCMC诊断工具
"""
module BayesianAnalysis

using LinearAlgebra
using Statistics
using Distributions
using Random
using ProgressMeter
using DataFrames

# 导出数据结构
export BayesianResult, MCMCChain
# 导出贝叶斯方法
export run_bayesian_evaluation, bayesA, bayesB, bayesC, bayesian_lasso
# 导出诊断工具
export mcmc_diagnostics, effective_sample_size, gelman_rubin
export plot_trace, plot_posterior

# ==================== 数据结构定义 ====================

"""
    MCMCChain

    MCMC采样链的存储结构
"""
mutable struct MCMCChain
    marker_effects::Matrix{Float64}                   # 标记效应链
    variance_components::Dict{String,Vector{Float64}} # 方差组分链
    hyperparameters::Dict{String,Vector{Float64}}     # 超参数链
    breeding_values::Matrix{Float64}                  # 育种值链
    log_likelihood::Vector{Float64}                   # 对数似然
    n_iter::Int                                       # 总迭代数
    burn_in::Int                                      # 燃烧期
    thin::Int                                         # 抽稀
end

"""
    BayesianResult

    贝叶斯分析结果结构
"""
mutable struct BayesianResult
    method::Symbol                                    # 方法名称
    marker_effects::Vector{Float64}                   # 标记效应均值
    marker_effects_sd::Vector{Float64}                # 标记效应标准差
    marker_pip::Vector{Float64}                       # 后验包含概率
    breeding_values::Vector{Float64}                  # 育种值均值
    breeding_values_sd::Vector{Float64}               # 育种值标准差
    variance_components::Dict{String,Float64}         # 方差组分
    hyperparameters::Dict{String,Float64}             # 超参数
    mcmc_chain::Union{Nothing,MCMCChain}             # MCMC链
    convergence_diagnostics::Dict{String,Any}         # 收敛诊断
    prediction_accuracy::Union{Nothing,Float64}       # 预测准确性
end

# ==================== BayesA实现 ====================

"""
    bayesA(y, X, Z; n_iter, burn_in, thin, df, scale) -> BayesianResult

    BayesA方法实现
"""
function bayesA(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64};
                n_iter::Int=10000, burn_in::Int=2000, thin::Int=5,
                df::Float64=4.0, scale::Float64=0.01,
                verbose::Bool=true, seed::Union{Nothing,Int}=nothing)

    !isnothing(seed) && Random.seed!(seed)
    n, p = size(Z); n_fixed = size(X, 2)
    println("\n开始BayesA分析: $n 个体, $p 标记...")

    β = randn(n_fixed); α = randn(p) * 0.01
    σ²_e = var(y) * 0.5; σ²_α = ones(p) * scale
    ν_e = 10.0; S_e = σ²_e * (ν_e - 2)
    ν_α = df; S_α = scale * (ν_α - 2)

    n_saved = div(n_iter - burn_in, thin)
    marker_effects_chain = zeros(n_saved, p)
    variance_chain = Dict("residual" => zeros(n_saved), "genetic" => zeros(n_saved), "mean_marker_var" => zeros(n_saved))
    breeding_values_chain = zeros(n_saved, n)

    XtX = X' * X; ZtZ = [dot(Z[:, j], Z[:, j]) for j in 1:p]
    iter_saved = 0
    progress = verbose ? Progress(n_iter, desc="BayesA MCMC: ") : nothing

    for iter in 1:n_iter
        residual = y - Z * α
        V_β = inv(XtX / σ²_e + I(n_fixed) * 1e-8)
        μ_β = V_β * (X' * residual / σ²_e)
        β = rand(MvNormal(μ_β, Symmetric(V_β)))

        residual = y - X * β
        for j in 1:p
            residual .+= Z[:, j] * α[j]
            v_j = 1 / (ZtZ[j] / σ²_e + 1 / σ²_α[j])
            μ_j = v_j * (dot(Z[:, j], residual) / σ²_e)
            α[j] = rand(Normal(μ_j, sqrt(v_j)))
            residual .-= Z[:, j] * α[j]
        end

        for j in 1:p
            shape = (ν_α + 1) / 2; scale_post = (ν_α * S_α + α[j]^2) / 2
            σ²_α[j] = rand(InverseGamma(shape, scale_post))
        end

        sse = sum((y - X * β - Z * α).^2)
        shape_e = (ν_e + n) / 2; scale_e = (ν_e * S_e + sse) / 2
        σ²_e = rand(InverseGamma(shape_e, scale_e))

        if iter > burn_in && mod(iter - burn_in, thin) == 0
            iter_saved += 1
            marker_effects_chain[iter_saved, :] = α
            variance_chain["residual"][iter_saved] = σ²_e
            variance_chain["genetic"][iter_saved] = var(Z * α)
            variance_chain["mean_marker_var"][iter_saved] = mean(σ²_α)
            breeding_values_chain[iter_saved, :] = Z * α
        end
        verbose && next!(progress)
    end

    marker_effects_mean = vec(mean(marker_effects_chain, dims=1))
    marker_effects_sd = vec(std(marker_effects_chain, dims=1))
    breeding_values_mean = vec(mean(breeding_values_chain, dims=1))
    breeding_values_sd = vec(std(breeding_values_chain, dims=1))

    mcmc_chain = MCMCChain(marker_effects_chain, variance_chain, Dict(), breeding_values_chain, zeros(n_saved), n_iter, burn_in, thin)

    return BayesianResult(:BayesA, marker_effects_mean, marker_effects_sd, ones(p), breeding_values_mean, breeding_values_sd,
        Dict("residual" => mean(variance_chain["residual"]), "genetic" => mean(variance_chain["genetic"])),
        Dict("df" => df, "scale" => scale), mcmc_chain, calculate_convergence_diagnostics(mcmc_chain), nothing)
end


# ==================== BayesB实现 ====================
function bayesB(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64};
                n_iter::Int=10000, burn_in::Int=2000, thin::Int=5, π::Float64=0.95,
                verbose::Bool=true, seed::Union{Nothing,Int}=nothing)

    !isnothing(seed) && Random.seed!(seed)
    n, p = size(Z); n_fixed = size(X, 2)
    println("\n开始BayesB分析 (π=$π)...")

    β = randn(n_fixed); α = randn(p) * 0.01; δ = rand(Bernoulli(1 - π), p)
    α .*= δ; σ²_e = var(y) * 0.5; σ²_α = ones(p) * 0.01
    ν_e = 10.0; S_e = σ²_e * (ν_e - 2); ν_α = 4.0; S_α = 0.01 * (ν_α - 2)

    n_saved = div(n_iter - burn_in, thin)
    marker_effects_chain = zeros(n_saved, p)
    marker_inclusion_chain = zeros(Bool, n_saved, p)
    variance_chain = Dict("residual" => zeros(n_saved), "genetic" => zeros(n_saved))
    breeding_values_chain = zeros(n_saved, n)
    hyperparameter_chain = Dict("pi" => zeros(n_saved))

    ZtZ = [dot(Z[:, j], Z[:, j]) for j in 1:p]
    iter_saved = 0
    progress = verbose ? Progress(n_iter, desc="BayesB MCMC: ") : nothing

    for iter in 1:n_iter
        residual = y - Z * (α .* δ)
        C_β = inv(X' * X / σ²_e + I(n_fixed) * 1e-8)
        μ_β = C_β * (X' * residual / σ²_e)
        β = rand(MvNormal(μ_β, Symmetric(C_β)))

        residual = y - X * β
        n_included = 0
        for j in 1:p
            residual .+= Z[:, j] * (α[j] * δ[j])
            v_j = 1 / (ZtZ[j] / σ²_e + 1 / σ²_α[j])
            μ_j = v_j * dot(Z[:, j], residual) / σ²_e
            log_odds_in = log(1 - π) - 0.5 * log(v_j * σ²_α[j]) + 0.5 * μ_j^2 / v_j
            prob_in = 1 / (1 + exp(log(π) - log_odds_in))
            δ[j] = rand(Bernoulli(prob_in))

            if δ[j] == 1
                α[j] = rand(Normal(μ_j, sqrt(v_j))); n_included += 1
                σ²_α[j] = rand(InverseGamma((ν_α + 1) / 2, (ν_α * S_α + α[j]^2) / 2))
            else
                α[j] = 0
                σ²_α[j] = rand(InverseGamma(ν_α / 2, ν_α * S_α / 2))
            end
            residual .-= Z[:, j] * (α[j] * δ[j])
        end

        π = rand(Beta(1 + p - n_included, 1 + n_included))
        sse = sum((y - X * β - Z * (α .* δ)).^2)
        σ²_e = rand(InverseGamma((ν_e + n) / 2, (ν_e * S_e + sse) / 2))

        if iter > burn_in && mod(iter - burn_in, thin) == 0
            iter_saved += 1
            marker_effects_chain[iter_saved, :] = α .* δ
            marker_inclusion_chain[iter_saved, :] = δ
            variance_chain["residual"][iter_saved] = σ²_e
            variance_chain["genetic"][iter_saved] = var(Z * (α .* δ))
            breeding_values_chain[iter_saved, :] = Z * (α .* δ)
            hyperparameter_chain["pi"][iter_saved] = π
        end
        verbose && next!(progress)
    end

    marker_effects_mean = vec(mean(marker_effects_chain, dims=1))
    marker_effects_sd = vec(std(marker_effects_chain, dims=1))
    marker_pip = vec(mean(marker_inclusion_chain, dims=1))
    breeding_values_mean = vec(mean(breeding_values_chain, dims=1))
    breeding_values_sd = vec(std(breeding_values_chain, dims=1))

    mcmc_chain = MCMCChain(marker_effects_chain, variance_chain, hyperparameter_chain, breeding_values_chain, zeros(n_saved), n_iter, burn_in, thin)

    return BayesianResult(:BayesB, marker_effects_mean, marker_effects_sd, marker_pip, breeding_values_mean, breeding_values_sd,
        Dict("residual" => mean(variance_chain["residual"]), "genetic" => mean(variance_chain["genetic"])),
        Dict("pi" => mean(hyperparameter_chain["pi"])), mcmc_chain, calculate_convergence_diagnostics(mcmc_chain), nothing)
end

# ==================== BayesC实现 ====================
function bayesC(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64};
                n_iter::Int=10000, burn_in::Int=2000, thin::Int=5, π::Float64=0.95,
                verbose::Bool=true, seed::Union{Nothing,Int}=nothing)

    !isnothing(seed) && Random.seed!(seed)
    n, p = size(Z); n_fixed = size(X, 2)
    println("\n开始BayesC分析 (共同方差)...")

    β = randn(n_fixed); α = randn(p) * 0.01; δ = rand(Bernoulli(1 - π), p)
    α .*= δ; σ²_e = var(y) * 0.5; σ²_α = 0.01
    ν_e = 10.0; S_e = σ²_e * (ν_e - 2); ν_α = 4.0; S_α = σ²_α * (ν_α - 2)

    n_saved = div(n_iter - burn_in, thin)
    marker_effects_chain = zeros(n_saved, p); marker_inclusion_chain = zeros(Bool, n_saved, p)
    variance_chain = Dict("residual" => zeros(n_saved), "genetic" => zeros(n_saved), "marker" => zeros(n_saved))
    breeding_values_chain = zeros(n_saved, n); hyperparameter_chain = Dict("pi" => zeros(n_saved))

    ZtZ = [dot(Z[:, j], Z[:, j]) for j in 1:p]
    iter_saved = 0
    progress = verbose ? Progress(n_iter, desc="BayesC MCMC: ") : nothing

    for iter in 1:n_iter
        residual = y - Z * (α .* δ); C_β = inv(X' * X / σ²_e + I(n_fixed) * 1e-8)
        μ_β = C_β * (X' * residual / σ²_e); β = rand(MvNormal(μ_β, Symmetric(C_β)))

        residual = y - X * β; n_included = 0
        for j in 1:p
            residual .+= Z[:, j] * (α[j] * δ[j])
            v_j = 1 / (ZtZ[j] / σ²_e + 1 / σ²_α); μ_j = v_j * dot(Z[:, j], residual) / σ²_e
            log_odds_in = log(1 - π) - 0.5 * log(v_j * σ²_α) + 0.5 * μ_j^2 / v_j
            prob_in = 1 / (1 + exp(log(π) - log_odds_in)); δ[j] = rand(Bernoulli(prob_in))

            if δ[j] == 1; α[j] = rand(Normal(μ_j, sqrt(v_j))); n_included += 1; else; α[j] = 0; end
            residual .-= Z[:, j] * (α[j] * δ[j])
        end

        if n_included > 0
            ss_α = sum(α[δ .== 1].^2)
            σ²_α = rand(InverseGamma((ν_α + n_included) / 2, (ν_α * S_α + ss_α) / 2))
        end

        π = rand(Beta(1 + p - n_included, 1 + n_included))
        sse = sum((y - X * β - Z * (α .* δ)).^2)
        σ²_e = rand(InverseGamma((ν_e + n) / 2, (ν_e * S_e + sse) / 2))

        if iter > burn_in && mod(iter - burn_in, thin) == 0
            iter_saved += 1
            marker_effects_chain[iter_saved, :] = α .* δ; marker_inclusion_chain[iter_saved, :] = δ
            variance_chain["residual"][iter_saved] = σ²_e; variance_chain["genetic"][iter_saved] = var(Z * (α .* δ))
            variance_chain["marker"][iter_saved] = σ²_α; breeding_values_chain[iter_saved, :] = Z * (α .* δ)
            hyperparameter_chain["pi"][iter_saved] = π
        end
        verbose && next!(progress)
    end

    marker_effects_mean = vec(mean(marker_effects_chain, dims=1)); marker_effects_sd = vec(std(marker_effects_chain, dims=1))
    marker_pip = vec(mean(marker_inclusion_chain, dims=1)); breeding_values_mean = vec(mean(breeding_values_chain, dims=1))
    breeding_values_sd = vec(std(breeding_values_chain, dims=1))
    mcmc_chain = MCMCChain(marker_effects_chain, variance_chain, hyperparameter_chain, breeding_values_chain, zeros(n_saved), n_iter, burn_in, thin)

    return BayesianResult(:BayesC, marker_effects_mean, marker_effects_sd, marker_pip, breeding_values_mean, breeding_values_sd,
        Dict("residual" => mean(variance_chain["residual"]), "genetic" => mean(variance_chain["genetic"]), "marker" => mean(variance_chain["marker"])),
        Dict("pi" => mean(hyperparameter_chain["pi"])), mcmc_chain, calculate_convergence_diagnostics(mcmc_chain), nothing)
end

# ==================== Bayesian LASSO实现 ====================
function bayesian_lasso(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64};
                       n_iter::Int=10000, burn_in::Int=2000, thin::Int=5,
                       verbose::Bool=true, seed::Union{Nothing,Int}=nothing)

    !isnothing(seed) && Random.seed!(seed)
    n, p = size(Z); n_fixed = size(X, 2)
    println("\n开始Bayesian LASSO分析...")

    β = randn(n_fixed); α = randn(p) * 0.01; σ²_e = var(y) * 0.5; τ² = ones(p); λ² = 1.0

    n_saved = div(n_iter - burn_in, thin)
    marker_effects_chain = zeros(n_saved, p)
    variance_chain = Dict("residual" => zeros(n_saved), "genetic" => zeros(n_saved))
    hyperparameter_chain = Dict("lambda" => zeros(n_saved))
    breeding_values_chain = zeros(n_saved, n)

    ZtZ = [dot(Z[:, j], Z[:, j]) for j in 1:p]
    iter_saved = 0; progress = verbose ? Progress(n_iter, desc="Bayesian LASSO: ") : nothing

    for iter in 1:n_iter
        residual = y - Z * α; C_β = inv(X' * X / σ²_e + I(n_fixed) * 1e-8); μ_β = C_β * (X' * residual / σ²_e); β = rand(MvNormal(μ_β, Symmetric(C_β)))

        residual = y - X * β
        for j in 1:p
            residual .+= Z[:, j] * α[j]
            v_j = 1 / (ZtZ[j] / σ²_e + 1 / τ²[j]); μ_j = v_j * dot(Z[:, j], residual) / σ²_e
            α[j] = rand(Normal(μ_j, sqrt(v_j)))
            residual .-= Z[:, j] * α[j]
        end

        for j in 1:p; τ²[j] = rand(InverseGamma(1, λ² / 2 + α[j]^2 / (2 * σ²_e))); end
        λ² = rand(Gamma((p + 1) / 2, 1 / (1 + sum(1 ./ τ²) / 2)))
        sse = sum((y - X * β - Z * α).^2)
        σ²_e = rand(InverseGamma((n + p) / 2, (sse + sum(α.^2 ./ τ²)) / 2))

        if iter > burn_in && mod(iter - burn_in, thin) == 0
            iter_saved += 1; marker_effects_chain[iter_saved, :] = α
            variance_chain["residual"][iter_saved] = σ²_e; variance_chain["genetic"][iter_saved] = var(Z * α)
            hyperparameter_chain["lambda"][iter_saved] = sqrt(λ²); breeding_values_chain[iter_saved, :] = Z * α
        end
        verbose && next!(progress)
    end

    marker_effects_mean = vec(mean(marker_effects_chain, dims=1)); marker_effects_sd = vec(std(marker_effects_chain, dims=1))
    breeding_values_mean = vec(mean(breeding_values_chain, dims=1)); breeding_values_sd = vec(std(breeding_values_chain, dims=1))
    mcmc_chain = MCMCChain(marker_effects_chain, variance_chain, hyperparameter_chain, breeding_values_chain, zeros(n_saved), n_iter, burn_in, thin)

    return BayesianResult(:BayesianLASSO, marker_effects_mean, marker_effects_sd, ones(p), breeding_values_mean, breeding_values_sd,
        Dict("residual" => mean(variance_chain["residual"]), "genetic" => mean(variance_chain["genetic"])),
        Dict("lambda" => mean(hyperparameter_chain["lambda"])), mcmc_chain, calculate_convergence_diagnostics(mcmc_chain), nothing)
end

# ==================== 主接口函数 ====================

"""
    run_bayesian_evaluation(y, X, Z; method=:BayesB, kwargs...) -> BayesianResult

    贝叶斯基因组评估的统一接口
"""
function run_bayesian_evaluation(y::Vector, X::Matrix, Z::Matrix; method::Symbol=:BayesB, kwargs...)
    println("\n" * "="^60, "\n 贝叶斯基因组评估 (方法: $method)\n", "="^60)

    valid_idx = .!ismissing.(y); y_clean = Float64.(y[valid_idx])
    X_clean = X[valid_idx, :]; Z_clean = Z[valid_idx, :]
    println("  有效样本: $(sum(valid_idx))/$(length(y))")

    Z_std = standardize_markers(Z_clean)

    func = if method == :BayesA; bayesA; elseif method == :BayesB; bayesB; elseif method == :BayesC; bayesC; elseif method == :BayesianLASSO; bayesian_lasso; else error("未知的贝叶斯方法: $method"); end
    result = func(y_clean, X_clean, Z_std; kwargs...)

    print_bayesian_summary(result)
    return result
end

# ==================== 收敛诊断函数 ====================
function calculate_convergence_diagnostics(chain::MCMCChain)
    diagnostics = Dict{String,Any}()
    if isempty(chain.variance_components["residual"]); return diagnostics; end

    ess_res = effective_sample_size(chain.variance_components["residual"])
    diagnostics["ess_residual"] = ess_res
    if haskey(chain.variance_components, "genetic")
        diagnostics["ess_genetic"] = effective_sample_size(chain.variance_components["genetic"])
    end

    n_samples = length(chain.variance_components["residual"])
    n1 = div(n_samples, 10); n2 = div(n_samples, 2)
    geweke_z = geweke_diagnostic(chain.variance_components["residual"][1:n1], chain.variance_components["residual"][end-n2:end])
    diagnostics["geweke_residual_variance"] = geweke_z
    diagnostics["converged"] = abs(geweke_z) < 1.96
    return diagnostics
end

function effective_sample_size(x::Vector{Float64})
    n = length(x)
    acf = autocorrelation(x, min(n - 1, 1000))
    first_negative = findfirst(acf .< 0)
    sum_acf = isnothing(first_negative) ? sum(acf) : sum(acf[1:first_negative-1])
    return max(1.0, n / (1 + 2 * sum_acf))
end

function autocorrelation(x::Vector{Float64}, max_lag::Int)
    n = length(x); x_centered = x .- mean(x)
    c0 = dot(x_centered, x_centered) / n
    acf = [dot(x_centered[1:n-k], x_centered[k+1:n]) / ((n - k) * c0) for k in 1:max_lag]
    return acf
end

function geweke_diagnostic(x1::Vector{Float64}, x2::Vector{Float64})
    mean1, var1 = mean(x1), var(x1) / length(x1)
    mean2, var2 = mean(x2), var(x2) / length(x2)
    return (mean1 - mean2) / sqrt(var1 + var2)
end

function gelman_rubin end # Placeholder for multi-chain diagnostics

# ==================== MCMC诊断和可视化 ====================
function mcmc_diagnostics(result::BayesianResult; parameter::String="variance")
    # Implementation placeholder
    println("MCMC diagnostics for $parameter...")
end

function plot_trace(result::BayesianResult; parameter::String="residual_variance")
    # Implementation placeholder
    println("Plotting trace for $parameter...")
    return Dict()
end

function plot_posterior(result::BayesianResult; parameter::String="heritability")
    # Implementation placeholder
    println("Plotting posterior for $parameter...")
    return Dict()
end

# ==================== 工具函数 ====================
function standardize_markers(Z::Matrix)
    Z_std = copy(convert(Matrix{Float64}, Z))
    for j in 1:size(Z_std, 2)
        μ = mean(Z_std[:, j]); σ = std(Z_std[:, j])
        if σ > 0; Z_std[:, j] = (Z_std[:, j] .- μ) / σ; else; Z_std[:, j] .= 0; end
    end
    return Z_std
end

function print_bayesian_summary(result::BayesianResult)
    println("\n" * "="^60, "\n 贝叶斯分析结果摘要: $(result.method)\n", "="^60)
    println("\n方差组分:"); for (c, v) in result.variance_components; println("  $c: $(round(v, digits=4))"); end
    if haskey(result.variance_components, "genetic") && haskey(result.variance_components, "residual")
        h2 = result.variance_components["genetic"] / (result.variance_components["genetic"] + result.variance_components["residual"])
        println("  遗传力: $(round(h2, digits=3))")
    end
    if !isempty(result.hyperparameters); println("\n超参数:"); for (p, v) in result.hyperparameters; println("  $p: $(round(v, digits=4))"); end; end
    println("\n标记效应:\n  均值: $(round(mean(result.marker_effects), digits=6)),  标准差: $(round(std(result.marker_effects), digits=6))")
    if result.method in [:BayesB, :BayesC]; println("  PIP > 0.5: $(sum(result.marker_pip .> 0.5)) 个标记"); end
    println("\n育种值:\n  均值: $(round(mean(result.breeding_values), digits=2)),  标准差: $(round(std(result.breeding_values), digits=2))")
    if haskey(result.convergence_diagnostics, "converged"); println("\n收敛状态: $(result.convergence_diagnostics["converged"] ? "✓ 通过" : "⚠ 需要更多迭代")"); end
    println("="^60)
end

end # module BayesianAnalysis