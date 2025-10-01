module Bayesian

using LinearAlgebra
using Statistics
using Random
using DataFrames
using Distributions

import ..DataManager: DataRepository
import ..ModelSpec: ModelSpec, find_trait

export BayesResult, run_bayesian_evaluation, mcmc_diagnostics

struct BayesResult
    method::Symbol
    trait::Symbol
    posterior_means::Dict{Symbol,Any}
    draws::Dict{Symbol,Vector{Float64}}
    marker_effects::Vector{Float64}
    marker_ids::Vector{Symbol}
    intercept::Float64
end

function run_bayesian_evaluation(model::ModelSpec, repo::DataRepository; trait::Union{Symbol,AbstractString} = model.traits[1].name,
        method::Symbol = :BayesC, n_iter::Int = 4000, burn_in::Int = 1000, thin::Int = 5,
        π::Float64 = 0.95, df::Float64 = 4.2, scale::Float64 = 0.5, λ::Float64 = 0.1,
        rng::AbstractRNG = Random.default_rng())
    y, X, Z, marker_ids = _prepare_design(model, repo, Symbol(trait))
    result = _bayes_sampler(y, X, Z; method, n_iter, burn_in, thin, π, df, scale, λ, rng)
    return BayesResult(method, Symbol(trait), result.posterior_means, result.draws,
        result.posterior_means[:marker_effects], marker_ids, result.posterior_means[:intercept])
end

function run_bayesian_evaluation(y::AbstractVector, X::AbstractMatrix, Z::AbstractMatrix; kwargs...)
    result = _bayes_sampler(Float64.(y), Matrix{Float64}(X), Matrix{Float64}(Z); kwargs...)
    marker_ids = Symbol.(("marker" .* string.(1:size(Z, 2))))
    return BayesResult(get(kwargs, :method, :BayesC), :unspecified, result.posterior_means, result.draws,
        result.posterior_means[:marker_effects], marker_ids, result.posterior_means[:intercept])
end

function mcmc_diagnostics(result::BayesResult; parameter::Symbol = :marker_variance)
    draws = get(result.draws, parameter, Float64[])
    isempty(draws) && return Dict(:mean => NaN, :std => NaN, :n => 0)
    return Dict(:mean => mean(draws), :std => std(draws), :n => length(draws))
end

struct _SamplerState
    posterior_means::Dict{Symbol,Any}
    draws::Dict{Symbol,Vector{Float64}}
end

function _bayes_sampler(y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}; method::Symbol,
        n_iter::Int, burn_in::Int, thin::Int, π::Float64, df::Float64, scale::Float64, λ::Float64,
        rng::AbstractRNG)
    n, p = size(X)
    m = size(Z, 2)
    XtX = X' * X
    ZtZ_diag = vec(sum(abs2, eachcol(Z)))
    β = zeros(p)
    b = zeros(m)
    δ = ones(Int, m)
    marker_var = ones(m) .* scale
    σe2 = var(y)
    σg2 = var(y) * 0.5
    residual = y - X * β - Z * (b .* δ)
    intercept_draws = Float64[]
    var_draws = Float64[]
    residual_draws = Float64[]
    sample_count = 0
    accum_b = zeros(m)
    accum_intercept = 0.0
    for iter in 1:n_iter
        # update fixed effects via conjugate normal prior with large variance
        precision = XtX / σe2 + I * 1e-6
        covβ = inv(precision)
        meanβ = covβ * (X' * (y - Z * (b .* δ)) / σe2)
        β = meanβ + cholesky(Symmetric(covβ)).L * randn(rng, p)
        residual = y - X * β - Z * (b .* δ)
        # update marker effects sequentially
        for j in 1:m
            z = view(Z, :, j)
            residual .+= z .* (b[j] * δ[j])
            τ = marker_var[j]
            vj = ZtZ_diag[j]
            denom = vj + σe2 / max(τ, eps())
            mean_j = dot(z, residual) / denom
            var_j = σe2 / denom
            if method in (:BayesB, :BayesC)
                prob_ratio = ((1 - π) / max(π, 1e-6)) * sqrt(σe2 / (σe2 + τ * vj)) *
                    exp(0.5 * (dot(z, residual)^2) / (σe2 * (σe2 + τ * vj)))
                prob_inclusion = prob_ratio / (1 + prob_ratio)
                if rand(rng) > prob_inclusion
                    δ[j] = 0
                    b[j] = 0.0
                    residual .-= z .* (b[j] * δ[j])
                    continue
                else
                    δ[j] = 1
                end
            else
                δ[j] = 1
            end
            b[j] = mean_j + sqrt(max(var_j, 1e-8)) * randn(rng)
            residual .-= z .* (b[j] * δ[j])
            if method == :BayesA
                shape = (df + 1) / 2
                rate = (df * scale + (b[j]^2) / σe2) / 2
                marker_var[j] = rand(InverseGamma(shape, rate))
            elseif method == :BayesianLASSO
                rate = λ^2 / 2
                marker_var[j] = 1 / rand(Gamma(1.0, 1 / rate))
            end
        end
        σe_shape = (n + 2) / 2
        σe_rate = (sum(residual .^ 2) + scale) / 2
        σe2 = rand(InverseGamma(σe_shape, σe_rate))
        σg_shape = (sum(δ) + 2) / 2
        σg_rate = (sum((b .* δ) .^ 2) + scale) / 2
        σg2 = rand(InverseGamma(σg_shape, σg_rate))
        if iter > burn_in && (iter - burn_in) % thin == 0
            sample_count += 1
            accum_b .+= b .* δ
            accum_intercept += β[1]
            push!(intercept_draws, β[1])
            push!(var_draws, σg2)
            push!(residual_draws, σe2)
        end
    end
    posterior_means = Dict{Symbol,Any}(
        :marker_effects => accum_b ./ max(sample_count, 1),
        :intercept => accum_intercept / max(sample_count, 1),
        :marker_variance => isempty(var_draws) ? NaN : mean(var_draws),
        :residual_variance => isempty(residual_draws) ? NaN : mean(residual_draws)
    )
    draws = Dict{Symbol,Vector{Float64}}(
        :intercept => intercept_draws,
        :marker_variance => var_draws,
        :residual_variance => residual_draws
    )
    return _SamplerState(posterior_means, draws)
end

function _prepare_design(model::ModelSpec, repo::DataRepository, trait::Symbol)
    trait_spec = find_trait(model, trait)
    df = repo.phenotypes
    haskey(df, trait) || throw(ArgumentError("Trait column $(trait) missing"))
    y = Float64.(coalesce.(df[!, trait], mean(skipmissing(df[!, trait]))))
    intercept = ones(length(y))
    X = reshape(intercept, :, 1)
    if isempty(repo.genotypes)
        throw(ArgumentError("Genotypes required for Bayesian genomic evaluation"))
    end
    geno = repo.genotypes
    ids = String.(coalesce.(df[!, :animal], ""))
    marker_ids = Symbol.(setdiff(names(geno), [:animal]))
    idx = Dict(String(geno[i, :animal]) => i for i in 1:nrow(geno))
    Z = zeros(Float64, length(y), length(marker_ids))
    for (row_idx, animal) in enumerate(ids)
        geno_row = get(idx, animal, nothing)
        geno_row === nothing && throw(ArgumentError("Missing genotype for animal $(animal)"))
        for (j, marker) in enumerate(marker_ids)
            Z[row_idx, j] = Float64(geno[geno_row, marker])
        end
    end
    return y, X, Z, marker_ids
end

end
