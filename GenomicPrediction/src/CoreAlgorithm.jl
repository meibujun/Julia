# CoreAlgorithm.jl: 核心算法模块
module CoreAlgorithm

using LinearAlgebra
using Statistics
using Distributions
using Random
using DataFrames
using GLMNet
using ..DataProcessing
import ..AbstractModel

@doc raw"""
    GBLUPModel(lambda::Float64) <: AbstractModel
"""
mutable struct GBLUPModel <: AbstractModel
    lambda::Float64
    effects::Vector{Float64}
    intercept::Float64
    allele_freqs::Vector{Float64}
    GBLUPModel(lambda::Float64) = new(lambda, [], 0.0, [])
end

@doc raw"""
    BayesAModel(;iterations=2000, burnin=500) <: AbstractModel
"""
mutable struct BayesAModel <: AbstractModel
    iterations::Int
    burnin::Int
    effects::Vector{Float64}
    intercept::Float64
    BayesAModel(;iterations=2000, burnin=500) = new(iterations, burnin, [], 0.0)
end

@doc raw"""
    BayesBModel(;iterations=2000, burnin=500, pi=0.05) <: AbstractModel
"""
mutable struct BayesBModel <: AbstractModel
    iterations::Int
    burnin::Int
    pi::Float64
    effects::Vector{Float64}
    intercept::Float64
    BayesBModel(;iterations=2000, burnin=500, pi=0.05) = new(iterations, burnin, pi, [], 0.0)
end

@doc raw"""
    BayesCModel(;iterations=2000, burnin=500, pi=0.05) <: AbstractModel

Implements the BayesC model, which assumes a common variance for all non-zero SNP effects.

# Arguments
- `iterations::Int`: The total number of MCMC iterations.
- `burnin::Int`: The number of initial iterations to discard.
- `pi::Float64`: The prior probability that a SNP has a non-zero effect.
"""
mutable struct BayesCModel <: AbstractModel
    iterations::Int
    burnin::Int
    pi::Float64
    effects::Vector{Float64}
    intercept::Float64
    BayesCModel(;iterations=2000, burnin=500, pi=0.05) = new(iterations, burnin, pi, [], 0.0)
end

@doc raw"""
    BayesRModel(;iterations=2000, burnin=500, proportions=[0.95, 0.02, 0.02, 0.01], variance_fractions=[0.0, 0.0001, 0.001, 0.01]) <: AbstractModel

Implements the BayesR model, which assumes SNP effects are drawn from a mixture of normal distributions.

# Arguments
- `iterations::Int`: The total number of MCMC iterations.
- `burnin::Int`: The number of initial iterations to discard.
- `proportions::Vector{Float64}`: Prior proportions of SNPs in each variance class. Must sum to 1.
- `variance_fractions::Vector{Float64}`: Fractions of genetic variance for each class.
"""
mutable struct BayesRModel <: AbstractModel
    iterations::Int
    burnin::Int
    proportions::Vector{Float64}
    variance_fractions::Vector{Float64}
    effects::Vector{Float64}
    intercept::Float64
    function BayesRModel(;iterations=2000, burnin=500, proportions=[0.95, 0.02, 0.02, 0.01], variance_fractions=[0.0, 0.0001, 0.001, 0.01])
        @assert sum(proportions) ≈ 1.0 "Proportions must sum to 1."
        @assert length(proportions) == length(variance_fractions) "Proportions and variance_fractions must have the same length."
        new(iterations, burnin, proportions, variance_fractions, [], 0.0)
    end
end

@doc raw"""
    LASSOModel(lambda::Float64) <: AbstractModel
"""
mutable struct LASSOModel <: AbstractModel
    lambda::Float64
    path::Any
    LASSOModel(lambda::Float64) = new(lambda)
end

@doc raw"""
    ElasticNetModel(lambda::Float64, alpha::Float64) <: AbstractModel
"""
mutable struct ElasticNetModel <: AbstractModel
    lambda::Float64
    alpha::Float64
    path::Any
    ElasticNetModel(lambda::Float64, alpha::Float64) = new(lambda, alpha)
end

function _standardize_genotypes(G::Matrix, freqs::Vector{Float64})
    M = G .- (2 .* freqs')
    return M
end

function fit!(model::GBLUPModel, data::GenomicData)
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    p = size(G, 2)
    model.allele_freqs = vec(mean(G, dims=1) ./ 2)
    Z = _standardize_genotypes(G, model.allele_freqs)
    model.intercept = mean(y)
    y_centered = y .- model.intercept
    lhs = Z' * Z
    for i in 1:p; lhs[i, i] += model.lambda; end
    rhs = Z' * y_centered
    model.effects = lhs \ rhs
    return nothing
end

function predict(model::GBLUPModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    Z_new = _standardize_genotypes(G_new, model.allele_freqs)
    return model.intercept .+ Z_new * model.effects
end

function fit!(model::BayesAModel, data::GenomicData; rng::AbstractRNG = Random.GLOBAL_RNG)
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    n, p = size(G)
    μ = mean(y)
    β = zeros(p)
    σ²_e = var(y) * 0.5
    σ²_β = ones(p) .* 0.01
    β_samples = zeros(p, model.iterations)
    println("开始 BayesA MCMC 抽样 (迭代次数: $(model.iterations))...")
    for iter in 1:model.iterations
        y_res = y - G * β
        μ = rand(rng, Normal(mean(y_res), sqrt(σ²_e / n)))
        y_res_no_mu = y .- μ
        for j in 1:p
            y_res_j = y_res_no_mu - G[:, 1:end .!= j] * β[1:end .!= j]
            Gj = G[:, j]
            rhs = dot(Gj, y_res_j)
            lhs = dot(Gj, Gj) + σ²_e / σ²_β[j]
            mean_βj = rhs / lhs
            var_βj = σ²_e / lhs
            β[j] = rand(rng, Normal(mean_βj, sqrt(var_βj)))
            v0 = 4.0; s20 = 0.002
            shape = (v0 + 1) / 2
            scale = (β[j]^2 + v0 * s20) / 2
            σ²_β[j] = 1.0 / rand(rng, Gamma(shape, 1.0 / scale))
        end
        y_pred = μ .+ G * β
        e = y - y_pred
        shape_e = (n / 2)
        scale_e = dot(e, e) / 2
        σ²_e = 1.0 / rand(rng, Gamma(shape_e, 1.0 / scale_e))
        β_samples[:, iter] = β
        if iter % 100 == 0; println("迭代: $iter / $(model.iterations)"); end
    end
    println("MCMC 抽样完成。")
    model.intercept = μ
    model.effects = mean(β_samples[:, (model.burnin+1):end], dims=2)[:]
    return nothing
end

function predict(model::BayesAModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    return model.intercept .+ G_new * model.effects
end

function fit!(model::BayesBModel, data::GenomicData; rng::AbstractRNG = Random.GLOBAL_RNG)
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    n, p = size(G)
    μ = mean(y)
    β = zeros(p)
    σ²_e = var(y) * 0.5
    σ²_β = ones(p) .* 0.01
    β_samples = zeros(p, model.iterations)
    println("开始 BayesB MCMC 抽样 (迭代次数: $(model.iterations), pi: $(model.pi))...")
    for iter in 1:model.iterations
        y_res = y - G * β
        μ = rand(rng, Normal(mean(y_res), sqrt(σ²_e / n)))
        y_res_no_mu = y .- μ
        for j in 1:p
            if rand(rng) < model.pi
                y_res_j = y_res_no_mu - G[:, 1:end .!= j] * β[1:end .!= j]
                Gj = G[:, j]
                rhs = dot(Gj, y_res_j)
                lhs = dot(Gj, Gj) + σ²_e / σ²_β[j]
                mean_βj = rhs / lhs
                var_βj = σ²_e / lhs
                β[j] = rand(rng, Normal(mean_βj, sqrt(var_βj)))
                v0 = 4.0; s20 = 0.002
                shape = (v0 + 1) / 2
                scale = (β[j]^2 + v0 * s20) / 2
                σ²_β[j] = 1.0 / rand(rng, Gamma(shape, 1.0 / scale))
            else
                β[j] = 0.0
            end
        end
        y_pred = μ .+ G * β
        e = y - y_pred
        shape_e = (n / 2); scale_e = dot(e, e) / 2
        σ²_e = 1.0 / rand(rng, Gamma(shape_e, 1.0 / scale_e))
        β_samples[:, iter] = β
        if iter % 100 == 0; println("迭代: $iter / $(model.iterations)"); end
    end
    println("MCMC 抽样完成。")
    model.intercept = μ
    model.effects = mean(β_samples[:, (model.burnin+1):end], dims=2)[:]
    return nothing
end

function predict(model::BayesBModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    return model.intercept .+ G_new * model.effects
end

function fit!(model::BayesCModel, data::GenomicData; rng::AbstractRNG = Random.GLOBAL_RNG)
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    n, p = size(G)

    μ = mean(y)
    β = zeros(p)
    σ²_e = var(y) * 0.5
    σ²_β = 0.01 # Common variance component

    β_samples = zeros(p, model.iterations)

    println("开始 BayesC MCMC 抽样 (迭代次数: $(model.iterations), pi: $(model.pi))...")

    for iter in 1:model.iterations
        # Update intercept
        y_res = y - G * β
        μ = rand(rng, Normal(mean(y_res), sqrt(σ²_e / n)))
        y_res_no_mu = y .- μ

        # Update SNP effects
        for j in 1:p
            # Calculate probability of SNP j being in the model
            log_p_zero = log(1 - model.pi)
            log_p_nonzero = log(model.pi) + 0.5 * log(σ²_e / (dot(G[:,j], G[:,j])*σ²_β + σ²_e)) -
                            (dot(G[:,j], y_res_no_mu - G[:, 1:end .!=j]*β[1:end .!= j])^2) / (2 * (dot(G[:,j], G[:,j])*σ²_β + σ²_e))

            prob_nonzero = 1 / (1 + exp(log_p_zero - log_p_nonzero))

            if rand(rng) < prob_nonzero
                # Sample effect from its posterior
                y_res_j = y_res_no_mu - G[:, 1:end .!= j] * β[1:end .!= j]
                Gj = G[:, j]
                rhs = dot(Gj, y_res_j)
                lhs = dot(Gj, Gj) + σ²_e / σ²_β
                mean_βj = rhs / lhs
                var_βj = σ²_e / lhs
                β[j] = rand(rng, Normal(mean_βj, sqrt(var_βj)))
            else
                β[j] = 0.0
            end
        end

        # Update common variance σ²_β
        non_zero_effects = β[β .!= 0]
        v0 = 4.0; s20 = 0.002
        shape_beta = (v0 + length(non_zero_effects)) / 2
        scale_beta = (sum(abs2, non_zero_effects) + v0 * s20) / 2
        σ²_β = 1.0 / rand(rng, Gamma(shape_beta, 1.0 / scale_beta))

        # Update error variance σ²_e
        y_pred = μ .+ G * β
        e = y - y_pred
        shape_e = (n / 2)
        scale_e = dot(e, e) / 2
        σ²_e = 1.0 / rand(rng, Gamma(shape_e, 1.0 / scale_e))

        β_samples[:, iter] = β
        if iter % 100 == 0; println("迭代: $iter / $(model.iterations)"); end
    end

    println("MCMC 抽样完成。")
    model.intercept = μ
    model.effects = mean(β_samples[:, (model.burnin+1):end], dims=2)[:]
    return nothing
end

function predict(model::BayesCModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    return model.intercept .+ G_new * model.effects
end

function fit!(model::BayesRModel, data::GenomicData; rng::AbstractRNG = Random.GLOBAL_RNG)
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    n, p = size(G)

    μ = mean(y)
    β = zeros(p)
    σ²_e = var(y) * 0.5
    total_genetic_var = var(y) * 0.5 # Assume 50% heritability initially
    σ²_β_classes = model.variance_fractions .* total_genetic_var

    β_samples = zeros(p, model.iterations)
    snp_class_indicators = zeros(Int, p) # To which class each SNP belongs

    println("开始 BayesR MCMC 抽样 (迭代次数: $(model.iterations))...")

    for iter in 1:model.iterations
        # Update intercept
        y_res = y - G * β
        μ = rand(rng, Normal(mean(y_res), sqrt(σ²_e / n)))
        y_res_no_mu = y .- μ

        # Update SNP effects and their class assignments
        current_y_res = y_res_no_mu
        for j in 1:p
            current_y_res += G[:, j] * β[j] # Remove effect of SNP j

            log_probs = zeros(length(model.proportions))
            Gj_sq = dot(G[:,j], G[:,j])

            for k in 1:length(model.proportions)
                if model.variance_fractions[k] == 0.0 # Zero variance class
                    log_probs[k] = log(model.proportions[k])
                else
                    var_k = σ²_β_classes[k]
                    log_probs[k] = log(model.proportions[k]) + 0.5 * log(σ²_e / (Gj_sq * var_k + σ²_e)) -
                                   (dot(G[:,j], current_y_res)^2) / (2 * (Gj_sq * var_k + σ²_e))
                end
            end

            # Sample class for SNP j
            probs = exp.(log_probs .- maximum(log_probs))
            probs ./= sum(probs)
            snp_class_indicators[j] = rand(rng, Categorical(probs))

            # Sample effect for SNP j based on its class
            class_k = snp_class_indicators[j]
            if model.variance_fractions[class_k] == 0.0
                β[j] = 0.0
            else
                var_k = σ²_β_classes[class_k]
                lhs = Gj_sq + σ²_e / var_k
                rhs = dot(G[:,j], current_y_res)
                mean_βj = rhs / lhs
                var_βj = σ²_e / lhs
                β[j] = rand(rng, Normal(mean_βj, sqrt(var_βj)))
            end

            current_y_res -= G[:, j] * β[j] # Add back new effect of SNP j
        end

        # Update variance components
        total_genetic_var = sum(abs2, β)
        σ²_β_classes = model.variance_fractions .* total_genetic_var

        # Update error variance σ²_e
        y_pred = μ .+ G * β
        e = y - y_pred
        shape_e = (n / 2)
        scale_e = dot(e, e) / 2
        σ²_e = 1.0 / rand(rng, Gamma(shape_e, 1.0 / scale_e))

        β_samples[:, iter] = β
        if iter % 100 == 0; println("迭代: $iter / $(model.iterations)"); end
    end

    println("MCMC 抽样完成。")
    model.intercept = μ
    model.effects = mean(β_samples[:, (model.burnin+1):end], dims=2)[:]
    return nothing
end

function predict(model::BayesRModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    return model.intercept .+ G_new * model.effects
end


function fit!(model::LASSOModel, data::GenomicData)
    y = data.phenotypes[!, 1]
    X = Matrix(data.genotypes)
    model.path = glmnet(X, y; alpha=1.0, lambda=[model.lambda])
    return nothing
end

function predict(model::LASSOModel, new_data::DataFrame)
    X_new = Matrix(new_data)
    return GLMNet.predict(model.path, X_new)[:]
end

function fit!(model::ElasticNetModel, data::GenomicData)
    y = data.phenotypes[!, 1]
    X = Matrix(data.genotypes)
    model.path = glmnet(X, y; alpha=model.alpha, lambda=[model.lambda])
    return nothing
end

function predict(model::ElasticNetModel, new_data::DataFrame)
    X_new = Matrix(new_data)
    return GLMNet.predict(model.path, X_new)[:]
end

end # module CoreAlgorithm
