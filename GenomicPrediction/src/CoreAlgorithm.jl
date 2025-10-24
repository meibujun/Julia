# CoreAlgorithm.jl: 核心算法模块
# ---------------------------------
# ... (header comments) ...

module CoreAlgorithm

using LinearAlgebra
using Statistics
using Distributions
using Random
using DataFrames
using GLMNet
using ..DataProcessing

abstract type AbstractModel end

# --- Model Structs ---
mutable struct GBLUPModel <: AbstractModel
    lambda::Float64
    effects::Vector{Float64}
    intercept::Float64
    allele_freqs::Vector{Float64}
    GBLUPModel(lambda::Float64) = new(lambda, [], 0.0, [])
end

mutable struct BayesAModel <: AbstractModel
    iterations::Int
    burnin::Int
    effects::Vector{Float64}
    intercept::Float64
    BayesAModel(;iterations=2000, burnin=500) = new(iterations, burnin, [], 0.0)
end

mutable struct LASSOModel <: AbstractModel
    lambda::Float64
    path::Any # Use Any since GlmNetPath is not exported
    LASSOModel(lambda::Float64) = new(lambda)
end

mutable struct ElasticNetModel <: AbstractModel
    lambda::Float64
    alpha::Float64
    path::Any # Use Any since GlmNetPath is not exported
    ElasticNetModel(lambda::Float64, alpha::Float64) = new(lambda, alpha)
end


# --- Helper Functions ---
function _standardize_genotypes(G::Matrix, freqs::Vector{Float64})
    M = G .- (2 .* freqs')
    return M
end

# --- GBLUP Implementation ---
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


# --- BayesA Implementation ---
function fit!(model::BayesAModel, data::GenomicData; rng::AbstractRNG = Random.GLOBAL_RNG)
    # ... (BayesA fit! implementation is lengthy and unchanged) ...
    y = data.phenotypes[!, 1]
    G = Matrix(data.genotypes)
    n, p = size(G)
    μ = mean(y)
    β = zeros(p)
    σ²_e = var(y) * 0.5
    σ²_β = ones(p) .* 0.01
    β_samples = zeros(p, model.iterations)
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
    end
    model.intercept = μ
    model.effects = mean(β_samples[:, (model.burnin+1):end], dims=2)[:]
    return nothing
end
function predict(model::BayesAModel, new_data::DataFrame)
    G_new = Matrix(new_data)
    return model.intercept .+ G_new * model.effects
end

# --- LASSO Implementation ---
function fit!(model::LASSOModel, data::GenomicData)
    y = data.phenotypes[!, 1]
    X = Matrix(data.genotypes)
    # GLMNet requires alpha=1.0 for LASSO
    model.path = glmnet(X, y; alpha=1.0, lambda=[model.lambda])
    return nothing
end

function predict(model::LASSOModel, new_data::DataFrame)
    X_new = Matrix(new_data)
    # The result of predict is a matrix, needs to be converted to a vector
    return GLMNet.predict(model.path, X_new)[:]
end

# --- ElasticNet Implementation ---
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
