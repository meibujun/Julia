module MathUtils

using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using Distributions

function standardize!(X::AbstractMatrix; dims::Integer = 1)
    if dims == 1
        μ = mapslices(mean, X; dims = dims)
        σ = mapslices(std, X; dims = dims)
        for j in axes(X, 2)
            σj = σ[j]
            σj = σj == 0 ? one(eltype(X)) : σj
            X[:, j] .= (X[:, j] .- μ[j]) ./ σj
        end
    else
        μ = mean(X, dims = dims)
        σ = std(X, dims = dims)
        for i in axes(X, 1)
            σi = σ[i]
            σi = σi == 0 ? one(eltype(X)) : σi
            X[i, :] .= (X[i, :] .- μ[i]) ./ σi
        end
    end
    return X
end

function symmetrize!(K::AbstractMatrix)
    @inbounds for i in axes(K, 1)
        for j in i+1:size(K, 1)
            v = (K[i, j] + K[j, i]) / 2
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

logit(p::Real; eps = 1e-8) = log((clamp(p, eps, 1 - eps)) / (1 - clamp(p, eps, 1 - eps)))
logistic(x::Real) = 1 / (1 + exp(-x))

function simulate_genotype_matrix(n_samples::Int, n_markers::Int; rare_rate::Float64 = 0.01)
    maf = clamp.(rand(Beta(rare_rate * 2, 5), n_markers), 1e-4, 0.5)
    G = zeros(Float64, n_samples, n_markers)
    for j in 1:n_markers
        p = maf[j]
        G[:, j] .= rand(Binomial(2, p), n_samples)
    end
    return G
end

function simulate_phenotypes(G::AbstractMatrix, β::AbstractVector, h2::Real; noise::Symbol = :gaussian)
    genetic = G * β
    σg = var(genetic)
    σe = σg * (1 - h2) / h2
    noisevec = noise === :gaussian ? randn(size(G, 1)) .* sqrt(σe) : rand(TDist(5), size(G, 1)) .* sqrt(σe)
    return genetic .+ noisevec
end

end # module
