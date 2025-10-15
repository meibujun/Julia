module RKHS

using LinearAlgebra
using Distributions
using DataFrames
using Statistics
using SparseArrays
using Random
using ..MathUtils

"""
    rkhs_epistasis!(G, y; kernel = :poly, degree = 2, λ = 1.0)

基于再生核希尔伯特空间 (RKHS) 的上位性分析，实现多项式核、径向基核与谱核。
返回拟合值、随机效应估计与方差成分。
"""
function rkhs_epistasis!(G::AbstractMatrix, y::AbstractVector; kernel::Symbol=:poly, degree::Int=2, λ::Float64=1.0)
    K = compute_kernel(G; kernel, degree)
    MathUtils.symmetrize!(K)
    α = (K + λ * I) \ y
    fitted = K * α
    σg = var(fitted)
    σe = var(y - fitted)
    return DataFrame(kernel = kernel, degree = degree, genetic_variance = σg, residual_variance = σe)
end

function compute_kernel(G::AbstractMatrix; kernel::Symbol, degree::Int)
    if kernel == :poly
        return (G * G') .^ degree
    elseif kernel == :rbf
        dists = pairwise_squared_dist(G)
        γ = rbf_gamma(dists)
        return exp.(-γ .* dists)
    elseif kernel == :spectral
        U, S, _ = svd(G)
        return U * diagm(0 => S .^ degree) * U'
    else
        error("未支持的核函数: $kernel")
    end
end

function pairwise_squared_dist(G::AbstractMatrix)
    norms = sum(abs2, G; dims = 2)
    d = norms .+ norms' .- 2 * (G * G')
    return max.(d, 0.0)
end

function rbf_gamma(dists::AbstractMatrix{<:Real})
    total = 0.0
    count = 0
    for val in dists
        if val > 0
            total += val
            count += 1
        end
    end
    scale = count == 0 ? 1.0 : total / count
    return 1 / max(scale, eps())
end

end # module
