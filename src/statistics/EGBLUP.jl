module EGBLUP

using LinearAlgebra
using DataFrames
using Statistics
using Random
using SparseArrays
using ..MathUtils

"""
    egblup!(G, y; interaction_order = 2, λ = 1e-4)

实现扩展的基因组最佳线性无偏预测 (EG-BLUP)，显式建模高阶上位性效应。
返回个体遗传效应预测及方差成分估计。
"""
function egblup!(G::AbstractMatrix, y::AbstractVector; interaction_order::Int=2, λ::Float64=1e-4)
    n = size(G, 1)
    K = build_eg_relationship(G, interaction_order)
    MathUtils.symmetrize!(K)
    α = (K + λ * I) \ y
    ghat = K * α
    σg = var(ghat)
    σe = var(y - ghat)
    reliability = σg ./ (σg .+ σe / n)
    return DataFrame(individual = 1:n, ghat = ghat, reliability = reliability)
end

function build_eg_relationship(G::AbstractMatrix, order::Int)
    n, p = size(G)
    if order == 1
        return (G * G') / p
    end
    base = (G * G') / p
    accum = copy(base)
    current = base
    for o in 2:order
        current = current .* base
        accum .+= current
    end
    return accum
end

end # module
