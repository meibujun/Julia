module LinearModelsChapter4

using LinearAlgebra
using Random
using Statistics
using Distributions

import ..LinearModelsChapter1: LinearModelDesign, simulate_response

export MixedModelStructure, simulate_mixed_model, solve_mixed_model_equations,
       reml_loglikelihood, predict_random_effects

"""
    struct MixedModelStructure

描述混合线性模型 `y = Xβ + Zu + e` 的关键矩阵与方差分量，用于构建亨德森混合模型方程。
`G` 为随机效应协方差，默认假设为标量倍数的单位阵，`R` 为残差协方差。
"""
struct MixedModelStructure
    X::Matrix{Float64}
    Z::Matrix{Float64}
    G::Matrix{Float64}
    R::Matrix{Float64}

    function MixedModelStructure(X::AbstractMatrix, Z::AbstractMatrix, G::AbstractMatrix, R::AbstractMatrix)
        size(X, 1) == size(Z, 1) || throw(ArgumentError("固定效应与随机效应必须拥有相同行数"))
        size(G, 1) == size(G, 2) || throw(ArgumentError("随机效应协方差需为方阵"))
        size(R, 1) == size(R, 2) || throw(ArgumentError("残差协方差需为方阵"))
        size(Z, 2) == size(G, 1) || throw(ArgumentError("Z 与 G 维度不兼容"))
        size(X, 1) == size(R, 1) || throw(ArgumentError("X 与 R 维度不兼容"))
        return new(Matrix{Float64}(X), Matrix{Float64}(Z), Matrix{Float64}(G), Matrix{Float64}(R))
    end
end

"""
    simulate_mixed_model(structure, β, rng)

根据混合模型结构模拟响应，其中随机效应 `u` 与误差 `e` 分别服从多元正态分布。
"""
function simulate_mixed_model(structure::MixedModelStructure, β::AbstractVector; rng::AbstractRNG = Random.default_rng())
    μ_u = zeros(size(structure.Z, 2))
    μ_e = zeros(size(structure.X, 1))
    u = rand(rng, MvNormal(μ_u, structure.G))
    e = rand(rng, MvNormal(μ_e, structure.R))
    y = structure.X * β + structure.Z * u + e
    return y, u, e
end

"""
    solve_mixed_model_equations(structure, y)

求解亨德森混合模型方程，返回固定效应估计 `β̂` 与随机效应预测 `û`，并同时提供联合方差矩阵。
"""
function solve_mixed_model_equations(structure::MixedModelStructure, y::AbstractVector)
    X, Z, G, R = structure.X, structure.Z, structure.G, structure.R
    Rinv = inv(R)
    Ginv = inv(G)
    A = [X' * Rinv * X  X' * Rinv * Z; Z' * Rinv * X  Z' * Rinv * Z + Ginv]
    b = [X' * Rinv * y; Z' * Rinv * y]
    sol = A \ b
    p = size(X, 2)
    β̂ = sol[1:p]
    û = sol[p+1:end]
    V = inv(A)
    return β̂, û, V
end

"""
    predict_random_effects(structure, y)

基于BLUP思想返回随机效应预测向量，并附带其预测均方误差矩阵，方便开展后续评价。
"""
function predict_random_effects(structure::MixedModelStructure, y::AbstractVector)
    β̂, û, V = solve_mixed_model_equations(structure, y)
    p = size(structure.X, 2)
    Cuu = V[p+1:end, p+1:end]
    return û, Cuu
end

"""
    reml_loglikelihood(structure, y)

计算给定方差分量下的REML对数似然值，为方差分量估计与超参数调优提供依据。
"""
function reml_loglikelihood(structure::MixedModelStructure, y::AbstractVector)
    X, Z, G, R = structure.X, structure.Z, structure.G, structure.R
    V = Z * G * Z' + R
    Vinv = inv(V)
    β̂ = (X' * Vinv * X) \ (X' * Vinv * y)
    r = y - X * β̂
    p = size(X, 2)
    n = length(y)
    sign_det, logdetV = logabsdet(V)
    sign_xtvx, logdet_xtvx = logabsdet(X' * Vinv * X)
    sign_det == 1 || throw(ArgumentError("协方差矩阵不正定"))
    sign_xtvx == 1 || throw(ArgumentError("信息矩阵不正定"))
    return -0.5 * ((n - p) * log(2π) + logdetV + logdet_xtvx + r' * Vinv * r)
end

end # module
