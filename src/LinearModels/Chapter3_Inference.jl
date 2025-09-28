module LinearModelsChapter3

using LinearAlgebra
using Statistics
using Distributions

import ..LinearModelsChapter1: LinearModelDesign
import ..LinearModelsChapter2: least_squares_estimator, fitted_and_residual

export hypothesis_f_test, parameter_t_test, confidence_interval,
       estimable_contrast_matrix

"""
    estimable_contrast_matrix(X, C)

检测线性约束 `Cβ = d` 是否可估，并返回约束空间的正交基。
该函数强调线性模型理论中“可估性”概念，帮助读者理解约束推断的代数基础。
"""
function estimable_contrast_matrix(X::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real})
    Q, _ = qr(X)
    rankX = rank(X)
    Q1 = Q[:, 1:rankX]
    proj = I - Q1 * Q1'
    residual = proj * C'
    norms = vec(sum(abs2, eachcol(residual)))
    all(norms .< sqrt(eps(Float64))) || throw(ArgumentError("存在不可估的线性约束"))
    return Q1
end

"""
    hypothesis_f_test(design, y, C, d)

针对一般线性假设 `H₀: Cβ = d` 构造F检验，返回统计量、自由度和p值。
"""
function hypothesis_f_test(design::LinearModelDesign, y::AbstractVector,
                           C::AbstractMatrix{<:Real}, d::AbstractVector)
    β̂, κ = least_squares_estimator(design, y)
    ŷ, r, rss, mse, _ = fitted_and_residual(design, β̂, y)
    estimable_contrast_matrix(design.X, C)
    diff = C * β̂ .- d
    middle = C * (design.X' * design.X) \ C'
    q = rank(C)
    fstat = (diff' * (middle \ diff)) / q / mse
    pvalue = 1 - cdf(FDist(q, length(y) - size(design.X, 2)), fstat)
    return (; fstat, pvalue, df1 = q, df2 = length(y) - size(design.X, 2), κ, fitted = ŷ, residual = r, rss, mse)
end

"""
    parameter_t_test(design, y, index; value = 0.0)

对单个参数 `β_index` 进行t检验，可指定检验值 `value`，并返回估计值、标准误和p值。
"""
function parameter_t_test(design::LinearModelDesign, y::AbstractVector, index::Integer; value::Real = 0.0)
    β̂, _ = least_squares_estimator(design, y)
    _, _, rss, mse, _ = fitted_and_residual(design, β̂, y)
    XtX_inv = inv(design.X' * design.X)
    se = sqrt(mse * XtX_inv[index, index])
    tstat = (β̂[index] - value) / se
    ν = length(y) - size(design.X, 2)
    pvalue = 2 * (1 - cdf(TDist(ν), abs(tstat)))
    return (; estimate = β̂[index], se, tstat, pvalue, ν)
end

"""
    confidence_interval(design, y, index; level = 0.95)

计算指定参数的置信区间，默认置信水平为95%。
"""
function confidence_interval(design::LinearModelDesign, y::AbstractVector, index::Integer; level::Real = 0.95)
    res = parameter_t_test(design, y, index)
    ν = res.ν
    half = quantile(TDist(ν), (1 + level) / 2) * res.se
    return (lower = res.estimate - half, upper = res.estimate + half, level)
end

end # module
