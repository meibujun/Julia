module LinearModelsChapter2

using LinearAlgebra
using Statistics

import ..LinearModelsChapter1: LinearModelDesign, simulate_response, projection_matrix

export least_squares_estimator, fitted_and_residual, leave_one_out_diagnostics,
       generalized_inverse_solution

"""
    least_squares_estimator(design, y)

基于第一章定义的设计对象计算普通最小二乘估计 `β̂ = (X'X)^{-1}X'y`。
函数内部自动检测秩亏情况，并在必要时返回数值稳定的伪逆解，同时提供条件数信息。
"""
function least_squares_estimator(design::LinearModelDesign, y::AbstractVector)
    X = design.X
    Xty = X' * y
    XtX = Symmetric(X' * X)
    # 使用Cholesky检测数值稳定性
    try
        chol = cholesky(XtX; check = true)
        β̂ = chol \ Xty
        return β̂, cond(XtX)
    catch err
        err isa PosDefException || rethrow()
        return generalized_inverse_solution(X, y)
    end
end

"""
    generalized_inverse_solution(X, y)

当设计矩阵不可逆时，使用奇异值分解给出最小范数解，并返回估计向量及其条件数。
"""
function generalized_inverse_solution(X::AbstractMatrix{<:Real}, y::AbstractVector)
    F = svd(X)
    tol = maximum(size(X)) * eps() * maximum(F.S)
    idx = F.S .> tol
    Sinv = zeros(eltype(F.S), length(F.S))
    Sinv[idx] .= 1 ./ F.S[idx]
    β̂ = F.Vt' * Diagonal(Sinv) * F.U' * y
    κ = isempty(F.S) ? 0.0 : maximum(F.S[idx]) / minimum(F.S[idx])
    return β̂, κ
end

"""
    fitted_and_residual(design, β̂, y)

计算拟合值与残差，并返回残差平方和、均方误差和帽子矩阵，以便后续章节开展推断与诊断。
"""
function fitted_and_residual(design::LinearModelDesign, β̂::AbstractVector, y::AbstractVector)
    X = design.X
    ŷ = X * β̂
    r = y .- ŷ
    rss = sum(abs2, r)
    dfe = length(y) - size(X, 2)
    mse = rss / dfe
    H = projection_matrix(X)
    return ŷ, r, rss, mse, H
end

"""
    leave_one_out_diagnostics(design, y)

利用帽子矩阵实现留一交叉验证残差与预测误差的快速计算，用于评估模型的稳健性。
"""
function leave_one_out_diagnostics(design::LinearModelDesign, y::AbstractVector)
    β̂, _ = least_squares_estimator(design, y)
    ŷ, r, _, _, H = fitted_and_residual(design, β̂, y)
    leverage = diag(H)
    adj = 1 .- leverage
    any(adj .== 0) && throw(ArgumentError("存在杠杆值为1的观测，无法计算留一残差"))
    loo_residual = r ./ adj
    loo_prediction = y .- loo_residual
    cv_mse = mean(abs2, loo_residual)
    return (; β̂, fitted = ŷ, residual = r, leverage, loo_residual, loo_prediction, cv_mse)
end

end # module
