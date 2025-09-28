module LinearModelsChapter8

using LinearAlgebra
using Statistics

export henderson_method3, em_reml_variance_components, log_reml_likelihood

"""
    henderson_method3(Z, R, G, y, X)

实现章节八对方差分量估计的亨德森三法方程求解，返回随机效应预测 `û` 与固定效应估计 `β̂`。
其中 `Z` 为随机效应设计矩阵，`R` 与 `G` 分别为残差与随机效应协方差矩阵。
"""
function henderson_method3(Z::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real})
    Zmat = Matrix{Float64}(Z)
    Rmat = Matrix{Float64}(R)
    Gmat = Matrix{Float64}(G)
    yvec = Vector{Float64}(y)
    Xmat = Matrix{Float64}(X)
    Rinv = inv(Rmat)
    Ginv = inv(Gmat)
    A11 = Xmat' * Rinv * Xmat
    A12 = Xmat' * Rinv * Zmat
    A21 = Zmat' * Rinv * Xmat
    A22 = Zmat' * Rinv * Zmat + Ginv
    rhs1 = Xmat' * Rinv * yvec
    rhs2 = Zmat' * Rinv * yvec
    K = [A11 A12; A21 A22]
    rhs = vcat(rhs1, rhs2)
    sol = K \ rhs
    β̂ = sol[1:size(Xmat, 2)]
    û = sol[size(Xmat, 2)+1:end]
    return (β̂ = β̂, û = û)
end

"""
    em_reml_variance_components(X, Z, y; σe2_init = 1.0, σu2_init = 1.0, tol = 1e-6, maxiter = 200)

基于章节八的EM-REML推导，迭代估计随机效应与残差方差分量。
函数返回 `(σu2, σe2, iterations, converged)`。
"""
function em_reml_variance_components(X::AbstractMatrix{<:Real}, Z::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; σe2_init::Real = 1.0, σu2_init::Real = 1.0, tol::Real = 1e-6, maxiter::Int = 200)
    Xmat = Matrix{Float64}(X)
    Zmat = Matrix{Float64}(Z)
    yvec = Vector{Float64}(y)
    n, q = size(Zmat)
    p = size(Xmat, 2)
    σe2 = float(σe2_init)
    σu2 = float(σu2_init)
    for iter in 1:maxiter
        V = σu2 .* (Zmat * Zmat') + σe2 .* I(n)
        Vinv = inv(V)
        XtVinvX = Xmat' * Vinv * Xmat
        β̂ = XtVinvX \ (Xmat' * Vinv * yvec)
        resid = yvec - Xmat * β̂
        û = σu2 .* (Zmat' * Vinv * resid)
        C = σu2 .* I(q) - σu2^2 .* (Zmat' * Vinv * Zmat)
        σu2_new = (dot(û, û) + trace(C)) / q
        Py = Vinv * resid
        σe2_new = (dot(resid, Py) + σe2 * (n - p)) / n
        if abs(σu2_new - σu2) < tol && abs(σe2_new - σe2) < tol
            return (σu2 = σu2_new, σe2 = σe2_new, iterations = iter, converged = true)
        end
        σu2, σe2 = σu2_new, σe2_new
    end
    return (σu2 = σu2, σe2 = σe2, iterations = maxiter, converged = false)
end

"""
    log_reml_likelihood(X, Z, y, σu2, σe2)

计算REML对数似然，用于章节八对模型比较与方差分量估计评估的讨论。
"""
function log_reml_likelihood(X::AbstractMatrix{<:Real}, Z::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, σu2::Real, σe2::Real)
    Xmat = Matrix{Float64}(X)
    Zmat = Matrix{Float64}(Z)
    yvec = Vector{Float64}(y)
    n = length(yvec)
    p = size(Xmat, 2)
    V = σu2 .* (Zmat * Zmat') + σe2 .* I(n)
    Vinv = inv(V)
    XtVinvX = Xmat' * Vinv * Xmat
    β̂ = XtVinvX \ (Xmat' * Vinv * yvec)
    resid = yvec - Xmat * β̂
    logdetV = logdet(Symmetric(V))[1]
    logdetXtVinvX = logdet(Symmetric(XtVinvX))[1]
    quad = dot(resid, Vinv * resid)
    return -0.5 * (logdetV + logdetXtVinvX + quad + (n - p) * log(2π))
end

end # module
