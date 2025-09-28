module LinearModelsChapter7

using LinearAlgebra
using Statistics
using Random
using Distributions

export prediction_interval, cross_validated_rmse, predictive_residuals,
       bootstrap_prediction_intervals

"""
    prediction_interval(x_new, β̂, σ̂2, XtX_inv; α = 0.05)

依据章节七关于线性预测的推断结果，为单个新样本 `x_new` 提供 `1-α` 预测区间。
其中 `XtX_inv` 为 `(X'X)^{-1}`，可复用前面章节的矩阵分解结果。
"""
function prediction_interval(x_new::AbstractVector{<:Real}, β̂::AbstractVector{<:Real}, σ̂2::Real, XtX_inv::AbstractMatrix{<:Real}; α::Real = 0.05)
    xvec = Vector{Float64}(x_new)
    βvec = Vector{Float64}(β̂)
    ν = length(βvec)
    t_value = quantile(TDist(ν - 1), 1 - α / 2)
    pred = dot(xvec, βvec)
    margin = t_value * sqrt(σ̂2 * (1 + xvec' * XtX_inv * xvec))
    return (lower = pred - margin, upper = pred + margin, estimate = pred)
end

"""
    cross_validated_rmse(X, y; k = 5, rng = Random.default_rng())

执行K折交叉验证并返回均方根误差，体现章节七对预测评估的讨论。
"""
function cross_validated_rmse(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; k::Int = 5, rng::AbstractRNG = Random.default_rng())
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    n = length(yvec)
    k > 1 || throw(ArgumentError("交叉验证折数需大于1"))
    perm = randperm(rng, n)
    fold_sizes = fill(div(n, k), k)
    for i in 1:mod(n, k)
        fold_sizes[i] += 1
    end
    splits = Vector{Vector{Int}}()
    start_idx = 1
    for size in fold_sizes
        stop_idx = start_idx + size - 1
        push!(splits, perm[start_idx:stop_idx])
        start_idx = stop_idx + 1
    end
    errors = Float64[]
    for fold in splits
        test_idx = fold
        train_mask = trues(n)
        train_mask[test_idx] .= false
        Xtrain = Xmat[train_mask, :]
        ytrain = yvec[train_mask]
        β̂ = Xtrain \ ytrain
        preds = Xmat[test_idx, :] * β̂
        push!(errors, sqrt(mean((yvec[test_idx] .- preds) .^ 2)))
    end
    return mean(errors)
end

"""
    predictive_residuals(X, y)

计算预测残差（PRESS残差），展示章节七对留一法预测评估的推导。
"""
function predictive_residuals(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    β̂ = Xmat \ yvec
    XtX_inv = inv(Xmat' * Xmat)
    hat = vec(sum((Xmat * XtX_inv) .* Xmat, dims = 2))
    res = yvec - Xmat * β̂
    return res ./ (1 .- hat)
end

"""
    bootstrap_prediction_intervals(X, y, x_new; B = 500, rng = Random.default_rng())

使用百分位Bootstrap方法为新样本预测构造经验置信区间，呼应章节七对重采样方法的扩展讨论。
"""
function bootstrap_prediction_intervals(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, x_new::AbstractVector{<:Real}; B::Int = 500, rng::AbstractRNG = Random.default_rng())
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    xvec = Vector{Float64}(x_new)
    n = length(yvec)
    preds = zeros(B)
    for b in 1:B
        idx = rand(rng, 1:n, n)
        Xb = Xmat[idx, :]
        yb = yvec[idx]
        β̂ = Xb \ yb
        preds[b] = dot(xvec, β̂)
    end
    return (lower = quantile(preds, 0.025), upper = quantile(preds, 0.975), draws = preds)
end

end # module
