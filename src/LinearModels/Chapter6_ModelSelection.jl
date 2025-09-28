module LinearModelsChapter6

using LinearAlgebra

export model_selection_criteria, forward_stepwise_selection, information_weights

"""
    model_selection_criteria(y, X, β̂; σ̂2 = nothing)

根据章节六对模型选择准则的讨论，计算 AIC、BIC 与调整后的R²。
若未提供 `σ̂2`，函数自动依据残差平方和与自由度进行估计。
"""
function model_selection_criteria(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β̂::AbstractVector{<:Real}; σ̂2::Union{Nothing, Real} = nothing)
    yvec = Vector{Float64}(y)
    Xmat = Matrix{Float64}(X)
    βvec = Vector{Float64}(β̂)
    n, p = size(Xmat)
    res = yvec - Xmat * βvec
    rss = sum(abs2, res)
    σ̂2_val = σ̂2 === nothing ? rss / (n - p) : float(σ̂2)
    loglik = -0.5 * n * (log(2π * σ̂2_val) + 1)
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + p * log(n)
    tss = sum(abs2, yvec .- mean(yvec))
    adj_r2 = 1 - (rss / (n - p)) / (tss / (n - 1))
    return (AIC = aic, BIC = bic, adjusted_R2 = adj_r2, σ̂2 = σ̂2_val)
end

"""
    forward_stepwise_selection(X, y, candidate_columns; criterion = :AIC)

实现章节六介绍的逐步向前选择算法，基于给定准则挑选最优特征集合。
返回所选列索引及对应准则值的变化路径。
"""
function forward_stepwise_selection(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, candidate_columns::Vector{Int}; criterion::Symbol = :AIC)
    remaining = copy(candidate_columns)
    selected = Int[]
    path = Float64[]
    Xmat = Matrix{Float64}(X)
    yvec = Vector{Float64}(y)
    while !isempty(remaining)
        scores = Float64[]
        for col in remaining
            cols = sort(vcat(selected, col))
            Xsub = Xmat[:, cols]
            β̂ = Xsub \ yvec
            crit = model_selection_criteria(yvec, Xsub, β̂)
            push!(scores, getfield(crit, criterion))
        end
        best_idx = argmin(scores)
        best_col = remaining[best_idx]
        push!(selected, best_col)
        push!(path, scores[best_idx])
        deleteat!(remaining, best_idx)
    end
    return (selected = selected, path = path)
end

"""
    information_weights(criteria)

根据章节六讨论的Akaike权重，将一系列模型准则值转换为归一化权重，便于模型集成或模型平均分析。
"""
function information_weights(criteria::AbstractVector{<:Real})
    crit = Vector{Float64}(criteria)
    Δ = crit .- minimum(crit)
    weights = exp.(-0.5 .* Δ)
    weights ./= sum(weights)
    return weights
end

end # module
