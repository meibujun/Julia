module LinearModelsChapter5

using LinearAlgebra

export residual_diagnostics, influence_measures, durbin_watson_statistic,
       partial_residuals

"""
    residual_diagnostics(y, X, β̂)

计算线性模型的残差、标准化残差以及外推学生化残差，用于章节五对诊断图与残差分析的系统介绍。
返回一个命名元组 `(residuals, standardized, studentized, hatvalues, σ̂2)`，以便后续绘图或统计检验。
"""
function residual_diagnostics(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β̂::AbstractVector{<:Real})
    yvec = Vector{Float64}(y)
    Xmat = Matrix{Float64}(X)
    βvec = Vector{Float64}(β̂)
    fitted = Xmat * βvec
    res = yvec - fitted
    n, p = size(Xmat)
    XtX = Xmat' * Xmat
    XtX_inv = inv(XtX)
    hat = vec(sum((Xmat * XtX_inv) .* Xmat, dims = 2))
    σ̂2 = sum(abs2, res) / (n - p)
    std_res = res ./ sqrt.(σ̂2 .* (1 .- hat))
    # 外推学生化残差：使用每个观测点剔除后的均方误差估计
    s2_minus_i = max.(((n - p) .* σ̂2 .- res.^2 ./ (1 .- hat)) ./ (n - p - 1), eps())
    studentized = res ./ sqrt.(s2_minus_i .* (1 .- hat))
    return (residuals = res, standardized = std_res, studentized = studentized, hatvalues = hat, σ̂2 = σ̂2)
end

"""
    influence_measures(y, X, β̂)

结合帽子矩阵、残差方差与杠杆值给出库克距离与DFBETAS，以量化影响点和高杠杆点对模型的影响。
返回 `(cooks_distance, dfbetas)`，其中 `dfbetas` 为一个矩阵，每列对应一个预测变量。
"""
function influence_measures(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β̂::AbstractVector{<:Real})
    diag_data = residual_diagnostics(y, X, β̂)
    res = diag_data.residuals
    hat = diag_data.hatvalues
    σ̂2 = diag_data.σ̂2
    Xmat = Matrix{Float64}(X)
    n, p = size(Xmat)
    XtX_inv = inv(Xmat' * Xmat)
    cooks = ((res .^ 2) ./ (p * σ̂2)) .* (hat ./ (1 .- hat).^2)
    dfbetas = zeros(n, p)
    for i in 1:n
        xi = view(Xmat, i, :)
        scale = sqrt(σ̂2 * (1 - hat[i]))
        dfbetas[i, :] .= (res[i] / scale) .* (xi * XtX_inv)
    end
    return (cooks_distance = cooks, dfbetas = dfbetas)
end

"""
    durbin_watson_statistic(residuals)

计算Durbin-Watson统计量，用于检测章节五中讨论的残差自相关问题。
"""
function durbin_watson_statistic(residuals::AbstractVector{<:Real})
    res = Vector{Float64}(residuals)
    length(res) > 1 || throw(ArgumentError("Durbin-Watson统计量至少需要两个残差"))
    num = sum((res[2:end] .- res[1:end-1]) .^ 2)
    den = sum(res .^ 2)
    return num / den
end

"""
    partial_residuals(y, X, β̂)

给出部分残差（component + residual）以构造加性模型诊断图。
返回矩阵，每列对应一个解释变量的部分残差。
"""
function partial_residuals(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β̂::AbstractVector{<:Real})
    yvec = Vector{Float64}(y)
    Xmat = Matrix{Float64}(X)
    βvec = Vector{Float64}(β̂)
    res = yvec - Xmat * βvec
    return res .* ones(1, size(Xmat, 2)) .+ Xmat .* βvec'
end

end # module
