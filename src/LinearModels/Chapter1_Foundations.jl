module LinearModelsChapter1

using LinearAlgebra
using Random

export LinearModelDesign, simulate_response, gram_schmidt_orthonormal_basis,
       projection_matrix, center_and_scale

"""
    struct LinearModelDesign

保存线性模型的基本要素，包括设计矩阵 `X`、真实参数 `β` 以及误差方差 `σ²`。
该结构体在后续章节中被重复使用，用于模拟数据和评估估计方法的性能。
"""
struct LinearModelDesign
    X::Matrix{Float64}
    β::Vector{Float64}
    σ2::Float64

    function LinearModelDesign(X::AbstractMatrix, β::AbstractVector, σ2::Real)
        size(X, 2) == length(β) || throw(ArgumentError("设计矩阵列数必须与参数个数一致"))
        σ2 > 0 || throw(ArgumentError("误差方差必须为正"))
        return new(Matrix{Float64}(X), Vector{Float64}(β), float(σ2))
    end
end

"""
    simulate_response(design; rng=Random.default_rng())

根据指定的设计对象模拟响应向量 `y = Xβ + ε`，其中 `ε ~ N(0, σ²I)`。
该函数作为全书的统一数据入口，方便在不同章节中复用同一套模拟流程。
"""
function simulate_response(design::LinearModelDesign; rng::AbstractRNG = Random.default_rng())
    noise = sqrt(design.σ2) .* randn(rng, size(design.X, 1))
    return design.X * design.β + noise
end

"""
    gram_schmidt_orthonormal_basis(X)

对设计矩阵执行改进的格拉姆-施密特正交化，返回列正交的矩阵 `Q` 与上三角矩阵 `R`。
通过数值稳定的正交分解，展示线性模型分析中常用的QR分解技术。
"""
function gram_schmidt_orthonormal_basis(X::AbstractMatrix{<:Real})
    A = Matrix{Float64}(X)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)
    for j in 1:n
        v = copy(view(A, :, j))
        for i in 1:j-1
            R[i, j] = dot(Q[:, i], v)
            v .-= R[i, j] .* Q[:, i]
        end
        R[j, j] = norm(v)
        R[j, j] > 0 || throw(ArgumentError("设计矩阵存在线性相关列，无法获得正交基"))
        Q[:, j] .= v ./ R[j, j]
    end
    return Q, R
end

"""
    projection_matrix(X)

返回设计矩阵 `X` 的列空间投影矩阵 `P = X(X'X)^{-1}X'`。
该函数强调正交投影在最小二乘估计中的核心作用，可用于理解残差与拟合值的几何关系。
"""
function projection_matrix(X::AbstractMatrix{<:Real})
    Q, R = gram_schmidt_orthonormal_basis(X)
    return Q * Q'
end

"""
    center_and_scale(X)

对设计矩阵进行列中心化与标准化，返回处理后的矩阵以及对应的中心与缩放系数。
该步骤可提升数值稳定性，并为后续章节的正则化与贝叶斯扩展提供基础。
"""
function center_and_scale(X::AbstractMatrix{<:Real})
    Xmat = Matrix{Float64}(X)
    μ = mapslices(mean, Xmat; dims=1)
    σ = mapslices(std, Xmat; dims=1)
    σ[σ .== 0.0] .= 1.0
    Xscaled = (Xmat .- μ) ./ σ
    return Xscaled, vec(μ), vec(σ)
end

end # module
