module LinearModelsChapter10

using LinearAlgebra

export conjugate_gradient_solver, sweep_operator!, block_cholesky_update!,
       sparse_woodbury_inverse

"""
    conjugate_gradient_solver(A, b; tol = 1e-8, maxiter = 10_000)

实现章节十对迭代求解的讨论，利用共轭梯度法求解对称正定线性方程组 `Ax = b`。
返回 `(x, iterations, converged)`。
"""
function conjugate_gradient_solver(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}; tol::Real = 1e-8, maxiter::Int = 10_000)
    Amat = Matrix(A)
    bvec = Vector{Float64}(b)
    x = zeros(length(bvec))
    r = bvec - Amat * x
    p = copy(r)
    rsold = dot(r, r)
    for k in 1:maxiter
        Ap = Amat * p
        α = rsold / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)
        if sqrt(rsnew) < tol
            return (x = x, iterations = k, converged = true)
        end
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return (x = x, iterations = maxiter, converged = false)
end

"""
    sweep_operator!(S, k)

对对称矩阵 `S` 执行扫掠算子（SWEEP Operator），对应章节十在模型选择与更新公式中的快速矩阵更新技巧。
函数就地修改矩阵，并返回被扫掠的主元素值。
"""
function sweep_operator!(S::AbstractMatrix{<:Real}, k::Int)
    n = size(S, 1)
    k in 1:n || throw(ArgumentError("索引超出矩阵范围"))
    akk = S[k, k]
    isapprox(akk, 0.0) && throw(ArgumentError("扫掠算子要求主元素非零"))
    for i in 1:n
        for j in 1:n
            if i != k && j != k
                S[i, j] -= S[i, k] * S[k, j] / akk
            end
        end
    end
    for i in 1:n
        if i != k
            S[i, k] /= akk
            S[k, i] = S[i, k]
        end
    end
    S[k, k] = -1 / akk
    return akk
end

"""
    block_cholesky_update!(L, X)

对已有的Cholesky分解 `L` 进行块更新，与章节十对增量式最小二乘算法保持一致。
`X` 表示新增样本的设计矩阵行块。
"""
function block_cholesky_update!(L::Cholesky{Float64, Matrix{Float64}}, X::AbstractMatrix{<:Real})
    Xmat = Matrix{Float64}(X)
    A_old = Symmetric(L.U' * L.U)
    A_new = Symmetric(Matrix(A_old) + Xmat' * Xmat)
    return cholesky(A_new)
end

"""
    sparse_woodbury_inverse(A, U, C, V)

实现Woodbury矩阵恒等式，快速计算 `A + UCV` 的逆矩阵，用于章节十的广义逆与高维更新讨论。
"""
function sparse_woodbury_inverse(A::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real}, C::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real})
    Amat = Matrix(A)
    Umat = Matrix(U)
    Cmat = Matrix(C)
    Vmat = Matrix(V)
    Ainv = inv(Amat)
    mid = inv(Cmat + Vmat * Ainv * Umat)
    return Ainv - Ainv * Umat * mid * Vmat * Ainv
end

end # module
