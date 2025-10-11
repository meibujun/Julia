# ============================================================================
# BLUP评估模块 - REML方差组分估计
# AnimalBreeding.jl
# ============================================================================

"""
    REMLResult

存储REML方差组分估计结果的结构体。

# 字段
- `variance_components::Dict{String,Float64}`: 估计出的方差组分字典。
- `log_likelihood::Float64`: 最终的对数似然值 (当前为占位符)。
- `iterations::Int`: 完成的迭代次数。
- `converged::Bool`: 是否达到收敛标准。
"""
mutable struct REMLResult
    variance_components::Dict{String,Float64}
    log_likelihood::Float64
    iterations::Int
    converged::Bool
end

"""
    estimate_variances_reml(...) -> REMLResult

使用基于EM算法的REML方法来迭代估计方差组分。

# 算法思想
REML通过最大化残差（或限制性）似然函数来获得方差组分的无偏估计。
本实现采用EM（期望-最大化）算法进行迭代求解，该算法在每次迭代中：
1.  **E-步**: 基于当前的方差组分估计，求解混合模型方程，获得随机效应的期望值。
2.  **M-步**: 基于E-步的结果，最大化似然函数，更新方差组分的估计值。

# 参数
- `X, Z_dict, y, dm, model`: 模型和数据组件。
- `max_iter::Int`: 最大迭代次数。
- `tol::Float64`: 相对收敛阈值。
- `relaxation_param::Float64`: 松弛因子 (0 < α <= 1)。值越小，更新越平滑。

# 返回
- `REMLResult`: 包含估计结果的对象。
"""
function estimate_variances_reml(
    X::Matrix, Z_dict::Dict, y::Vector, dm::DataManager, model::ModelSpec;
    max_iter::Int=100,
    tol::Float64=1e-6,
    relaxation_param::Float64=0.7
)
    @info "开始REML方差组分估计 (max_iter=$(max_iter), tol=$(tol), relaxation=$(relaxation_param))..."

    if isempty(model.random_effects)
        error("REML估计需要至少一个随机效应。")
    end

    n_obs = length(y)
    n_fixed = size(X, 2)

    total_variance = var(y)
    n_random = length(model.random_effects)

    variances = Dict{String, Float64}("residual" => total_variance * 0.6)
    for effect in model.random_effects
        variances[effect.name] = total_variance * 0.4 / n_random
    end

    @info "初始方差组分: " * join(["$(k)=$(round(v, digits=4))" for (k,v) in variances], ", ")

    ranges = _random_effect_ranges(model, Z_dict, n_fixed)
    random_offsets = zeros(Float64, n_obs)
    penalty_dense_cache = nothing

    p = Progress(max_iter, desc="REML迭代: ")
    for iter in 1:max_iter
        variances_old = copy(variances)

        C, rhs = setup_mme(X, Z_dict, y, dm, model, variances)
        C_dense = Matrix(C)
        factor = cholesky(Symmetric(C_dense); check=false)
        solutions = factor \ rhs
        C_inv = Matrix(inv(factor))

        _accumulate_random_offsets!(random_offsets, model, Z_dict, solutions, ranges)
        residuals = y - X * solutions[1:n_fixed] - random_offsets

        variances_new = Dict{String, Float64}()

        denom_res = max(n_obs - n_fixed, 1)
        variances_new["residual"] = max(1e-9, dot(residuals, residuals) / denom_res)

        for effect in model.random_effects
            name = effect.name
            rng = ranges[name]
            u = solutions[rng]
            dim = length(rng)

            if effect.type == :additive
                if penalty_dense_cache === nothing
                    penalty_dense_cache = Matrix(_relationship_inverse(dm))
                end
                penalty = penalty_dense_cache
            else
                penalty = Matrix{Float64}(I, dim, dim)
            end

            trace_term = sum(C_inv[rng, rng] .* penalty)
            quad_form = dot(u, penalty * u)
            variances_new[name] = max(1e-9, (quad_form + trace_term) / dim)
        end

        max_rel_change = 0.0
        for (key, old_val) in variances_old
            if haskey(variances_new, key) && abs(old_val) > 1e-9
                rel_change = abs(variances_new[key] - old_val) / old_val
                max_rel_change = max(max_rel_change, rel_change)
            end
        end

        next!(p, showvalues=[(:iter, iter), (:max_rel_change, round(max_rel_change, digits=7))])

        if max_rel_change < tol
            finish!(p)
            @info "REML在第 $(iter) 次迭代后收敛。"
            for (k, v) in variances_new
                variances[k] = v
            end
            return REMLResult(copy(variances), 0.0, iter, true)
        end

        for (k, old_val) in variances_old
            if haskey(variances_new, k)
                new_val = relaxation_param * variances_new[k] + (1 - relaxation_param) * old_val
                variances[k] = max(1e-9, new_val)
            end
        end
    end

    @warn "REML在 $(max_iter) 次迭代后未收敛。"
    return REMLResult(variances, 0.0, max_iter, false)
end
