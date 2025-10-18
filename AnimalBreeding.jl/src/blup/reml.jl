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
    @info "开始REML方差组分估计 (max_iter=$max_iter, tol=$tol, relaxation=$relaxation_param)..."

    # 1. 初始化方差组分
    total_variance = var(y)
    n_random = length(model.random_effects)

    variances = Dict{String, Float64}(
        "residual" => total_variance * 0.6
    )
    for effect in model.random_effects
        variances[effect.name] = total_variance * 0.4 / n_random
    end

    @info "初始方差组分: " * join(["$k=$(round(v, digits=4))" for (k,v) in variances], ", ")

    # 2. 迭代求解
    p = Progress(max_iter, desc="REML迭代: ")
    n_obs, n_fixed = size(X)
    included_effects = _included_random_effects(model, Z_dict)
    blocks = _random_effect_blocks(n_fixed, included_effects, Z_dict)
    relationship_inv = nothing
    if any(effect -> effect.type == :additive, included_effects)
        relationship_inv = _relationship_inverse_matrix(dm)
    end

    for iter in 1:max_iter
        variances_old = copy(variances)

        # --- E-步: 求解MME ---
        C, rhs = setup_mme(X, Z_dict, y, dm, model, variances)
        C_inv = inv(Matrix(C)) # 警告: 这是计算瓶颈，更高级的AI-REML会避免它
        solutions = C_inv * rhs

        # --- M-步: 更新方差 ---
        variances_new = Dict{String, Float64}()

        fitted_random = zeros(Float64, n_obs)
        for block in blocks
            u = solutions[block.range]
            fitted_random .+= Z_dict[block.effect.name] * u
        end
        residuals = y - X * solutions[1:n_fixed] - fitted_random
        s_y = dot(residuals, residuals)
        tr_term = tr(C_inv[1:n_fixed, 1:n_fixed] * (X' * X))
        variances_new["residual"] = (s_y + tr_term) / n_obs

        for block in blocks
            effect = block.effect
            name = effect.name
            dim = length(block.range)
            u = solutions[block.range]

            if effect.type == :additive && !isnothing(relationship_inv)
                rel_inv = relationship_inv
                u_Ainv_u = dot(u, rel_inv * u)
                tr_term_u = tr(C_inv[block.range, block.range] * rel_inv)
                variances_new[name] = (u_Ainv_u + tr_term_u) / dim
            end
        end

        # 3. 检查收敛
        max_rel_change = 0.0
        for (k, old_val) in variances_old
            if haskey(variances_new, k) && abs(old_val) > 1e-9
                rel_change = abs(variances_new[k] - old_val) / old_val
                max_rel_change = max(max_rel_change, rel_change)
            end
        end

        next!(p, showvalues=[(:iter, iter), (:max_rel_change, round(max_rel_change, digits=7))])

        if max_rel_change < tol
            finish!(p)
            final_vars = copy(variances_old)
            for (k, v) in variances_new
                final_vars[k] = max(1e-9, v)
            end
            @info "REML在第 $iter 次迭代后收敛。"
            return REMLResult(final_vars, 0.0, iter, true)
        end

        # 4. 松弛更新
        for (k, old_val) in variances_old
            if haskey(variances_new, k)
                new_val = relaxation_param * variances_new[k] + (1 - relaxation_param) * old_val
                variances[k] = max(1e-9, new_val)
            end
        end
    end

    @warn "REML在 $max_iter 次迭代后未收敛。"
    return REMLResult(copy(variances), 0.0, max_iter, false)
end
