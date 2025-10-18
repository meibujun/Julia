# =============================================================================
# 选择与交配规划模块
# =============================================================================

"""
    sample(collection, n; replace=false, rng=Random.default_rng())

提供一个轻量级的抽样函数，默认不放回抽样。支持 `AbstractVector` 或
可索引的迭代器（如 `UnitRange`）。
"""
function sample(collection, n::Integer; replace::Bool=false, rng::Random.AbstractRNG=Random.default_rng())
    n < 0 && error("n 必须为非负整数。")
    values = collect(collection)
    total = length(values)
    total == 0 && error("无法从空集合中抽样。")

    if replace
        return [values[rand(rng, 1:total)] for _ in 1:n]
    else
        n > total && error("不放回抽样的样本量不能超过集合大小。")
        perm = randperm(rng, total)
        return values[perm[1:n]]
    end
end

# -----------------------------------------------------------------------------
# 内部辅助函数
# -----------------------------------------------------------------------------
function _average_relationship(matrix::AbstractMatrix, indices::Vector{Int})
    m = length(indices)
    m <= 1 && return 0.0
    total = 0.0
    count = 0
    for i in 1:(m - 1)
        idx_i = indices[i]
        for j in (i + 1):m
            total += matrix[idx_i, indices[j]]
            count += 1
        end
    end
    return count == 0 ? 0.0 : total / count
end

# -----------------------------------------------------------------------------
# 最优贡献选择 (OCS)
# -----------------------------------------------------------------------------
"""
    optimal_contribution_selection(ebv, relationship_matrix, n_select; max_relationship=0.2)

根据育种值向量和关系矩阵执行简单的贪心最优贡献选择。函数会尽量选择
`n_select` 个个体，同时控制平均关系度不超过 `max_relationship`。返回字典
包含选择索引、平均关系度以及建议贡献比例。
"""
function optimal_contribution_selection(ebv::AbstractVector{<:Real},
                                        relationship_matrix::AbstractMatrix{<:Real},
                                        n_select::Integer;
                                        max_relationship::Real=0.2,
                                        tolerance::Real=1e-6)
    n_select < 1 && error("n_select 必须至少为 1。")
    length(ebv) == size(relationship_matrix, 1) || error("关系矩阵行数必须与育种值长度一致。")
    size(relationship_matrix, 1) == size(relationship_matrix, 2) || error("关系矩阵必须为方阵。")

    order = sortperm(ebv; rev=true)
    selected = Int[]
    deferred = Int[]

    for idx in order
        push!(selected, idx)
        avg_rel = _average_relationship(relationship_matrix, selected)
        if avg_rel <= max_relationship + tolerance || length(selected) == 1
            if length(selected) == n_select
                break
            end
        else
            pop!(selected)
            push!(deferred, idx)
        end
    end

    relationship_violations = false
    if length(selected) < n_select
        for idx in deferred
            push!(selected, idx)
            avg_rel = _average_relationship(relationship_matrix, selected)
            if avg_rel > max_relationship + tolerance
                relationship_violations = true
            end
            if length(selected) == n_select
                break
            end
        end
    end

    if length(selected) < n_select
        remaining = setdiff(order, vcat(selected, deferred))
        for idx in remaining
            push!(selected, idx)
            avg_rel = _average_relationship(relationship_matrix, selected)
            if avg_rel > max_relationship + tolerance
                relationship_violations = true
            end
            length(selected) == n_select && break
        end
    end

    if length(selected) < n_select
        error("无法选择足够的个体，请放宽 max_relationship 或减少 n_select。")
    end

    avg_rel = _average_relationship(relationship_matrix, selected)
    contributions = fill(1.0 / length(selected), length(selected))
    selected_ebv = ebv[selected]

    if relationship_violations
        @warn "平均关系度约束无法完全满足，部分个体可能超过阈值。"
    end

    return Dict(
        "selected_indices" => selected,
        "mean_relationship" => avg_rel,
        "contributions" => contributions,
        "selected_ebv" => selected_ebv,
    )
end

# -----------------------------------------------------------------------------
# 交配计划设计
# -----------------------------------------------------------------------------
"""
    design_mating_plan(sires, dams, relationship_matrix, sire_ebv, dam_ebv; max_inbreeding=0.0625)

根据候选种公畜与母畜的育种值及关系矩阵设计交配计划。返回一个包含配对
信息的数据框，包括预期近交系数与子代育种值。
"""
function design_mating_plan(sire_indices::AbstractVector{<:Integer},
                            dam_indices::AbstractVector{<:Integer},
                            relationship_matrix::AbstractMatrix{<:Real},
                            sire_ebv::AbstractVector{<:Real},
                            dam_ebv::AbstractVector{<:Real};
                            max_inbreeding::Real=0.0625)
    length(sire_indices) == length(sire_ebv) || error("种公畜索引与育种值长度不一致。")
    length(dam_indices) == length(dam_ebv) || error("母畜索引与育种值长度不一致。")

    n = size(relationship_matrix, 1)
    size(relationship_matrix, 1) == size(relationship_matrix, 2) || error("关系矩阵必须为方阵。")

    for idx in vcat(sire_indices, dam_indices)
        (1 <= idx <= n) || error("索引 $idx 超出关系矩阵维度。")
    end

    sire_ids = Int[]
    dam_ids = Int[]
    rel_values = Float64[]
    inbreeding_values = Float64[]
    offspring_ebv = Float64[]
    sire_ebv_values = Float64[]
    dam_ebv_values = Float64[]
    feasible_flags = Bool[]

    violations = 0
    for (dam_pos, dam_idx) in enumerate(dam_indices)
        feasible_candidate = nothing
        fallback_candidate = nothing
        dam_value = dam_ebv[dam_pos]

        for (sire_pos, sire_idx) in enumerate(sire_indices)
            rel = relationship_matrix[sire_idx, dam_idx]
            rel = isnan(rel) ? 0.0 : rel
            inbreeding = max(rel, 0.0) / 2
            expected_offspring = (sire_ebv[sire_pos] + dam_value) / 2
            candidate = (; sire_idx, dam_idx, rel, inbreeding, expected_offspring,
                          sire_value = sire_ebv[sire_pos], dam_value, sire_pos)

            if inbreeding <= max_inbreeding + 1e-8
                if isnothing(feasible_candidate) || candidate.expected_offspring > feasible_candidate.expected_offspring
                    feasible_candidate = candidate
                end
            end

            if isnothing(fallback_candidate) || rel < fallback_candidate.rel ||
               (rel ≈ fallback_candidate.rel && expected_offspring > fallback_candidate.expected_offspring)
                fallback_candidate = candidate
            end
        end

        chosen = feasible_candidate
        feasible = true
        if isnothing(chosen)
            chosen = fallback_candidate
            feasible = false
            violations += 1
        end

        push!(sire_ids, chosen.sire_idx)
        push!(dam_ids, chosen.dam_idx)
        push!(rel_values, chosen.rel)
        push!(inbreeding_values, chosen.inbreeding)
        push!(offspring_ebv, chosen.expected_offspring)
        push!(sire_ebv_values, chosen.sire_value)
        push!(dam_ebv_values, chosen.dam_value)
        push!(feasible_flags, feasible)
    end

    if violations > 0
        @warn string(violations, " 个配对超过近交约束，已使用关系度最低的候选种公畜。")
    end

    return DataFrame(
        sire = sire_ids,
        dam = dam_ids,
        relationship = rel_values,
        expected_inbreeding = inbreeding_values,
        expected_offspring_ebv = offspring_ebv,
        sire_ebv = sire_ebv_values,
        dam_ebv = dam_ebv_values,
        within_inbreeding_limit = feasible_flags,
    )
end
