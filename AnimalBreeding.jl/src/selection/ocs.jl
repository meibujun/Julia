# =============================================================================
# Optimal Contribution Selection (OCS)
# =============================================================================

"""
    _coerce_breeding_values(ebvs)

Standardise different container types of breeding values to a plain
`Vector{Float64}` while returning a mapping back to the original
identifier ordering.
"""
function _coerce_breeding_values(ebvs)
    if ebvs isa AbstractVector{<:Real}
        return collect(float.(ebvs)), nothing
    elseif ebvs isa DataFrame
        if :EBV ∉ names(ebvs)
            throw(ArgumentError("提供的DataFrame缺少EBV列，无法进行最优贡献选择。"))
        end
        ids = nothing
        if :animal_id ∈ names(ebvs)
            ids = copy(ebvs.animal_id)
        elseif :animal ∈ names(ebvs)
            ids = copy(ebvs.animal)
        end
        return Float64.(ebvs.EBV), ids
    else
        throw(ArgumentError("不支持的育种值数据类型: $(typeof(ebvs))"))
    end
end

"""
    _mean_pairwise_relationship(R, indices)

Compute the average off-diagonal relationship for the provided indices.
"""
function _mean_pairwise_relationship(R::AbstractMatrix, indices::Vector{Int})
    n = length(indices)
    if n <= 1
        return 0.0
    end
    sub = view(R, indices, indices)
    total = sum(sub) - sum(diag(sub))
    return total / (n * (n - 1))
end

"""
    _normalise_contributions!(w)

Project the contribution vector onto the simplex (non-negative and summing to one).
"""
function _normalise_contributions!(w::Vector{Float64})
    for i in eachindex(w)
        if !isfinite(w[i]) || w[i] < 0
            w[i] = 0.0
        end
    end
    total = sum(w)
    if total ≤ 0
        fill!(w, 1 / length(w))
    else
        @inbounds w ./= total
    end
    return w
end

"""
    _adjust_contributions!(w, R, limit)

Heuristically shrink contributions so that `w' * R * w` does not exceed the
specified limit. Returns a Boolean flag indicating whether the constraint could
be met.
"""
function _adjust_contributions!(w::Vector{Float64}, R::AbstractMatrix, limit::Float64; max_iter::Int=1_000, tol::Float64=1e-6)
    constraint = (w' * R * w)
    if constraint ≤ limit + tol
        return true
    end

    uniform = fill(1 / length(w), length(w))
    converged = false
    for _ in 1:max_iter
        @. w = 0.5 * (w + uniform)
        _normalise_contributions!(w)
        constraint = (w' * R * w)
        if constraint ≤ limit + tol
            converged = true
            break
        end
    end
    return converged
end

"""
    optimal_contribution_selection(ebvs, relationship_matrix, n_select; kwargs...)

Perform a heuristic optimal contribution selection (OCS) using the supplied
breeding values and relationship matrix.

# Arguments
- `ebvs`: A vector of breeding values or a DataFrame containing an `:EBV` column.
- `relationship_matrix`: Symmetric matrix (A, G, or H) describing relationships.
- `n_select`: Number of individuals to keep.

# Keyword Arguments
- `max_relationship`: Upper bound on the average pairwise relationship
  amongst the selected group (default `0.25`).
- `return_contributions`: When `true`, includes the contributions vector in the
  returned dictionary (default `true`).

# Returns
A dictionary containing the selected indices, their EBVs, average relationship
and the expected genetic gain under the derived contributions.
"""
function optimal_contribution_selection(ebvs,
                                        relationship_matrix::AbstractMatrix,
                                        n_select::Integer;
                                        max_relationship::Float64=0.25,
                                        return_contributions::Bool=true)
    n_select > 0 || throw(ArgumentError("n_select 必须为正整数。"))

    values, ids = _coerce_breeding_values(ebvs)
    n = length(values)
    size(relationship_matrix, 1) == n || throw(ArgumentError("关系矩阵的维度与育种值数量不匹配。"))
    size(relationship_matrix, 1) == size(relationship_matrix, 2) || throw(ArgumentError("关系矩阵必须是方阵。"))

    order = sortperm(values; rev=true)
    selected = Int[]
    for idx in order
        push!(selected, idx)
        if length(selected) == n_select
            if _mean_pairwise_relationship(relationship_matrix, selected) ≤ max_relationship
                break
            else
                pop!(selected)
            end
        end
    end

    # Fallback: if constraint prevents reaching the required count, relax greedily.
    if length(selected) < n_select
        for idx in order
            if idx in selected
                continue
            end
            push!(selected, idx)
            if length(selected) == n_select
                break
            end
        end
    end

    sort!(selected)
    selected_values = values[selected]
    contributions = fill(1.0, length(selected))
    contributions .= selected_values .- minimum(selected_values)
    _normalise_contributions!(contributions)

    subR = view(relationship_matrix, selected, selected)
    constraint_ok = _adjust_contributions!(contributions, subR, max_relationship)
    avg_relationship = contributions' * subR * contributions
    expected_gain = dot(selected_values, contributions)

    output = Dict{String,Any}(
        "selected_indices" => selected,
        "selected_ebvs" => selected_values,
        "avg_relationship" => avg_relationship,
        "expected_gain" => expected_gain,
        "constraint_satisfied" => constraint_ok,
    )

    if ids !== nothing
        output["selected_ids"] = ids[selected]
    end

    if return_contributions
        output["contributions"] = contributions
    end

    return output
end

export optimal_contribution_selection
