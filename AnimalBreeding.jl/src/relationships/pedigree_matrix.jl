# ============================================================================
# 关系矩阵模块 - 谱系关系矩阵 (A)
# AnimalBreeding.jl
# ============================================================================

"""
    sort_pedigree_for_A_matrix(pedigree::DataFrame) -> (DataFrame, Dict)

在计算A矩阵之前，对谱系进行排序并建立ID映射。

# 流程
1.  识别所有出现在谱系中的独立个体（包括仅作为亲本的个体）。
2.  构建父母-子代图并执行拓扑排序，确保父母总是在子代之前。
3.  创建一个从动物ID到其在排序后列表中的索引的映射。

# 参数
- `pedigree::DataFrame`: 原始的谱系数据框。

# 返回
- `DataFrame`: 按世代排序后的完整谱系。
- `Dict`: 从动物ID到排序后索引的映射。
"""
function sort_pedigree_for_A_matrix(pedigree::DataFrame)
    animals = unique(vcat(pedigree.animal, pedigree.sire, pedigree.dam))
    filter!(x -> !ismissing(x) && x != 0, animals)

    parent_lookup = Dict{Any, Tuple{Any,Any}}()
    for row in eachrow(pedigree)
        sire = ismissing(row.sire) ? 0 : row.sire
        dam = ismissing(row.dam) ? 0 : row.dam
        parent_lookup[row.animal] = (sire, dam)
    end

    indegree = Dict(id => 0 for id in animals)
    children = Dict{Any, Vector{Any}}()
    for id in animals
        sire, dam = get(parent_lookup, id, (0, 0))
        for parent in (sire, dam)
            if ismissing(parent) || parent == 0
                continue
            end
            indegree[id] += 1
            push!(get!(() -> Any[], children, parent), id)
        end
    end

    queue = [id for id in animals if indegree[id] == 0]
    sorted_ids = Any[]
    while !isempty(queue)
        current = popfirst!(queue)
        push!(sorted_ids, current)
        for child in get(children, current, Any[])
            indegree[child] -= 1
            if indegree[child] == 0
                push!(queue, child)
            end
        end
    end

    if length(sorted_ids) != length(animals)
        error("谱系包含环路，无法完成拓扑排序。请先调用 `validate_pedigree` 检查数据。")
    end

    sire_col = Vector{Any}(undef, length(sorted_ids))
    dam_col = Vector{Any}(undef, length(sorted_ids))
    for (idx, id) in enumerate(sorted_ids)
        sire, dam = get(parent_lookup, id, (0, 0))
        sire_col[idx] = ismissing(sire) ? 0 : sire
        dam_col[idx] = ismissing(dam) ? 0 : dam
    end

    sorted_ped = DataFrame(animal=sorted_ids, sire=sire_col, dam=dam_col)
    sorted_id_map = Dict(id => i for (i, id) in enumerate(sorted_ped.animal))

    return sorted_ped, sorted_id_map
end

"""
    compute_A_matrix(pedigree::DataFrame) -> (Matrix{Float64}, DataFrame, Dict)

根据谱系数据计算加性遗传关系矩阵 (A矩阵)。
"""
function compute_A_matrix(pedigree::DataFrame)
    @info "计算谱系关系矩阵 A..."

    sorted_ped, id_map = sort_pedigree_for_A_matrix(pedigree)
    n = nrow(sorted_ped)
    A = zeros(Float64, n, n)

    @showprogress desc="计算A矩阵: " for i in 1:n
        sire_idx = get(id_map, sorted_ped.sire[i], 0)
        dam_idx = get(id_map, sorted_ped.dam[i], 0)

        for j in 1:(i-1)
            val = 0.0
            if sire_idx > 0
                val += 0.5 * A[sire_idx, j]
            end
            if dam_idx > 0
                val += 0.5 * A[dam_idx, j]
            end
            A[i, j] = A[j, i] = val
        end

        inbreeding = (sire_idx > 0 && dam_idx > 0) ? 0.5 * A[sire_idx, dam_idx] : 0.0
        A[i, i] = 1.0 + inbreeding
    end

    @info "A矩阵计算完成。"
    return A, sorted_ped, id_map
end

"""
    compute_A_inv_matrix(pedigree::DataFrame) -> (SparseMatrixCSC, DataFrame, Dict)

高效地直接计算谱系关系矩阵的逆 (A⁻¹)。
"""
function compute_A_inv_matrix(sorted_ped::DataFrame, id_map::Dict{Any,Int})
    n = nrow(sorted_ped)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    diag_vals = zeros(Float64, n)

    @showprogress desc="构建A-inverse: " for i in 1:n
        sire_idx = get(id_map, sorted_ped.sire[i], 0)
        dam_idx = get(id_map, sorted_ped.dam[i], 0)

        if sire_idx == 0 && dam_idx == 0
            diag_vals[i] += 1.0
        elseif xor(sire_idx == 0, dam_idx == 0)
            parent_idx = sire_idx == 0 ? dam_idx : sire_idx
            diag_vals[i] += 1.5
            diag_vals[parent_idx] += 0.25
            push!(rows, i); push!(cols, parent_idx); push!(vals, -0.5)
            push!(rows, parent_idx); push!(cols, i); push!(vals, -0.5)
        else
            diag_vals[i] += 2.0
            diag_vals[sire_idx] += 0.25
            diag_vals[dam_idx] += 0.25
            push!(rows, i); push!(cols, sire_idx); push!(vals, -0.5)
            push!(rows, sire_idx); push!(cols, i); push!(vals, -0.5)
            push!(rows, i); push!(cols, dam_idx); push!(vals, -0.5)
            push!(rows, dam_idx); push!(cols, i); push!(vals, -0.5)
            push!(rows, sire_idx); push!(cols, dam_idx); push!(vals, 0.25)
            push!(rows, dam_idx); push!(cols, sire_idx); push!(vals, 0.25)
        end
    end

    for i in 1:n
        push!(rows, i)
        push!(cols, i)
        push!(vals, diag_vals[i])
    end

    A_inv = sparse(rows, cols, vals, n, n)

    return A_inv
end

function compute_A_inv_matrix(pedigree::DataFrame)
    @info "直接计算谱系关系矩阵的逆 (A-inverse)..."

    sorted_ped, id_map = sort_pedigree_for_A_matrix(pedigree)
    A_inv = compute_A_inv_matrix(sorted_ped, id_map)

    @info "A-inverse 矩阵计算完成。"
    return A_inv, sorted_ped, id_map
end
