# ============================================================================
# 关系矩阵模块 - 谱系关系矩阵 (A)
# AnimalBreeding.jl
# ============================================================================

"""
    sort_pedigree_for_A_matrix(pedigree::DataFrame) -> (DataFrame, Dict)

在计算A矩阵之前，对谱系进行排序并建立ID映射。

# 流程
1.  识别所有出现在谱系中的独立个体（包括仅作为亲本的个体）。
2.  基于亲子关系计算每个个体的世代数。
3.  根据世代数对所有个体进行排序，确保父母总是在子代之前。
4.  创建一个从动物ID到其在排序后列表中的索引的映射。

# 参数
- `pedigree::DataFrame`: 原始的谱系数据框。

# 返回
- `DataFrame`: 按世代排序后的完整谱系。
- `Dict`: 从动物ID到排序后索引的映射。
"""
function sort_pedigree_for_A_matrix(pedigree::DataFrame)
    # 1. 识别所有独立个体
    all_ids = unique(vcat(pedigree.animal, pedigree.sire, pedigree.dam))
    filter!(x -> !ismissing(x) && x != 0, all_ids)

    # 2. 创建临时完整谱系和ID映射
    id_map_temp = Dict(id => i for (i, id) in enumerate(all_ids))
    full_ped = DataFrame(animal=all_ids)
    full_ped = leftjoin(full_ped, pedigree, on=:animal)
    full_ped.sire = coalesce.(full_ped.sire, 0)
    full_ped.dam = coalesce.(full_ped.dam, 0)

    # 3. 计算世代数
    generations = zeros(Int, length(all_ids))
    for i in 1:length(all_ids) # 迭代直到世代数稳定
        changed = false
        for (idx, row) in enumerate(eachrow(full_ped))
            sire_idx = get(id_map_temp, row.sire, 0)
            dam_idx = get(id_map_temp, row.dam, 0)
            sire_gen = sire_idx > 0 ? generations[sire_idx] : 0
            dam_gen = dam_idx > 0 ? generations[dam_idx] : 0
            new_gen = max(sire_gen, dam_gen) + 1
            if generations[idx] != new_gen
                generations[idx] = new_gen
                changed = true
            end
        end
        if !changed; break; end
    end

    # 4. 排序并创建最终映射
    sorted_indices = sortperm(generations)
    sorted_ped = full_ped[sorted_indices, :]
    sorted_id_map = Dict(id => i for (i, id) in enumerate(sorted_ped.animal))

    return sorted_ped, sorted_id_map
end

"""
    compute_A_matrix(pedigree::DataFrame) -> (Matrix{Float64}, DataFrame)

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

        # 非对角线元素
        for j in 1:(i-1)
            val = 0.0
            if sire_idx > 0; val += 0.5 * A[sire_idx, j]; end
            if dam_idx > 0; val += 0.5 * A[dam_idx, j]; end
            A[i, j] = A[j, i] = val
        end

        # 对角线元素
        inbreeding = (sire_idx > 0 && dam_idx > 0) ? 0.5 * A[sire_idx, dam_idx] : 0.0
        A[i, i] = 1.0 + inbreeding
    end

    @info "A矩阵计算完成。"
    return A, sorted_ped
end

"""
    compute_A_inv_matrix(pedigree::DataFrame) -> (SparseMatrixCSC, DataFrame)

高效地直接计算谱系关系矩阵的逆 (A⁻¹)。
"""
function compute_A_inv_matrix(pedigree::DataFrame)
    @info "直接计算谱系关系矩阵的逆 (A-inverse)..."

    sorted_ped, id_map = sort_pedigree_for_A_matrix(pedigree)
    n = nrow(sorted_ped)

    I, J, V = Int[], Int[], Float64[]

    @showprogress desc="构建A-inverse: " for i in 1:n
        sire_idx = get(id_map, sorted_ped.sire[i], 0)
        dam_idx = get(id_map, sorted_ped.dam[i], 0)

        d_ii = 0.0
        if sire_idx > 0 && dam_idx > 0; d_ii = 2.0
        elseif sire_idx > 0 || dam_idx > 0; d_ii = 1.5
        else; d_ii = 1.0; end

        push!(I, i); push!(J, i); push!(V, d_ii)

        if sire_idx > 0
            push!(I, i, sire_idx); push!(J, sire_idx, i); push!(V, -1.0, -1.0)
            push!(I, sire_idx); push!(J, sire_idx); push!(V, 0.5)
        end
        if dam_idx > 0
            push!(I, i, dam_idx); push!(J, dam_idx, i); push!(V, -1.0, -1.0)
            push!(I, dam_idx); push!(J, dam_idx); push!(V, 0.5)
        end
        if sire_idx > 0 && dam_idx > 0
            push!(I, sire_idx, dam_idx); push!(J, dam_idx, sire_idx); push!(V, 0.5, 0.5)
        end
    end

    A_inv = sparse(I, J, V, n, n)

    @info "A-inverse 矩阵计算完成。"
    return A_inv, sorted_ped
end