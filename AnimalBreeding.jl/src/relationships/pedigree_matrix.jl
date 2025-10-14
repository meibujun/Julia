# ============================================================================
# 关系矩阵模块 - 谱系关系矩阵 (A)
# AnimalBreeding.jl
# ============================================================================

"""
    sort_pedigree_for_A_matrix(pedigree::DataFrame) -> (DataFrame, Dict{Int,Int})

在计算A矩阵之前，对谱系进行排序并建立ID映射。

# 流程
1.  补全在父母列出现但缺少记录的基础动物。
2.  使用拓扑排序确保父母总是在子代之前。
3.  创建一个从动物ID到其在排序后列表中的索引的映射。

# 返回
- `DataFrame`: 按拓扑顺序排序后的完整谱系。
- `Dict{Int,Int}`: 从动物ID到排序后索引的映射。
"""
function sort_pedigree_for_A_matrix(pedigree::DataFrame)
    complete_ped = _complete_pedigree(pedigree)
    order = _topological_sort_pedigree(complete_ped)

    order_index = Dict(id => idx for (idx, id) in enumerate(order))
    n = length(order)
    sorted_ped = DataFrame(
        animal = copy(order),
        sire   = zeros(Int, n),
        dam    = zeros(Int, n),
    )

    sire_lookup = Dict(complete_ped.animal .=> complete_ped.sire)
    dam_lookup  = Dict(complete_ped.animal .=> complete_ped.dam)

    for (i, id) in enumerate(order)
        sorted_ped.sire[i] = get(sire_lookup, id, 0)
        sorted_ped.dam[i]  = get(dam_lookup, id, 0)
    end

    return sorted_ped, order_index
end

"""
    compute_A_matrix(pedigree::DataFrame) -> (Matrix{Float64}, DataFrame, Dict{Int,Int})

根据谱系数据计算加性遗传关系矩阵 (A矩阵)。
"""
function compute_A_matrix(pedigree::DataFrame)
    @info "计算谱系关系矩阵 A..."

    sorted_ped, id_map = sort_pedigree_for_A_matrix(pedigree)
    n = nrow(sorted_ped)
    A = Matrix{Float64}(I, n, n)

    sire_idx = [get(id_map, sorted_ped.sire[i], 0) for i in 1:n]
    dam_idx  = [get(id_map, sorted_ped.dam[i], 0) for i in 1:n]

    @inbounds for i in 1:n
        s_idx = sire_idx[i]
        d_idx = dam_idx[i]

        if s_idx == 0 && d_idx == 0
            A[i, i] = 1.0
            continue
        end

        if i > 1
            row_view = view(A, 1:i-1, i)
            fill!(row_view, 0.0)
            if s_idx > 0
                row_view .+= 0.5 .* view(A, 1:i-1, s_idx)
            end
            if d_idx > 0
                row_view .+= 0.5 .* view(A, 1:i-1, d_idx)
            end
            view(A, i, 1:i-1) .= row_view
        end

        if s_idx > 0 && d_idx > 0
            A[i, i] = 1.0 + 0.5 * A[s_idx, d_idx]
        else
            A[i, i] = 1.0
        end
    end

    @info "A矩阵计算完成。"
    return A, sorted_ped, id_map
end

"""
    compute_A_inv_matrix(pedigree::DataFrame) -> (SparseMatrixCSC{Float64,Int}, DataFrame, Dict{Int,Int})

使用Henderson方法高效计算谱系关系矩阵的逆 (A⁻¹)。
"""
function compute_A_inv_matrix(pedigree::DataFrame)
    @info "直接计算谱系关系矩阵的逆 (A-inverse)..."

    sorted_ped, id_map = sort_pedigree_for_A_matrix(pedigree)
    n = nrow(sorted_ped)

    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    @inbounds for i in 1:n
        animal = sorted_ped.animal[i]
        sire = sorted_ped.sire[i]
        dam  = sorted_ped.dam[i]
        s_idx = get(id_map, sire, 0)
        d_idx = get(id_map, dam, 0)

        if s_idx == 0 && d_idx == 0
            # 基础动物
            push!(I_idx, i); push!(J_idx, i); push!(V_val, 1.0)
        elseif xor(s_idx > 0, d_idx > 0)
            # 只有一个亲本已知
            parent_idx = s_idx > 0 ? s_idx : d_idx
            push!(I_idx, i); push!(J_idx, i); push!(V_val, 1.5)
            push!(I_idx, i); push!(J_idx, parent_idx); push!(V_val, -0.5)
            push!(I_idx, parent_idx); push!(J_idx, i); push!(V_val, -0.5)
            push!(I_idx, parent_idx); push!(J_idx, parent_idx); push!(V_val, 0.25)
        else
            # 双亲均已知
            push!(I_idx, i); push!(J_idx, i); push!(V_val, 2.0)
            for parent_idx in (s_idx, d_idx)
                push!(I_idx, i); push!(J_idx, parent_idx); push!(V_val, -1.0)
                push!(I_idx, parent_idx); push!(J_idx, i); push!(V_val, -1.0)
                push!(I_idx, parent_idx); push!(J_idx, parent_idx); push!(V_val, 0.5)
            end
            push!(I_idx, s_idx); push!(J_idx, d_idx); push!(V_val, 0.5)
            push!(I_idx, d_idx); push!(J_idx, s_idx); push!(V_val, 0.5)
        end
    end

    A_inv = sparse(I_idx, J_idx, V_val, n, n)

    @info "A-inverse 矩阵计算完成。"
    return A_inv, sorted_ped, id_map
end
