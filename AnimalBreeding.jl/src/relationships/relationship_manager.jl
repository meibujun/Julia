# ============================================================================
# 关系矩阵调度工具
# AnimalBreeding.jl
# ----------------------------------------------------------------------------
# 提供 `compute_relationship_matrix` 统一入口，负责根据类型调用具体的
# A、A⁻¹、G 以及单步 H⁻¹ 的计算函数，并自动将结果缓存回 DataManager。
# ============================================================================

"""
    compute_relationship_matrix(dm::DataManager; type::Symbol=:pedigree, kwargs...)

根据给定的类型计算相应的关系矩阵，并将结果写回 `DataManager` 中。

# 支持的类型
- `:pedigree` 或 `:A`: 计算谱系关系矩阵 A；可通过 `compute_inverse=true`
  同时计算并缓存 A⁻¹。
- `:pedigree_inv` 或 `:A_inv`: 直接计算谱系关系矩阵的逆。
- `:genomic` 或 `:G`: 计算基因组关系矩阵 G，额外的关键字参数会透传给
  `compute_G_matrix`。
- `:singlestep`, `:H_inv` 或 `:single_step`: 计算单步关系矩阵的逆
  (H⁻¹)，关键字参数会透传给 `compute_H_matrix_inv`。

# 返回
计算得到的矩阵对象，同时该矩阵也会存储在 `dm` 对应的字段中。
"""
function compute_relationship_matrix(dm::DataManager; type::Symbol=:pedigree, kwargs...)
    matrix_type = Symbol(lowercase(String(type)))
    kwargs_nt = (; kwargs...)

    if matrix_type in (:pedigree, :a)
        isnothing(dm.pedigree) && error("在计算A矩阵之前，请先加载谱系数据到DataManager中。")

        A, sorted_ped, id_map = compute_A_matrix(dm.pedigree)
        dm.A_matrix = A
        dm.pedigree = sorted_ped
        dm.animal_map = id_map

        compute_inverse = haskey(kwargs_nt, :compute_inverse) ? kwargs_nt[:compute_inverse] : false
        if compute_inverse
            A_inv, sorted_ped_inv, id_map_inv = compute_A_inv_matrix(dm.pedigree)
            dm.A_inv_matrix = A_inv
            dm.pedigree = sorted_ped_inv
            dm.animal_map = id_map_inv
            return dm.A_matrix
        end

        return dm.A_matrix
    elseif matrix_type in (:pedigree_inv, :a_inv)
        isnothing(dm.pedigree) && error("在计算A⁻¹之前，请先加载谱系数据到DataManager中。")

        A_inv, sorted_ped, id_map = compute_A_inv_matrix(dm.pedigree)
        dm.A_inv_matrix = A_inv
        dm.pedigree = sorted_ped
        dm.animal_map = id_map
        return dm.A_inv_matrix
    elseif matrix_type in (:genomic, :g)
        isnothing(dm.genotypes) && error("在计算G矩阵之前，请先加载基因型数据到DataManager中。")

        g_kwargs = _filter_kwargs(kwargs_nt, (:compute_inverse,))
        dm.G_matrix = isempty(keys(g_kwargs)) ? compute_G_matrix(dm.genotypes) : compute_G_matrix(dm.genotypes; g_kwargs...)
        return dm.G_matrix
    elseif matrix_type in (:singlestep, :h_inv, :single_step)
        h_kwargs = _filter_kwargs(kwargs_nt, (:compute_inverse,))
        dm.H_inv_matrix = isempty(keys(h_kwargs)) ? compute_H_matrix_inv(dm) : compute_H_matrix_inv(dm; h_kwargs...)
        return dm.H_inv_matrix
    else
        error("不支持的关系矩阵类型: $type")
    end
end

"""
    _filter_kwargs(kwargs::NamedTuple, exclude::Tuple{Vararg{Symbol}})

过滤 `kwargs` 中的特定键，返回一个 `NamedTuple` 便于再次作为关键字参数传递。
"""
function _filter_kwargs(kwargs::NamedTuple, exclude::Tuple{Vararg{Symbol}})
    return (; (k => getfield(kwargs, k) for k in keys(kwargs) if !(k in exclude))...)
end

