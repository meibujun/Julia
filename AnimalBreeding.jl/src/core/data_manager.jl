"""
    DataManager

数据管理器，是所有育种数据的中心容器。它负责存储和管理谱系、基因型、
表型、多组学数据以及计算出的关系矩阵。

# 字段
- `pedigree::Union{DataFrame,Nothing}`: 谱系数据，包含animal, sire, dam列。
- `genotypes::Union{DataFrame,Nothing}`: 基因型数据矩阵。
- `phenotypes::Union{DataFrame,Nothing}`: 表型数据。
- `omics_data::Dict{String,DataFrame}`: 多组学数据字典，键为组学类型（如"transcriptome"）。
- `metadata::Dict{String,Any}`: 用于存储额外信息的元数据字典。
- `A_matrix::Union{AbstractMatrix,Nothing}`: 谱系关系矩阵 (A)。
- `A_inv_matrix::Union{AbstractMatrix,Nothing}`: 谱系关系矩阵的逆 (A⁻¹)。
- `G_matrix::Union{AbstractMatrix,Nothing}`: 基因组关系矩阵 (G)。
- `H_inv_matrix::Union{AbstractMatrix,Nothing}`: 单步关系矩阵的逆 (H⁻¹)。
- `validated::Bool`: 一个布尔标志，指示数据是否已通过验证。
- `animal_map::Dict{Any, Int}`: 一个将所有动物ID映射到整数索引的字典，用于对齐所有矩阵。
"""
mutable struct DataManager
    pedigree::Union{DataFrame,Nothing}
    genotypes::Union{DataFrame,Nothing}
    phenotypes::Union{DataFrame,Nothing}
    omics_data::Dict{String,DataFrame}
    metadata::Dict{String,Any}
    A_matrix::Union{AbstractMatrix,Nothing}
    A_inv_matrix::Union{AbstractMatrix,Nothing}
    G_matrix::Union{AbstractMatrix,Nothing}
    H_inv_matrix::Union{AbstractMatrix,Nothing}
    validated::Bool
    animal_map::Dict{Any, Int}

    """
        DataManager()

    创建一个空的DataManager实例。
    """
    function DataManager()
        return new(
            nothing,
            nothing,
            nothing,
            Dict{String,DataFrame}(),
            Dict{String,Any}(),
            nothing,
            nothing,
            nothing,
            nothing,
            false,
            Dict{Any,Int}(),
        )
    end
end

"""
    update_animal_map_from_pedigree!(dm::DataManager) -> Dict{Any,Int}

根据当前谱系数据更新 `DataManager` 内部的动物索引映射。该映射是构建
设计矩阵、关系矩阵时保持个体顺序一致的关键。

# 注意事项
- 谱系表必须已经按照 `compute_A_matrix` 或 `compute_A_inv_matrix` 使用的顺序
  进行排序，否则后续矩阵与设计矩阵的行列将无法对齐。
- 若谱系为空，则回退到使用已加载的基因型或表型构建映射。
"""
function update_animal_map_from_pedigree!(dm::DataManager)
    if !isnothing(dm.pedigree)
        animal_ids = collect(dm.pedigree.animal)
        dm.animal_map = Dict(id => idx for (idx, id) in enumerate(animal_ids))
        return dm.animal_map
    end

    # 当尚未加载谱系时，根据基因型或表型数据构造映射，确保程序仍可运行
    candidates = Vector{Any}()
    if !isnothing(dm.genotypes)
        append!(candidates, dm.genotypes[!, first(names(dm.genotypes))])
    end
    if !isnothing(dm.phenotypes)
        append!(candidates, dm.phenotypes.animal)
    end

    if isempty(candidates)
        error("无法更新animal_map：请先加载谱系、基因型或表型数据。")
    end

    unique_ids = collect(Set(candidates))
    sort!(unique_ids, by = identity)
    dm.animal_map = Dict(id => idx for (idx, id) in enumerate(unique_ids))
    return dm.animal_map
end

"""
    compute_relationship_matrix(dm::DataManager; type=:pedigree, kwargs...) -> Any

根据指定的类型计算关系矩阵，并自动更新 `DataManager` 中的缓存字段。
支持的类型包括：

- `:pedigree` / `:A`: 计算谱系关系矩阵 A。
- `:pedigree_inverse` / `:A_inv`: 计算谱系关系矩阵的逆 A⁻¹。
- `:genomic` / `:G`: 计算基因组关系矩阵 G。
- `:singlestep` / `:H_inv`: 计算单步关系矩阵的逆 H⁻¹。
- `:all`: 顺序计算以上所有矩阵并返回字典。

对于 `:genomic` 类型，可以额外传入 `method`、`impute`、`scaling` 等关键字参数；
对于 `:singlestep` 类型，可以传入 `blending_factor` 与 `ridge` 参数。

# 返回
- 如果只请求一个矩阵，直接返回该矩阵；
- 如果请求多个矩阵（例如 `type=:all` 或向量类型），返回一个以类型为键的字典。
"""
function compute_relationship_matrix(
    dm::DataManager;
    type::Union{Symbol,AbstractString,AbstractVector{<:Union{Symbol,AbstractString}}}=:pedigree,
    kwargs...
)
    function normalize_types(t)
        if t isa Symbol
            return t === :all ? [:pedigree, :pedigree_inverse, :genomic, :singlestep] : [Symbol(t)]
        elseif t isa AbstractString
            sym = Symbol(t)
            return sym === :all ? [:pedigree, :pedigree_inverse, :genomic, :singlestep] : [sym]
        else
            converted = Symbol.(t)
            if :all in converted
                filtered = filter(x -> x != :all, converted)
                converted = vcat(filtered, [:pedigree, :pedigree_inverse, :genomic, :singlestep])
            end
            return unique(converted)
        end
    end

    types_to_compute = normalize_types(type)

    alias_map = Dict(
        :pedigree => :pedigree,
        :A => :pedigree,
        :pedigree_inverse => :pedigree_inverse,
        :A_inv => :pedigree_inverse,
        :Ainverse => :pedigree_inverse,
        :genomic => :genomic,
        :G => :genomic,
        :singlestep => :singlestep,
        :H_inv => :singlestep,
        :Hinverse => :singlestep,
        :HInverse => :singlestep,
    )

    genomic_kw = (; (k => v for (k, v) in kwargs if k in (:method, :impute, :scaling))...)
    singlestep_kw = (; (k => v for (k, v) in kwargs if k in (:blending_factor, :ridge))...)

    allowed_kw = Set([:method, :impute, :scaling, :blending_factor, :ridge])
    for (k, _) in kwargs
        if k ∉ allowed_kw
            msg = string("Keyword '", k, "' is not used in compute_relationship_matrix and will be ignored.")
            @warn msg
        end
    end

    results = Dict{Symbol,Any}()
    computed = Set{Symbol}()

    for t in types_to_compute
        canonical = get(alias_map, Symbol(t), nothing)
        if canonical === nothing
            err_msg = string(
                "Unsupported relationship matrix type '",
                t,
                "'. Supported types are :pedigree, :pedigree_inverse, :genomic, :singlestep, and :all."
            )
            error(err_msg)
        end
        if canonical in computed
            continue
        end

        push!(computed, canonical)

        if canonical == :pedigree
            if isnothing(dm.pedigree)
                error("Pedigree data must be loaded before computing the A matrix.")
            end
            A, sorted_ped, id_map = compute_A_matrix(dm.pedigree)
            dm.A_matrix = A
            dm.pedigree = sorted_ped
            dm.animal_map = id_map
            results[:pedigree] = A
        elseif canonical == :pedigree_inverse
            if isnothing(dm.pedigree)
                error("Pedigree data must be loaded before computing the inverse A matrix.")
            end
            A_inv, sorted_ped, id_map = compute_A_inv_matrix(dm.pedigree)
            dm.A_inv_matrix = A_inv
            dm.pedigree = sorted_ped
            dm.animal_map = id_map
            results[:pedigree_inverse] = A_inv
        elseif canonical == :genomic
            if isnothing(dm.genotypes)
                error("Genotype data must be loaded before computing the G matrix.")
            end
            G = compute_G_matrix(dm.genotypes; genomic_kw...)
            dm.G_matrix = G
            results[:genomic] = G
        elseif canonical == :singlestep
            if isnothing(dm.A_matrix)
                if isnothing(dm.pedigree)
                    error("Pedigree data is required to compute the single-step inverse matrix.")
                end
                A, sorted_ped, id_map = compute_A_matrix(dm.pedigree)
                dm.A_matrix = A
                dm.pedigree = sorted_ped
                dm.animal_map = id_map
            end
            if isnothing(dm.A_inv_matrix)
                if isnothing(dm.pedigree)
                    error("Pedigree data is required to compute the single-step inverse matrix.")
                end
                A_inv, sorted_ped, id_map = compute_A_inv_matrix(dm.pedigree)
                dm.A_inv_matrix = A_inv
                dm.pedigree = sorted_ped
                dm.animal_map = id_map
            end
            if isnothing(dm.G_matrix)
                if isnothing(dm.genotypes)
                    error("Genotype data is required to compute the single-step inverse matrix.")
                end
                G = compute_G_matrix(dm.genotypes; genomic_kw...)
                dm.G_matrix = G
            end
            H_inv = compute_H_matrix_inv(dm; singlestep_kw...)
            dm.H_inv_matrix = H_inv
            results[:singlestep] = H_inv
        end
    end

    if length(results) == 1
        return first(values(results))
    else
        return results
    end
end
