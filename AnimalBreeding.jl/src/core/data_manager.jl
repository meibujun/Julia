# ============================================================================
# 核心模块 - 数据管理器
# AnimalBreeding.jl
# ============================================================================

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
        new(nothing, nothing, nothing, Dict{String,DataFrame}(),
            Dict{String,Any}(), nothing, nothing, nothing, nothing, false, Dict{Any,Int}())
    end
end

"""
    reset!(dm::DataManager)

将数据管理器重置为初始状态，同时保留已分配的字典对象，避免重复分配。返回被重置的`dm`，以便链式调用。
"""
function reset!(dm::DataManager)
    dm.pedigree = nothing
    dm.genotypes = nothing
    dm.phenotypes = nothing
    empty!(dm.omics_data)
    empty!(dm.metadata)
    dm.A_matrix = nothing
    dm.A_inv_matrix = nothing
    dm.G_matrix = nothing
    dm.H_inv_matrix = nothing
    dm.validated = false
    empty!(dm.animal_map)
    return dm
end

"""
    ensure_animal_map!(dm::DataManager)

确保 `animal_map` 与当前谱系顺序保持同步。如果尚未构建谱系，则保持映射为空。
"""
function ensure_animal_map!(dm::DataManager)
    if isnothing(dm.pedigree)
        empty!(dm.animal_map)
        return dm
    end

    animals = dm.pedigree.animal
    if length(dm.animal_map) == length(animals) && all(get(dm.animal_map, animal, 0) == i for (i, animal) in enumerate(animals))
        return dm
    end

    empty!(dm.animal_map)
    for (idx, animal) in enumerate(animals)
        dm.animal_map[animal] = idx
    end
    return dm
end

"""
    align_genotypes_to_pedigree!(dm::DataManager)

按照谱系顺序重新排列基因型矩阵的行，如果存在未在谱系中的个体则保持在末尾。返回一个向量，给出重排后的行索引。
"""
function align_genotypes_to_pedigree!(dm::DataManager)
    if isnothing(dm.genotypes)
        return Int[]
    end
    ensure_animal_map!(dm)

    id_col = names(dm.genotypes)[1]
    ids = dm.genotypes[!, id_col]
    target_order = [get(dm.animal_map, id, 0) for id in ids]
    paired = collect(enumerate(target_order))
    sort!(paired; by = x -> x[2] == 0 ? typemax(Int) : x[2])
    perm = [idx for (idx, _) in paired]
    dm.genotypes = dm.genotypes[perm, :]
    return perm
end

"""
    ordered_animals(dm::DataManager) -> Vector

按照 `animal_map` 中的索引返回动物ID的排序列表。如果尚未构建映射，将尝试基于当前谱系自动构建。
"""
function ordered_animals(dm::DataManager)
    ensure_animal_map!(dm)
    n = length(dm.animal_map)
    animals = Vector{Any}(undef, n)
    for (animal, idx) in dm.animal_map
        if 1 <= idx <= n
            animals[idx] = animal
        end
    end
    return animals
end
