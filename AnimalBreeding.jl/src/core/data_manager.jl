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
