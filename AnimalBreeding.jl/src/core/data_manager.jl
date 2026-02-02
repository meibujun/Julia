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