# src/types.jl

using DataFrames

"""
SNPInfo 结构体
用于存储单个 SNP 的详细信息。

字段:
- `chr::String`: 染色体编号
- `id::String`: SNP 标识符
- `pos::Int`: 物理位置
- `ref::String`: 参考等位基因
- `alt::String`: 备选等位基因
"""
struct SNPInfo
    chr::String
    id::String
    pos::Int
    ref::String
    alt::String
end

"""
GenomicData 结构体
用于存储基因组数据。

字段:
- `genotypes::Matrix{Int8}`: 基因型矩阵，个体 x SNP
- `snp_info::Vector{SNPInfo}`: SNP 信息向量
- `sample_ids::Vector{String}`: 样本 ID 向量
"""
struct GenomicData
    genotypes::Matrix{Int8}
    snp_info::Vector{SNPInfo}
    sample_ids::Vector{String}
end

"""
PhenotypeData 结构体
用于存储表型数据。

字段:
- `phenotypes::DataFrame`: 表型数据，包含样本 ID 和表型值
- `sample_ids::Vector{String}`: 样本 ID 向量
"""
struct PhenotypeData
    phenotypes::DataFrame
    sample_ids::Vector{String}
end

"""
PopulationStructure 结构体
用于存储群体结构信息。

字段:
- `kinship_matrix::Matrix{Float64}`: 亲缘关系矩阵
- `principal_components::Matrix{Float64}`: 主成分
- `sample_ids::Vector{String}`: 样本 ID 向量
"""
struct PopulationStructure
    kinship_matrix::Matrix{Float64}
    principal_components::Matrix{Float64}
    sample_ids::Vector{String}
end
