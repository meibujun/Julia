# DataProcessing.jl - 数据处理模块
# ==========================================================
# 负责数据的加载、预处理、质量控制和基因组关系矩阵（GRM）的计算。
#
# 本文件经过修正，在 GenomicData 结构中增加了 pedigree 字段，
# 并更新了 load_csv 函数以支持加载系谱数据，从而修复 ssGBLUP 的架构问题。
# ==========================================================

module DataProcessing

# --- 1. 导入依赖 ---
using DataFrames
using CSV
using Statistics
using LinearAlgebra

# --- 2. 模块接口 ---
export GenomicData, load_csv, calculate_grm, filter_markers, impute_mean

# --- 3. 核心数据结构 ---
@doc raw"""
    GenomicData(genotypes::DataFrame, phenotypes::DataFrame, covariates::Union{DataFrame, Nothing}=nothing, pedigree::Union{DataFrame, Nothing}=nothing)

一个封装基因组预测所需全部数据的核心数据结构。

# Fields
- `genotypes::DataFrame`: 基因型数据。第一列应为个体ID，其余列为标记。
- `phenotypes::DataFrame`: 表型数据。第一列应为个体ID，其余列为表型性状。
- `covariates::Union{DataFrame, Nothing}`: (可选) 协变量数据。
- `pedigree::Union{DataFrame, Nothing}`: (可选) 系谱数据。应包含 ID, Sire, Dam 列。
"""
struct GenomicData
    genotypes::DataFrame
    phenotypes::DataFrame
    covariates::Union{DataFrame, Nothing}
    pedigree::Union{DataFrame, Nothing}
end

# --- 4. 功能实现 ---

@doc raw"""
    load_csv(geno_path::String, pheno_path::String; cov_path=nothing, ped_path=nothing, header_geno=true, header_pheno=true) -> GenomicData

从 CSV 文件中加载基因型、表型、协变量和（可选的）系谱数据。

# Arguments
- `geno_path`: 基因型 CSV 文件的路径。
- `pheno_path`: 表型 CSV 文件的路径。
- `cov_path`: (可选) 协变量 CSV 文件的路径。
- `ped_path`: (可选) 系谱 CSV 文件的路径。
- `header_geno`: 基因型文件是否有表头。
- `header_pheno`: 表型文件是否有表头。

# Returns
- 一个 `GenomicData` 对象。
"""
function load_csv(geno_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing, ped_path::Union{String, Nothing}=nothing, header_geno=true, header_pheno=true)
    # --- 加载数据 ---
    geno_df = CSV.read(geno_path, DataFrame, header=header_geno)
    pheno_df = CSV.read(pheno_path, DataFrame, header=header_pheno)

    cov_df = nothing
    if !isnothing(cov_path)
        cov_df = CSV.read(cov_path, DataFrame)
    end

    ped_df = nothing
    if !isnothing(ped_path)
        ped_df = CSV.read(ped_path, DataFrame)
    end

    # --- 标准化列名 ---
    rename!(geno_df, names(geno_df)[1] => :ID)
    rename!(pheno_df, names(pheno_df)[1] => :ID)
    if !isnothing(cov_df); rename!(cov_df, names(cov_df)[1] => :ID); end
    if !isnothing(ped_df)
        rename!(ped_df, names(ped_df)[1:3] .=> [:ID, :Sire, :Dam])
    end

    # --- 对齐基因型和表型数据 ---
    # 大多数模型只使用同时具有基因型和表型的个体
    common_ids = innerjoin(geno_df[!, [:ID]], pheno_df[!, [:ID]], on=:ID).ID
    geno_aligned = filter(:ID => id -> id in common_ids, geno_df)
    pheno_aligned = filter(:ID => id -> id in common_ids, pheno_df)

    cov_aligned = nothing
    if !isnothing(cov_df)
        cov_aligned = filter(:ID => id -> id in common_ids, cov_df)
    end

    println("数据加载成功，创建 GenomicData 对象...")
    # 系谱数据 (ped_df) 保持原样，不进行对齐，因为 ssGBLUP 需要所有个体的信息
    return GenomicData(geno_aligned, pheno_aligned, cov_aligned, ped_df)
end

@doc raw"""
    calculate_grm(G::Matrix{<:Real}; scale=true) -> Matrix{Float64}

根据给定的基因型矩阵 `G` 计算基因组关系矩阵 (GRM)。使用 VanRaden (2008) 的方法一。

# Arguments
- `G::Matrix`: 基因型矩阵，个体为行，标记为列。数值应为 0, 1, 2。
- `scale::Bool`: 是否对 GRM 进行标准化。

# Returns
- `Matrix{Float64}`: 基因组关系矩阵。
"""
function calculate_grm(G::Matrix{<:Real}; scale=true)
    n, m = size(G)

    # 1. 计算每个标记的等位基因频率 p
    # G 的编码是 0, 1, 2，代表次等位基因的数量
    p = mean(G, dims=1) ./ 2  # 频率 p 是 (sum of allele counts) / (2 * n), 这里简化

    # 2. 创建中心化矩阵 M
    P = 2 .* p
    M = G .- P

    # 3. 计算 GRM
    # 分母是遗传方差的期望
    denominator = 2 * sum(p .* (1 .- p))
    if denominator == 0; error("基因型数据没有变异，无法计算 GRM。"); end

    GRM = (M * M') / denominator

    return GRM
end

@doc raw"""
    filter_markers(geno_df::DataFrame; maf_threshold=0.05) -> DataFrame
"""
function filter_markers(geno_df::DataFrame; maf_threshold=0.05)
    G = Matrix(geno_df[!, 2:end])
    freqs = mean(G, dims=1) ./ 2
    maf = min.(freqs, 1 .- freqs)

    keep_indices = findall(m -> m >= maf_threshold, vec(maf))

    return geno_df[!, [1; keep_indices .+ 1]]
end

@doc raw"""
    impute_mean(geno_df::DataFrame) -> DataFrame
"""
function impute_mean(geno_df::DataFrame)
    G = copy(geno_df)
    for col in names(G)[2:end]
        if any(ismissing, G[!, col])
            mean_val = round(Int, mean(skipmissing(G[!, col])))
            G[!, col] = coalesce.(G[!, col], mean_val)
        end
    end
    return G
end

end # module DataProcessing
