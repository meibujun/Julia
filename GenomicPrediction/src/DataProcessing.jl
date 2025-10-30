# DataProcessing.jl - 数据处理模块
# ==========================================================
# 负责数据的加载、预处理、质量控制和基因组关系矩阵（GRM）的计算。
#
# 本文件经过优化，`calculate_grm` 函数现在使用多线程来加速计算。
# ==========================================================

module DataProcessing

# --- 1. 导入依赖 ---
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Base.Threads

# --- 2. 模块接口 ---
export GenomicData, load_csv, calculate_grm, filter_markers, impute_mean

# --- 3. 核心数据结构 ---
@doc raw"""
    GenomicData(genotypes::DataFrame, phenotypes::DataFrame, covariates::Union{DataFrame, Nothing}=nothing, pedigree::Union{DataFrame, Nothing}=nothing)
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
"""
function load_csv(geno_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing, ped_path::Union{String, Nothing}=nothing, header_geno=true, header_pheno=true)
    geno_df = CSV.read(geno_path, DataFrame, header=header_geno)
    pheno_df = CSV.read(pheno_path, DataFrame, header=header_pheno)
    cov_df = !isnothing(cov_path) ? CSV.read(cov_path, DataFrame) : nothing
    ped_df = !isnothing(ped_path) ? CSV.read(ped_path, DataFrame) : nothing

    rename!(geno_df, names(geno_df)[1] => :ID)
    rename!(pheno_df, names(pheno_df)[1] => :ID)
    if !isnothing(cov_df); rename!(cov_df, names(cov_df)[1] => :ID); end
    if !isnothing(ped_df); rename!(ped_df, names(ped_df)[1:3] .=> [:ID, :Sire, :Dam]); end

    common_ids = innerjoin(geno_df[!, [:ID]], pheno_df[!, [:ID]], on=:ID).ID
    geno_aligned = filter(:ID => id -> id in common_ids, geno_df)
    pheno_aligned = filter(:ID => id -> id in common_ids, pheno_df)
    cov_aligned = !isnothing(cov_df) ? filter(:ID => id -> id in common_ids, cov_df) : nothing

    return GenomicData(geno_aligned, pheno_aligned, cov_aligned, ped_df)
end

@doc raw"""
    calculate_grm(G::Matrix{<:Real}; scale=true) -> Matrix{Float64}

根据给定的基因型矩阵 `G` 计算基因组关系矩阵 (GRM)。使用 VanRaden (2008) 的方法一。
此实现利用多线程并行计算以提高大型数据集的处理速度。

# Arguments
- `G::Matrix`: 基因型矩阵，个体为行，标记为列。数值应为 0, 1, 2。
- `scale::Bool`: 是否对 GRM 进行标准化。

# Returns
- `Matrix{Float64}`: 基因组关系矩阵。
"""
function calculate_grm(G::Matrix{<:Real}; scale=true)
    n, m = size(G)

    # 1. 计算等位基因频率 p (此步很快，无需并行)
    p = mean(G, dims=1) ./ 2

    # 2. 创建中心化矩阵 M (此步很快，无需并行)
    P = 2 .* p
    M = G .- P

    # 3. 并行计算 GRM = M * M'
    # 这是计算密集型步骤，我们在此处使用多线程。
    GRM = zeros(Float64, n, n)

    # 使用 @threads 宏将外层循环（计算 GRM 的每一行）分配到多个线程
    # 由于 GRM 是对称的，我们只计算上三角部分以避免重复计算和线程间的竞争条件。
    @threads for i in 1:n
        for j in i:n
            # 使用 `dot` 和 `view` 高效计算点积 M[i,:]' * M[j,:]
            GRM[i, j] = dot(view(M, i, :), view(M, j, :))
        end
    end

    # 填充下三角部分（此步很快，单线程执行）
    for i in 1:n
        for j in (i + 1):n
            GRM[j, i] = GRM[i, j]
        end
    end

    # 4. 标准化 GRM
    denominator = 2 * sum(p .* (1 .- p))
    if denominator == 0
        error("基因型数据没有变异，无法计算 GRM。")
    end

    GRM ./= denominator

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
