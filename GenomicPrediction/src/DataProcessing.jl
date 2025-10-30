# DataProcessing.jl - 数据处理模块
# ==========================================================
# 本文件经过了重大的架构重构，引入了 AbstractGenomicData 抽象类型，
# 为支持内存映射文件等可扩展的数据后端奠定了基础。
# ==========================================================

module DataProcessing

# --- 1. 导入依赖 ---
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Base.Threads
using PGENFiles

# --- 2. 模块接口 ---
export AbstractGenomicData, InMemoryGenomicData, PGENGenomicData, load_csv, load_pgen, get_genotypes, get_phenotypes, get_pedigree, get_covariates
export calculate_grm, filter_markers, impute_mean

# --- 3. 核心数据结构 ---

@doc raw"""
    AbstractGenomicData
所有基因组数据容器的抽象父类型。
该抽象类型为不同的数据后端（如内存、内存映射文件）提供了一个统一的接口。
"""
abstract type AbstractGenomicData end

@doc raw"""
    InMemoryGenomicData <: AbstractGenomicData
一个将所有数据完整加载到内存中的数据容器。
"""
struct InMemoryGenomicData <: AbstractGenomicData
    genotypes::DataFrame
    phenotypes::DataFrame
    covariates::Union{DataFrame, Nothing}
    pedigree::Union{DataFrame, Nothing}
end

@doc raw"""
    PGENGenomicData <: AbstractGenomicData
一个用于处理 PGEN 格式基因型数据的数据容器。
它不会将基因型矩阵完全加载到内存中，而是通过内存映射按需访问。
"""
struct PGENGenomicData <: AbstractGenomicData
    pgen::Pgen
    phenotypes::DataFrame
    covariates::Union{DataFrame, Nothing}
    pedigree::Union{DataFrame, Nothing}
end

# --- 4. 数据访问器接口 ---

@doc raw"""
    get_genotypes(data::AbstractGenomicData) -> DataFrame
"""
get_genotypes(data::InMemoryGenomicData) = data.genotypes
get_genotypes(data::PGENGenomicData) = data.pgen

@doc raw"""
    get_phenotypes(data::AbstractGenomicData) -> DataFrame
"""
get_phenotypes(data::InMemoryGenomicData) = data.phenotypes

@doc raw"""
    get_pedigree(data::AbstractGenomicData) -> Union{DataFrame, Nothing}
"""
get_pedigree(data::InMemoryGenomicData) = data.pedigree

@doc raw"""
    get_covariates(data::AbstractGenomicData) -> Union{DataFrame, Nothing}
"""
get_covariates(data::InMemoryGenomicData) = data.covariates


# --- 5. 功能实现 ---

@doc raw"""
    load_csv(...) -> InMemoryGenomicData
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

    return InMemoryGenomicData(geno_aligned, pheno_aligned, cov_aligned, ped_df)
end

@doc raw"""
    load_pgen(pgen_path::String, pheno_path::String; cov_path=nothing, ped_path=nothing) -> PGENGenomicData
加载 PGEN 格式的基因型数据以及其他相关的 CSV 数据。
"""
function load_pgen(pgen_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing, ped_path::Union{String, Nothing}=nothing)
    # PGEN 文件需要 .pvar 和 .psam 文件在同一目录下
    pgen = Pgen(pgen_path)

    # 从 .psam 文件获取样本 ID
    psam_df = CSV.read(replace(pgen_path, ".pgen" => ".psam"), DataFrame)
    sample_ids = psam_df[!, 1]

    pheno_df = CSV.read(pheno_path, DataFrame)
    rename!(pheno_df, names(pheno_df)[1] => :ID)

    # (此处省略了数据对齐的逻辑，简化实现)

    cov_df = !isnothing(cov_path) ? CSV.read(cov_path, DataFrame) : nothing
    ped_df = !isnothing(ped_path) ? CSV.read(ped_path, DataFrame) : nothing
    if !isnothing(cov_df); rename!(cov_df, names(cov_df)[1] => :ID); end
    if !isnothing(ped_df); rename!(ped_df, names(ped_df)[1:3] .=> [:ID, :Sire, :Dam]); end

    return PGENGenomicData(pgen, pheno_df, cov_df, ped_df)
end

@doc raw"""
    calculate_grm(data::AbstractGenomicData; scale=true) -> Matrix{Float64}
"""
function calculate_grm(data::AbstractGenomicData; scale=true)
    # 这是一个 dispatch，将根据数据类型调用不同的实现
    _calculate_grm(get_genotypes(data), scale)
end

# 内存版本的实现
function _calculate_grm(geno_df::DataFrame, scale::Bool)
    G = Matrix(geno_df[!, 2:end])
    n, m = size(G)
    p = mean(G, dims=1) ./ 2
    M = G .- (2 .* p)
    GRM = M * M'

    denominator = 2 * sum(p .* (1 .- p))
    if denominator == 0; error("基因型数据没有变异，无法计算 GRM。"); end

    return GRM ./ denominator
end

# PGEN 版本的实现 (内存高效)
function _calculate_grm(pgen::Pgen, scale::Bool; chunk_size=1000)
    n_samples, n_variants = n_samples(pgen), n_variants(pgen)
    GRM = zeros(Float64, n_samples, n_samples)

    # 计算等位基因频率 (需要一次完整的遍历)
    p = zeros(n_variants)
    for i in 1:n_variants
        p[i] = mean(convert(Vector{Float32}, @view(pgen[:, i]))) / 2
    end

    # 分块计算 M*M'
    for i in 1:chunk_size:n_variants
        last = min(i + chunk_size - 1, n_variants)

        # 读取一个数据块
        G_chunk = convert(Matrix{Float32}, @view(pgen[:, i:last]))
        p_chunk = @view p[i:last]

        # 中心化
        M_chunk = G_chunk .- (2 .* p_chunk')

        # 累加到 GRM
        GRM .+= M_chunk * M_chunk'
    end

    denominator = 2 * sum(p .* (1 .- p))
    if denominator == 0; error("基因型数据没有变异，无法计算 GRM。"); end

    return GRM ./ denominator
end

@doc raw"""
    filter_markers(data::AbstractGenomicData; maf_threshold=0.05) -> InMemoryGenomicData
"""
function filter_markers(data::AbstractGenomicData; maf_threshold=0.05)
    geno_df = get_genotypes(data)
    G = Matrix(geno_df[!, 2:end])
    freqs = mean(G, dims=1) ./ 2
    maf = min.(freqs, 1 .- freqs)
    keep_indices = findall(m -> m >= maf_threshold, vec(maf))

    new_geno_df = geno_df[!, [1; keep_indices .+ 1]]

    return InMemoryGenomicData(new_geno_df, get_phenotypes(data), get_covariates(data), get_pedigree(data))
end

@doc raw"""
    impute_mean(data::AbstractGenomicData) -> InMemoryGenomicData
"""
function impute_mean(data::AbstractGenomicData)
    geno_df = get_genotypes(data)
    G = copy(geno_df)
    for col in names(G)[2:end]
        if any(ismissing, G[!, col])
            mean_val = round(Int, mean(skipmissing(G[!, col])))
            G[!, col] = coalesce.(G[!, col], mean_val)
        end
    end
    return InMemoryGenomicData(G, get_phenotypes(data), get_covariates(data), get_pedigree(data))
end

end # module DataProcessing
