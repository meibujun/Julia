# DataProcessing.jl: 数据处理模块
# ---------------------------------
# ... (header comments) ...

module DataProcessing

using CSV, DataFrames, Statistics, LinearAlgebra

export GenomicData, load_csv, calculate_grm, filter_markers, impute_mean

# --- 结构体定义 ---

@doc raw"""
    GenomicData

一个用于存储基因组预测所需数据的复合类型。

该结构体旨在将不同来源的数据（基因型、表型、协变量、系谱）整合到一个标准化的容器中，
方便后续分析模块统一调用。

# 字段
- `genotypes::DataFrame`: 基因型数据。通常，行代表个体，列代表分子标记 (SNP)。
- `phenotypes::DataFrame`: 表型数据。通常，行代表个体，列代表不同的性状。
- `covariates::Union{DataFrame, Nothing}`: 协变量数据（可选）。例如，环境因素、固定效应等。
- `pedigree::Union{DataFrame, Nothing}`: 系谱数据（可选），用于构建加性亲缘关系矩阵 A。
"""
struct GenomicData
    genotypes::DataFrame
    phenotypes::DataFrame
    covariates::Union{DataFrame, Nothing}
    pedigree::Union{DataFrame, Nothing}
end

GenomicData(genotypes::DataFrame, phenotypes::DataFrame) = GenomicData(genotypes, phenotypes, nothing, nothing)
GenomicData(genotypes::DataFrame, phenotypes::DataFrame, covariates::DataFrame) = GenomicData(genotypes, phenotypes, covariates, nothing)


# --- 函数定义 ---

@doc raw"""
    load_csv(geno_path, pheno_path; cov_path=nothing, ped_path=nothing) -> GenomicData

从 CSV 文件中加载基因型、表型和（可选的）协变量、系谱数据，并返回一个 `GenomicData` 对象。

该函数是数据导入的主要入口点，简化了从标准 CSV 格式创建 `GenomicData` 实例的过程。

# 参数
- `geno_path::String`: 基因型数据 CSV 文件的路径。
- `pheno_path::String`: 表型数据 CSV 文件的路径。
- `cov_path::Union{String, Nothing}`: (可选) 协变量数据 CSV 文件的路径。
- `ped_path::Union{String, Nothing}`: (可选) 系谱数据 CSV 文件的路径。

# 返回
- `GenomicData`: 一个包含所有已加载数据的 `GenomicData` 结构体实例。

# 示例
```julia
# 加载基因型和表型数据
data = load_csv("data/genotypes.csv", "data/phenotypes.csv")

# 加载所有类型的数据
data_full = load_csv("data/genotypes.csv", "data/phenotypes.csv", cov_path="data/covariates.csv", ped_path="data/pedigree.csv")
```
"""
function load_csv(geno_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing, ped_path::Union{String, Nothing}=nothing)
    println("正在从 $geno_path 加载基因型数据...")
    genotypes = CSV.read(geno_path, DataFrame)
    println("基因型数据加载完成：$(size(genotypes, 1)) 个体，$(size(genotypes, 2)) 个标记。")

    println("正在从 $pheno_path 加载表型数据...")
    phenotypes = CSV.read(pheno_path, DataFrame)
    println("表型数据加载完成：$(size(phenotypes, 1)) 个体，$(size(phenotypes, 2)) 个性状。")

    if size(genotypes, 1) != size(phenotypes, 1)
        error("基因型数据和表型数据的个体数量（行数）不匹配！")
    end

    covariates = nothing
    if cov_path !== nothing
        println("正在从 $cov_path 加载协变量数据...")
        covariates = CSV.read(cov_path, DataFrame)
        println("协变量数据加载完成：$(size(covariates, 1)) 个体，$(size(covariates, 2)) 个协变量。")
        if size(genotypes, 1) != size(covariates, 1)
            error("协变量数据的个体数量（行数）与基因型/表型数据不匹配！")
        end
    end

    pedigree = nothing
    if ped_path !== nothing
        println("正在从 $ped_path 加载系谱数据...")
        pedigree = CSV.read(ped_path, DataFrame)
        println("系谱数据加载完成：$(size(pedigree, 1)) 个体。")
    end

    println("数据加载成功，正在创建 GenomicData 对象...")
    return GenomicData(genotypes, phenotypes, covariates, pedigree)
end

@doc raw"""
    calculate_grm(G::Matrix{<:Real}) -> Matrix{Float64}

根据给定的基因型矩阵计算基因组关系矩阵 (GRM)。

该函数实现了 VanRaden (2008) 的方法一。

# 参数
- `G::Matrix{<:Real}`: 基因型矩阵，个体为行，标记为列。编码通常为 0, 1, 2。

# 返回
- `Matrix{Float64}`: 计算得到的 GRM。
"""
function calculate_grm(G::Matrix{<:Real})
    n, p = size(G)

    # 1. 计算等位基因频率
    freqs = vec(mean(G, dims=1) ./ 2)

    # 2. 创建中心化的标记矩阵 M
    P = 2 .* freqs'
    M = G .- P

    # 3. 计算 GRM 分母
    denom = 2 * sum(freqs .* (1 .- freqs))
    if denom == 0
        @warn "等位基因频率方差之和为零，GRM 可能无意义。"
        return zeros(n, n)
    end

    # 4. 多线程计算 GRM (MM')
    GRM = zeros(Float64, n, n)
    M_t = M' # 预先转置以优化内存访问模式

    Threads.@threads for i in 1:n
        # 每个线程负责计算 GRM 的一部分行
        for j in i:n
            # 利用对称性，只计算上三角部分
            dot_product = dot(M[i, :], M[j, :])
            GRM[i, j] = dot_product
        end
    end

    # 填充下三角部分并除以分母
    for i in 1:n
        for j in (i+1):n
            GRM[j, i] = GRM[i, j]
        end
    end

    GRM ./= denom

    return GRM
end


@doc raw"""
    filter_markers(data::GenomicData; maf_threshold=0.05, call_rate_threshold=0.95) -> GenomicData

根据次要等位基因频率 (MAF) 和标记调用率过滤基因型数据。

# 参数
- `data::GenomicData`: 原始 `GenomicData` 对象。
- `maf_threshold::Float64`: MAF 的最小阈值。低于此值的标记将被移除。
- `call_rate_threshold::Float64`: 标记调用率的最小阈值。低于此值的标记将被移除。

# 返回
- `GenomicData`: 包含过滤后基因型数据的新 `GenomicData` 对象。
"""
function filter_markers(data::GenomicData; maf_threshold=0.05, call_rate_threshold=0.95)
    geno_df = copy(data.genotypes)
    G = Matrix(geno_df[!, 2:end])
    n, p = size(G)

    markers_to_keep = trues(p)

    for j in 1:p
        marker_data = G[:, j]

        # Call Rate
        non_missing = count(!ismissing, marker_data)
        call_rate = non_missing / n
        if call_rate < call_rate_threshold
            markers_to_keep[j] = false
            continue
        end

        # MAF
        # Skip MAF calculation if all values are missing (handled by call rate)
        if non_missing == 0; continue; end

        clean_marker_data = collect(skipmissing(marker_data))
        freq = mean(clean_marker_data) / 2
        maf = min(freq, 1 - freq)

        if maf < maf_threshold
            markers_to_keep[j] = false
        end
    end

    println("QC: 移除了 $(p - sum(markers_to_keep)) / $p 个标记。")

    # +1 to account for the ID column
    new_geno_df = geno_df[:, [true; markers_to_keep]]

    return GenomicData(new_geno_df, data.phenotypes, data.covariates, data.pedigree)
end


@doc raw"""
    impute_mean(data::GenomicData) -> GenomicData

使用每个标记的平均值（四舍五入到最接近的整数基因型）来填充缺失的基因型数据。

# 参数
- `data::GenomicData`: 包含缺失值的 `GenomicData` 对象。

# 返回
- `GenomicData`: 包含填充后基因型数据的新 `GenomicData` 对象。
"""
function impute_mean(data::GenomicData)
    geno_df = copy(data.genotypes)
    G = geno_df[!, 2:end] # Exclude ID column

    for j in 1:ncol(G)
        marker_col = G[!, j]
        if any(ismissing, marker_col)
            mean_val = mean(skipmissing(marker_col))
            imputed_val = round(Int, mean_val)
            marker_col[ismissing.(marker_col)] .= imputed_val
        end
    end

    println("使用平均值进行了缺失值填充。")
    return GenomicData(geno_df, data.phenotypes, data.covariates, data.pedigree)
end


end # module DataProcessing
