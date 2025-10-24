# DataProcessing.jl: 数据处理模块
# ---------------------------------
#
# 本模块提供全面的数据处理功能，是整个基因组预测流程的起点。
# 其核心职责包括：
# - **数据导入/导出**: 支持多种常见格式（如 VCF, PLINK, CSV）的基因型和表型数据。
# - **质量控制 (QC)**: 实现对基因型和表型数据的质量控制，如标记筛选、样本筛选等。
# - **缺失值处理**: 提供多种基因型数据填充（Imputation）方法。
# - **矩阵构建**: 计算基因组关系矩阵 (GRM) 等核心数据结构。
# - **数据模拟**: 内置数据模拟器，用于生成具有特定遗传结构的数据集，方便算法测试和评估。

module DataProcessing

using CSV, DataFrames, Statistics

export GenomicData, load_csv

# --- 结构体定义 ---

"""
    GenomicData

一个用于存储基因组预测所需数据的复合类型。

该结构体旨在将不同来源的数据（基因型、表型、协变量）整合到一个标准化的容器中，
方便后续分析模块统一调用。

# 字段
- `genotypes::DataFrame`: 基因型数据。通常，行代表个体，列代表分子标记 (SNP)。
- `phenotypes::DataFrame`: 表型数据。通常，行代表个体，列代表不同的性状。
- `covariates::Union{DataFrame, Nothing}`: 协变量数据（可选）。例如，环境因素、固定效应等。
"""
struct GenomicData
    genotypes::DataFrame
    phenotypes::DataFrame
    covariates::Union{DataFrame, Nothing}
end

# 默认构造函数，当没有协变量时使用
GenomicData(genotypes::DataFrame, phenotypes::DataFrame) = GenomicData(genotypes, phenotypes, nothing)


# --- 函数定义 ---

"""
    load_csv(geno_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing) -> GenomicData

从 CSV 文件中加载基因型、表型和（可选的）协变量数据，并返回一个 `GenomicData` 对象。

该函数是数据导入的主要入口点，简化了从标准 CSV 格式创建 `GenomicData` 实例的过程。

# 参数
- `geno_path::String`: 基因型数据 CSV 文件的路径。
- `pheno_path::String`: 表型数据 CSV 文件的路径。
- `cov_path::Union{String, Nothing}`: (可选) 协变量数据 CSV 文件的路径。

# 返回
- `GenomicData`: 一个包含所有已加载数据的 `GenomicData` 结构体实例。

# 示例
```julia
# 加载基因型和表型数据
data = load_csv("data/genotypes.csv", "data/phenotypes.csv")

# 加载基因型、表型和协变量数据
data_with_cov = load_csv("data/genotypes.csv", "data/phenotypes.csv", cov_path="data/covariates.csv")
```
"""
function load_csv(geno_path::String, pheno_path::String; cov_path::Union{String, Nothing}=nothing)
    # --- 1. 加载基因型数据 ---
    # 使用 CSV.read 读取文件，并将其转换为 DataFrame
    # header=true 表示文件第一行为列名
    # delim=',' 指定逗号为分隔符
    println("正在从 $geno_path 加载基因型数据...")
    genotypes = CSV.read(geno_path, DataFrame)
    println("基因型数据加载完成：$(size(genotypes, 1)) 个体，$(size(genotypes, 2)) 个标记。")

    # --- 2. 加载表型数据 ---
    println("正在从 $pheno_path 加载表型数据...")
    phenotypes = CSV.read(pheno_path, DataFrame)
    println("表型数据加载完成：$(size(phenotypes, 1)) 个体，$(size(phenotypes, 2)) 个性状。")

    # --- 3. 检查数据一致性 ---
    # 确保基因型和表型数据的个体数量（行数）一致
    if size(genotypes, 1) != size(phenotypes, 1)
        error("基因型数据和表型数据的个体数量（行数）不匹配！")
    end

    # --- 4. 加载协变量数据 (如果提供了路径) ---
    covariates = nothing # 默认为 nothing
    if cov_path !== nothing
        println("正在从 $cov_path 加载协变量数据...")
        covariates = CSV.read(cov_path, DataFrame)
        println("协变量数据加载完成：$(size(covariates, 1)) 个体，$(size(covariates, 2)) 个协变量。")

        # 检查协变量数据的个体数量一致性
        if size(genotypes, 1) != size(covariates, 1)
            error("协变量数据的个体数量（行数）与基因型/表型数据不匹配！")
        end
    end

    # --- 5. 创建并返回 GenomicData 对象 ---
    println("数据加载成功，正在创建 GenomicData 对象...")
    return GenomicData(genotypes, phenotypes, covariates)
end




end # module DataProcessing
