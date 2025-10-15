# `RareVariantEpistasis.jl` 教程

欢迎使用 `RareVariantEpistasis.jl`！本教程将引导你完成一个完整的数据分析流程，从数据加载到执行多种统计分析。

## 1. 准备工作

在开始之前，请确保你已经安装了 `RareVariantEpistasis.jl`。如果尚未安装，请在 Julia REPL 中运行：

```julia
using Pkg
Pkg.add("RareVariantEpistasis")
```

## 2. 数据加载

本软件包支持多种数据格式。下面我们以加载 PLINK 文件为例。

```julia
using RareVariantEpistasis

# PLINK 文件路径
bed_path = "data/example.bed"
bim_path = "data/example.bim"
fam_path = "data/example.fam"

# 加载基因组数据
genomic_data = load_plink(bed_path, bim_path, fam_path)
```

## 3. 表型数据

表型数据通常存储在 CSV 或 DataFrame 中。

```julia
using CSV, DataFrames

# 创建一个虚拟的表型 DataFrame
phenotypes_df = DataFrame(
    sample_id = genomic_data.sample_ids,
    trait1 = randn(length(genomic_data.sample_ids)),
    trait2 = rand(length(genomic_data.sample_ids))
)

phenotype_data = PhenotypeData(phenotypes_df, "sample_id")
```

## 4. 执行分析

### 4.1 折叠法 (Collapsing Analysis)

```julia
# 执行 CAST 分析
cast_results = collapsing_analysis(genomic_data, phenotype_data, method="CAST")
println("CAST Results:")
println(cast_results)

# 执行 WSS 分析
wss_results = collapsing_analysis(genomic_data, phenotype_data, method="WSS")
println("\nWSS Results:")
println(wss_results)
```

### 4.2 贝叶斯回归 (Bayesian Regression)

```julia
bayesian_results = bayesian_regression_analysis(genomic_data, phenotype_data)
println("\nBayesian Regression Results:")
println(bayesian_results)
```

### 4.3 RKHS

```julia
using KernelFunctions

# 使用高斯核
kernel = GaussianKernel()
rkhs_results = rkhs_analysis(genomic_data, phenotype_data, kernel)
println("\nRKHS Results:")
println(rkhs_results)
```

### 4.4 EG-BLUP

```julia
egblup_results = eg_blup_analysis(genomic_data, phenotype_data)
println("\nEG-BLUP Results:")
println(egblup_results)
```

## 5. 结果解释

每种分析方法都会返回一个 `DataFrame`，其中包含了相应的结果，如 p-value、SNP 效应、方差组分等。你可以使用 Julia 的数据处理工具（如 `DataFrames.jl`）对这些结果进行进一步的分析和可视化。

---

本教程提供了一个基本的分析流程。更多高级功能和选项，请参阅 API 文档。
