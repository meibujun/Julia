# RareVariantEpistasis.jl

**家畜全基因组稀有变异上位性检验的 Meta 分析**

[![Build Status](https://travis-ci.org/your-username/RareVariantEpistasis.jl.svg?branch=main)](https://travis-ci.org/your-username/RareVariantEpistasis.jl)
[![Coverage](https://coveralls.io/repos/github/your-username/RareVariantEpistasis.jl/badge.svg?branch=main)](https://coveralls.io/github/your-username/RareVariantEpistasis.jl?branch=main)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`RareVariantEpistasis.jl` 是一个为动物育种领域设计的高性能 Julia 软件包，专门用于全基因组稀有变异的上位性检验和 Meta 分析。本软件包整合了多种前沿的统计方法，旨在提供一个功能全面、计算高效、用户友好的分析平台。

## 核心功能

- **多种数据格式支持**: 支持 PLINK, VCF, CSV 等多种常见的基因组数据格式。
- **丰富的统计方法**: 实现了多种稀有变异分析方法，包括：
    - **折叠法 (Collapsing Methods)**: 如 CAST, WSS 等。
    - **贝叶斯多元回归 (Bayesian Multivariate Regression)**: 灵活的贝叶斯模型，用于估计 SNP 效应。
    - **再生核希尔伯特空间 (RKHS)**: 基于核方法的非参数模型，用于捕获复杂的遗传结构。
    - **EG-BLUP (Genomic Best Linear Unbiased Prediction)**: 用于估计基因组育种值。
- **高性能计算**: 核心算法经过优化，支持大规模数据集的并行计算。
- **可扩展性**: 模块化的设计，方便用户添加新的分析方法和功能。

## 安装

你可以通过 Julia 的包管理器来安装 `RareVariantEpistasis.jl`：

```julia
using Pkg
Pkg.add("RareVariantEpistasis")
```

## 快速上手

下面是一个使用本软件包进行基本分析的简单示例：

```julia
using RareVariantEpistasis

# 1. 加载数据
# 假设我们有一个 CSV 格式的基因型文件和表型文件
genomic_data = load_csv("path/to/genotypes.csv", sample_id_col="ID", snp_cols=2:101)
phenotype_data = PhenotypeData(CSV.read("path/to/phenotypes.csv"), "ID")

# 2. 执行 CAST 折叠分析
results = collapsing_analysis(genomic_data, phenotype_data, method="CAST")

# 3. 查看结果
println(results)
```

## 文档

详细的文档和教程，请访问 [Documentation](https://your-username.github.io/RareVariantEpistasis.jl/dev/)。

## 贡献

我们欢迎任何形式的贡献，包括 bug 报告、功能建议和代码提交。请在提交 pull request 前，先在 GitHub Issues 中进行讨论。

## 许可证

本软件包在 [MIT License](LICENSE) 下发布。
