module RareVariantEpistasis

# 模块功能概述
# 本模块旨在提供一个全面的工具集，用于家畜全基因组稀有变异上位性检验的 Meta 分析。
# 它整合了多种先进的统计方法，并支持多种常见的基因组数据格式。

# 核心模块导入
using DataFrames, CSV, SnpArrays, VariantCallFormat
using StatsBase, Distributions, LinearAlgebra, Random
using KernelFunctions

# 包含核心文件
include("types.jl")
include("io.jl")
include("collapsing.jl")
include("bayesian_regression.jl")
include("rkhs.jl")
include("eg_blup.jl")

# 导出函数和类型，供用户使用
export greet, SNPInfo, GenomicData, PhenotypeData, PopulationStructure
export load_plink, load_vcf, load_csv
export collapsing_analysis
export bayesian_regression_analysis
export rkhs_analysis
export eg_blup_analysis

# greet 函数，用于测试
greet() = print("Hello World!")

end # module
