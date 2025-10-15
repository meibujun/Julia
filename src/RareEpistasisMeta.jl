module RareEpistasisMeta

"""
    RareEpistasisMeta

高性能家畜全基因组稀有变异上位性检验 Meta 分析软件包的主模块。
该模块组织并导出数据加载、预处理、模型拟合、元分析与可视化等功能。
"""

using LinearAlgebra
using SparseArrays
using Random
using StatsBase
using StatsModels
using DataFrames
using CSV
using Distributions
using ProgressLogging

include("utils/Threading.jl")
include("utils/MathUtils.jl")
include("data/Formats.jl")
include("preprocessing/QualityControl.jl")
include("statistics/Collapsing.jl")
include("statistics/BayesianMVR.jl")
include("statistics/RKHS.jl")
include("statistics/EGBLUP.jl")
include("statistics/MetaAnalysis.jl")
include("io/Reporting.jl")

using .utils.MathUtils: simulate_genotype_matrix, simulate_phenotypes
using .data.Formats: load_genotypes, load_phenotypes, load_covariates
using .preprocessing.QualityControl: qc_filter_variants!, qc_filter_samples!
using .statistics.Collapsing: collapsing_test
using .statistics.BayesianMVR: bayesian_mvr!
using .statistics.RKHS: rkhs_epistasis!
using .statistics.EGBLUP: egblup!
using .statistics.MetaAnalysis: meta_analyze, summarize_results
using .io.Reporting: save_report

export load_genotypes, load_phenotypes, load_covariates,
       qc_filter_variants!, qc_filter_samples!,
       collapsing_test, bayesian_mvr!, rkhs_epistasis!, egblup!,
       meta_analyze, summarize_results, save_report,
       simulate_genotype_matrix, simulate_phenotypes

end # module
