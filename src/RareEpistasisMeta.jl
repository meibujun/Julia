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
include("statistics/EpistasisModels.jl")
include("statistics/RKHS.jl")
include("statistics/EGBLUP.jl")
include("statistics/MetaAnalysis.jl")
include("omics/Integration.jl")
include("applications/BreedingPrograms.jl")
include("applications/LLMAssist.jl")
include("io/Reporting.jl")
include("visualization/Visualization.jl")

using .MathUtils: simulate_genotype_matrix, simulate_phenotypes
using .Formats: load_genotypes, load_phenotypes, load_covariates
using .QualityControl: qc_filter_variants!, qc_filter_samples!
using .Collapsing: collapsing_test, list_rvat_methods
using .BayesianMVR: bayesian_mvr!, bayesian_blasso!, bayesian_bayesb!
using .EpistasisModels: epistasis_scan, list_epistasis_methods
using .RKHS: rkhs_epistasis!
using .EGBLUP: egblup!
using .MetaAnalysis: meta_analyze, summarize_results, list_meta_models
using .Integration: simulate_multiomics, integrate_multiomics!, omics_kernel
using .BreedingPrograms: simulate_breeding_pipeline, optimize_breeding_scheme
using .LLMAssist: configure_llm!, llm_explain_results, llm_rank_candidate_genes, llm_optimize_scheme
using .Reporting: save_report
using .Visualization: manhattan_plot, qq_plot, forest_plot, network_plot, save_plot

export load_genotypes, load_phenotypes, load_covariates,
       qc_filter_variants!, qc_filter_samples!,
       collapsing_test, list_rvat_methods,
       bayesian_mvr!, bayesian_blasso!, bayesian_bayesb!,
       rkhs_epistasis!, egblup!,
       epistasis_scan, list_epistasis_methods,
       meta_analyze, summarize_results, list_meta_models,
       simulate_multiomics, integrate_multiomics!, omics_kernel,
       simulate_breeding_pipeline, optimize_breeding_scheme,
       configure_llm!, llm_explain_results, llm_rank_candidate_genes, llm_optimize_scheme,
       manhattan_plot, qq_plot, forest_plot, network_plot, save_plot,
       save_report,
       simulate_genotype_matrix, simulate_phenotypes,
       sample

@doc """
    sample(args...; kwargs...)

封装并重导出 `StatsBase.sample`，用于在分析流程中执行带或不带权重的高性能抽样操作，
便于在 HPC 场景下快速构建交叉验证折叠、Bootstrap 样本等随机子集。
""" sample

end # module
