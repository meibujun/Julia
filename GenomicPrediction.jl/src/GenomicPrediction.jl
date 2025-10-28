module GenomicPrediction

"""
    GenomicPrediction

顶层模块聚合基因组预测的全部功能模块, 包括数据处理、统计与贝叶斯模型、深度学习训练、性能评估、FAIR 元数据管理以及 AutoGS 自动建模。
"""

# -------------------------- 基础依赖导入 --------------------------
#= 采用显式 using/import 便于 Documenter 收集 API 并保持命名空间清晰。 =#
using LinearAlgebra
using Random
using Statistics
using Dates
using Logging

using CSV
using DataFrames
using Distributions
using Flux
using StatsBase
using StatsModels
using Tables
using BSON

# -------------------------- 子模块引入 --------------------------
include("DataProcessing.jl")
include("CoreAlgorithm.jl")
include("DeepLearning.jl")
include("Evaluation.jl")
include("FAIRModeling.jl")
include("AutoGS.jl")

# -------------------------- 对外导出符号 --------------------------
export GenomicDataset,
       QualityReport,
       load_genomic_table,
       load_phenotype_table,
       merge_genomic_phenotype,
       quality_control!,
       standardize_genotypes!,
       impute_missing!,
       build_grm,
       kfold_split,
       make_holdout_split,
       simulate_genomic_data,

       PredictionModel,
       GBLUPModel,
       RidgeRegressionModel,
       LassoModel,
       ElasticNetModel,
       BayesAModel,
       BayesBModel,
       fit!,
       predict,

       TrainingHistory,
       build_mlp,
       build_cnn,
       build_transformer,
       train_deep_model!,

       evaluate_metrics,
       cross_validate,
       summarize_cv,

       ModelMetadata,
       create_metadata,
       enrich_metadata!,
       save_model,
       load_model,

       AutoGSPipeline,
       run_autogs,
       default_workflow

end # module GenomicPrediction
