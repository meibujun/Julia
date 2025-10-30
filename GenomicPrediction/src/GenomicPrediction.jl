# GenomicPrediction.jl - 基因组预测理论与方法综合软件包
# ==========================================================
# 本文件已更新，集成了动态模型注册系统，以增强扩展性。
# ==========================================================

module GenomicPrediction

# --- 1. 核心依赖与抽象类型 ---
using DataFrames; using Random; using Statistics; using LinearAlgebra
abstract type AbstractModel end
abstract type AbstractGenomicData end

# --- 2. 通用 API 函数存根 ---
function fit! end; function predict end; function get_genotypes end; function get_phenotypes end; function get_pedigree end; function get_covariates end

# --- 3. 引入子模块 ---
include("DataProcessing.jl")
include("CoreAlgorithm.jl")
include("KernelModels.jl")
include("DeepLearning.jl")
include("Evaluation.jl")
include("FAIRModeling.jl")
include("AutoGS.jl")
include("ModelRegistry.jl")
include("ExperimentTracking.jl") # 引入实验追踪系统
include("Pipeline.jl")

# --- 4. 从子模块导入符号 ---
using .DataProcessing: InMemoryGenomicData, PGENGenomicData, load_csv, load_pgen, calculate_grm
using .CoreAlgorithm: GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel
using .KernelModels: ssGBLUPModel
using .DeepLearning: FNNModel, CNNModel, TransformerModel, GNNModel
using .Evaluation: cross_validate, accuracy, mse
using .FAIRModeling: save_model, load_model
using .AutoGS: grid_search, bayesian_optimization
using .ModelRegistry: register_model, create_model, list_models
using .ExperimentTracking: init_db, log_experiment!, summarize_experiments
using .Pipeline: run_pipeline

# --- 5. 自动注册内置模型 ---
function __init__()
    # 在模块加载时，自动将所有内置模型注册到模型注册表
    println("正在注册内置模型...")
    register_model("gblup", params -> GBLUPModel(params[:lambda]))
    register_model("bayes_a", params -> BayesAModel(; params...))
    register_model("bayes_b", params -> BayesBModel(; params...))
    register_model("bayes_c", params -> BayesCModel(; params...))
    register_model("bayes_r", params -> BayesRModel(; params...))
    register_model("lasso", params -> LASSOModel(; params...))
    register_model("elastic_net", params -> ElasticNetModel(; params...))
    register_model("ssgblup", params -> ssGBLUPModel(params[:lambda]))
    register_model("fnn", params -> FNNModel(params[:input_dim]; Dict(k => v for (k, v) in params if k != :input_dim)...))
    register_model("cnn", params -> CNNModel(params[:input_dim]; Dict(k => v for (k, v) in params if k != :input_dim)...))
    register_model("transformer", params -> TransformerModel(params[:input_dim]; Dict(k => v for (k, v) in params if k != :input_dim)...))
    register_model("gnn", params -> GNNModel(params[:input_dim]; Dict(k => v for (k, v) in params if k != :input_dim)...))
end

# --- 6. 定义非破坏性的 `fit` 函数 ---
function fit(model::AbstractModel, data::AbstractGenomicData; kwargs...); new_model = deepcopy(model); fit!(new_model, data; kwargs...); return new_model; end

# --- 7. 导出统一的公共 API ---
export AbstractGenomicData, InMemoryGenomicData, PGENGenomicData, load_csv, load_pgen
export get_genotypes, get_phenotypes, get_pedigree, get_covariates
export calculate_grm
export AbstractModel, GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel, ssGBLUPModel, FNNModel, CNNModel, TransformerModel, GNNModel
export fit!, fit, predict
export cross_validate, accuracy, mse
export save_model, load_model
export grid_search, bayesian_optimization
export register_model, create_model, list_models
export init_db, log_experiment!, summarize_experiments
export run_pipeline

end # module GenomicPrediction
