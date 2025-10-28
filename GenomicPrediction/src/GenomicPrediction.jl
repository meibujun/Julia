# GenomicPrediction.jl - 基因组预测理论与方法综合软件包
# ==========================================================
# 这是软件包的主模块，负责整合所有子模块并提供统一的公共 API。
# 职责:
# 1. 定义核心抽象类型 (AbstractModel, GenomicData)。
# 2. 定义通用的 API 函数存根 (fit!, predict)。
# 3. 包含所有功能的子模块。
# 4. 导出所有公共 API，供用户使用。
# ==========================================================

module GenomicPrediction

# --- 1. 导入核心依赖 ---
# 整个包几乎都会用到的依赖可以放在这里
using DataFrames
using Random
using Statistics
using LinearAlgebra

# --- 2. 定义核心抽象类型和结构 ---

@doc raw"""
    AbstractModel
所有基因组预测模型的抽象父类型。
"""
abstract type AbstractModel end

# --- 3. 定义通用的公共 API 函数存根 ---
# 这些是通用函数，子模块将为具体的模型类型提供实现。
# 我们在这里定义它们，以便子模块可以扩展。

@doc raw"""
    fit!(model::AbstractModel, data)
就地训练模型。这是每个模型类型必须实现的核心函数。
"""
function fit! end

@doc raw"""
    predict(model::AbstractModel, new_data)
使用训练好的模型进行预测。
"""
function predict end


# --- 4. 引入子模块 ---
# 每个文件都包含一个独立的子模块，负责一块具体的功能。
include("DataProcessing.jl")
include("CoreAlgorithm.jl")
include("KernelModels.jl")
include("DeepLearning.jl")
include("Evaluation.jl")
include("FAIRModeling.jl")
include("AutoGS.jl")


# --- 5. 从子模块导入所有公共符号，以便重新导出 ---
using .DataProcessing: GenomicData, load_csv, calculate_grm, filter_markers, impute_mean
using .CoreAlgorithm: GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel
using .KernelModels: ssGBLUPModel
using .DeepLearning: FNNModel, CNNModel, TransformerModel, GNNModel
using .Evaluation: cross_validate, accuracy, mse
using .FAIRModeling: save_model, load_model, view_model_metadata
using .AutoGS: grid_search, bayesian_optimization


# --- 6. 定义一个非破坏性的 `fit` 函数 ---
@doc raw"""
    fit(model::AbstractModel, data; kwargs...) -> AbstractModel
训练模型并返回一个新的、训练好的模型副本，原始模型不变。
"""
function fit(model::AbstractModel, data; kwargs...)
    new_model = deepcopy(model)
    fit!(new_model, data; kwargs...)
    return new_model
end


# --- 7. 导出统一的公共 API ---
# 这里列出了用户可以直接使用的所有功能。
export GenomicData, load_csv, calculate_grm, filter_markers, impute_mean
export AbstractModel, GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel, ssGBLUPModel, FNNModel, CNNModel, TransformerModel, GNNModel
export fit!, fit, predict
export cross_validate, accuracy, mse
export save_model, load_model, view_model_metadata
export grid_search, bayesian_optimization

end # module GenomicPrediction
