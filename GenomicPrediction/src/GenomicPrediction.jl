# GenomicPrediction.jl - 基因组预测理论与方法综合软件包
# ==========================================================
# ... (header comments) ...

module GenomicPrediction

# ... (using statements) ...
using DataFrames
using Random
using IterTools

# --- 1. 定义核心抽象类型 ---
@doc raw"""
    AbstractModel
所有基因组预测模型的抽象父类型。
"""
abstract type AbstractModel end

# --- 2. 引入子模块 ---
# ... (includes) ...
include("DataProcessing.jl")
include("CoreAlgorithm.jl")
include("DeepLearning.jl")
include("Evaluation.jl")
include("FAIRModeling.jl")
include("AutoGS.jl")
include("KernelModels.jl")


# --- 3. 显式函数转发和类型别名 ---
# ... (forwarding for types) ...
const GenomicData = DataProcessing.GenomicData
const load_csv = DataProcessing.load_csv
const GBLUPModel = CoreAlgorithm.GBLUPModel
const BayesAModel = CoreAlgorithm.BayesAModel
const BayesBModel = CoreAlgorithm.BayesBModel
const BayesCModel = CoreAlgorithm.BayesCModel
const BayesRModel = CoreAlgorithm.BayesRModel
const LASSOModel = CoreAlgorithm.LASSOModel
const ElasticNetModel = CoreAlgorithm.ElasticNetModel
const FNNModel = DeepLearning.FNNModel
const CNNModel = DeepLearning.CNNModel
const TransformerModel = DeepLearning.TransformerModel
const GNNModel = DeepLearning.GNNModel
const ssGBLUPModel = KernelModels.ssGBLUPModel
const accuracy = Evaluation.accuracy
const mse = Evaluation.mse
const save_model = FAIRModeling.save_model
const load_model = FAIRModeling.load_model
const view_model_metadata = FAIRModeling.view_model_metadata


# 通用函数 (转发到各自的实现)
@doc raw"""
    fit!(model::AbstractModel, data::GenomicData)

使用提供的数据训练一个模型。该函数会原地修改 `model` 对象。

# 参数
- `model::AbstractModel`: 一个模型实例，例如 `GBLUPModel`。
- `data::GenomicData`: 用于训练的 `GenomicData` 对象。
"""
function fit! end

@doc raw"""
    fit(model::AbstractModel, data::GenomicData) -> AbstractModel

A non-mutating version of `fit!`. This function creates a deep copy of the model,
trains it, and returns the trained copy. Useful for functional programming patterns
and for interfaces with languages like Python where mutating functions can be awkward.
"""
function fit(model::AbstractModel, data::GenomicData; rng = Random.GLOBAL_RNG)
    new_model = deepcopy(model)
    fit!(new_model, data; rng=rng)
    return new_model
end

fit!(model::GBLUPModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::BayesAModel, data::GenomicData; rng = Random.GLOBAL_RNG) = CoreAlgorithm.fit!(model, data; rng=rng)
fit!(model::BayesBModel, data::GenomicData; rng = Random.GLOBAL_RNG) = CoreAlgorithm.fit!(model, data; rng=rng)
fit!(model::BayesCModel, data::GenomicData; rng = Random.GLOBAL_RNG) = CoreAlgorithm.fit!(model, data; rng=rng)
fit!(model::BayesRModel, data::GenomicData; rng = Random.GLOBAL_RNG) = CoreAlgorithm.fit!(model, data; rng=rng)
fit!(model::LASSOModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::ElasticNetModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::FNNModel, data::GenomicData) = DeepLearning.fit!(model, data)
fit!(model::CNNModel, data::GenomicData) = DeepLearning.fit!(model, data)
fit!(model::TransformerModel, data::GenomicData) = DeepLearning.fit!(model, data)
fit!(model::GNNModel, data::GenomicData) = DeepLearning.fit!(model, data)
fit!(model::ssGBLUPModel, data::GenomicData) = KernelModels.fit!(model, data)

@doc raw"""
    predict(model::AbstractModel, new_data::DataFrame) -> Vector

使用一个训练好的模型进行预测。

# 参数
- `model::AbstractModel`: 一个已训练的模型实例。
- `new_data::DataFrame`: 用于预测的新基因型数据。

# 返回
- `Vector`: 预测的表型值。
"""
function predict end

predict(model::GBLUPModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::BayesAModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::BayesBModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::BayesCModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::BayesRModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::LASSOModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::ElasticNetModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::FNNModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)
predict(model::CNNModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)
predict(model::TransformerModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)
predict(model::GNNModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)
predict(model::ssGBLUPModel, new_ids::Vector{Int}) = KernelModels.predict(model, new_ids)

# ... (cross_validate and grid_search wrappers) ...
function cross_validate(model_generator, data::GenomicData, k::Int; rng = Random.GLOBAL_RNG)
    return Evaluation.cross_validate(model_generator, data, k, fit!, predict; rng=rng)
end
function grid_search(model_generator, data::GenomicData, hyperparameters; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG)
    return AutoGS.grid_search(model_generator, data, hyperparameters, cross_validate; k=k, metric=metric, rng=rng)
end

# --- 4. 导出统一的公共 API ---
export GenomicData, load_csv
export AbstractModel, GBLUPModel, BayesAModel, BayesBModel, BayesCModel, BayesRModel, LASSOModel, ElasticNetModel, FNNModel, CNNModel, TransformerModel, GNNModel, ssGBLUPModel
export fit!, fit, predict
export accuracy, mse
export save_model, load_model, view_model_metadata
export cross_validate
export grid_search

end # module GenomicPrediction
