# GenomicPrediction.jl - 基因组预测理论与方法综合软件包
# ==========================================================
# ... (header comments) ...

module GenomicPrediction

using DataFrames
using Random
using IterTools

# --- 1. 定义核心抽象类型 ---
abstract type AbstractModel end

# --- 2. 引入子模块 ---
include("DataProcessing.jl")
include("CoreAlgorithm.jl")
include("DeepLearning.jl")
include("Evaluation.jl")
include("FAIRModeling.jl")
include("AutoGS.jl")

# --- 3. 显式函数转发和类型别名 ---

# DataProcessing
const GenomicData = DataProcessing.GenomicData
const load_csv = DataProcessing.load_csv

# CoreAlgorithm
const GBLUPModel = CoreAlgorithm.GBLUPModel
const BayesAModel = CoreAlgorithm.BayesAModel
const LASSOModel = CoreAlgorithm.LASSOModel
const ElasticNetModel = CoreAlgorithm.ElasticNetModel

# DeepLearning
const FNNModel = DeepLearning.FNNModel
const CNNModel = DeepLearning.CNNModel

# Evaluation
const accuracy = Evaluation.accuracy
const mse = Evaluation.mse

# FAIRModeling
const save_model = FAIRModeling.save_model
const load_model = FAIRModeling.load_model

# 通用函数 (转发到各自的实现)
# fit!
fit!(model::GBLUPModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::BayesAModel, data::GenomicData; rng = Random.GLOBAL_RNG) = CoreAlgorithm.fit!(model, data; rng=rng)
fit!(model::LASSOModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::ElasticNetModel, data::GenomicData) = CoreAlgorithm.fit!(model, data)
fit!(model::FNNModel, data::GenomicData) = DeepLearning.fit!(model, data)
fit!(model::CNNModel, data::GenomicData) = DeepLearning.fit!(model, data)

# predict
predict(model::GBLUPModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::BayesAModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::LASSOModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::ElasticNetModel, new_data::DataFrame) = CoreAlgorithm.predict(model, new_data)
predict(model::FNNModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)
predict(model::CNNModel, new_data::DataFrame) = DeepLearning.predict(model, new_data)

# cross_validate
function cross_validate(model_generator, data::GenomicData, k::Int; rng = Random.GLOBAL_RNG)
    return Evaluation.cross_validate(model_generator, data, k, fit!, predict; rng=rng)
end

# AutoGS
function grid_search(model_generator, data::GenomicData, hyperparameters; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG)
    return AutoGS.grid_search(model_generator, data, hyperparameters, cross_validate; k=k, metric=metric, rng=rng)
end

# --- 4. 导出统一的公共 API ---
export GenomicData, load_csv
export AbstractModel, GBLUPModel, BayesAModel, LASSOModel, ElasticNetModel, FNNModel, CNNModel
export fit!, predict
export accuracy, mse
export save_model, load_model
export cross_validate
export grid_search

end # module GenomicPrediction
