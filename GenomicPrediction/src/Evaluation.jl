# Evaluation.jl: 评估与解释模块
# -----------------------------------
#
# 本模块提供了一系列用于评估模型性能和解释预测结果的工具。
#
# 主要功能:
# - **性能指标**: 计算标准的回归和分类指标，如均方误差 (MSE)、相关系数 (Accuracy)、AUC 等。
# - **交叉验证**: 实现 K-折交叉验证，以提供对模型泛化能力更稳健的评估。
# - **模型解释**: (未来实现) 集成如 SHAP (SHapley Additive exPlanations) 等方法，
#   用于解释复杂模型（如深度学习模型）的预测依据。

module Evaluation

using Statistics
using Random
using DataFrames
using ..DataProcessing
# Note: We need access to the model types and fit!/predict,
# but we cannot import the parent module `GenomicPrediction` due to circular dependencies.
# The `cross_validate` function will rely on the methods being available in its call scope.
# This is handled by the explicit forwarding in the main module.

export accuracy, mse, cross_validate

"""
    accuracy(predictions::Vector, truth::Vector) -> Float64

计算预测值与真实值之间的皮尔逊相关系数，作为预测准确性的度量。
"""
function accuracy(predictions::Vector, truth::Vector)
    return cor(predictions, truth)
end

"""
    mse(predictions::Vector, truth::Vector) -> Float64

计算预测值与真实值之间的均方误差 (Mean Squared Error)。
"""
function mse(predictions::Vector, truth::Vector)
    return mean((predictions .- truth).^2)
end


"""
    cross_validate(model_generator, data::GenomicData, k::Int; rng::AbstractRNG = Random.GLOBAL_RNG)

对给定的模型进行 k-折交叉验证。

# 参数
- `model_generator`: 一个函数，每次调用时返回一个新的、未训练的模型实例 (例如 `() -> GBLUPModel(5.0)`)。
- `data::GenomicData`: 包含基因型和表型的数据集。
- `k::Int`: 交叉验证的折数。
- `rng::AbstractRNG`: 用于随机数据分割的随机数生成器。

# 返回
- `Dict{String, Float64}`: 包含在 k-折中平均的评估指标的字典 (例如 `mean_accuracy`, `mean_mse`)。

# 流程
1. 将数据集的索引随机打乱并分成 k 份（折）。
2. 迭代 k 次，每次选择一折作为验证集，其余 k-1 折作为训练集。
3. 在训练集上训练一个新模型。
4. 在验证集上进行预测并计算性能指标。
5. 将 k 次迭代的指标进行平均，并返回结果。
"""
function cross_validate(model_generator, data::GenomicData, k::Int, fit_function, predict_function; rng::AbstractRNG = Random.GLOBAL_RNG)
    n_individuals = size(data.genotypes, 1)
    indices = randperm(rng, n_individuals)
    fold_size = floor(Int, n_individuals / k)

    accuracies = []
    mses = []

    println("开始 $k-折交叉验证...")

    for i in 1:k
        println("  正在处理第 $i/$k 折...")

        # --- 1. 划分训练集和验证集 ---
        start_idx = (i - 1) * fold_size + 1
        end_idx = i < k ? i * fold_size : n_individuals

        val_indices = indices[start_idx:end_idx]
        train_indices = setdiff(1:n_individuals, val_indices)

        train_geno = data.genotypes[train_indices, :]
        train_pheno = data.phenotypes[train_indices, :]
        train_data = GenomicData(train_geno, train_pheno)

        val_geno = data.genotypes[val_indices, :]
        val_pheno_true = data.phenotypes[val_indices, 1]

        # --- 2. 训练模型 ---
        model = model_generator()
        fit_function(model, train_data)

        # --- 3. 预测与评估 ---
        predictions = predict_function(model, val_geno)

        push!(accuracies, accuracy(predictions, val_pheno_true))
        push!(mses, mse(predictions, val_pheno_true))
    end

    println("交叉验证完成。")

    # --- 4. 计算并返回平均指标 ---
    return Dict(
        "mean_accuracy" => mean(accuracies),
        "mean_mse" => mean(mses)
    )
end


end # module Evaluation
