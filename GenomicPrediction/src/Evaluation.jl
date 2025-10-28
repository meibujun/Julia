# Evaluation.jl - 评估与解释模块
# ==========================================================
# 负责模型性能的评估，包括交叉验证和各种准确性指标的计算。
#
# 本文件经过修正，简化了 `cross_validate` 函数的 API，使其直接
# 使用 `GenomicPrediction` 模块中定义的通用 `fit!` 和 `predict` 函数。
# ==========================================================

module Evaluation

using ..GenomicPrediction: AbstractModel, GenomicData, fit!, predict
using Statistics
using Random
using ProgressMeter
using DataFrames

export cross_validate, accuracy, mse

@doc raw"""
    accuracy(y_true::Vector, y_pred::Vector) -> Float64

计算预测值与真实值之间的皮尔逊相关系数，作为预测准确性的度量。
这在基因组预测领域是评估预测模型性能的常用指标。
"""
function accuracy(y_true::Vector, y_pred::Vector)
    # 确保没有 NaN 或 Inf 值，这可能在相关性计算中导致错误
    if any(!isfinite, y_true) || any(!isfinite, y_pred)
        @warn "输入向量中包含非有限值 (NaN/Inf)，相关性可能为 NaN。"
        return NaN
    end
    if length(y_true) < 2
        @warn "向量长度小于2，无法计算相关性。"
        return NaN
    end
    return cor(y_true, y_pred)
end

@doc raw"""
    mse(y_true::Vector, y_pred::Vector) -> Float64

计算预测值与真实值之间的均方误差 (Mean Squared Error)。
"""
function mse(y_true::Vector, y_pred::Vector)
    return mean((y_true .- y_pred).^2)
end

@doc raw"""
    cross_validate(model_prototype::AbstractModel, data::GenomicData; k::Int=5, rng::AbstractRNG = Random.GLOBAL_RNG)

对给定的模型原型执行 k-折交叉验证。

该函数会自动处理数据的划分、模型的训练和评估。它利用了 `GenomicPrediction`
模块的通用 `fit!` 和 `predict` 接口。

# Arguments
- `model_prototype::AbstractModel`: 一个未训练的模型实例，将作为每折训练的模板。
- `data::GenomicData`: 完整的 `GenomicData` 对象。
- `k::Int`: 交叉验证的折数。默认为 5。
- `rng::AbstractRNG`: 用于数据混洗的随机数生成器。

# Returns
- 一个包含平均准确性、平均 MSE 和每折详细指标的 NamedTuple。
"""
function cross_validate(model_prototype::AbstractModel, data::GenomicData; k::Int=5, rng::AbstractRNG = Random.GLOBAL_RNG)
    n = size(data.genotypes, 1)
    indices = shuffle(rng, 1:n)
    fold_size = floor(Int, n / k)

    metrics_per_fold = []

    println("开始 $k-折交叉验证 (模型: $(typeof(model_prototype)))...")
    p = Progress(k, 1, "交叉验证进度:")

    for i in 1:k
        # 1. 划分训练集和验证集索引
        start_idx = (i - 1) * fold_size + 1
        end_idx = i < k ? i * fold_size : n
        val_indices = indices[start_idx:end_idx]
        train_indices = setdiff(1:n, val_indices)

        # 2. 创建训练数据和验证数据
        train_data = GenomicData(data.genotypes[train_indices, :], data.phenotypes[train_indices, :])
        val_geno = data.genotypes[val_indices, :]
        val_pheno_vec = data.phenotypes[val_indices, 2]

        # 3. 训练模型 (从原型创建新实例)
        model_for_fold = deepcopy(model_prototype)

        # 使用通用的 fit! 函数
        fit!(model_for_fold, train_data; rng=rng)

        # 4. 预测并评估
        # 使用通用的 predict 函数
        predictions = predict(model_for_fold, val_geno)

        acc = accuracy(val_pheno_vec, predictions)
        ms_error = mse(val_pheno_vec, predictions)

        push!(metrics_per_fold, (accuracy=acc, mse=ms_error))
        next!(p)
    end

    println("\n交叉验证完成。")

    # 5. 计算并返回平均指标
    mean_acc = mean(m.accuracy for m in metrics_per_fold if !isnan(m.accuracy))
    mean_mse = mean(m.mse for m in metrics_per_fold if !isnan(m.mse))

    results = (
        mean_accuracy = mean_acc,
        mean_mse = mean_mse,
        fold_metrics = metrics_per_fold
    )

    println("平均准确性 (相关系数): $(round(mean_acc, digits=4))")
    println("平均均方误差 (MSE): $(round(mean_mse, digits=4))")

    return results
end

end # module Evaluation
