# Evaluation.jl - 评估与解释模块
# ==========================================================
# 负责模型性能的评估，包括交叉验证和各种准确性指标的计算。
#
# 本文件经过优化，`cross_validate` 函数现在使用多线程来并行处理
# 不同的数据折，从而显著加快评估过程。
# ==========================================================

module Evaluation

using ..GenomicPrediction: AbstractModel, GenomicData, fit!, predict
using Statistics
using Random
using ProgressMeter
using DataFrames
using Base.Threads

export cross_validate, accuracy, mse

@doc raw"""
    accuracy(y_true::Vector, y_pred::Vector) -> Float64
"""
function accuracy(y_true::Vector, y_pred::Vector)
    if any(!isfinite, y_true) || any(!isfinite, y_pred); return NaN; end
    if length(y_true) < 2; return NaN; end
    return cor(y_true, y_pred)
end

@doc raw"""
    mse(y_true::Vector, y_pred::Vector) -> Float64
"""
function mse(y_true::Vector, y_pred::Vector)
    return mean((y_true .- y_pred).^2)
end

@doc raw"""
    cross_validate(model_prototype::AbstractModel, data::GenomicData; k::Int=5, rng::AbstractRNG = Random.GLOBAL_RNG)

对给定的模型原型执行并行的 k-折交叉验证。

该函数利用多线程将每一折的计算分配到不同的核心，从而显著加快
对于计算密集型模型的评估速度。

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
    fold_indices = [indices[floor(Int, (i-1)*n/k)+1:floor(Int, i*n/k)] for i in 1:k]

    # 创建一个线程安全的结果收集器
    metrics_per_fold = Vector{Any}(undef, k)

    println("开始并行的 $k-折交叉验证 (模型: $(typeof(model_prototype)), 线程数: $(nthreads()))...")

    # 使用 @threads 宏并行处理每一折
    @threads for i in 1:k
        println("  线程 $(threadid()) 正在处理第 $i 折...")

        # 1. 划分训练集和验证集
        val_indices = fold_indices[i]
        train_indices = setdiff(1:n, val_indices)

        train_data = GenomicData(data.genotypes[train_indices, :], data.phenotypes[train_indices, :], data.covariates, data.pedigree)
        val_geno = data.genotypes[val_indices, :]
        val_pheno_vec = data.phenotypes[val_indices, 2]

        # 2. 训练模型 (每个线程使用模型的深拷贝以避免竞争)
        model_for_fold = deepcopy(model_prototype)

        # 创建一个线程本地的随机数生成器，以确保随机过程的线程安全
        local_rng = Random.MersenneTwister(rand(rng, UInt))
        fit!(model_for_fold, train_data; rng=local_rng)

        # 3. 预测并评估
        predictions = predict(model_for_fold, val_geno)

        acc = accuracy(val_pheno_vec, predictions)
        ms_error = mse(val_pheno_vec, predictions)

        # 将结果存入预分配的向量中
        metrics_per_fold[i] = (accuracy=acc, mse=ms_error)
        println("  线程 $(threadid()) 完成第 $i 折, 准确性: $(round(acc, digits=4))")
    end

    println("\n交叉验证完成。")

    # 4. 计算并返回平均指标
    valid_metrics = filter(m -> !isnothing(m) && !isnan(m.accuracy), metrics_per_fold)
    mean_acc = mean(m.accuracy for m in valid_metrics)
    mean_mse = mean(m.mse for m in valid_metrics)

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
