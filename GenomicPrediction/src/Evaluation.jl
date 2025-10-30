# Evaluation.jl - 评估与解释模块
# ==========================================================
# 本文件已更新，以与 AbstractGenomicData 架构兼容。
# ==========================================================

module Evaluation

using ..GenomicPrediction: AbstractModel, AbstractGenomicData, InMemoryGenomicData, fit!, predict, get_genotypes, get_phenotypes, get_covariates, get_pedigree
using Statistics
using Random
using ProgressMeter
using DataFrames
using Base.Threads

export cross_validate, accuracy, mse

function accuracy(y_true::Vector, y_pred::Vector)
    if any(!isfinite, y_true) || any(!isfinite, y_pred); return NaN; end
    if length(y_true) < 2; return NaN; end
    return cor(y_true, y_pred)
end

function mse(y_true::Vector, y_pred::Vector)
    return mean((y_true .- y_pred).^2)
end

@doc raw"""
    cross_validate(model_prototype::AbstractModel, data::AbstractGenomicData; k::Int=5, rng::AbstractRNG = Random.GLOBAL_RNG)
"""
function cross_validate(model_prototype::AbstractModel, data::AbstractGenomicData; k::Int=5, rng::AbstractRNG = Random.GLOBAL_RNG)
    geno_df = get_genotypes(data)
    pheno_df = get_phenotypes(data)
    n = size(geno_df, 1)

    indices = shuffle(rng, 1:n)
    fold_indices = [indices[floor(Int, (i-1)*n/k)+1:floor(Int, i*n/k)] for i in 1:k]

    metrics_per_fold = Vector{Any}(undef, k)

    println("开始并行的 $k-折交叉验证 (模型: $(typeof(model_prototype)), 线程数: $(nthreads()))...")

    @threads for i in 1:k
        println("  线程 $(threadid()) 正在处理第 $i 折...")

        val_indices = fold_indices[i]
        train_indices = setdiff(1:n, val_indices)

        # 创建 InMemoryGenomicData 用于训练，因为大多数模型需要内存中的数据
        train_data = InMemoryGenomicData(
            geno_df[train_indices, :],
            pheno_df[train_indices, :],
            get_covariates(data),
            get_pedigree(data)
        )
        val_geno = geno_df[val_indices, :]
        val_pheno_vec = pheno_df[val_indices, 2]

        model_for_fold = deepcopy(model_prototype)
        local_rng = Random.MersenneTwister(rand(rng, UInt))
        fit!(model_for_fold, train_data; rng=local_rng)

        predictions = predict(model_for_fold, val_geno)

        acc = accuracy(val_pheno_vec, predictions)
        ms_error = mse(val_pheno_vec, predictions)

        metrics_per_fold[i] = (accuracy=acc, mse=ms_error)
        println("  线程 $(threadid()) 完成第 $i 折, 准确性: $(round(acc, digits=4))")
    end

    println("\n交叉验证完成。")

    valid_metrics = filter(m -> !isnothing(m) && !isnan(m.accuracy), metrics_per_fold)
    mean_acc = mean(m.accuracy for m in valid_metrics)
    mean_mse = mean(m.mse for m in valid_metrics)

    results = (mean_accuracy = mean_acc, mean_mse = mean_mse, fold_metrics = metrics_per_fold)

    println("平均准确性 (相关系数): $(round(mean_acc, digits=4))")
    println("平均均方误差 (MSE): $(round(mean_mse, digits=4))")

    return results
end

end # module Evaluation
