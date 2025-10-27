# Evaluation.jl
module Evaluation
using Statistics
using Random
using DataFrames
using ..DataProcessing

export accuracy, mse, cross_validate, get_snp_effects

@doc raw"""
    accuracy(predictions::Vector, truth::Vector) -> Float64
计算预测值与真实值之间的皮尔逊相关系数。
"""
function accuracy(predictions::Vector, truth::Vector)
    return cor(predictions, truth)
end
@doc raw"""
    mse(predictions::Vector, truth::Vector) -> Float64
计算预测值与真实值之间的均方误差 (MSE)。
"""
function mse(predictions::Vector, truth::Vector)
    return mean((predictions .- truth).^2)
end
@doc raw"""
    cross_validate(model_generator, data::GenomicData, k::Int; ...) -> Dict
对给定的模型进行 k-折交叉验证。
# 参数
- `model_generator`: 返回新模型实例的函数。
- `data::GenomicData`: 数据集。
- `k::Int`: 折数。
# 返回
- `Dict{String, Float64}`: 平均评估指标。
"""
function cross_validate(model_template, data::GenomicData, k::Int, fit_function, predict_function; rng::AbstractRNG = Random.GLOBAL_RNG)
    n_individuals = size(data.genotypes, 1)
    indices = randperm(rng, n_individuals)
    fold_size = floor(Int, n_individuals / k)

    accuracies = []
    mses = []

    println("开始 $k-折交叉验证...")

    for i in 1:k
        println("  正在处理第 $i/$k 折...")

        start_idx = (i - 1) * fold_size + 1
        end_idx = i < k ? i * fold_size : n_individuals

        val_indices = indices[start_idx:end_idx]
        train_indices = setdiff(1:n_individuals, val_indices)

        train_geno = data.genotypes[train_indices, :]
        train_pheno = data.phenotypes[train_indices, :]
        train_data = GenomicData(train_geno, train_pheno)

        val_geno = data.genotypes[val_indices, :]
        val_pheno_true = data.phenotypes[val_indices, 1]

        # 使用 deepcopy 创建一个全新的模型实例以避免各折之间的信息泄露
        model = deepcopy(model_template)
        fit_function(model, train_data)

        predictions = predict_function(model, val_geno)

        push!(accuracies, accuracy(predictions, val_pheno_true))
        push!(mses, mse(predictions, val_pheno_true))
    end

    println("交叉验证完成。")

    # 返回一个包含所有指标的字典
    metrics = Dict("mean_accuracy" => mean(accuracies), "mean_mse" => mean(mses))
    return (metrics=metrics, raw_accuracies=accuracies, raw_mses=mses)
end

@doc raw"""
    get_snp_effects(model::AbstractModel, data::GenomicData) -> DataFrame

从一个训练好的模型中提取 SNP (标记) 效应。

# 参数
- `model`: 一个已训练的模型实例，必须包含 `effects` 字段。
- `data::GenomicData`: 原始 `GenomicData` 对象，用于获取标记名称。

# 返回
- `DataFrame`: 一个包含两列的 DataFrame：`MarkerName` 和 `Effect`。
"""
function get_snp_effects(model, data::GenomicData)
    if !hasfield(typeof(model), :effects)
        error("该模型类型 ($(typeof(model))) 不支持提取 SNP 效应，因为它没有 `effects` 字段。")
    end

    # 第 1 列是 ID，所以标记名称从第 2 列开始
    marker_names = names(data.genotypes)[2:end]

    if length(marker_names) != length(model.effects)
        error("标记数量与模型效应数量不匹配。")
    end

    return DataFrame(MarkerName = marker_names, Effect = model.effects)
end
end
