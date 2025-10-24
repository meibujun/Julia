# AutoGS.jl: 自动基因组选择模块
# -----------------------------------
#
# 本模块旨在提供自动化机器学习 (AutoML) 功能，以简化和优化基因组预测的建模流程。
#
# 主要功能:
# - **超参数调优**: 实现如网格搜索 (Grid Search)、随机搜索 (Random Search)
#   和贝叶斯优化 (Bayesian Optimization) 等方法，自动寻找最佳模型超参数。
# - **模型选择**: (未来实现) 自动在多个模型（例如 GBLUP, BayesA, CNN）之间
#   进行比较，并推荐性能最佳的模型。
# - **集成学习**: (未来实现) 提供模型堆叠 (Stacking) 或集成 (Ensembling)
#   的功能，以进一步提升预测准确性。

module AutoGS

using IterTools: product
using Random

# We need access to GenomicData, cross_validate, and model types.
# These will be passed as arguments or will be available in the calling scope
# defined in the main GenomicPrediction module.

export grid_search

"""
    grid_search(model_generator, data, hyperparameters; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG)

对给定的模型和超参数网格执行网格搜索，以找到最佳的超参数组合。

# 参数
- `model_generator`: 一个函数，接收超参数作为关键字参数，并返回一个新的模型实例。
  例如: `params -> GBLUPModel(lambda=params[:lambda])`
- `data::GenomicData`: 用于评估的数据集。
- `hyperparameters::Dict`: 一个字典，键是超参数的名称 (作为 Symbol)，值是待搜索值的向量。
- `k::Int`: 交叉验证的折数。
- `metric::String`: 用于评估和选择最佳模型的指标名称 (来自交叉验证的结果)。
- `rng::AbstractRNG`: 用于交叉验证中的随机分割。

# 返回
- `NamedTuple`: 包含两部分内容:
  - `best_params`: 找到的最佳超参数组合的字典。
  - `results`: 一个包含了所有已评估的超参数组合及其对应性能的向量。
"""
function grid_search(model_generator, data, hyperparameters, cross_validate_func; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG)

    param_names = keys(hyperparameters)
    param_values = values(hyperparameters)

    # 生成所有超参数组合
    param_combinations = Iterators.product(param_values...)

    best_params = nothing
    best_score = -Inf
    all_results = []

    println("开始网格搜索 (共 $(length(collect(param_combinations))) 种组合)...")

    for params in param_combinations
        current_params = Dict(zip(param_names, params))

        # 创建一个使用当前参数的模型生成器
        current_model_gen() = model_generator(current_params)

        println("  正在评估参数: $current_params")

        # 执行交叉验证
        cv_results = cross_validate_func(current_model_gen, data, k; rng=rng)
        current_score = cv_results[metric]

        push!(all_results, (params=current_params, metrics=cv_results))

        if current_score > best_score
            best_score = current_score
            best_params = current_params
        end
    end

    println("网格搜索完成。")
    println("最佳参数: $best_params")
    println("最佳得分 ($metric): $best_score")

    return (best_params=best_params, results=all_results)
end


end # module AutoGS
