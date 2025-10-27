# AutoGS.jl - 自动基因组选择 (Automated Genomic Selection) 模块
# ----------------------------------------------------------------
# 提供自动化超参数调优和模型选择的功能。
# ----------------------------------------------------------------

module AutoGS

using ..DataProcessing
using ..CoreAlgorithm
using ..Evaluation
using IterTools
using Random
using Statistics

export grid_search, bayesian_optimization

"""
    grid_search(model_generator, data::GenomicData, hyperparameters::Dict, cross_validate_func; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG) -> NamedTuple

对给定的模型和超参数网格执行网格搜索，以找到最佳超参数组合。

该函数通过交叉验证评估每组超参数的性能。

# 参数
- `model_generator`: 一个函数，接收超参数字典并返回一个新的模型实例。例如 `params -> GBLUPModel(lambda=params[:lambda])`。
- `data::GenomicData`: 用于评估的 `GenomicData` 对象。
- `hyperparameters::Dict`: 一个字典，键是超参数名称 (Symbol)，值是待测试值的向量。
- `cross_validate_func`: 用于执行交叉验证的函数，通常是 `Evaluation.cross_validate`。
- `k::Int`: 交叉验证的折数。
- `metric::String`: 用于选择最佳模型的评估指标名称。该名称必须是 `cross_validate` 返回结果的 `metrics` 字典中的一个键。
- `rng`: 随机数生成器，用于确保交叉验证数据划分的可复现性。

# 返回
- `NamedTuple`: 包含两个字段：
    - `best_params`: 性能最佳的超参数字典。
    - `results`: 一个包含每次运行的详细结果（参数和指标）的向量。

# 示例
```julia
# hyper_grid = Dict(:lambda => [1.0, 10.0, 100.0])
# model_gen(p) = GBLUPModel(lambda=p[:lambda])
# result = grid_search(model_gen, my_data, hyper_grid, cross_validate)
# println("最佳 lambda: ", result.best_params[:lambda])
```
"""
function grid_search(model_generator, data::GenomicData, hyperparameters::Dict, cross_validate_func; k=3, metric="mean_accuracy", rng=Random.GLOBAL_RNG)
    param_names = keys(hyperparameters)
    param_values = values(hyperparameters)
    # 修正：使用 Iterators.product 来避免弃用警告
    param_combinations = Iterators.product(param_values...)

    all_results = []
    best_score = -Inf
    best_params = nothing

    println("开始网格搜索，总共 ", length(param_combinations), " 种参数组合...")

    for params_tuple in param_combinations
        current_params = Dict(zip(param_names, params_tuple))

        println("  正在测试参数: ", current_params)

        model_instance = model_generator(current_params)

        # 修正：将关键字参数 k=k 改为位置参数 k
        cv_results = cross_validate_func(model_instance, data, k, rng=rng)

        current_score = cv_results.metrics[metric]

        push!(all_results, (params=current_params, metrics=cv_results.metrics))

        if current_score > best_score
            best_score = current_score
            best_params = current_params
        end
    end

    println("网格搜索完成。")
    println("最佳得分 ($metric): $best_score")
    println("最佳参数: ", best_params)

    # 确保返回结果
    return (best_params = best_params, results = all_results)
end

using Hyperopt

@doc raw"""
    bayesian_optimization(model_generator, data::GenomicData, search_space, cross_validate_func; k=3, metric="mean_accuracy", max_iters=50, rng=Random.GLOBAL_RNG) -> NamedTuple

使用贝叶斯优化 (通过 Hyperopt.jl) 来寻找最佳超参数。

# 参数
- `model_generator`: 创建模型实例的函数。
- `data::GenomicData`: 数据集。
- `search_space`: `Hyperopt` 搜索空间。例如 `Dict(:lambda => @hyperopt(hp.loguniform(log(1.0), log(1000.0))))`。
- `cross_validate_func`: 交叉验证函数。
- `max_iters::Int`: 优化的最大迭代次数。

# 返回
- `NamedTuple`: 包含 `best_params` 和 `results`。
"""
function bayesian_optimization(model_generator, data::GenomicData, search_space, cross_validate_func; k=3, metric="mean_accuracy", max_iters=50, rng=Random.GLOBAL_RNG)
    println("开始贝叶斯优化 (最大迭代次数: $max_iters)...")

    # Hyperopt 最小化目标，所以我们需要返回负的准确率
    function objective(params)
        model = model_generator(params)
        cv_results = cross_validate_func(model, data, k; rng=rng)
        score = cv_results.metrics[metric]
        # Hyperopt.jl expects a dictionary with a :loss key
        return Dict(:loss => -score, :status => "ok")
    end

    # 创建 Hyperopt 对象
    ho = Hyperopt(objective, search_space)

    # 运行优化
    best_params = @hyperopt for i=max_iters, params=search_space
        objective(params)
    end

    println("贝叶斯优化完成。")
    # 注意：@hyperopt 宏直接返回最佳参数字典
    # 为了与 grid_search 的输出保持一致，我们不存储详细的迭代结果

    return (best_params = best_params, results = [])
end


end # module AutoGS
