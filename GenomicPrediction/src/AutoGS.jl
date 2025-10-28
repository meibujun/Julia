# AutoGS.jl - 自动基因组选择模块
# ==========================================================
# 提供自动化超参数调优和模型选择的功能。
#
# 本文件经过修正，以移除对 `cross_validate_func` 的依赖，
# 转而直接调用包内定义的 `cross_validate` 函数，从而简化 API。
# ==========================================================

module AutoGS

# --- 1. 导入依赖 ---
using ..GenomicPrediction: AbstractModel, GenomicData, cross_validate
using Hyperopt
using Random
using ProgressMeter
using IterTools

# --- 2. 模块接口 ---
export grid_search, bayesian_optimization

@doc raw"""
    grid_search(model_constructor, data::GenomicData, hyperparameters; k=5, metric=:mean_accuracy, rng=Random.GLOBAL_RNG)

使用网格搜索，通过交叉验证来寻找最佳超参数组合。

# Arguments
- `model_constructor`: 一个接受关键字参数并返回 `AbstractModel` 实例的函数。例如 `params -> GBLUPModel(params[:lambda])`。
- `data::GenomicData`: 完整的 `GenomicData` 对象。
- `hyperparameters::Dict`: 一个字典，键是超参数的符号，值是待测试值的向量。
- `k::Int`: 交叉验证的折数。
- `metric::Symbol`: 用于评估性能的指标 (`:mean_accuracy` 或 `:mean_mse`)。
- `rng::AbstractRNG`: 用于交叉验证数据混洗的随机数生成器。

# Returns
- 包含最佳参数、最佳得分和所有结果的 NamedTuple。
"""
function grid_search(model_constructor, data::GenomicData, hyperparameters; k=5, metric=:mean_accuracy, rng=Random.GLOBAL_RNG)
    param_names = keys(hyperparameters)
    param_combinations = [Dict(zip(param_names, values)) for values in product(values(hyperparameters)...)]

    # 根据指标确定是最大化还是最小化
    lower_is_better = (metric == :mean_mse)
    best_score = lower_is_better ? Inf : -Inf
    best_params = nothing
    all_results = []

    println("开始网格搜索，总共 $(length(param_combinations)) 种参数组合...")
    p = Progress(length(param_combinations), 1, "网格搜索进度:")

    for params in param_combinations
        # 1. 使用当前参数组合构造模型
        model_prototype = model_constructor(params)

        # 2. 直接调用 cross_validate 函数
        cv_results = cross_validate(model_prototype, data; k=k, rng=rng)
        score = getfield(cv_results, metric)

        push!(all_results, (params=params, metrics=cv_results))

        # 3. 更新最佳参数
        if (lower_is_better && score < best_score) || (!lower_is_better && score > best_score)
            best_score = score
            best_params = params
        end
        next!(p)
    end

    println("\n网格搜索完成。")
    println("最佳得分 ($metric): $(round(best_score, digits=4))")
    println("最佳参数: $best_params")

    return (best_params=best_params, best_score=best_score, results=all_results)
end

@doc raw"""
    bayesian_optimization(model_constructor, data::GenomicData, search_space; k=5, max_iters=30, metric=:mean_accuracy, rng=Random.GLOBAL_RNG)

使用贝叶斯优化，通过交叉验证来寻找最佳超参数。

# Arguments
- `model_constructor`: 一个接受关键字参数并返回 `AbstractModel` 实例的函数。
- `data::GenomicData`: 完整的 `GenomicData` 对象。
- `search_space::Dict`: `Hyperopt` 的搜索空间定义。
- `k::Int`: 交叉验证的折数。
- `max_iters::Int`: 贝叶斯优化的最大迭代次数。
- `metric::Symbol`: 用于评估性能的指标 (`:mean_accuracy` 或 `:mean_mse`)。
- `rng::AbstractRNG`: 随机数生成器。

# Returns
- 包含最佳参数和最佳得分的 NamedTuple。
"""
function bayesian_optimization(model_constructor, data::GenomicData, search_space; k=5, max_iters=30, metric=:mean_accuracy, rng=Random.GLOBAL_RNG)

    lower_is_better = (metric == :mean_mse)

    # 定义 Hyperopt 的目标函数
    function objective(params)
        # Hyperopt 返回的是 Tuple，需要转换为 Dict
        params_dict = Dict(params)

        model_prototype = model_constructor(params_dict)

        cv_results = cross_validate(model_prototype, data; k=k, rng=rng)
        score = getfield(cv_results, metric)

        # Hyperopt 总是最小化，所以如果指标是越大越好，我们需要取其负值
        return lower_is_better ? score : -score
    end

    println("开始贝叶斯优化，最大迭代次数: $max_iters...")

    # 执行优化
    ho = @hyperopt for i=max_iters, sampler=RandomSampler(rng=rng), kwargs=search_space
        # 在每次迭代打印信息
        println("  [迭代 $i/$max_iters] 测试参数: $kwargs")
        objective(kwargs)
    end

    best_params_dict = Dict(ho.minimizer)
    best_score = lower_is_better ? ho.minimum : -ho.minimum

    println("\n贝叶斯优化完成。")
    println("最佳得分 ($metric): $(round(best_score, digits=4))")
    println("最佳参数: $best_params_dict")

    return (best_params=best_params_dict, best_score=best_score)
end


end # module AutoGS
