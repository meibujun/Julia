#=###############################################################################
# AutoGS 自动建模
# 自动搜索多种模型与超参数组合, 输出最优方案并提供 FAIR 元数据。
###############################################################################=#

using DataFrames
using Random
using Statistics
using Flux

"""
    mutable struct AutoGSPipeline

描述自动基因组选择工作流的配置与评估状态。
"""
Base.@kwdef mutable struct AutoGSPipeline
    candidates::Vector{NamedTuple}
    metric::Symbol = :rmse
    goal::Symbol = :auto
    k::Int = 5
    results::DataFrame = DataFrame(name = String[], mean = Float64[], std = Float64[], min = Float64[], max = Float64[])
    best_model::Any = nothing
    best_history::Any = nothing
    best_metadata::Union{Nothing,ModelMetadata} = nothing
end

_default_goal(metric::Symbol) = metric in (:mse, :rmse, :mae) ? :min : :max

"""
    AutoGSPipeline(candidates; metric=:rmse, goal=:auto, k=5)

创建自动建模流程。`candidates` 需包含 `name`, `builder`, `trainer`, `predictor`, `hyperparams` 等字段。
"""
function AutoGSPipeline(candidates::Vector{NamedTuple}; metric::Symbol = :rmse, goal::Symbol = :auto, k::Integer = 5)
    pipeline_goal = goal == :auto ? _default_goal(metric) : goal
    return AutoGSPipeline(candidates = candidates, metric = metric, goal = pipeline_goal, k = k)
end

"""
    run_autogs(pipeline, dataset; rng=Random.default_rng())

执行候选模型评估, 更新最佳模型与结果表。
"""
function run_autogs(pipeline::AutoGSPipeline, dataset::GenomicDataset; rng::AbstractRNG = Random.default_rng())
    empty!(pipeline.results)
    best_score = pipeline.goal == :min ? Inf : -Inf
    best_model = nothing
    best_history = nothing
    best_metadata = nothing
    for cand in pipeline.candidates
        cv = cross_validate(cand.builder, dataset, pipeline.k;
                            trainer = cand.trainer,
                            predictor = cand.predictor,
                            metric = pipeline.metric,
                            rng = rng,
                            retain_hist = true)
        summary = summarize_cv(cv)
        push!(pipeline.results, (cand.name, summary[:mean], summary[:std], summary[:min], summary[:max]))
        score = summary[:mean]
        isnan(score) && continue
        better = pipeline.goal == :min ? score < best_score : score > best_score
        if better
            best_score = score
            trained = cand.trainer(cand.builder(), dataset.genotype, dataset.phenotype)
            if trained isa Tuple
                best_history = trained[1]
                best_model = trained[end]
            else
                best_model = trained
                best_history = nothing
            end
            metadata = create_metadata(cand.name;
                                       metrics = Dict(string(pipeline.metric) => score),
                                       hyperparameters = haskey(cand, :hyperparams) ? cand.hyperparams : Dict{String,Any}(),
                                       data_sources = [get(dataset.metadata, "source", "unknown")],
                                       random_seed = haskey(cand, :seed) ? cand.seed : nothing)
            best_metadata = metadata
        end
    end
    pipeline.best_model = best_model
    pipeline.best_history = best_history
    pipeline.best_metadata = best_metadata
    return pipeline
end

"""
    default_workflow(dataset)

构建包含经典统计模型与 MLP 的默认候选集。
"""
function default_workflow(dataset::GenomicDataset)
    input_dim = size(dataset.genotype, 2)
    mlp_builder = () -> build_mlp(input_dim, [256, 128, 64]; dropout_rate = 0.2, output_dim = 1)
    mlp_trainer = (model, X, y) -> train_deep_model!(model, X, y; epochs = 30, batch_size = 64,
                                                     device = :cpu,
                                                     metric_fns = Dict("rmse" => (ŷ, t) -> sqrt(mean((ŷ .- t) .^ 2))))
    mlp_predictor = (trained, X) -> vec(Flux.cpu(trained(permutedims(Float32.(X)))))

    candidates = [
        (name = "GBLUP",
         builder = () -> GBLUPModel(λ = 1.0),
         trainer = (model, X, y) -> fit!(model, X, y),
         predictor = (model, X) -> predict(model, X),
         hyperparams = Dict("λ" => 1.0)),
        (name = "Ridge",
         builder = () -> RidgeRegressionModel(λ = 0.5),
         trainer = (model, X, y) -> fit!(model, X, y; λ = 0.5),
         predictor = (model, X) -> predict(model, X),
         hyperparams = Dict("λ" => 0.5)),
        (name = "ElasticNet",
         builder = () -> ElasticNetModel(λ1 = 0.05, λ2 = 0.05),
         trainer = (model, X, y) -> fit!(model, X, y; λ1 = 0.05, λ2 = 0.05),
         predictor = (model, X) -> predict(model, X),
         hyperparams = Dict("λ1" => 0.05, "λ2" => 0.05)),
        (name = "BayesA",
         builder = () -> BayesAModel(iterations = 1500, burn_in = 300),
         trainer = (model, X, y) -> fit!(model, X, y),
         predictor = (model, X) -> predict(model, X),
         hyperparams = Dict("iterations" => 1500)),
        (name = "MLP",
         builder = mlp_builder,
         trainer = mlp_trainer,
         predictor = mlp_predictor,
         hyperparams = Dict("architecture" => "[256,128,64]", "dropout" => 0.2))
    ]
    return AutoGSPipeline(candidates; metric = :rmse, goal = :min, k = 5)
end

