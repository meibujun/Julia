#=###############################################################################
# 评估模块
# 提供常用指标、交叉验证框架与结果汇总。
###############################################################################=#

using LinearAlgebra
using Random
using Statistics
using StatsBase

const _EPS = 1e-12

"""
    evaluate_metrics(y_true, y_pred; metrics=[:mse, :rmse, :mae, :r2, :pearson]) -> Dict

根据给定的指标名称计算评估值。支持的指标:
- `:mse`, `:rmse`, `:mae`, `:mape`
- `:r2`, `:pearson`, `:spearman`
- `:auc` (二分类), `:accuracy`, `:balanced_accuracy`
"""
function evaluate_metrics(y_true::AbstractVector, y_pred::AbstractVector;
                          metrics::Vector{Symbol} = [:mse, :rmse, :mae, :r2, :pearson])
    length(y_true) == length(y_pred) || throw(DimensionMismatch("预测与真实值长度不一致"))
    results = Dict{Symbol,Float64}()
    residuals = y_pred .- y_true
    for metric in metrics
        if metric == :mse
            results[:mse] = mean(residuals .^ 2)
        elseif metric == :rmse
            results[:rmse] = sqrt(mean(residuals .^ 2))
        elseif metric == :mae
            results[:mae] = mean(abs.(residuals))
        elseif metric == :mape
            results[:mape] = mean(abs.(residuals ./ (y_true .+ _EPS))) * 100
        elseif metric == :r2
            ss_res = sum((y_true .- y_pred) .^ 2)
            ss_tot = sum((y_true .- mean(y_true)) .^ 2)
            results[:r2] = 1 - ss_res / (ss_tot + _EPS)
        elseif metric == :pearson
            results[:pearson] = cor(y_true, y_pred)
        elseif metric == :spearman
            results[:spearman] = corspearman(y_true, y_pred)
        elseif metric == :auc
            results[:auc] = compute_auc(y_true, y_pred)
        elseif metric == :accuracy
            results[:accuracy] = mean((y_pred .>= 0.5) .== (y_true .>= 0.5))
        elseif metric == :balanced_accuracy
            results[:balanced_accuracy] = balanced_accuracy(y_true, y_pred)
        else
            @warn "未知指标 $(metric), 已跳过"
        end
    end
    return results
end

"""
    compute_auc(y_true, y_score)

计算二分类 ROC AUC。`y_true` 允许为 0/1 或 Bool。
"""
function compute_auc(y_true::AbstractVector, y_score::AbstractVector)
    length(y_true) == length(y_score) || throw(DimensionMismatch("长度不一致"))
    labels = Float64.(y_true .> 0)
    order = sortperm(y_score; rev = true)
    sorted_labels = labels[order]
    pos = sum(sorted_labels)
    neg = length(sorted_labels) - pos
    (pos == 0 || neg == 0) && return 1.0
    tp = 0.0
    fp = 0.0
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_score = typemax(Float64)
    for idx in order
        if y_score[idx] != prev_score
            auc += trapezoid_area(fpr, fpr + fp / max(neg, _EPS), tpr, tpr + tp / max(pos, _EPS))
            fpr += fp / max(neg, _EPS)
            tpr += tp / max(pos, _EPS)
            tp = 0.0
            fp = 0.0
            prev_score = y_score[idx]
        end
        if labels[idx] > 0
            tp += 1
        else
            fp += 1
        end
    end
    auc += trapezoid_area(fpr, fpr + fp / max(neg, _EPS), tpr, tpr + tp / max(pos, _EPS))
    return clamp(auc, 0.0, 1.0)
end

"""
    balanced_accuracy(y_true, y_score)

按类均衡计算准确率, 防止类别不平衡导致的偏差。
"""
function balanced_accuracy(y_true::AbstractVector, y_score::AbstractVector)
    labels = y_true .>= 0.5
    preds = y_score .>= 0.5
    tp = sum(labels .& preds)
    tn = sum(.!labels .& .!preds)
    fp = sum(.!labels .& preds)
    fn = sum(labels .& .!preds)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return (sensitivity + specificity) / 2
end

trapezoid_area(x1, x2, y1, y2) = (x2 - x1) * (y1 + y2) / 2

"""
    cross_validate(model_builder, dataset, k; metric=:rmse, rng=Random.default_rng(), retain_hist=false)

执行 k 折交叉验证。`model_builder` 应返回未训练模型, `metric` 使用 `evaluate_metrics` 计算。
返回包含各折得分、预测结果与可选训练历史的命名元组。
"""
function cross_validate(model_builder::Function, dataset::GenomicDataset, k::Integer;
                        trainer::Function = (model, X, y) -> fit!(model, X, y),
                        predictor::Function = (model, X) -> predict(model, X),
                        metric::Symbol = :rmse,
                        rng::AbstractRNG = Random.default_rng(),
                        retain_hist::Bool = false)
    splits = kfold_split(dataset, k; rng = rng)
    scores = Float64[]
    histories = Any[]
    predictions = Vector{Vector{Float64}}()
    for (train_idx, test_idx) in splits
        model = model_builder()
        trained = trainer(model, dataset.genotype[train_idx, :], dataset.phenotype[train_idx])
        trained_model = trained
        history = nothing
        if trained isa Tuple
            history = trained[1]
            trained_model = trained[end]
        end
        y_pred = predictor(trained_model, dataset.genotype[test_idx, :])
        metric_values = evaluate_metrics(dataset.phenotype[test_idx], y_pred; metrics = [metric])
        push!(scores, metric_values[metric])
        retain_hist && push!(histories, history)
        push!(predictions, y_pred)
    end
    return (metric = metric, scores = scores, predictions = predictions, history = histories)
end

"""
    summarize_cv(cv_result)

对交叉验证结果给出均值、标准差、极值等统计量。
"""
function summarize_cv(cv_result)
    scores = cv_result.scores
    return Dict(
        :metric => cv_result.metric,
        :mean => mean(scores),
        :std => std(scores),
        :min => minimum(scores),
        :max => maximum(scores),
        :nfolds => length(scores)
    )
end

