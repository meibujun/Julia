module MetaAnalysis

using LinearAlgebra
using DataFrames
using Statistics
using Random
using Distributions

"""
    meta_analyze(results; model = :random)

对多研究稀有变异上位性检验结果进行 Meta 分析，支持固定效应与随机效应模型。
输入应包含列 :effect, :variance, :study。
"""
function meta_analyze(results::DataFrame; model::Symbol=:random)
    effects = results.effect
    variances = results.variance
    weights = model == :fixed ? 1 ./ variances : random_effect_weights(effects, variances)
    pooled = sum(weights .* effects) / sum(weights)
    se = sqrt(1 / sum(weights))
    z = pooled / se
    p = 2 * (1 - cdf(Normal(), abs(z)))
    τ2 = model == :random ? estimate_tau2(effects, variances) : 0.0
    return DataFrame(model = model, pooled_effect = pooled, se = se, pvalue = p, tau2 = τ2)
end

"""
    random_effect_weights(effects, variances)

DerSimonian-Laird 随机效应加权策略。
"""
function random_effect_weights(effects, variances)
    τ2 = estimate_tau2(effects, variances)
    return 1 ./ (variances .+ τ2)
end

"""
    estimate_tau2(effects, variances)

DerSimonian-Laird 方差分量估计器。
"""
function estimate_tau2(effects, variances)
    w = 1 ./ variances
    mean_fixed = sum(w .* effects) / sum(w)
    Q = sum(w .* (effects .- mean_fixed).^2)
    df = length(effects) - 1
    c = sum(w) - sum(w.^2) / sum(w)
    return max((Q - df) / c, 0.0)
end

"""
    summarize_results(df)

对模型输出进行格式化展示。
"""
function summarize_results(df::DataFrame)
    return df
end

end # module
