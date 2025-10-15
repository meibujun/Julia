module MetaAnalysis

using LinearAlgebra
using DataFrames
using Statistics
using Random
using Distributions

"""
    list_meta_models()

列出支持的 5 种 Meta 分析框架，涵盖固定效应、随机效应、贝叶斯层级、
ACAT 组合与稳健加权模型，满足跨试验整合需求。
"""
list_meta_models() = (:fixed, :random, :bayesian, :acat, :robust)

"""
    meta_analyze(results; model = :random)

对多研究稀有变异上位性检验结果进行 Meta 分析。
输入表需包含 :effect, :variance, :study, :pvalue 等列。
"""
function meta_analyze(results::DataFrame; model::Symbol = :random)
    model in list_meta_models() || error("不支持的 Meta 模型: $model")
    if model === :fixed
        return fixed_effect(results)
    elseif model === :random
        return random_effect(results)
    elseif model === :bayesian
        return bayesian_hierarchical(results)
    elseif model === :acat
        return acat_meta(results)
    elseif model === :robust
        return robust_meta(results)
    end
end

"""
    summarize_results(df)

对 Meta 分析输出进行格式化，保留主要统计量。
"""
function summarize_results(df::DataFrame)
    return df
end

# ------------------------ 具体模型 ------------------------

function fixed_effect(results::DataFrame)
    w = 1 ./ results.variance
    pooled = sum(w .* results.effect) / sum(w)
    se = sqrt(1 / sum(w))
    z = pooled / se
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return DataFrame(model = :fixed, pooled_effect = pooled, se = se, z = z, pvalue = p, tau2 = 0.0)
end

function estimate_tau2(effects, variances)
    w = 1 ./ variances
    mean_fixed = sum(w .* effects) / sum(w)
    Q = sum(w .* (effects .- mean_fixed).^2)
    df = length(effects) - 1
    c = sum(w) - sum(w.^2) / sum(w)
    return max((Q - df) / c, 0.0)
end

function random_effect(results::DataFrame)
    τ2 = estimate_tau2(results.effect, results.variance)
    w = 1 ./ (results.variance .+ τ2)
    pooled = sum(w .* results.effect) / sum(w)
    se = sqrt(1 / sum(w))
    z = pooled / se
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return DataFrame(model = :random, pooled_effect = pooled, se = se, z = z, pvalue = p, tau2 = τ2)
end

function bayesian_hierarchical(results::DataFrame)
    μ0 = mean(results.effect)
    τ0 = std(results.effect)
    post_mean = Float64[]
    post_sd = Float64[]
    for i in 1:nrow(results)
        vi = results.variance[i]
        wi = 1 / (vi + τ0^2 + eps())
        μi = (wi * results.effect[i] + μ0 / τ0^2) / (wi + 1 / τ0^2)
        σi = sqrt(1 / (wi + 1 / τ0^2))
        push!(post_mean, μi)
        push!(post_sd, σi)
    end
    pooled = mean(post_mean)
    se = sqrt(mean(post_sd.^2))
    z = pooled / (se + eps())
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return DataFrame(model = :bayesian, pooled_effect = pooled, se = se, z = z, pvalue = p, tau2 = mean(post_sd .^ 2))
end

function acat_meta(results::DataFrame)
    ps = hasproperty(results, :pvalue) ? results.pvalue : @. 2 * (1 - cdf(Normal(), abs(results.effect) / sqrt(results.variance + eps())))
    transformed = tan.((0.5 .- ps) .* π)
    stat = mean(transformed)
    p = 0.5 - atan(stat) / π
    return DataFrame(model = :acat, pooled_effect = mean(results.effect), se = std(results.effect), z = stat, pvalue = clamp(p, 0.0, 1.0), tau2 = var(results.effect))
end

function robust_meta(results::DataFrame)
    med = median(results.effect)
    mad = median(abs.(results.effect .- med)) + eps()
    weights = 1 ./ (abs.(results.effect .- med) ./ mad .+ 1)
    pooled = sum(weights .* results.effect) / sum(weights)
    se = sqrt(sum(weights .^ 2 .* results.variance) / (sum(weights)^2 + eps()))
    z = pooled / (se + eps())
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return DataFrame(model = :robust, pooled_effect = pooled, se = se, z = z, pvalue = p, tau2 = var(results.effect))
end

end # module
