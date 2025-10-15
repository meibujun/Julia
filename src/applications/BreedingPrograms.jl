module BreedingPrograms

using Random
using Statistics
using DataFrames
using LinearAlgebra
using Distributions
using ..Integration: simulate_multiomics, integrate_multiomics!, omics_kernel
using ..EGBLUP: egblup!

"""
    simulate_breeding_pipeline(; generations = 5, n_animals = 400, selection_rate = 0.2, seed = 2025)

模拟肉羊/肉牛育种流程，通过多组学整合、EG-BLUP 预测与迭代选择估计遗传进展。
返回每一代的平均育种值、真实育种值与整合得分。
"""
function simulate_breeding_pipeline(; generations::Int = 5, n_animals::Int = 400,
                                     selection_rate::Float64 = 0.2, seed::Int = 2025)
    rng = MersenneTwister(seed)
    data = simulate_multiomics(n_samples = n_animals, seed = seed)
    integrate_multiomics!(data; method = :zscore)
    history = DataFrame(Generation = Int[], MeanGEBV = Float64[], MeanTBV = Float64[], IntegratedScore = Float64[])
    geno = data[:genomics]
    y = data[:phenomics].DailyGain
    tbv = data[:tbv]
    for gen in 1:generations
        gebv = egblup!(geno, y; interaction_order = 3)
        mean_ghat = mean(gebv.ghat)
        push!(history, (gen, mean_ghat, mean(tbv), mean(data[:integrated_score][:, end])))
        n_select = max(2, round(Int, selection_rate * length(y)))
        idx = partialsortperm(gebv.ghat, 1:n_select; rev = true)
        geno = geno[idx, :]
        y = y[idx] .+ 0.2 * randn(rng, n_select)
        tbv = tbv[idx] .+ 0.1 * randn(rng, n_select)
        data[:integrated_score] = data[:integrated_score][idx, :]
    end
    return history
end

"""
    optimize_breeding_scheme(phenos; trait = :DailyGain, heritability = 0.35,
                              selection_rates = 0.1:0.1:0.5, mating_ratios = 0.2:0.1:0.5)

基于经典遗传响应公式与经济权重，搜索最佳的育种方案组合，返回候选方案排序。
"""
function optimize_breeding_scheme(phenos::DataFrame; trait::Symbol = :DailyGain,
                                  heritability::Float64 = 0.35,
                                  selection_rates = 0.1:0.1:0.5,
                                  mating_ratios = 0.2:0.1:0.5)
    σp = std(phenos[!, trait])
    μ = mean(phenos[!, trait])
    results = DataFrame(SelectionRate = Float64[], MatingRatio = Float64[], ExpectedGain = Float64[], Response = Float64[])
    for s in selection_rates
        i = selection_intensity(s)
        for m in mating_ratios
            response = i * sqrt(heritability) * σp * m
            expected = μ + response
            push!(results, (s, m, expected, response))
        end
    end
    sort!(results, :ExpectedGain, rev = true)
    return results
end

function selection_intensity(rate::Float64)
    rate = clamp(rate, 1e-6, 0.5)
    return quantile(Normal(), 1 - rate)
end

end # module
