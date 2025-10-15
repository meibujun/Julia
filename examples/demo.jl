#!/usr/bin/env julia

using RareEpistasisMeta
using Random
using DataFrames
using Dates

Random.seed!(2025)

G = simulate_genotype_matrix(500, 1000; rare_rate = 0.02)
β = randn(1000) .* 0.05
phenotypes = simulate_phenotypes(G, β, 0.6)

variants = DataFrame(ID = 1:size(G, 2))
samples = DataFrame(ID = 1:size(G, 1))
G, variants, _ = qc_filter_variants!(G, variants; maf_threshold = 0.005)
G, samples, _ = qc_filter_samples!(G, samples)

collapsing_res = collapsing_test(G, phenotypes; method = :skat)
println("折叠法结果:")
println(collapsing_res)

bayes_res = bayesian_mvr!(G, phenotypes, nothing; iterations = 1000, burnin = 200)
println("贝叶斯多元回归前 5 行:")
println(first(bayes_res, 5))

rkhs_res = rkhs_epistasis!(G, phenotypes; kernel = :rbf)
println("RKHS 方差成分估计:")
println(rkhs_res)

Egblup_res = egblup!(G, phenotypes; interaction_order = 3)
println("EG-BLUP 个体效应前 5 行:")
println(first(Egblup_res, 5))

meta_input = DataFrame(effect = randn(5) .* 0.1, variance = rand(5) .* 0.02 .+ 0.01, study = 1:5)
meta_res = meta_analyze(meta_input)
println("Meta 分析结果:")
println(meta_res)

save_report("analysis_report.md", Dict("date" => string(Dates.now()), "n_samples" => size(G, 1)),
            Dict(:collapsing => collapsing_res, :meta => meta_res))
