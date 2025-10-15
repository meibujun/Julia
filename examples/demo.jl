#!/usr/bin/env julia

using RareEpistasisMeta
using Random
using DataFrames
using Dates

Random.seed!(2025)

println("=== 多组学模拟与预处理 ===")
multiomics = simulate_multiomics(n_samples = 300)
integrate_multiomics!(multiomics; method = :zscore)
G = multiomics[:genomics]
y = multiomics[:phenomics].DailyGain

variants = DataFrame(ID = 1:size(G, 2))
samples = DataFrame(ID = multiomics[:phenomics].Animal)
G, variants, keep_variants = qc_filter_variants!(G, variants; maf_threshold = 0.005)
G, samples, keep_samples = qc_filter_samples!(G, samples)
y = y[keep_samples]

println("=== 稀有变异检验（12 种方法接口示例） ===")
rvat_res = collapsing_test(G, y; method = :acatv)
println(rvat_res)
println("可用稀有变异方法: ", list_rvat_methods())

println("=== 贝叶斯分析（3 种贝叶斯框架） ===")
bayes_mvr = bayesian_mvr!(G, y, nothing; iterations = 800, burnin = 200)
println(first(bayes_mvr, 3))
blasso = bayesian_blasso!(G[:, 1:50], y; iterations = 600, burnin = 200)
println(first(blasso, 3))
bayesb = bayesian_bayesb!(G[:, 1:50], y; iterations = 600, burnin = 200)
println(first(bayesb, 3))

println("=== 上位性扫描（4 种检测策略） ===")
epi_res = epistasis_scan(G[:, 1:120], y; max_pairs = 50)
println(first(epi_res, 5))
println("支持的上位性方法: ", list_epistasis_methods())

println("=== RKHS / EG-BLUP 建模 ===")
rkhs_res = rkhs_epistasis!(G[:, 1:200], y; kernel = :rbf)
println(rkhs_res)
K_multi = omics_kernel(G, multiomics)
ebv = egblup!(G[:, 1:200], y; interaction_order = 3)
println(first(ebv, 5))

println("=== Meta 分析与报告 ===")
meta_input = DataFrame(effect = randn(6) .* 0.1, variance = rand(6) .* 0.02 .+ 0.01, study = 1:6)
meta_res = meta_analyze(meta_input; model = :bayesian)
println(meta_res)
println("可用 Meta 模型: ", list_meta_models())

println("=== 育种方案模拟与优化 ===")
trend = simulate_breeding_pipeline(generations = 3, n_animals = 200)
println(trend)
plan = optimize_breeding_scheme(multiomics[:phenomics]; trait = :DailyGain)
println(first(plan, 5))

println("=== 可视化示例 ===")
manhattan = manhattan_plot(DataFrame(Chr = repeat(1:5, inner = 20), Pos = collect(1:100) .* 1_000, P_value = rand(100)))
qq = qq_plot(rand(100))
save_plot("output/manhattan.png", manhattan)
save_plot("output/qq.png", qq)

println("=== LLM 辅助（需要配置 API Key） ===")
# configure_llm!(api_key = "sk-xxxxx")
# report_text = llm_explain_results(rvat_res, epi_res)
# println(report_text)

save_report("output/analysis_report.md",
    Dict("date" => string(Dates.now()), "n_samples" => size(G, 1), "methods" => list_rvat_methods()),
    Dict(:rvat => rvat_res, :meta => meta_res, :trend => trend))
