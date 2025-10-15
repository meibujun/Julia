using Test
using RareEpistasisMeta
using Random
using DataFrames

Random.seed!(1234)
G = simulate_genotype_matrix(100, 200)
β = randn(200) .* 0.1
phenotypes = simulate_phenotypes(G, β, 0.5)
variants = DataFrame(ID = 1:size(G, 2))
samples = DataFrame(ID = 1:size(G, 1))
G, variants, _ = qc_filter_variants!(G, variants)
G, samples, _ = qc_filter_samples!(G, samples)

@test size(G, 1) == nrow(samples)
@test size(G, 2) == nrow(variants)

collapsing_res = collapsing_test(G, phenotypes)
@test :mean_statistic in propertynames(collapsing_res)

bayes_res = bayesian_mvr!(G, phenotypes, nothing; iterations = 100, burnin = 20)
@test nrow(bayes_res) == size(G, 2)

rkhs_res = rkhs_epistasis!(G, phenotypes)
@test rkhs_res.genetic_variance[1] > 0

egblup_res = egblup!(G, phenotypes)
@test nrow(egblup_res) == size(G, 1)

meta_input = DataFrame(effect = randn(4), variance = rand(4) .+ 0.1, study = 1:4)
meta_res = meta_analyze(meta_input)
@test meta_res.pvalue[1] >= 0
