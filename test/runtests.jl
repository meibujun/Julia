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
G, variants, keep_variants = qc_filter_variants!(G, variants)
G, samples, keep_samples = qc_filter_samples!(G, samples)
phenotypes = phenotypes[keep_samples]

@test size(G, 1) == nrow(samples)
@test size(G, 2) == nrow(variants)

collapsing_res = collapsing_test(G, phenotypes; method = :burden)
@test :mean_statistic in propertynames(collapsing_res)
@test length(list_rvat_methods()) >= 10

bayes_res = bayesian_mvr!(G, phenotypes, nothing; iterations = 120, burnin = 20)
@test nrow(bayes_res) == size(G, 2)
blasso_res = bayesian_blasso!(G[:, 1:40], phenotypes; iterations = 150, burnin = 50)
@test :posterior_mean in propertynames(blasso_res)

rkhs_res = rkhs_epistasis!(G[:, 1:80], phenotypes)
@test rkhs_res.genetic_variance[1] > 0

epi_res = epistasis_scan(G[:, 1:40], phenotypes; max_pairs = 10)
@test nrow(epi_res) > 0
@test length(list_epistasis_methods()) == 4

ebv_res = egblup!(G[:, 1:80], phenotypes)
@test nrow(ebv_res) == size(G, 1)

meta_input = DataFrame(effect = randn(4), variance = rand(4) .+ 0.1, study = 1:4, pvalue = rand(4))
meta_res = meta_analyze(meta_input; model = :robust)
@test meta_res.pvalue[1] >= 0
@test length(list_meta_models()) >= 5

multiomics = simulate_multiomics(n_samples = 120)
integrate_multiomics!(multiomics)
@test haskey(multiomics, :integrated_score)
K = omics_kernel(multiomics[:genomics], multiomics)
@test size(K, 1) == size(K, 2) == size(multiomics[:genomics], 1)

trend = simulate_breeding_pipeline(generations = 2, n_animals = 120)
@test nrow(trend) == 2

plan = optimize_breeding_scheme(multiomics[:phenomics]; trait = :DailyGain)
@test plan.Response[1] > 0

manhattan = manhattan_plot(DataFrame(Chr = [1, 1], Pos = [1, 2], P_value = [0.01, 0.02]))
save_plot("output/test_plot.png", manhattan)
@test isfile("output/test_plot.png")

