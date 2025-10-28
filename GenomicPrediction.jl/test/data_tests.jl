using Test
using DataFrames
using Random
using GenomicPrediction

@testset "Data Processing" begin
    dataset = simulate_genomic_data(30, 15; h2 = 0.5, seed = 42)
    @test dataset.ids == dataset.sample_ids
    means, stds = standardize_genotypes!(dataset)
    @test length(means) == 15
    @test all(stds .> 0)
    @test haskey(dataset.metadata, "scaling")
    @test dataset.metadata["scaling"]["means"] == means
    @test size(build_grm(dataset)) == (30, 30)

    # introduce missing values
    dataset.genotype[1, 1] = NaN
    dataset.genotype[2, 2] = NaN
    impute_missing!(dataset; method = :mean)
    @test !any(isnan, dataset.genotype)
    @test haskey(dataset.metadata, "imputation")

    dataset, report = quality_control!(dataset; maf_threshold = 0.0, missing_rate = 1.0, return_report = true)
    @test report isa QualityReport
    @test report.retained_markers <= report.initial_markers
    @test haskey(dataset.metadata, "quality_control")
    @test sum(values(report.removal_reasons)) == report.initial_markers

    train_idx, test_idx = make_holdout_split(dataset; test_ratio = 0.2, rng = MersenneTwister(7))
    @test length(train_idx) + length(test_idx) == size(dataset.genotype, 1)

    folds = kfold_split(dataset, 5; rng = MersenneTwister(7))
    @test length(folds) == 5
    seen = reduce(vcat, last.(folds))
    @test length(unique(seen)) == size(dataset.genotype, 1)

    geno_df = DataFrame(id = dataset.ids, snp1 = dataset.genotype[:, 1], snp2 = dataset.genotype[:, 2])
    pheno_df = DataFrame(id = dataset.ids, phenotype = dataset.phenotype)
    merged = merge_genomic_phenotype(geno_df, pheno_df)
    @test size(merged.genotype, 1) == size(geno_df, 1)

    # 强制触发 KNN 回退路径
    fallback_dataset = simulate_genomic_data(12, 8; seed = 7)
    fallback_dataset.genotype[1, 1] = NaN
    impute_missing!(fallback_dataset; method = :knn, max_knn_samples = 5)
    @test fallback_dataset.metadata["imputation"]["fallback"] === true
    @test !any(isnan, fallback_dataset.genotype)
end
