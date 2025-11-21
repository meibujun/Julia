using Test
using GenomicPro

@testset "Configuration" begin
    config = read_pipeline_config("test/config/test_pipeline.json")
    @test length(config.dataset_configs) == 2
    @test config.integration_config.strategy == :concatenate
    @test config.analysis_config.cluster_count == 2
end

@testset "Dataset loading" begin
    config = read_pipeline_config("test/config/test_pipeline.json")
    ds = load_omics_dataset(config.dataset_configs[1])
    @test size(ds.matrix, 1) == 4
    @test size(ds.matrix, 2) == 3
    @test ds.features[1] == "Gene1"
end

@testset "Preprocessing" begin
    config = read_pipeline_config("test/config/test_pipeline.json")
    ds = load_omics_dataset(config.dataset_configs[1])
    impute_missing!(ds, :mean)
    @test !any(isnan, ds.matrix)
    normalize!(ds, :zscore)
    @test isapprox(mean(ds.matrix[1, :]), 0.0; atol=1e-8)
end

@testset "Integration and analysis" begin
    config = read_pipeline_config("test/config/test_pipeline.json")
    result = run_pipeline(config)
    @test length(result.integrated.samples) == 3
    @test size(result.integrated.matrix, 1) == 8
    @test result.analysis.pca !== nothing
    @test result.analysis.clustering !== nothing
    @test length(result.analysis.clustering.assignments) == 3
    @test haskey(result.report, :datasets)
    @test haskey(result.report, :analysis)
end
