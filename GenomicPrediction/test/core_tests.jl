# test/core_tests.jl
# ==========================================================
# Unit tests for CoreAlgorithm.jl module.
#
# This file has been updated to include tests for the newly implemented
# LASSOModel and ElasticNetModel.
# ==========================================================

using Test
using DataFrames
using Random
using .GenomicPrediction

@testset "CoreAlgorithm.jl" begin

    # --- Test Data Setup ---
    Random.seed!(42)
    G = rand(0:2, 20, 10)
    y = rand(20)
    geno_df = DataFrame(hcat(1:20, G), :auto)
    pheno_df = DataFrame(ID=1:20, y=y)
    # The `nothing` arguments are for covariates and pedigree, which are not used here
    mock_data = GenomicData(geno_df, pheno_df, nothing, nothing)
    new_geno_df = DataFrame(hcat(21:22, rand(0:2, 2, 10)), :auto)
    n_markers = 10

    @testset "GBLUPModel" begin
        model = GBLUPModel(lambda=10.0)
        @test model isa GBLUPModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    # Use minimal iterations for Bayesian models to speed up tests
    @testset "BayesAModel" begin
        model = BayesAModel(iterations=20, burn_in=10, thin=2)
        @test model isa BayesAModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        @test length(model.beta_samples) == (20-10)÷2
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    @testset "BayesBModel" begin
        model = BayesBModel(pi=0.9, iterations=20, burn_in=10, thin=2)
        @test model isa BayesBModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    @testset "BayesCModel" begin
        model = BayesCModel(pi=0.9, iterations=20, burn_in=10, thin=2)
        @test model isa BayesCModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    @testset "BayesRModel" begin
        model = BayesRModel(iterations=20, burn_in=10, thin=2)
        @test model isa BayesRModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    @testset "LASSOModel" begin
        model = LASSOModel(lambda=0.1, max_iters=10)
        @test model isa LASSOModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        # LASSO should produce sparse effects
        @test sum(model.effects .== 0) > 0
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

    @testset "ElasticNetModel" begin
        model = ElasticNetModel(lambda=0.1, alpha=0.5, max_iters=10)
        @test model isa ElasticNetModel
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == n_markers
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 2
    end

end
