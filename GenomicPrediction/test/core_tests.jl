# test/core_tests.jl - CoreAlgorithm 模块单元测试
# ----------------------------------------------------
# ... (header comments) ...

using Test
using DataFrames
using LinearAlgebra
using Statistics
using Random

@testset "CoreAlgorithm.jl - 核心算法模块测试" begin

    # --- GBLUP 模型测试 ---
    @testset "GBLUP 模型" begin
        # ... (GBLUP tests) ...
        G = [1.0 0.0; 0.0 1.0; 1.0 1.0]
        y = [10.0, 11.0, 12.0]
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)
        lambda = 5.0
        μ = mean(y)
        y_centered = y .- μ
        p = [1/3, 1/3]
        Z = G .- (2 .* p')
        LHS = Z' * Z
        for i in 1:size(LHS, 1); LHS[i, i] += lambda; end
        RHS = Z' * y_centered
        expected_effects = LHS \ RHS
        model = GenomicPrediction.GBLUPModel(lambda)
        GenomicPrediction.fit!(model, mock_data)
        @test model.intercept ≈ μ atol=1e-6
        @test model.effects ≈ expected_effects atol=1e-6
        @test model.allele_freqs ≈ p atol=1e-6
        predictions = GenomicPrediction.predict(model, geno_df)
        expected_predictions = μ .+ Z * expected_effects
        @test predictions ≈ expected_predictions atol=1e-6
        new_model = GenomicPrediction.GBLUPModel(1.0)
        @test_throws DimensionMismatch GenomicPrediction.predict(new_model, geno_df)
    end

    # --- BayesA 模型测试 ---
    @testset "BayesA 模型" begin
        # ... (BayesA tests) ...
        Random.seed!(42)
        G = rand([0.0, 1.0, 2.0], 10, 5)
        u_true = [0.5, -0.3, 0.0, 0.2, 0.0]
        y = G * u_true + randn(10) * 0.1
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)
        model = GenomicPrediction.BayesAModel(iterations=200, burnin=50)
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.effects) == 5
        @test isfinite(model.intercept)
        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 10
        @test all(isfinite, predictions)
    end

    # --- 正则化回归模型测试 ---
    @testset "正则化回归 (LASSO, Elastic Net)" begin
        Random.seed!(123)
        G = rand(50, 20)
        y = G[:, 1] * 2.5 - G[:, 5] * 1.5 + randn(50) * 0.5
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        @testset "LASSO 模型" begin
            model = GenomicPrediction.LASSOModel(0.1)
            GenomicPrediction.fit!(model, mock_data)
            @test model.path isa Any # GLMNet.GlmNetPath is not exported

            predictions = GenomicPrediction.predict(model, geno_df)
            @test length(predictions) == 50
            @test all(isfinite, predictions)
        end

        @testset "Elastic Net 模型" begin
            model = GenomicPrediction.ElasticNetModel(0.1, 0.5)
            GenomicPrediction.fit!(model, mock_data)
            @test model.path isa Any

            predictions = GenomicPrediction.predict(model, geno_df)
            @test length(predictions) == 50
            @test all(isfinite, predictions)
        end
    end

end
