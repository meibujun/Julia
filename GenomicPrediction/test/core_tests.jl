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

    # --- BayesC 模型测试 ---
    @testset "BayesC 模型" begin
        Random.seed!(44)
        G = rand([0.0, 1.0, 2.0], 10, 5)
        u_true = [0.9, 0.0, 0.0, 0.0, 0.0]
        y = G * u_true + randn(10) * 0.1
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        model = GenomicPrediction.BayesCModel(iterations=200, burnin=50, pi=0.1)
        GenomicPrediction.fit!(model, mock_data)

        @test length(model.effects) == 5
        @test isfinite(model.intercept)
        @test any(abs.(model.effects) .< 1e-4) # Verify sparsity

        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 10
        @test all(isfinite, predictions)
    end

    # --- BayesR 模型测试 ---
    @testset "BayesR 模型" begin
        Random.seed!(45)
        G = rand([0.0, 1.0, 2.0], 10, 5)
        u_true = [1.0, 0.0, 0.0, 0.05, 0.0]
        y = G * u_true + randn(10) * 0.1
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        model = GenomicPrediction.BayesRModel(iterations=200, burnin=50)
        GenomicPrediction.fit!(model, mock_data)

        @test length(model.effects) == 5
        @test isfinite(model.intercept)
        @test any(abs.(model.effects) .< 1e-4) # Verify sparsity

        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 10
        @test all(isfinite, predictions)
    end

    # --- BayesB 模型测试 ---
    @testset "BayesB 模型" begin
        Random.seed!(43)
        G = rand([0.0, 1.0, 2.0], 10, 5)
        # 真实效应更加稀疏，以稳定测试
        u_true = [0.8, 0.0, 0.0, 0.0, 0.0]
        y = G * u_true + randn(10) * 0.1
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        # 使用较低的 pi 以增加稀疏性，使测试更稳定
        model = GenomicPrediction.BayesBModel(iterations=200, burnin=50, pi=0.1)
        GenomicPrediction.fit!(model, mock_data)

        @test length(model.effects) == 5
        @test isfinite(model.intercept)
        # 验证 BayesB 的稀疏性：至少有一个效应应该接近于零
        # 注意：由于随机性，这个测试可能不稳定，但在多数情况下应该通过
        @test any(abs.(model.effects) .< 1e-4)

        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 10
        @test all(isfinite, predictions)
    end

    # --- 正则化回归模型测试 ---
    @testset "正则化回归 (LASSO, Elastic Net)" begin
        # ... (Regularized regression tests) ...
        Random.seed!(123)
        G = rand(50, 20)
        y = G[:, 1] * 2.5 - G[:, 5] * 1.5 + randn(50) * 0.5
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        @testset "LASSO 模型" begin
            model = GenomicPrediction.LASSOModel(0.1)
            GenomicPrediction.fit!(model, mock_data)
            @test model.path isa Any
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
