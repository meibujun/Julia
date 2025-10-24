# test/dl_tests.jl - DeepLearning 模块单元测试
# -------------------------------------------------
# ... (header comments) ...

using Test
using DataFrames
using Random

@testset "DeepLearning.jl - 深度学习模块测试" begin

    # --- FNNModel 测试 ---
    @testset "FNNModel" begin
        Random.seed!(123)
        G = rand(10, 5)
        y = rand(10)
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)
        input_dim = size(G, 2)
        model = GenomicPrediction.FNNModel(input_dim, epochs=5)
        @test model isa GenomicPrediction.FNNModel
        @test model.epochs == 5
        GenomicPrediction.fit!(model, mock_data)
        @test length(model.history) == 5
        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 10
        @test all(isfinite, predictions)
    end

    # --- CNN 模型测试 ---
    @testset "CNN 模型" begin
        Random.seed!(456)
        # CNNs often work better with more markers
        G = rand(20, 30)
        y = rand(20)
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        input_len = size(G, 2)
        model = GenomicPrediction.CNNModel(input_len, epochs=5)
        @test model isa GenomicPrediction.CNNModel

        GenomicPrediction.fit!(model, mock_data)
        @test length(model.history) == 5

        predictions = GenomicPrediction.predict(model, geno_df)
        @test length(predictions) == 20
        @test all(isfinite, predictions)
    end

    # --- Transformer 模型测试 (占位符) ---
    @testset "Transformer 模型" begin
        @test true
    end

    # --- GPU 兼容性测试 (占位符) ---
    @testset "GPU 兼容性" begin
        @test true
    end

end
