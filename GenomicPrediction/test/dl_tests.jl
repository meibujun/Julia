# test/dl_tests.jl
# ==========================================================
# 本文件已更新，以包含对 GPU 加速功能的条件化测试。
# 测试将自动检测 CUDA GPU 的可用性，并仅在可用时运行 GPU 相关测试。
# ==========================================================

using Test
using DataFrames
using Random
using GenomicPrediction
using CUDA
using Flux

# --- 辅助函数：检查是否可以运行 GPU 测试 ---
can_run_gpu_tests() = CUDA.functional()

@testset "DeepLearning.jl" begin

    # --- Test Data Setup ---
    Random.seed!(42)
    n_train = 50; n_markers = 10; n_test = 5
    G_train = rand(0:2, n_train, n_markers)
    y_train = rand(Float32, n_train) * 5
    geno_df_train = DataFrame(hcat(1:n_train, G_train), :auto)
    pheno_df_train = DataFrame(ID=1:n_train, y=y_train)
    mock_data = GenomicData(geno_df_train, pheno_df_train, nothing, nothing)
    G_test = rand(0:2, n_test, n_markers)
    new_geno_df = DataFrame(hcat((n_train+1):(n_train+n_test), G_test), :auto)
    input_dim = n_markers

    @testset "CPU Execution" begin
        @testset "FNNModel" begin
            model = FNNModel(input_dim, epochs=2, hidden_layers=[8, 4])
            GenomicPrediction.fit!(model, mock_data)
            @test !isempty(model.history) && length(model.history) == 2
            preds = GenomicPrediction.predict(model, new_geno_df)
            @test length(preds) == n_test && all(isfinite, preds)
        end

        @testset "CNNModel" begin
            model = CNNModel(input_dim, epochs=2)
            GenomicPrediction.fit!(model, mock_data)
            @test !isempty(model.history) && length(model.history) == 2
            preds = GenomicPrediction.predict(model, new_geno_df)
            @test length(preds) == n_test && all(isfinite, preds)
        end

        @testset "TransformerModel" begin
            model = TransformerModel(input_dim, epochs=2, d_model=4, n_head=1, n_layers=1)
            GenomicPrediction.fit!(model, mock_data)
            @test !isempty(model.history) && length(model.history) == 2
            preds = GenomicPrediction.predict(model, new_geno_df)
            @test length(preds) == n_test && all(isfinite, preds)
        end

        @testset "GNNModel" begin
            G_gnn = rand(0:2, 20, 5); y_gnn = rand(Float32, 20)
            geno_df_gnn = DataFrame(hcat(1:20, G_gnn), :auto)
            pheno_df_gnn = DataFrame(ID=1:20, y=y_gnn)
            mock_data_gnn = GenomicData(geno_df_gnn, pheno_df_gnn, nothing, nothing)
            model = GNNModel(5, epochs=2, gcn_dims=[8])
            GenomicPrediction.fit!(model, mock_data_gnn)
            @test !isempty(model.history) && length(model.history) == 2
            preds = GenomicPrediction.predict(model, new_geno_df)
            @test length(preds) == 20 && all(isfinite, preds)
        end
    end

    @testset "GPU Execution" begin
        if can_run_gpu_tests()
            println("\n[GPU Tests] 检测到 CUDA GPU，正在运行 GPU 测试...")

            @testset "FNNModel on GPU" begin
                model = FNNModel(input_dim, epochs=2, hidden_layers=[8, 4])
                # fit! 函数现在应该会自动使用 GPU
                GenomicPrediction.fit!(model, mock_data)

                # 验证模型已被移至 GPU
                @test model.chain isa Flux.Chain
                @test first(model.chain.layers).weight isa CuArray

                @test !isempty(model.history)

                # 验证预测（输入 CPU 数据，应自动处理）
                preds = GenomicPrediction.predict(model, new_geno_df)
                @test preds isa Vector{Float32}
                @test length(preds) == n_test
            end

        else
            @warn "\n[GPU Tests] 未检测到功能正常的 CUDA GPU，跳过 GPU 测试。"
            # 创建一个假的测试集以避免 Test 报错
            @test true
        end
    end

end
