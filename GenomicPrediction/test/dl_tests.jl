# test/dl_tests.jl
# ==========================================================
# Unit tests for DeepLearning.jl module.
#
# This file has been updated to enable all model training tests.
# Epochs and model complexity are kept to a minimum to ensure tests run quickly.
# Assertions have been added to verify that training actually occurs.
# ==========================================================

using Test
using DataFrames
using Random
using GenomicPrediction

@testset "DeepLearning.jl" begin

    # --- Test Data Setup ---
    Random.seed!(42)
    # Use a slightly larger, more realistic dataset for stability
    n_train = 50
    n_markers = 10
    n_test = 5

    G_train = rand(0:2, n_train, n_markers)
    y_train = rand(Float32, n_train) * 5
    geno_df_train = DataFrame(hcat(1:n_train, G_train), :auto)
    pheno_df_train = DataFrame(ID=1:n_train, y=y_train)
    mock_data = GenomicData(geno_df_train, pheno_df_train, nothing, nothing)

    G_test = rand(0:2, n_test, n_markers)
    new_geno_df = DataFrame(hcat((n_train+1):(n_train+n_test), G_test), :auto)

    input_dim = n_markers

    @testset "FNNModel" begin
        # Test with minimal parameters for speed
        model = FNNModel(input_dim, epochs=2, hidden_layers=[8, 4])
        @test model isa FNNModel

        # Enable the training step
        GenomicPrediction.fit!(model, mock_data)

        # Verify training happened
        @test !isempty(model.history)
        @test length(model.history) == 2
        @test model.history[1] > model.history[2] # Loss should decrease

        # Test prediction
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == n_test
        @test all(isfinite, preds)
    end

    @testset "CNNModel" begin
        model = CNNModel(input_dim, epochs=2)
        @test model isa CNNModel

        # Enable the training step
        GenomicPrediction.fit!(model, mock_data)

        @test !isempty(model.history)
        @test length(model.history) == 2
        @test model.history[1] > model.history[2]

        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == n_test
        @test all(isfinite, preds)
    end

    @testset "TransformerModel" begin
        # Use minimal parameters: 1 layer, 1 head, small model dimension
        model = TransformerModel(input_dim, epochs=2, d_model=4, n_head=1, n_layers=1)
        @test model isa TransformerModel

        GenomicPrediction.fit!(model, mock_data)

        @test !isempty(model.history)
        @test length(model.history) == 2

        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == n_test
        @test all(isfinite, preds)
    end

    @testset "GNNModel" begin
        # GNN test on a smaller subset to speed up GRM calculation
        G_gnn = rand(0:2, 20, 5)
        y_gnn = rand(Float32, 20)
        geno_df_gnn = DataFrame(hcat(1:20, G_gnn), :auto)
        pheno_df_gnn = DataFrame(ID=1:20, y=y_gnn)
        mock_data_gnn = GenomicData(geno_df_gnn, pheno_df_gnn, nothing, nothing)

        model = GNNModel(5, epochs=2, gcn_dims=[8])
        @test model isa GNNModel

        GenomicPrediction.fit!(model, mock_data_gnn)

        @test !isempty(model.history)
        @test length(model.history) == 2

        # GNN predict returns predictions for all nodes in the training graph
        preds = GenomicPrediction.predict(model, new_geno_df)
        @test length(preds) == 20 # Corrected assertion
        @test all(isfinite, preds)
    end

end
