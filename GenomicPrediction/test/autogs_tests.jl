# test/autogs_tests.jl
# ==========================================================
# Unit tests for AutoGS.jl module.
#
# This file has been updated to enable the bayesian_optimization test
# with minimal iterations to ensure it completes quickly.
# The API calls have been updated to reflect the refactored AutoGS module.
# ==========================================================

using Test
using DataFrames
using Random
using GenomicPrediction
using Hyperopt

@testset "AutoGS.jl" begin

    Random.seed!(42)
    # Use a small but sufficient dataset
    G = rand(0:2, 30, 10)
    y = rand(30)
    geno_df = DataFrame(hcat(1:30, G), :auto)
    pheno_df = DataFrame(ID=1:30, y=y)
    mock_data = GenomicData(geno_df, pheno_df, nothing, nothing)

    @testset "grid_search" begin
        # Define a constructor that takes a dictionary of parameters
        model_constructor(params) = GBLUPModel(params[:lambda])
        hyperparameters = Dict(:lambda => [1.0, 10.0, 20.0])

        # Call the refactored grid_search function
        search_result = grid_search(model_constructor, mock_data, hyperparameters; k=3)

        @test search_result.best_params isa Dict
        @test haskey(search_result.best_params, :lambda)
        @test length(search_result.results) == 3
        @test search_result.best_score > -Inf
    end

    @testset "bayesian_optimization" begin
        model_constructor(params) = GBLUPModel(params[:lambda])
        # Define the search space for Hyperopt
        search_space = Dict(
            :lambda => Hyperopt.hp.loguniform("lambda", log(0.1), log(100.0))
        )

        # Enable the test with a very small number of iterations
        opt_result = bayesian_optimization(model_constructor, mock_data, search_space; k=2, max_iters=2)

        @test opt_result.best_params isa Dict
        @test haskey(opt_result.best_params, :lambda)
        @test 0.1 <= opt_result.best_params[:lambda] <= 100.0
        @test opt_result.best_score > -Inf
    end
end
