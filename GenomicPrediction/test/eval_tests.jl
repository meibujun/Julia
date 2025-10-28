# test/eval_tests.jl

using Test
using DataFrames
using Random
using GenomicPrediction

@testset "Evaluation.jl" begin

    @testset "Metrics" begin
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.8, 4.3, 5.1]

        @test accuracy(y_true, y_pred) isa Float64
        @test isapprox(accuracy(y_true, y_pred), 0.99, atol=0.01)

        @test mse(y_true, y_pred) isa Float64
        @test isapprox(mse(y_true, y_pred), 0.06, atol=0.01)
    end

    @testset "cross_validate" begin
        Random.seed!(123)
        G = rand(0:2, 20, 10)
        y = rand(20)
        geno_df = DataFrame(ID=1:20, G, :auto)
        pheno_df = DataFrame(ID=1:20, y=y)
        mock_data = GenomicData(geno_df, pheno_df)

        gblup_generator() = GBLUPModel(lambda=10.0)

        cv_results = cross_validate(gblup_generator, mock_data, 5, fit!, predict)

        @test cv_results.mean_accuracy isa Float64
        @test length(cv_results.fold_metrics) == 5
    end
end
