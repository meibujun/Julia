using Test
using GenomicPrediction

@testset "Core Algorithms" begin
    dataset = simulate_genomic_data(60, 25; h2 = 0.7, seed = 123)
    X = dataset.genotype
    y = dataset.phenotype

    @testset "GBLUP" begin
        model = GBLUPModel(λ = 0.5)
        fit!(model, X, y)
        ŷ = predict(model, X)
        @test model.fitted
        @test length(ŷ) == length(y)
    end

    @testset "Ridge" begin
        model = RidgeRegressionModel(λ = 0.5)
        fit!(model, X, y)
        ŷ = predict(model, X)
        @test model.fitted
        @test length(ŷ) == length(y)
    end

    @testset "LASSO" begin
        model = LassoModel(λ = 0.1)
        fit!(model, X, y; maxiter = 200)
        ŷ = predict(model, X)
        @test model.fitted
        @test length(ŷ) == length(y)
    end

    @testset "ElasticNet" begin
        model = ElasticNetModel(λ1 = 0.1, λ2 = 0.1)
        fit!(model, X, y; maxiter = 200)
        ŷ = predict(model, X)
        @test model.fitted
        @test length(ŷ) == length(y)
    end

    @testset "Bayesian Models" begin
        model_a = BayesAModel(ν = 4.0, S = 1.0, iterations = 400, burn_in = 80, thinning = 5)
        fit!(model_a, X, y)
        ŷa = predict(model_a, X)
        @test model_a.fitted
        @test length(ŷa) == length(y)

        model_b = BayesBModel(π = 0.3, ν = 4.0, S = 1.0, iterations = 400, burn_in = 80, thinning = 5)
        fit!(model_b, X, y)
        ŷb = predict(model_b, X)
        @test model_b.fitted
        @test length(ŷb) == length(y)
    end
end
