using Test
using Random
using GenomicPrediction

@testset "Evaluation" begin
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.2, 1.9, 2.8]
    metrics = evaluate_metrics(y_true, y_pred; metrics = [:mse, :rmse, :mae, :r2, :pearson])
    @test metrics[:mse] ≥ 0
    @test metrics[:rmse] ≈ sqrt(metrics[:mse]) atol=1e-8
    @test -1 ≤ metrics[:pearson] ≤ 1

    dataset = simulate_genomic_data(40, 10; h2 = 0.5, seed = 99)
    pipeline = default_workflow(dataset)
    cv = cross_validate(() -> GBLUPModel(λ = 0.5), dataset, 3; rng = MersenneTwister(11))
    summary = summarize_cv(cv)
    @test haskey(summary, :mean)
    @test length(cv.scores) == 3

    result = run_autogs(pipeline, dataset; rng = MersenneTwister(10))
    @test size(result.results, 1) == length(pipeline.candidates)
    @test result.best_metadata !== nothing
end
