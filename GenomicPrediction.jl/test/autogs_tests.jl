using Test
using Random
using GenomicPrediction

@testset "AutoGS" begin
    dataset = simulate_genomic_data(30, 8; seed = 55)
    pipeline = default_workflow(dataset)
    run_autogs(pipeline, dataset; rng = MersenneTwister(3))
    @test pipeline.best_model !== nothing
    @test size(pipeline.results, 1) == length(pipeline.candidates)
    @test pipeline.best_metadata !== nothing
end
