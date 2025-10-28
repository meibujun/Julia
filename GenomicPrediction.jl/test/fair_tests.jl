using Test
using GenomicPrediction
using Random

@testset "FAIR Modeling" begin
    metadata = create_metadata("GBLUP"; metrics = Dict("rmse" => 0.5), data_sources = ["simulated"])
    enrich_metadata!(metadata; notes = "unit test")
    model = GBLUPModel()
    dataset = simulate_genomic_data(20, 5; seed = 1)
    fit!(model, dataset)
    tmp = mktempdir()
    path = joinpath(tmp, "model.bson")
    save_model(model, path; metadata = metadata, extra = Dict("info" => "test"))
    loaded_model, loaded_metadata, extra = load_model(path)
    @test loaded_metadata.notes == "unit test"
    ŷ = predict(loaded_model, dataset.genotype)
    @test length(ŷ) == size(dataset.genotype, 1)
    @test extra["info"] == "test"
end
