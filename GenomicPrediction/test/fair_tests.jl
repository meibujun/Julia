# test/fair_tests.jl

using Test
using DataFrames
using Random
using GenomicPrediction
using Dates

@testset "FAIRModeling.jl" begin

    Random.seed!(123)
    G = rand(0:2, 10, 5)
    y = rand(10)
    geno_df = DataFrame(ID=1:10, G, :auto)
    pheno_df = DataFrame(ID=1:10, y=y)
    mock_data = GenomicData(geno_df, pheno_df)

    model = GBLUPModel(lambda=25.0)
    fit!(model, mock_data)

    temp_path = mktemp()[1]

    @testset "save_model and load_model" begin
        save_model(model, temp_path)
        @test isfile(temp_path)

        loaded_model = load_model(temp_path)
        @test loaded_model isa GBLUPModel
        @test loaded_model.lambda == model.lambda
        @test loaded_model.effects ≈ model.effects
    end

    @testset "view_model_metadata" begin
        metadata = view_model_metadata(temp_path)
        @test metadata isa Dict
        @test metadata[:model_type] == "GBLUPModel"
        @test metadata[:save_timestamp] isa DateTime
        @test isnothing(metadata[:package_version]) || metadata[:package_version] isa String
    end

    rm(temp_path)
end
