using Test
using Flux
using GenomicPrediction

@testset "Deep Learning" begin
    dataset = simulate_genomic_data(40, 12; h2 = 0.6, seed = 77)
    X = dataset.genotype
    y = dataset.phenotype

    @testset "MLP" begin
        model = build_mlp(size(X, 2), [32, 16], output_dim = 1)
        history, trained = train_deep_model!(model, X, y; epochs = 3, batch_size = 8, device = :cpu)
        preds = vec(trained(Float32.(permutedims(X))))
        @test length(history.loss) == 3
        @test length(preds) == size(X, 1)
    end

    @testset "CNN" begin
        cnn = build_cnn(1, size(X, 2), [(4, 3, 1)], [16])
        output = cnn(Float32.(permutedims(X)))
        @test size(output, 1) == 1
        @test size(output, 2) == size(X, 1)
    end

    @testset "Transformer" begin
        transformer = build_transformer(size(X, 2), 32, 4, 2)
        output = transformer(Float32.(permutedims(X)))
        @test size(output, 1) == 1
        @test size(output, 2) == size(X, 1)
    end
end
