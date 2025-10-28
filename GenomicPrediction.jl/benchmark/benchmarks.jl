using BenchmarkTools
using GenomicPrediction
using Random

const SUITE = BenchmarkGroup()

function _prepare_dataset(n::Int, p::Int)
    dataset = simulate_genomic_data(n, p; h2 = 0.6, seed = 2024)
    return dataset.genotype, dataset.phenotype
end

X_small, y_small = _prepare_dataset(200, 500)
X_medium, y_medium = _prepare_dataset(400, 800)

SUITE["GBLUP"] = BenchmarkGroup()
SUITE["GBLUP"]["fit_small"] = @benchmarkable begin
    model = GBLUPModel(λ = 1.0)
    fit!(model, $X_small, $y_small)
end

SUITE["GBLUP"]["fit_medium"] = @benchmarkable begin
    model = GBLUPModel(λ = 1.0)
    fit!(model, $X_medium, $y_medium)
end

SUITE["DeepLearning"] = BenchmarkGroup()
mlp_builder = () -> build_mlp(size(X_small, 2), [128, 64])
SUITE["DeepLearning"]["train_mlp"] = @benchmarkable begin
    model = mlp_builder()
    train_deep_model!(model, $X_small, $y_small; epochs = 5, batch_size = 32)
end

SUITE
