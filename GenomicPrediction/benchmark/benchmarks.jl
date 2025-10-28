# benchmark/benchmarks.jl
#
# Performance benchmarks for the GenomicPrediction.jl package.

using BenchmarkTools
using GenomicPrediction
using DataFrames
using Random

const SUITE = BenchmarkGroup()

SUITE["DataProcessing"] = BenchmarkGroup()
SUITE["CoreAlgorithm"] = BenchmarkGroup()

function generate_benchmark_data(n_individuals::Int, n_markers::Int)
    Random.seed!(42)
    G = rand(0:2, n_individuals, n_markers)
    y = randn(n_individuals)

    geno_df = DataFrame(G, :auto)
    insertcols!(geno_df, 1, :ID => 1:n_individuals)

    pheno_df = DataFrame(ID = 1:n_individuals, y = y)

    return GenomicData(geno_df, pheno_df)
end

const data_small = generate_benchmark_data(100, 500)
const data_medium = generate_benchmark_data(500, 2000)

# Benchmarks for DataProcessing
grm_group = SUITE["DataProcessing"]
grm_group["small"] = @benchmarkable calculate_grm($(Matrix(data_small.genotypes[!, 2:end])))
grm_group["medium"] = @benchmarkable calculate_grm($(Matrix(data_medium.genotypes[!, 2:end])))

# Benchmarks for CoreAlgorithm
SUITE["CoreAlgorithm"]["GBLUP_small"] = @benchmarkable fit!(model, $data_small) setup=(model=GBLUPModel(lambda=50.0))
SUITE["CoreAlgorithm"]["GBLUP_medium"] = @benchmarkable fit!(model, $data_medium) setup=(model=GBLUPModel(lambda=50.0))
SUITE["CoreAlgorithm"]["BayesA_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesAModel(iterations=100))

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running benchmark suite for GenomicPrediction.jl...")
    results = run(SUITE, verbose = true)
    show(results)
    println()
end
