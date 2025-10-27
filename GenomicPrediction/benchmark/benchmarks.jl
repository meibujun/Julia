# benchmark/benchmarks.jl
#
# Performance benchmarks for the GenomicPrediction.jl package.
#
# This script uses BenchmarkTools.jl to define a suite of benchmarks for
# performance-critical functions. It can be run using PkgBenchmark.jl to
# compare performance between different versions of the code.
#
# To run this benchmark suite manually:
#
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# include("benchmark/benchmarks.jl")

using BenchmarkTools
using GenomicPrediction
using DataFrames
using Random

# --- Benchmark Suite Setup ---

const SUITE = BenchmarkGroup()

SUITE["DataProcessing"] = BenchmarkGroup(["GRM"])
SUITE["CoreAlgorithm"] = BenchmarkGroup(["GBLUP"])

# --- Data Generation for Benchmarks ---

function generate_benchmark_data(n_individuals::Int, n_markers::Int)
    Random.seed!(42)
    G = rand(0:2, n_individuals, n_markers)
    y = randn(n_individuals)

    geno_df = DataFrame(G, :auto)
    # Add an ID column, which is expected by the data structures
    insertcols!(geno_df, 1, :ID => 1:n_individuals)

    pheno_df = DataFrame(ID = 1:n_individuals, y = y)

    return GenomicData(geno_df, pheno_df)
end

# Create datasets of different sizes
const data_small = generate_benchmark_data(100, 500)
const data_medium = generate_benchmark_data(500, 2000)

# --- Benchmark Definitions ---

# 1. Benchmarks for DataProcessing module
# We focus on the most computationally intensive function: calculate_grm
grm_group = SUITE["DataProcessing"]
grm_group["small"] = @benchmarkable calculate_grm($(Matrix(data_small.genotypes[!, 2:end])))
grm_group["medium"] = @benchmarkable calculate_grm($(Matrix(data_medium.genotypes[!, 2:end])))

# 2. Benchmarks for CoreAlgorithm module
SUITE["CoreAlgorithm"]["GBLUP_small"] = @benchmarkable fit!(model, $data_small) setup=(model=GBLUPModel(lambda=50.0))
SUITE["CoreAlgorithm"]["GBLUP_medium"] = @benchmarkable fit!(model, $data_medium) setup=(model=GBLUPModel(lambda=50.0))

# Add a benchmark for a Bayesian model
# We use a small number of iterations to keep the benchmark runtime reasonable
SUITE["CoreAlgorithm"]["BayesA_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesAModel(iterations=100))


# --- Running the Benchmarks (if the script is run directly) ---

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running benchmark suite for GenomicPrediction.jl...")
    results = run(SUITE, verbose = true)
    show(results)
    println()
end
