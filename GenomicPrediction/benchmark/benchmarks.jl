# benchmark/benchmarks.jl
# ==========================================================
# Performance benchmarks for the GenomicPrediction.jl package.
#
# This file has been expanded to include a comprehensive suite of benchmarks
# for all major algorithms and performance-critical functions.
# ==========================================================

using BenchmarkTools
using GenomicPrediction
using DataFrames
using Random

const SUITE = BenchmarkGroup()

# --- 1. 数据生成 ---
function generate_benchmark_data(n_individuals::Int, n_markers::Int)
    Random.seed!(42)
    G = rand(0:2, n_individuals, n_markers)
    y = randn(n_individuals)

    geno_df = DataFrame(G, :auto)
    insertcols!(geno_df, 1, :ID => 1:n_individuals)

    pheno_df = DataFrame(ID = 1:n_individuals, y = y)

    # Generate a simple pedigree for ssGBLUP benchmarks
    ped_df = DataFrame(ID = 1:n_individuals, Sire = 0, Dam = 0)

    return GenomicData(geno_df, pheno_df, nothing, ped_df)
end

println("正在生成基准测试数据...")
const data_small = generate_benchmark_data(100, 500)
const data_medium = generate_benchmark_data(500, 1000)
# const data_large = generate_benchmark_data(1000, 5000) # Optional for more intensive tests

# --- 2. 基准测试组定义 ---
SUITE["DataProcessing"] = BenchmarkGroup(["GRM", "QC"])
SUITE["CoreAlgorithm"] = BenchmarkGroup(["GBLUP", "Bayesian", "Penalized"])
SUITE["KernelModels"] = BenchmarkGroup(["ssGBLUP"])
SUITE["Evaluation"] = BenchmarkGroup(["CV"])

# --- 3. 添加具体基准测试 ---

# a. DataProcessing: calculate_grm (并行化后)
grm_group = SUITE["DataProcessing"]
grm_group["calculate_grm_small"] = @benchmarkable calculate_grm($(Matrix(data_small.genotypes[!, 2:end])))
grm_group["calculate_grm_medium"] = @benchmarkable calculate_grm($(Matrix(data_medium.genotypes[!, 2:end])))

# b. CoreAlgorithm: GBLUP (Cholesky 优化后)
gblup_group = SUITE["CoreAlgorithm"]
gblup_group["GBLUP_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=GBLUPModel(lambda=10.0))
gblup_group["GBLUP_fit_medium"] = @benchmarkable fit!(model, $data_medium) setup=(model=GBLUPModel(lambda=10.0))

# c. CoreAlgorithm: Bayesian Models (minimal iterations)
bayes_group = SUITE["CoreAlgorithm"]
bayes_group["BayesA_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesAModel(iterations=20, burn_in=10))
bayes_group["BayesB_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesBModel(iterations=20, burn_in=10))
bayes_group["BayesC_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesCModel(iterations=20, burn_in=10))
bayes_group["BayesR_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=BayesRModel(iterations=20, burn_in=10))

# d. CoreAlgorithm: Penalized Models
penalized_group = SUITE["CoreAlgorithm"]
penalized_group["LASSO_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=LASSOModel(max_iters=20))
penalized_group["ElasticNet_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=ElasticNetModel(max_iters=20))

# e. KernelModels: ssGBLUP (避免求逆优化后)
ssgblup_group = SUITE["KernelModels"]
ssgblup_group["ssGBLUP_fit_small"] = @benchmarkable fit!(model, $data_small) setup=(model=ssGBLUPModel(lambda=10.0))
ssgblup_group["ssGBLUP_fit_medium"] = @benchmarkable fit!(model, $data_medium) setup=(model=ssGBLUPModel(lambda=10.0))

# f. Evaluation: cross_validate (并行化后)
cv_group = SUITE["Evaluation"]
cv_group["cross_validate_GBLUP_small"] = @benchmarkable cross_validate(model, $data_small; k=3) setup=(model=GBLUPModel(lambda=10.0))

# --- 4. 运行基准测试 ---
# 如果直接运行此文件，则执行基准测试
if abspath(PROGRAM_FILE) == @__FILE__
    println("正在运行 GenomicPrediction.jl 的性能基准测试套件...")
    # 预热，确保 JIT 编译完成
    warmup(SUITE)
    # 运行基准测试
    results = run(SUITE, verbose = true, seconds = 10)

    println("\n--- 基准测试结果摘要 ---")
    show(results)
    println()
end
