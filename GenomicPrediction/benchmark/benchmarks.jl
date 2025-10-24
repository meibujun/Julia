# benchmark/benchmarks.jl

using BenchmarkTools
using GenomicPrediction

# 创建一个基准测试套件
const SUITE = BenchmarkGroup()

SUITE["GBLUP"] = BenchmarkGroup()

# --- 准备基准测试数据 ---
n_ind = 200
n_snp = 500
G = rand(0:2, n_ind, n_snp)
y = rand(n_ind)
geno_df = GenomicPrediction.DataFrames.DataFrame(G, :auto)
pheno_df = GenomicPrediction.DataFrames.DataFrame(y = y)
data = GenomicPrediction.GenomicData(geno_df, pheno_df)

# --- 添加基准测试 ---
SUITE["GBLUP"]["fit"] = @benchmarkable GenomicPrediction.fit!(model, \$data) setup=(model = GenomicPrediction.GBLUPModel(10.0))
SUITE["GBLUP"]["predict"] = @benchmarkable GenomicPrediction.predict(model, \$geno_df) setup=(model = (m = GenomicPrediction.GBLUPModel(10.0); GenomicPrediction.fit!(m, data); m))
