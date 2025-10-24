# examples/scripts/basic_workflow.jl

using GenomicPrediction
using DataFrames
using Random

println("--- 1. 生成模拟数据 ---")
Random.seed!(123)
n_ind = 100
n_snp = 200
G = rand(0:2, n_ind, n_snp)
y = rand(n_ind)

geno_df = DataFrame(G, :auto)
pheno_df = DataFrame(y = y)
data = GenomicPrediction.GenomicData(geno_df, pheno_df)

println("--- 2. 初始化并训练 GBLUP 模型 ---")
model = GenomicPrediction.GBLUPModel(10.0)
GenomicPrediction.fit!(model, data)

println("--- 3. 进行预测 ---")
predictions = GenomicPrediction.predict(model, data.genotypes)

println("--- 4. 评估模型 ---")
acc = GenomicPrediction.accuracy(predictions, data.phenotypes.y)
println("预测准确性 (correlation): ", acc)

println("\n示例脚本运行完成。")
