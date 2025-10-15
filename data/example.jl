# RareVariantEpistasis.jl - Example Script

using RareVariantEpistasis
using CSV, DataFrames, KernelFunctions

# --- 1. 数据准备 ---

# 创建一个虚拟的数据集用于演示
# 在实际应用中，你应该加载真实的数据

# 模拟基因组数据
n_samples = 100
n_snps = 500
mock_snp_info = [SNPInfo("1", "rs$i", 1000+i, "A", "T") for i in 1:n_snps]
mock_genotypes = rand(Int8[0, 1, 2], n_samples, n_snps)
mock_sample_ids = ["ID_$i" for i in 1:n_samples]
genomic_data = GenomicData(mock_genotypes, mock_snp_info, mock_sample_ids)

# 模拟表型数据
phenotypes_df = DataFrame(
    sample_id = mock_sample_ids,
    phenotype = randn(n_samples)
)
phenotype_data = PhenotypeData(phenotypes_df, "sample_id")

println("数据准备完成。")
println("样本数量: $n_samples")
println("SNP 数量: $n_snps")

# --- 2. 执行分析 ---

println("\n--- 开始执行分析 ---")

# 2.1 折叠法 (CAST)
println("\n执行 CAST 分析...")
cast_results = collapsing_analysis(genomic_data, phenotype_data, method="CAST")
println("CAST 分析完成。")
# println(first(cast_results, 5)) # 显示前 5 行结果

# 2.2 贝叶斯回归
println("\n执行贝叶斯回归分析...")
bayesian_results = bayesian_regression_analysis(genomic_data, phenotype_data)
println("贝叶斯回归分析完成。")
# println(first(bayesian_results, 5))

# 2.3 RKHS (使用线性核)
println("\n执行 RKHS 分析...")
rkhs_results = rkhs_analysis(genomic_data, phenotype_data, LinearKernel())
println("RKHS 分析完成。")
# println(rkhs_results)

# 2.4 EG-BLUP
println("\n执行 EG-BLUP 分析...")
egblup_results = eg_blup_analysis(genomic_data, phenotype_data)
println("EG-BLUP 分析完成。")
# println(first(egblup_results, 5))

println("\n--- 所有分析已完成 ---")

# --- 3. 保存结果 ---

# 可以将结果保存到 CSV 文件
# CSV.write("results/cast_results.csv", cast_results)
# CSV.write("results/bayesian_results.csv", bayesian_results)
# CSV.write("results/rkhs_results.csv", rkhs_results)
# CSV.write("results/egblup_results.csv", egblup_results)

println("\n示例脚本执行完毕。")
