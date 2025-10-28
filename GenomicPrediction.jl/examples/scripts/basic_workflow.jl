#!/usr/bin/env julia
#=!
    基础示例脚本: 展示从数据模拟到模型评估的标准流程。
=#

using GenomicPrediction
using Random
using Statistics

# 设定随机种子以确保复现性
Random.seed!(2024)

println("=== 生成模拟数据 ===")
dataset = simulate_genomic_data(200, 400; h2 = 0.6)
println("基因型矩阵尺寸: ", size(dataset.genotype))

println("=== 数据清洗 ===")
quality_control!(dataset; maf_threshold = 0.02, missing_rate = 0.2)
impute_missing!(dataset; method = :mean)

println("=== 训练 GBLUP 模型 ===")
model = GBLUPModel(λ = 0.8)
fit!(model, dataset.genotype, dataset.phenotype)

println("=== 评估模型 ===")
preds = predict(model, dataset.genotype)
metrics = evaluate_metrics(dataset.phenotype, preds)
println("评估指标: ", metrics)

println("=== 保存模型与元数据 ===")
metadata = create_metadata("GBLUP"; metrics = Dict("rmse" => metrics[:rmse]), data_sources = [string(dataset.metadata["source"])])
save_model(model, "gblup_model.bson"; metadata = metadata)
println("模型已保存至 gblup_model.bson")
