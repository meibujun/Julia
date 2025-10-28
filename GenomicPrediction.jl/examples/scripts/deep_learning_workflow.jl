#!/usr/bin/env julia
#=!
    深度学习示例: 构建 MLP 与 CNN, 对比不同网络结构的预测表现。
=#

using GenomicPrediction
using Random
using Statistics
using Flux

Random.seed!(2025)
dataset = simulate_genomic_data(180, 256; h2 = 0.55)
quality_control!(dataset; maf_threshold = 0.01)
impute_missing!(dataset)

println("=== 构建模型 ===")
mlp = build_mlp(size(dataset.genotype, 2), [128, 64], output_dim = 1)
cnn = build_cnn(1, size(dataset.genotype, 2), [(8, 5, 1), (8, 3, 1)], [32])

println("=== 训练 MLP ===")
mlp_history, mlp_trained = train_deep_model!(mlp, dataset.genotype, dataset.phenotype; epochs = 25, batch_size = 32)
mlp_preds = vec(mlp_trained(permutedims(dataset.genotype)))
mlp_metrics = evaluate_metrics(dataset.phenotype, mlp_preds)
println("MLP 指标: ", mlp_metrics)

println("=== 训练 CNN ===")
cnn_history, cnn_trained = train_deep_model!(cnn, dataset.genotype, dataset.phenotype; epochs = 25, batch_size = 32)
cnn_preds = vec(cnn_trained(permutedims(dataset.genotype)))
cnn_metrics = evaluate_metrics(dataset.phenotype, cnn_preds)
println("CNN 指标: ", cnn_metrics)

println("=== 保存训练历史 ===")
save_model(mlp_trained, "mlp_model.bson"; metadata = create_metadata("MLP"; metrics = Dict("rmse" => mlp_metrics[:rmse]), data_sources = ["simulation"]), extra = mlp_history)
save_model(cnn_trained, "cnn_model.bson"; metadata = create_metadata("CNN"; metrics = Dict("rmse" => cnn_metrics[:rmse]), data_sources = ["simulation"]), extra = cnn_history)
