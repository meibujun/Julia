# 快速入门教程

```@meta
CurrentModule = GenomicPrediction
```

本教程通过模拟数据演示完整的基因组预测流程, 涵盖数据准备、模型训练、评估与模型保存。

## 1. 准备数据

```julia
using GenomicPrediction
using Random

Random.seed!(2025)
dataset = simulate_genomic_data(300, 800; h2 = 0.5)
```

`simulate_genomic_data` 将返回 `GenomicDataset` 对象, 其中包含基因型矩阵、表型向量及基础元数据。若使用真实数据, 可先调用 `load_genomic_table` 和 `load_phenotype_table` 读取, 再通过 `merge_genomic_phenotype` 组合。

## 2. 数据清洗

```julia
quality_control!(dataset; maf_threshold = 0.01, missing_rate = 0.1)
impute_missing!(dataset; method = :mean)
```

质量控制会剔除低频或缺失率过高的标记, 并自动同步标记名称。随后使用均值法填补缺失值。

## 3. 模型训练与评估

### 3.1 传统模型

```julia
model = RidgeRegressionModel(λ = 0.5)
fit!(model, dataset.genotype, dataset.phenotype)
preds = predict(model, dataset.genotype)
metrics = evaluate_metrics(dataset.phenotype, preds)
```

### 3.2 深度学习模型

```julia
mlp = build_mlp(size(dataset.genotype, 2), [256, 128], output_dim = 1)
history, trained = train_deep_model!(mlp, dataset.genotype, dataset.phenotype; epochs = 30, batch_size = 32)
```

训练完成后可通过 `history.loss` 可视化损失下降趋势。

### 3.3 交叉验证

```julia
cv = cross_validate(() -> GBLUPModel(λ = 0.8), dataset, 5; rng = MersenneTwister(1))
summary = summarize_cv(cv)
```

## 4. 自动化建模

```julia
pipeline = default_workflow(dataset)
run_autogs(pipeline, dataset)
pipeline.results
```

结果表格给出每个候选模型的平均指标及波动区间, `pipeline.best_model` 则保存表现最佳的模型对象。

## 5. 模型保存与共享

```julia
metadata = create_metadata("Ridge"; metrics = Dict("rmse" => summary[:mean]), data_sources = ["Simulated"], notes = "Tutorial example")
save_model(model, "ridge_model.bson"; metadata = metadata, extra = history)
```

该命令会生成 `.bson` 模型文件及对应的 `_metadata.json`, 以供外部系统读取。

## 6. 下一步

- 在 `examples/scripts` 中尝试更多工作流脚本。
- 在 `benchmark/` 中运行性能基准, 评估不同算法的效率。
- 使用 `interfaces/python` 与 `interfaces/R` 下的示例在其他语言中复用本包能力。
