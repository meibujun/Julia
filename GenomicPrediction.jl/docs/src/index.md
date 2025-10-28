# GenomicPrediction.jl 简介

```@meta
CurrentModule = GenomicPrediction
```

GenomicPrediction.jl 是一个面向现代育种与基因组学研究的 Julia 包, 提供端到端的基因组预测工作流。项目涵盖数据处理、核心算法、深度学习、模型评估、FAIR 元数据管理以及自动化模型选择等模块, 并配套示例、文档与基准测试工具。

## 主要特性

- **模块化架构**: `CoreAlgorithm`, `DataProcessing`, `DeepLearning`, `Evaluation`, `FAIRModeling`, `AutoGS` 六大子模块各司其职, 易于扩展。
- **多算法支持**: 实现 GBLUP、岭回归、LASSO、Elastic Net、BayesA/B 等经典方法, 并提供 MLP/CNN/Transformer 风格网络。
- **完善的评估体系**: 内置交叉验证、常见回归指标、AUC 等工具, 方便比较模型表现。
- **FAIR 原则落地**: 提供模型保存、元数据记录、JSON 导出等功能, 保障模型可复现与共享。
- **自动化建模**: AutoGS 模块可自动遍历候选模型, 根据指标挑选最优方案。

## 快速体验

以下示例展示了利用内置模拟数据进行训练和评估的完整流程:

```julia
using GenomicPrediction
using Random

Random.seed!(123)
dataset = simulate_genomic_data(200, 500; h2 = 0.6)
model = GBLUPModel(λ = 0.5)
fit!(model, dataset.genotype, dataset.phenotype)
preds = predict(model, dataset.genotype)
metrics = evaluate_metrics(dataset.phenotype, preds)
```

## 更多资源

- [快速入门教程](tutorial.md)
- [API 参考](api.md)
- `examples/` 目录下提供完整脚本与 Pluto Notebook, 帮助深入学习。
