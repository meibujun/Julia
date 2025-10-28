# GenomicPrediction.jl

GenomicPrediction.jl 是一个面向基因组选择与数量性状预测的 Julia 包, 提供从数据处理到模型评估的完整工具链。项目遵循模块化、可复现与高性能的设计理念, 支持传统统计模型、深度学习模型、自动化模型选择以及 FAIR 原则的模型管理。

## 快速开始

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/scripts/basic_workflow.jl
```

更多使用说明请参阅 `docs/` 与 `examples/` 目录。
