# AnimalBreeding.jl - 多物种动物育种软件系统

[![Julia Version](https://img.shields.io/badge/julia-1.11.6%2B-blue.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://your-docs-url.com)

一个功能全面、模块化、高性能的动物育种遗传评估软件系统，使用Julia语言开发，支持牛、猪、羊、家禽等多物种育种应用。

## ✨ 主要特性

### 核心功能

- **多数据源整合**: 支持谱系、基因型、表型、多组学等多种数据类型，并提供标准化数据验证和质量控制。
- **关系矩阵计算**: 高效计算谱系关系矩阵 (A)、基因组关系矩阵 (G, VanRaden方法) 和单步关系矩阵 (H)。
- **BLUP遗传评估**: 实现传统BLUP, GBLUP, 和单步GBLUP (SS-GBLUP)，并包含REML方差组分估计。

### 先进方法

- **贝叶斯基因组选择**: 实现BayesA, BayesB, BayesC, และ Bayesian LASSO，并提供完整的MCMC诊断工具。
- **机器学习集成**: 提供随机森林和神经网络模型用于非线性表型预测，并包含交叉验证框架。
- **高级遗传模型**:
  - **测定日模型**: 用于泌乳曲线等纵向数据分析。
  - **上位性分析**: 建模基因间的交互效应。
  - **G×E互作**: 使用反应范式模型分析基因组与环境的互作。
- **育种规划与选择**:
  - **选择指数**: 多性状综合选择。
  - **最优贡献选择 (OCS)**: 平衡遗传进展与近交。
  - **交配设计**: 制定考虑近交的优化交配方案。
  - **遗传进展预测**: 量化选择策略的长期效果。

### 技术特点

- **高性能计算**: 提供GPU加速 (CUDA)、多线程并行和稀疏矩阵优化等高性能计算方案。
- **模块化架构**: 清晰、松耦合的模块设计，易于维护和扩展。
- **科学严谨**: 所有核心算法均基于已发表的科学文献，并提供验证和诊断工具。
- **数据模拟**: 内置强大的数据模拟引擎，可生成包含谱系、基因型、表型和多组学数据的复杂数据集，用于测试和研究。

## 📦 安装

### 系统要求

- Julia 1.11.6 或更高版本
- 推荐至少 8GB 内存
- 可选: NVIDIA GPU 和 CUDA Toolkit (用于GPU加速)

### 安装步骤

1.  **安装 Julia**:
    从 [Julia官网](https://julialang.org/downloads/) 下载并安装。

2.  **安装 AnimalBreeding.jl**:
    启动 Julia REPL，然后使用内置的包管理器 `Pkg` 进行安装。

    ```julia
    using Pkg
    # 从GitHub安装 (推荐)
    Pkg.add(url="https://github.com/meibujun/Julia.git", subdir="AnimalBreeding.jl")

    # 或者，如果您已将代码克隆到本地，可以进行开发模式安装
    # 假设仓库克隆在 /path/to/Julia
    # Pkg.develop(path="/path/to/Julia/AnimalBreeding.jl")
    ```

## 🚀 快速开始

### 基础使用示例: GBLUP评估

```julia
using AnimalBreeding

# 1. 模拟一个标准数据集
dm, true_params = simulate_complete_dataset(
    n_generations=5,
    n_per_generation=200,
    n_markers=5000,
    h2=0.3
);

# 2. 计算基因组关系矩阵
compute_relationship_matrix(dm, type=:genomic);

# 3. 定义一个简单的加性模型
model = define_model(
    traits=["trait"],
    fixed=["herd"],
    random=[("animal", :additive)]
);

# 4. 运行GBLUP评估 (使用dm中的G矩阵)
result = run_evaluation(model, dm, method=:GBLUP, h2=0.3);

# 5. 查看结果摘要
println(result)

# 6. 保存育种值（合并谱系信息）
save_results(result, dm, "gblup_breeding_values.csv"; include_pedigree=true);
```

> ℹ️ `save_results(result, dm, path; include_pedigree=true, include_phenotypes=false)`
> 会自动拼接 DataManager 中的谱系或首条表型记录，便于下游分析。

### 高级功能示例: 育种规划

```julia
using AnimalBreeding
using Random

# ... (假设已有dm和评估结果result) ...

# 1. 最优贡献选择 (OCS)
# 选择20个个体，同时将平均关系控制在0.1以下
ocs_results = optimal_contribution_selection(
    result.breeding_values.EBV,
    dm.G_matrix,
    20,
    max_relationship=0.1
);
selected_sires_indices = ocs_results["selected_indices"];

# 2. 设计交配计划
# 假设我们有100头母畜
females_indices = sample(1:nrow(dm.pedigree), 100, replace=false);

mating_plan = design_mating_plan(
    selected_sires_indices,
    females_indices,
    dm.G_matrix, # 使用G矩阵控制近交
    result.breeding_values.EBV[selected_sires_indices],
    result.breeding_values.EBV[females_indices],
    max_inbreeding=0.0625 # 控制近交在同父半兄妹水平
);

println("生成的交配计划:")
println(first(mating_plan, 5))
```

## 📖 完整功能演示

要查看本软件所有功能的完整演示，请在安装后运行：

```julia
using AnimalBreeding

# 运行所有内置示例
run_all_examples()
```

## 🧪 测试

运行完整的测试套件以确保所有功能正常工作：

```julia
using Pkg
Pkg.test("AnimalBreeding")
```

## 🤝 贡献

我们欢迎任何形式的贡献！无论是报告问题、提出功能建议还是直接贡献代码，请通过本项目的GitHub页面进行。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 `LICENSE` 文件。