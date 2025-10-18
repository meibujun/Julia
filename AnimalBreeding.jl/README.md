# AnimalBreeding.jl - 动物育种遗传评估演示系统

[![Julia Version](https://img.shields.io/badge/julia-1.11.6%2B-blue.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AnimalBreeding.jl 是一个模块化的动物育种遗传评估演示系统，聚焦于谱系、
基因型与表型数据的管理、关系矩阵计算以及 BLUP/GBLUP 评估。项目所有核心
函数均提供中文文档，方便教学或入门学习。

## ✨ 主要特性

### 核心功能

- **数据模拟**：`simulate_complete_dataset` 生成包含谱系、基因型和表型的
  示例数据集，可直接用于教学或测试。
- **关系矩阵计算**：`compute_relationship_matrix` 自动计算并缓存 A、A⁻¹、
  G 以及单步 H⁻¹ 矩阵，支持一次请求多个矩阵。
- **BLUP 遗传评估**：`run_evaluation` 结合设计矩阵构建、MME 求解和 REML
  方差估计，完成 BLUP/GBLUP 分析。
- **结果导出**：`save_results` 将育种值和可靠性保存为 CSV 文件，便于后续
  处理。

### 技术特点

- **模块化架构**：数据管理、关系矩阵、BLUP 求解分层实现，结构清晰。
- **自动化示例**：提供 `quickstart_gblup` 一键运行完整流程，并返回所有中间
  对象，便于调试和探索。
- **中文注释**：核心源代码配有中文注释与文档字符串，降低学习门槛。

## 📦 安装

### 系统要求

- Julia 1.11.6 或更高版本
- 推荐至少 8GB 内存

### 安装步骤

1. **安装 Julia**：从 [Julia 官网](https://julialang.org/downloads/) 下载并安装。
2. **安装 AnimalBreeding.jl**：在 Julia REPL 中执行：

   ```julia
   using Pkg
   Pkg.add(url="https://github.com/your-repo/AnimalBreeding.jl")
   ```

   如果已克隆本仓库，可使用 `Pkg.develop(path="/path/to/AnimalBreeding.jl")`。

## 🚀 快速开始

### 手动运行流程

```julia
using AnimalBreeding

# 1. 模拟数据
dm, true_params = simulate_complete_dataset(
    n_generations=5,
    n_per_generation=200,
    n_markers=5000,
    h2=0.3,
)

# 2. 计算基因组关系矩阵
compute_relationship_matrix(dm, type=:genomic)

# 3. 定义模型
model = define_model(
    traits=["trait"],
    fixed=["herd"],
    random=[("animal", :additive)],
)

# 4. 运行 GBLUP 评估
result = run_evaluation(model, dm, method=:GBLUP, h2=0.3)

# 5. 查看结果并导出
println(result)
save_results(result, "gblup_breeding_values.csv")
```

### 一键运行

```julia
using AnimalBreeding

result, dm, true_params = quickstart_gblup()
```

`quickstart_gblup` 会自动执行上述步骤，并将育种值保存到 `result` 中。可通过
关键字参数自定义代数、个体数、标记数等设置。

## 📖 示例脚本

`examples/examples.jl` 展示了数据模拟、关系矩阵构建和 BLUP 求解的协同流程，
适合作为进一步探索的起点。

## 🧪 测试

使用 `Pkg.test("AnimalBreeding")` 运行测试套件，验证关系矩阵与数据层逻辑。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 以改进功能与文档。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 `LICENSE` 文件。
