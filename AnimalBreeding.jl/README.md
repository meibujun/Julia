# AnimalBreeding.jl - 多物种动物育种软件系统

[![Julia Version](https://img.shields.io/badge/Julia-1.11.6-blue.svg)](https://julialang.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个功能全面、模块化的动物育种遗传评估软件系统，使用Julia语言开发。本项目旨在提供一个从数据管理到高级分析（包括BLUP、GBLUP等）的完整解决方案，支持牛、猪、羊、家禽等多物种的育种应用。

## ✨ 主要特性

### v0.1.0 已实现功能
- **核心数据结构**: `DataManager` 用于统一管理所有数据。
- **多数据源导入**: 支持从CSV文件加载谱系、基因型和表型数据。
- **数据验证**: 提供数据一致性和完整性检查功能。
- **关系矩阵计算**:
  - 谱系关系矩阵 (A 矩阵)
  - 基因组关系矩阵 (G 矩阵, 基于VanRaden方法)
- **BLUP遗传评估**:
  - 支持单性状动物模型。
  - 基于已知遗传力（h²）的BLUP求解。
- **模块化架构**: 清晰的代码结构，易于维护和扩展。
- **完整的测试套件**: 提供单元测试和集成测试，确保代码质量。

### 计划中功能
- **方差组分估计**: REML (限制性最大似然法)。
- **高级模型**: 多性状模型、随机回归模型、阈值模型。
- **基因组选择**: 单步GBLUP (SS-GBLUP)、贝叶斯方法 (BayesA/B/C)。
- **性能优化**: GPU加速和多线程并行计算。
- **用户界面**: 提供Web图形界面和更强大的命令行工具。

## 📦 安装与配置

### 系统要求
- Julia 1.11.6 或更高版本。
- 推荐至少 8GB 内存。

### 安装步骤

1.  **安装 Julia**:
    如果您的系统中没有安装Julia，请从 [Julia官网](https://julialang.org/downloads/) 下载并安装。

2.  **安装 AnimalBreeding.jl**:
    打开 Julia REPL (命令行界面)，然后使用内置的包管理器 `Pkg` 来安装本软件。您需要将 `path/to/AnimalBreeding.jl` 替换为本仓库在您本地的实际路径。

    ```julia
    using Pkg
    Pkg.add(path="path/to/AnimalBreeding.jl")
    ```
    或者，如果您想以开发模式安装（方便修改代码），可以使用 `dev` 命令：
    ```julia
    using Pkg
    Pkg.dev("path/to/AnimalBreeding.jl")
    ```
    此命令会自动安装 `Project.toml` 文件中列出的所有依赖包。

## 🚀 快速开始

以下是一个完整的使用示例，展示了如何使用本软件进行一次简单的BLUP评估。

```julia
# 1. 导入 AnimalBreeding 模块
using AnimalBreeding

# 2. 准备数据 (请将 "path/to/data/" 替换为实际路径)
# 示例数据位于项目根目录下的 `data/` 文件夹中。
data_path = "path/to/AnimalBreeding.jl/data/"

# 创建数据管理器
dm = DataManager()

# 加载谱系、表型数据
dm.pedigree = load_pedigree(joinpath(data_path, "pedigree.csv"))
dm.phenotypes = load_phenotypes(joinpath(data_path, "phenotypes.csv"),
                                trait_cols=["milk"],
                                fixed_cols=["herd"])

# 3. 验证数据一致性
# 这是一个好习惯，确保数据质量
validate_data(dm)

# 4. 计算关系矩阵
# 对于普通BLUP，我们需要A矩阵
compute_relationship_matrix(dm, type=:A)

# 5. 定义统计模型
model = define_model(
    traits = ["milk"],
    fixed = ["herd"],
    random = [("animal", :additive)]
)

# 6. 运行遗传评估
# 这里我们假设遗传力 h² = 0.5
result = run_evaluation(model, dm, h2=0.5)

# 7. 查看和保存结果
println("评估完成，结果如下：")
println(result)

# 将育种值保存到文件
save_results(result, "breeding_values.csv")
println("育种值已保存到 breeding_values.csv 文件。")
```

## 🧪 测试

我们提供了一套完整的测试来保证软件的质量。您可以按以下步骤运行测试：

1.  确保您已经通过 `Pkg.dev` 或 `Pkg.add` 安装了本软件。
2.  在 Julia REPL 中，运行：

    ```julia
    using Pkg
    Pkg.test("AnimalBreeding")
    ```

这将自动执行 `test/runtests.jl` 文件中的所有测试用例。

## 🔬 主要算法简介

### BLUP (最佳线性无偏预测)
系统通过求解混合模型方程 (MME) 来获得育种值。MME的矩阵形式如下：

```
[ X'X   X'Z     ] [β̂]   [ X'y ]
[ Z'X   Z'Z+λA⁻¹ ] [û] = [ Z'y ]
```
其中:
- `y`: 表型向量
- `β̂`: 固定效应的估计值
- `û`: 随机效应 (育种值) 的预测值
- `X`, `Z`: 分别是固定和随机效应的设计矩阵
- `A⁻¹`: 谱系关系矩阵的逆
- `λ = σ²ₑ / σ²ₐ`: 方差组分的比率

## 🤝 贡献

我们欢迎任何形式的贡献，包括报告问题、提出功能建议或直接贡献代码！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 `LICENSE` 文件。