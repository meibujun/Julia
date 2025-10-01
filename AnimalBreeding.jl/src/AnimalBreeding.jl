"""
    AnimalBreeding

一个功能全面、模块化的动物育种遗传评估软件系统。
该模块是整个Julia包的入口点，整合了所有子模块的功能，并向用户提供统一的API。

# 核心功能
- **数据管理**: 加载、验证和管理谱系、基因型、表型数据。
- **关系矩阵**: 计算A、G等关系矩阵。
- **模型定义**: 提供灵活的接口来定义线性混合模型。
- **遗传评估**: 使用BLUP等方法进行育种值估计。

# 使用示例
```julia
using AnimalBreeding

# 1. 加载数据
dm = DataManager()
dm.pedigree = load_pedigree("path/to/pedigree.csv")
dm.phenotypes = load_phenotypes("path/to/phenotypes.csv", trait_cols=["milk"], fixed_cols=["herd"])

# 2. 验证数据
validate_data(dm)

# 3. 计算关系矩阵
compute_relationship_matrix(dm, type=:pedigree)

# 4. 定义模型
model = define_model(
    traits = ["milk"],
    fixed = ["herd"],
    random = [("animal", :additive)]
)

# 5. 运行评估
result = run_evaluation(model, dm, h2=0.3)

# 6. 保存结果
save_results(result, "breeding_values.csv")
```
"""
module AnimalBreeding

# --- 导出公共API ---
# `export` 关键字将指定的函数和类型暴露给用户，
# 当用户 `using AnimalBreeding` 时，这些函数可以直接使用。
export DataManager, load_pedigree, load_genotypes, load_phenotypes, validate_data
export compute_relationship_matrix
export ModelSpec, define_model # 导出ModelSpec以允许类型提示
export run_evaluation, save_results, EvaluationResult # 导出结果类型

# --- 导入依赖库 ---
# 这里列出了项目运行所需的所有外部 Julia 包。
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using Random
using ProgressMeter

# --- 包含子模块文件 ---
# `include` 语句将各个子模块的源文件代码插入到此处，
# 从而将整个项目组装成一个完整的模块。
# 这种模块化的结构使得代码更易于维护和扩展。
include("data_management.jl")
include("relationship_matrices.jl")
include("model_definition.jl")
include("blup.jl")

# --- 未来可扩展的模块 (当前注释掉) ---
# include("bayesian.jl")        # 贝叶斯分析模块
# include("ml.jl")              # 机器学习模块
# include("reml.jl")            # REML方差组分估计模块

end # module AnimalBreeding