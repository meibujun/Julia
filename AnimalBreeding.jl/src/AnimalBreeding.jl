# ============================================================================
# AnimalBreeding.jl - 多物种动物育种软件系统
# 主模块文件 - 正确的模块化结构
# 版本: 3.0.0 (结构重置)
# ============================================================================

"""
    AnimalBreeding

一个功能全面、模块化、高性能的动物育种遗传评估软件系统。
"""
module AnimalBreeding

# ============================================================================
# 依赖包导入
# ============================================================================
using LinearAlgebra, SparseArrays, Statistics, Random, Distributions
using DataFrames, CSV, ProgressMeter, Printf, Dates

# ============================================================================
# 子模块包含 (正确的相对路径)
# ============================================================================

# --- 核心模块 ---
# include("core/data_manager.jl")
# include("core/model_spec.jl")
# include("utils/helpers.jl")

# --- 数据处理 ---
# include("data/pedigree.jl")
# include("data/genotypes.jl")
# include("data/phenotypes.jl")
# include("data/omics.jl")
# include("data/validation.jl")

# --- 关系矩阵 ---
# include("relationships/pedigree_matrix.jl")
# include("relationships/genomic_matrix.jl")
# include("relationships/singlestep_matrix.jl")

# --- BLUP 评估 ---
# include("blup/mme_solver.jl")
# include("blup/reml.jl")
# include("blup/evaluation.jl")

# --- 贝叶斯分析 ---
# include("bayesian/bayes_a.jl")
# include("bayesian/bayes_b.jl")
# include("bayesian/bayes_c.jl")
# include("bayesian/mcmc.jl")
# include("bayesian/diagnostics.jl")

# --- 机器学习 ---
# include("ml/random_forest.jl")
# include("ml/neural_network.jl")
# include("ml/cross_validation.jl")

# --- 高级模型 ---
# include("advanced/test_day.jl")
# include("advanced/random_regression.jl")
# include("advanced/epistasis.jl")
# include("advanced/gxe.jl")

# --- 育种规划 ---
# include("selection/index.jl")
# include("selection/ocs.jl")
# include("selection/mating.jl")
# include("selection/genetic_gain.jl")

# --- 性能优化 ---
# include("performance/gpu.jl")
# include("performance/parallel.jl")

# --- 数据模拟 ---
# include("simulation/pedigree_sim.jl")
# include("simulation/genotype_sim.jl")
# include("simulation/phenotype_sim.jl")
# include("simulation/omics_sim.jl")
# include("simulation/complete.jl")

# --- IO 模块 ---
# include("io/import.jl")
# include("io/export.jl")


# ============================================================================
# 模块导出 (公共API)
# ============================================================================
# 在实现相应模块后，将在此处添加导出

end # module AnimalBreeding