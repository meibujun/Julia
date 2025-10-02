# =============================================================================
# AnimalBreeding.jl - 多物种动物育种软件系统
# 主模块文件 - 模块化结构
# 版本: 3.1.0
# =============================================================================

"""
    AnimalBreeding

一个功能全面、模块化、高性能的动物育种遗传评估软件系统。
"""
module AnimalBreeding

# =============================================================================
# 依赖包导入
# =============================================================================
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using DataFrames
using CSV
using ProgressMeter
using Printf
using Dates

# =============================================================================
# 子模块包含
# =============================================================================

# --- 核心模块 ---
include("core/data_manager.jl")
include("core/model_spec.jl")

# --- 数据处理 ---
include("data/pedigree.jl")
include("data/genotypes.jl")
include("data/phenotypes.jl")
include("data/validation.jl")

# --- 关系矩阵 ---
include("relationships/pedigree_matrix.jl")
include("relationships/genomic_matrix.jl")
include("relationships/singlestep_matrix.jl")

# --- BLUP 评估 ---
include("blup/mme_solver.jl")
include("blup/reml.jl")
include("blup/evaluation.jl")

# =============================================================================
# 模块导出 (公共 API)
# =============================================================================
export DataManager, reset!, ensure_animal_map!, align_genotypes_to_pedigree!, ordered_animals
export RandomEffect, ModelSpec, define_model
export load_pedigree, validate_pedigree
export load_genotypes, perform_genotype_qc
export load_phenotypes
export validate_data
export sort_pedigree_for_A_matrix, compute_A_matrix, compute_A_inv_matrix
export compute_G_matrix, compute_H_matrix_inv
export build_design_matrices, setup_mme, solve_mme
export run_evaluation, save_results

end # module AnimalBreeding
