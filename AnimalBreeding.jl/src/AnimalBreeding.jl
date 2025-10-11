# ============================================================================
# AnimalBreeding.jl - 多物种动物育种软件系统
# 主模块文件 - 负责将各子模块组织成统一的公开API
# 版本: 3.1.0
# ============================================================================

"""
    AnimalBreeding

一个功能全面、模块化、高性能的动物育种遗传评估软件系统。
"""
module AnimalBreeding

# ============================================================================
# 依赖包导入
# ----------------------------------------------------------------------------
# 此处集中导入包的好处是：
# 1. 统一管理依赖，便于在Project.toml中维护；
# 2. 子模块可以直接使用这些常用依赖，减少重复引用；
# 3. 方便利用预编译缓存提高性能。
# ============================================================================
using LinearAlgebra, SparseArrays, Statistics
using DataFrames, CSV, ProgressMeter, Printf

# ============================================================================
# 子模块包含 (按照功能分层加载)
# ----------------------------------------------------------------------------
# 相比旧版本中被注释掉的include语句，此处正式启用所有已实现的模块。
# 这样在使用 `using AnimalBreeding` 时，核心功能即可立即可用。
# ============================================================================

# --- 核心模块 ---
include("core/data_manager.jl")
include("core/model_spec.jl")

# --- 数据处理 ---
include("data/pedigree.jl")
include("data/genotypes.jl")
include("data/phenotypes.jl")
include("data/validation.jl")
include("data/simulation.jl")

# --- 关系矩阵 ---
include("relationships/pedigree_matrix.jl")
include("relationships/genomic_matrix.jl")
include("relationships/singlestep_matrix.jl")
include("relationships/relationship_manager.jl")

# --- BLUP 评估 ---
include("blup/mme_solver.jl")
include("blup/reml.jl")
include("blup/evaluation.jl")

# ============================================================================
# 模块导出 (公共API)
# ----------------------------------------------------------------------------
# 公开常用结构体与函数，方便下游用户直接调用。
# 若未来新增模块，只需在此追加对应导出即可。
# ============================================================================
export DataManager,
       update_animal_map_from_pedigree!,
       define_model,
       RandomEffect,
       ModelSpec,
       load_pedigree,
       validate_pedigree,
       load_genotypes,
       perform_genotype_qc,
       load_phenotypes,
       validate_data,
       compute_A_matrix,
       compute_A_inv_matrix,
       compute_G_matrix,
       compute_H_matrix_inv,
       compute_relationship_matrix,
       simulate_complete_dataset,
       build_design_matrices,
       setup_mme,
       solve_mme,
       estimate_variances_reml,
       run_evaluation,
       save_results

end # module AnimalBreeding
