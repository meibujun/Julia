"""
# 群体结构分析模块

提供群体遗传结构分析工具，包括：
- 主成分分析（PCA）
- 群体混合分析（ADMIXTURE）
- 系统发育树构建
- FST 计算

## 使用示例
```julia
using GenomicPro2.PopulationStructure

# PCA 分析
pca_results = perform_pca(genotypes, n_components=10)

# ADMIXTURE 分析
admix_results = perform_admixture(genotypes, K=3, niter=1000)

# 可视化
plot_pca(pca_results)
plot_admixture(admix_results)
```
"""
module PopulationStructure

using LinearAlgebra
using Statistics
using Random
using ..Core: GenotypeData, ValidationError
using ..Data: CompactGenotypes

# 包含子模块
include("pca.jl")
include("admixture.jl")

# 导出
export PCAResults, perform_pca, scree_plot, biplot_pca
export ADMIXTUREResults, perform_admixture, estimate_optimal_k
export compute_fst, compute_kinship

end # module PopulationStructure
