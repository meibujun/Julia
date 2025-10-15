# src/eg_blup.jl

using .RareVariantEpistasis
using LinearAlgebra

"""
    eg_blup_analysis(g::GenomicData, p::PhenotypeData)

执行 EG-BLUP (Genomic Best Linear Unbiased Prediction) 分析。

参数:
- `g::GenomicData`: 基因组数据
- `p::PhenotypeData`: 表型数据

返回:
- `DataFrame`: 包含 SNP 效应 (育种值) 的估计结果
"""
function eg_blup_analysis(g::GenomicData, p::PhenotypeData)
    # 1. 构建设计矩阵 X (中心化的基因型) 和响应向量 y
    X = convert(Matrix{Float64}, g.genotypes)
    X .-= mean(X, dims=1)
    y = p.phenotypes.phenotype

    n, p = size(X)

    # 2. 估计方差组分 (使用简化的 REML)
    # This is a very simplified approach. A more robust implementation would use a proper REML algorithm.
    λ = 1.0 # Placeholder for variance ratio (σ_e^2 / σ_g^2)

    # 3. 求解混合模型方程 (MME)
    # (X'X + λ*I) * β = X'y
    XtX = X' * X
    Xty = X' * y

    snp_effects = (XtX + λ * I) \ Xty

    results = DataFrame(
        snp_id = [snp.id for snp in g.snp_info],
        effect_size = vec(snp_effects)
    )

    return results
end
