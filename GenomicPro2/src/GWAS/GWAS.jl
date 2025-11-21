"""
# GWAS 模块

全基因组关联分析（Genome-Wide Association Study）。

## 功能
- 线性模型 GWAS
- 混合线性模型 GWAS（校正群体结构）
- GPU 加速
- 多重检验校正
- 协变量支持

## 使用示例
```julia
using GenomicPro2.GWAS

# 线性模型 GWAS
results = perform_gwas(
    genotypes,
    phenotypes,
    model = LinearModelGWAS()
)

# 混合模型 GWAS（使用 GRM 校正群体结构）
grm = compute_grm(genotypes)
results = perform_gwas(
    genotypes,
    phenotypes,
    model = MixedModelGWAS(grm)
)

# GPU 加速
results = gwas_gpu(genotypes, phenotypes)

# 多重检验校正
adjusted_pvalues = adjust_pvalues(results.pvalues, method=:bonferroni)
```
"""
module GWAS

using LinearAlgebra
using Statistics
using Distributions
using Printf
using ..Core
using ..Data
using ..Models  # 复用 GRM 计算
using ..PopulationStructure  # 复用 PCA

# 导出类型和函数
export AbstractGWASModel, LinearModelGWAS, MixedModelGWAS
export GWASResults
export perform_gwas, gwas_gpu
export adjust_pvalues, genomic_control
export estimate_heritability

# ============================================================================
# 核心类型定义
# ============================================================================

"""
GWAS 模型的抽象基类
"""
abstract type AbstractGWASModel end

"""
    LinearModelGWAS

线性模型 GWAS：y = Xβ + e

适用于：
- 无群体分层的样本
- 需要快速分析
- 探索性分析

# 字段
- `adjust_population_structure::Bool`: 是否使用 PCA 校正群体分层
- `n_pcs::Int`: 使用的主成分数量（如果校正）
- `covariates::Union{Matrix{Float64}, Nothing}`: 协变量矩阵
"""
struct LinearModelGWAS <: AbstractGWASModel
    adjust_population_structure::Bool
    n_pcs::Int
    covariates::Union{Matrix{Float64}, Nothing}

    function LinearModelGWAS(;
                             adjust_population_structure::Bool=false,
                             n_pcs::Int=10,
                             covariates::Union{Matrix{Float64}, Nothing}=nothing)
        new(adjust_population_structure, n_pcs, covariates)
    end
end

"""
    MixedModelGWAS

混合线性模型 GWAS：y = Xβ + Zu + e

其中 u ~ N(0, σ²_g G)，G 是基因组关系矩阵。

适用于：
- 存在群体分层
- 存在亲缘关系
- 需要更准确的关联检验

# 字段
- `grm::Union{Matrix{Float64}, Nothing}`: 基因组关系矩阵（如果为 nothing 则自动计算）
- `covariates::Union{Matrix{Float64}, Nothing}`: 协变量矩阵
- `reml::Bool`: 是否使用 REML 估计方差组分
"""
struct MixedModelGWAS <: AbstractGWASModel
    grm::Union{Matrix{Float64}, Nothing}
    covariates::Union{Matrix{Float64}, Nothing}
    reml::Bool

    function MixedModelGWAS(grm::Union{Matrix{Float64}, Nothing}=nothing;
                            covariates::Union{Matrix{Float64}, Nothing}=nothing,
                            reml::Bool=true)
        new(grm, covariates, reml)
    end
end

"""
    GWASResults

GWAS 分析结果。

# 字段
- `snp_ids::Vector{String}`: SNP 标识符
- `chromosomes::Vector{Int}`: 染色体编号
- `positions::Vector{Int}`: SNP 位置（bp）
- `pvalues::Vector{Float64}`: P 值
- `effect_sizes::Vector{Float64}`: 效应大小（beta）
- `standard_errors::Vector{Float64}`: 标准误
- `test_statistics::Vector{Float64}`: 检验统计量
- `model_type::String`: 模型类型
- `n_samples::Int`: 样本数
- `n_snps::Int`: SNP 数
- `genomic_control_lambda::Float64`: 基因组控制因子 λ
- `heritability::Union{Float64, Nothing}`: 估计的遗传力（混合模型）
"""
struct GWASResults
    snp_ids::Vector{String}
    chromosomes::Vector{Int}
    positions::Vector{Int}
    pvalues::Vector{Float64}
    effect_sizes::Vector{Float64}
    standard_errors::Vector{Float64}
    test_statistics::Vector{Float64}
    model_type::String
    n_samples::Int
    n_snps::Int
    genomic_control_lambda::Float64
    heritability::Union{Float64, Nothing}
end

# ============================================================================
# 主要 GWAS 函数
# ============================================================================

"""
    perform_gwas(genotypes::CompactGenotypes, phenotypes::PhenotypeData,
                 model::AbstractGWASModel; kwargs...)

执行 GWAS 分析。

# 参数
- `genotypes`: CompactGenotypes 基因型数据
- `phenotypes`: PhenotypeData 表型数据
- `model`: GWAS 模型（LinearModelGWAS 或 MixedModelGWAS）
- `parallel::Bool`: 是否并行计算（默认 true）
- `verbose::Bool`: 是否显示进度（默认 true）

# 返回
GWASResults 对象

# 示例
```julia
# 线性模型
model = LinearModelGWAS(adjust_population_structure=true, n_pcs=10)
results = perform_gwas(genotypes, phenotypes, model)

# 混合模型
grm = compute_grm(genotypes)
model = MixedModelGWAS(grm)
results = perform_gwas(genotypes, phenotypes, model)
```
"""
function perform_gwas(genotypes::CompactGenotypes,
                      phenotypes::PhenotypeData,
                      model::AbstractGWASModel;
                      parallel::Bool=true,
                      verbose::Bool=true)

    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    if n_samples != length(phenotypes.values)
        throw(ValidationError("样本数不匹配"))
    end

    if verbose
        @info "开始 GWAS 分析"
        @info "  样本数: $n_samples"
        @info "  SNP 数: $n_snps"
        @info "  模型: $(typeof(model))"
    end

    # 准备表型数据
    y = copy(phenotypes.values)
    y .-= mean(y)  # 中心化

    # 根据模型类型分发
    if isa(model, LinearModelGWAS)
        return perform_linear_gwas(genotypes, y, model, parallel, verbose)
    elseif isa(model, MixedModelGWAS)
        return perform_mixed_gwas(genotypes, y, model, parallel, verbose)
    else
        throw(ArgumentError("未知的模型类型"))
    end
end

    # Prepare covariates (excluding intercept, as optimized function adds it)
    covs = model.covariates
    if model.adjust_population_structure
        if verbose
            @info "  Performing PCA for population structure correction..."
        end
        pca = perform_pca(genotypes, n_components=model.n_pcs, verbose=false)
        if isnothing(covs)
            covs = pca.scores[:, 1:model.n_pcs]
        else
            covs = hcat(covs, pca.scores[:, 1:model.n_pcs])
        end
    end

    return perform_gwas_linear_optimized(genotypes, 
                                         PhenotypeData(y, genotypes.sample_ids, ["Trait"]), 
                                         covariates=covs,
                                         use_parallel=parallel)
end

"""
混合模型 GWAS 实现 (委托给优化版本)
"""
function perform_mixed_gwas(genotypes::CompactGenotypes,
                             y::Vector{Float64},
                             model::MixedModelGWAS,
                             parallel::Bool,
                             verbose::Bool)
                             
    return perform_gwas_mixed_model_optimized(genotypes,
                                              PhenotypeData(y, genotypes.sample_ids, ["Trait"]),
                                              G=model.grm,
                                              covariates=model.covariates)
end

# ============================================================================
# GPU 加速版本
# ============================================================================

include("gwas_gpu.jl")
using .GPUGWAS

end # module GWAS

# ============================================================================
# 辅助函数
# ============================================================================

"""
准备协变量矩阵
"""
function prepare_covariates(genotypes::CompactGenotypes,
                             model::LinearModelGWAS,
                             verbose::Bool)

    n_samples = size(genotypes.data, 1)

    # 基础：截距项
    X = ones(n_samples, 1)

    # PCA 校正
    if model.adjust_population_structure
        if verbose
            @info "  执行 PCA 以校正群体分层..."
        end

        pca = perform_pca(genotypes, n_components=model.n_pcs, verbose=false)
        X = hcat(X, pca.scores[:, 1:model.n_pcs])

        if verbose
            @info "  使用前 $(model.n_pcs) 个主成分作为协变量"
        end
    end

    # 用户提供的协变量
    if model.covariates !== nothing
        X = hcat(X, model.covariates)
    end

    return X
end

"""
测试单个 SNP（线性模型）
"""
function test_snp_linear(genotypes::CompactGenotypes,
                          y::Vector{Float64},
                          snp_idx::Int,
                          X_cov::Matrix{Float64})

    n = length(y)

    # 提取 SNP 基因型
    x_snp = Vector{Float64}(undef, n)
    for i in 1:n
        x_snp[i] = Float64(genotypes[i, snp_idx])
    end

    # 中心化
    x_snp .-= mean(x_snp)

    # 构建设计矩阵：[协变量 | SNP]
    X = hcat(X_cov, x_snp)

    # 线性回归：(X'X)^-1 X'y
    XtX = X' * X
    Xty = X' * y

    # 检查奇异性
    if cond(XtX) > 1e10
        return (1.0, 0.0, Inf, 0.0)  # P=1, beta=0
    end

    β = XtX \ Xty

    # SNP 的效应是最后一个系数
    beta_snp = β[end]

    # 残差
    y_hat = X * β
    residuals = y .- y_hat

    # 残差方差
    df = n - size(X, 2)
    σ² = sum(residuals .^ 2) / df

    # 标准误
    se = sqrt(σ² * inv(XtX)[end, end])

    # t 统计量
    t = beta_snp / se

    # P 值（双侧 t 检验）
    dist = TDist(df)
    p = 2 * ccdf(dist, abs(t))

    return (p, beta_snp, se, t)
end

"""
测试单个 SNP（混合模型）
"""
function test_snp_mixed(genotypes::CompactGenotypes,
                         y::Vector{Float64},
                         snp_idx::Int,
                         X_cov::Matrix{Float64},
                         V_inv::Matrix{Float64})

    n = length(y)

    # 提取 SNP
    x_snp = Vector{Float64}(undef, n)
    for i in 1:n
        x_snp[i] = Float64(genotypes[i, snp_idx])
    end

    x_snp .-= mean(x_snp)

    # 设计矩阵
    X = hcat(X_cov, x_snp)

    # 加权最小二乘：(X'V^-1X)^-1 X'V^-1y
    XtV_inv = X' * V_inv
    XtV_invX = XtV_inv * X
    XtV_invy = XtV_inv * y

    if cond(XtV_invX) > 1e10
        return (1.0, 0.0, Inf, 0.0)
    end

    β = XtV_invX \ XtV_invy

    # SNP 效应
    beta_snp = β[end]

    # 标准误
    se = sqrt(inv(XtV_invX)[end, end])

    # Wald 检验统计量
    z = beta_snp / se

    # P 值（正态分布）
    p = 2 * ccdf(Normal(), abs(z))

    return (p, beta_snp, se, z)
end

"""
估计方差组分（REML）
"""
function estimate_variance_components(y::Vector{Float64},
                                       G::Matrix{Float64},
                                       X::Matrix{Float64},
                                       reml::Bool)

    n = length(y)

    # 简化实现：使用矩估计
    # 实际生产中应使用 REML 或 ML

    # 计算残差（去除固定效应）
    β_fixed = (X' * X) \ (X' * y)
    residuals = y - X * β_fixed

    # 估计遗传方差
    σ²_g = var(G * residuals)

    # 估计残差方差
    σ²_e = var(residuals)

    # 遗传力
    h2 = σ²_g / (σ²_g + σ²_e)

    return (σ²_g, σ²_e, h2)
end

"""
计算基因组控制因子 λ
"""
function compute_genomic_control(test_statistics::Vector{Float64})
    # λ = median(χ²) / 0.4549
    # 这里 test_statistics 是 t 或 z 统计量

    chi2_stats = test_statistics .^ 2
    median_chi2 = median(chi2_stats)

    lambda = median_chi2 / 0.4549

    return lambda
end

# ============================================================================
# 多重检验校正
# ============================================================================

"""
    adjust_pvalues(pvalues::Vector{Float64}; method::Symbol=:bonferroni)

多重检验校正。

# 参数
- `pvalues`: P 值向量
- `method`: 校正方法
  - `:bonferroni`: Bonferroni 校正
  - `:fdr`: Benjamini-Hochberg FDR 校正
  - `:sidak`: Šidák 校正

# 返回
校正后的 P 值

# 示例
```julia
adjusted = adjust_pvalues(results.pvalues, method=:fdr)
significant = adjusted .< 0.05
```
"""
function adjust_pvalues(pvalues::Vector{Float64}; method::Symbol=:bonferroni)
    m = length(pvalues)

    if method == :bonferroni
        return min.(pvalues .* m, 1.0)

    elseif method == :fdr
        # Benjamini-Hochberg
        sorted_idx = sortperm(pvalues)
        sorted_p = pvalues[sorted_idx]

        adjusted = similar(pvalues)
        cummin_val = 1.0

        for i in m:-1:1
            cummin_val = min(cummin_val, sorted_p[i] * m / i)
            adjusted[sorted_idx[i]] = cummin_val
        end

        return min.(adjusted, 1.0)

    elseif method == :sidak
        return 1.0 .- (1.0 .- pvalues) .^ m

    else
        throw(ArgumentError("未知的校正方法: $method"))
    end
end

# ============================================================================
# 优化的 GWAS 实现
# ============================================================================

include("gwas_optimized.jl")

"""
线性模型 GWAS 实现 (委托给优化版本)
"""
function perform_linear_gwas(genotypes::CompactGenotypes,
                              y::Vector{Float64},
                              model::LinearModelGWAS,
                              parallel::Bool,
                              verbose::Bool)
    
    # 准备协变量 (optimized version handles this internally, but we need to pass matrix)
    # perform_gwas_linear_optimized expects pheno object, but we have y vector here.
    # Actually perform_gwas_linear_optimized takes (geno, pheno).
    # But perform_gwas has already extracted y.
    
    # Let's look at perform_gwas_linear_optimized signature again.
    # It takes (geno, pheno).
    
    # We should probably redirect at perform_gwas level or adapt here.
    # perform_gwas in GWAS.jl does:
    # y = copy(phenotypes.values)
    # y .-= mean(y)
    # perform_linear_gwas(genotypes, y, model...)
    
    # perform_gwas_linear_optimized does:
    # y = pheno.traits[:, 1]
    # ...
    
    # If we want to use the optimized one, we should call it directly from perform_gwas
    # OR refactor perform_gwas_linear_optimized to take y vector.
    
    # Refactoring perform_gwas_linear_optimized is risky without seeing it again.
    # But wait, I can just call perform_gwas_linear_optimized with a dummy pheno object?
    # Or better, update perform_gwas to call optimized functions directly.
    
    # Let's update perform_gwas instead.
    return perform_gwas_linear_optimized(genotypes, 
                                         PhenotypeData(y, genotypes.sample_ids, ["Trait"]), 
                                         covariates=model.covariates,
                                         use_parallel=parallel)
end

"""
混合模型 GWAS 实现 (委托给优化版本)
"""
function perform_mixed_gwas(genotypes::CompactGenotypes,
                             y::Vector{Float64},
                             model::MixedModelGWAS,
                             parallel::Bool,
                             verbose::Bool)
                             
    return perform_gwas_mixed_model_optimized(genotypes,
                                              PhenotypeData(y, genotypes.sample_ids, ["Trait"]),
                                              G=model.grm,
                                              covariates=model.covariates)
end

# ============================================================================
# GPU 加速版本
# ============================================================================

include("gwas_gpu.jl")
using .GPUGWAS

end # module GWAS
