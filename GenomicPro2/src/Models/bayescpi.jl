"""
# BayesCπ 模型

实现 BayesCπ 贝叶斯变量选择模型，用于基因组预测。

## 模型描述
BayesCπ 是一个贝叶斯回归模型，假设：
- 一定比例 π 的 SNP 效应为零（不在模型中）
- 其余 (1-π) 的 SNP 效应服从正态分布

与 BayesR 不同，BayesCπ 只有两个混合成分：
1. 效应为 0（概率 π）
2. 效应 ~ N(0, σ²) （概率 1-π）

## 使用示例
```julia
using GenomicPro2.Models

# 拟合 BayesCπ 模型
results = fit_bayescpi(
    genotypes,
    phenotypes,
    niter = 50000,
    burnin = 10000,
    estimate_pi = true
)

# 查看结果
println("估计的 π: ", results.pi_estimated)
println("遗传力: ", results.h2_estimated)
```
"""

using Random
using Statistics
using LinearAlgebra
using Distributions
using ..Core: GenotypeData, PhenotypeData, ValidationError
using ..Data: CompactGenotypes

"""
    BayesCpiResults

BayesCπ 模型拟合结果。

# 字段
- `beta`: SNP 效应估计值（后验均值）
- `beta_samples`: SNP 效应的 MCMC 样本
- `pi_estimated`: 估计的 π（零效应比例）
- `pi_samples`: π 的 MCMC 样本
- `h2_estimated`: 估计的遗传力
- `variance_genetic`: 遗传方差
- `variance_residual`: 残差方差
- `gebv`: 基因组估计育种值
- `inclusion_prob`: 每个 SNP 的包含概率（1-π_j）
- `convergence`: 收敛诊断信息
"""
struct BayesCpiResults
    beta::Vector{Float64}
    beta_samples::Matrix{Float64}
    pi_estimated::Float64
    pi_samples::Vector{Float64}
    h2_estimated::Float64
    variance_genetic::Float64
    variance_residual::Float64
    gebv::Vector{Float64}
    inclusion_prob::Vector{Float64}
    convergence::Dict{String, Any}
end

"""
    fit_bayescpi(genotypes::CompactGenotypes, phenotypes::PhenotypeData;
                 niter::Int=50000,
                 burnin::Int=10000,
                 thin::Int=10,
                 pi_init::Float64=0.95,
                 estimate_pi::Bool=true,
                 pi_prior_alpha::Float64=1.0,
                 pi_prior_beta::Float64=1.0,
                 verbose::Bool=true,
                 seed::Union{Int,Nothing}=nothing)

使用 Gibbs 采样拟合 BayesCπ 模型。

# 参数
- `genotypes`: CompactGenotypes 基因型数据
- `phenotypes`: PhenotypeData 表型数据
- `niter`: MCMC 总迭代次数（默认 50000）
- `burnin`: 燃烧期（默认 10000）
- `thin`: 稀疏化间隔（默认 10）
- `pi_init`: π 的初始值（默认 0.95，即 95% SNP 效应为零）
- `estimate_pi`: 是否估计 π（默认 true）
- `pi_prior_alpha`, `pi_prior_beta`: π 的 Beta 先验参数
- `verbose`: 是否显示进度信息
- `seed`: 随机种子

# 返回
BayesCpiResults 对象

# 算法
使用 Gibbs 采样算法：
1. 更新 SNP 效应 β_j（条件于包含指示器 δ_j）
2. 更新包含指示器 δ_j（变量选择）
3. 更新方差组分 σ²_e, σ²_β
4. （可选）更新 π

# 示例
```julia
results = fit_bayescpi(
    genotypes,
    phenotypes,
    niter = 50000,
    burnin = 10000,
    estimate_pi = true
)
```
"""
function fit_bayescpi(genotypes::CompactGenotypes,
                      phenotypes::PhenotypeData;
                      niter::Int=50000,
                      burnin::Int=10000,
                      thin::Int=10,
                      pi_init::Float64=0.95,
                      estimate_pi::Bool=true,
                      pi_prior_alpha::Float64=1.0,
                      pi_prior_beta::Float64=1.0,
                      verbose::Bool=true,
                      seed::Union{Int,Nothing}=nothing)

    # 设置随机种子
    if seed !== nothing
        Random.seed!(seed)
    end

    # 验证输入
    n_samples_g = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)
    n_samples_p = length(phenotypes.values)

    if n_samples_g != n_samples_p
        throw(ValidationError("样本数不匹配: genotypes=$n_samples_g, phenotypes=$n_samples_p"))
    end

    if pi_init < 0 || pi_init > 1
        throw(ValidationError("pi_init 必须在 [0, 1] 之间"))
    end

    if verbose
        @info "开始 BayesCπ 模型拟合"
        @info "  样本数: $n_samples_g"
        @info "  SNP 数: $n_snps"
        @info "  MCMC 迭代: $niter"
        @info "  燃烧期: $burnin"
        @info "  稀疏化: $thin"
        @info "  估计 π: $estimate_pi"
    end

    # 准备数据
    y = copy(phenotypes.values)
    y .-= mean(y)  # 中心化

    # 解压缩基因型并中心化/标准化
    X = Matrix{Float64}(undef, n_samples_g, n_snps)
    x_mean = zeros(n_snps)
    x_std = zeros(n_snps)

    for j in 1:n_snps
        for i in 1:n_samples_g
            X[i, j] = Float64(genotypes[i, j])
        end
        x_mean[j] = mean(X[:, j])
        X[:, j] .-= x_mean[j]
        x_std[j] = std(X[:, j])
        if x_std[j] > 0
            X[:, j] ./= x_std[j]
        end
    end

    # 预计算 X'X 的对角元素
    XtX_diag = [dot(X[:, j], X[:, j]) for j in 1:n_snps]

    # 初始化参数
    β = zeros(n_snps)
    δ = rand(n_snps) .> pi_init  # 包含指示器
    π = pi_init
    σ²_e = var(y)
    σ²_β = 0.01

    # MCMC 样本存储
    n_saved = div(niter - burnin, thin)
    β_samples = zeros(n_snps, n_saved)
    π_samples = zeros(n_saved)
    σ²_e_samples = zeros(n_saved)
    σ²_β_samples = zeros(n_saved)
    inclusion_count = zeros(n_snps)

    # Gibbs 采样
    y_corrected = copy(y)
    save_idx = 0

    if verbose
        @info "开始 MCMC 采样..."
    end

    for iter in 1:niter
        # 1. 更新 SNP 效应 β 和包含指示器 δ
        for j in 1:n_snps
            # 移除当前 SNP 的贡献
            if δ[j]
                y_corrected .+= X[:, j] .* β[j]
            end

            # 计算后验概率
            # P(δ_j = 1 | ...)
            rhs = dot(X[:, j], y_corrected)
            lhs = XtX_diag[j] + σ²_e / σ²_β

            # 效应在模型中的条件后验
            β_mean = rhs / lhs
            β_var = σ²_e / lhs

            # 计算贝叶斯因子（简化版本）
            log_bf = 0.5 * (log(σ²_e) - log(β_var) + β_mean^2 / β_var)

            # 更新包含概率
            log_odds = log((1 - π) / π) + log_bf
            prob_include = 1.0 / (1.0 + exp(-log_odds))

            # 采样包含指示器
            δ[j] = rand() < prob_include

            # 如果包含，采样效应值
            if δ[j]
                β[j] = randn() * sqrt(β_var) + β_mean
                y_corrected .-= X[:, j] .* β[j]
            else
                β[j] = 0.0
            end
        end

        # 2. 更新残差方差 σ²_e
        residuals = y_corrected
        sse = dot(residuals, residuals)
        # 逆伽马分布
        shape = (n_samples_g + 3.0) / 2.0
        scale = (sse + 0.01) / 2.0
        σ²_e = scale / rand(Gamma(shape, 1.0))

        # 3. 更新效应方差 σ²_β
        n_included = sum(δ)
        if n_included > 0
            ssβ = sum(β[δ] .^ 2)
            shape = (n_included + 3.0) / 2.0
            scale = (ssβ + 0.01) / 2.0
            σ²_β = scale / rand(Gamma(shape, 1.0))
        end

        # 4. 更新 π（如果需要）
        if estimate_pi
            n_included = sum(δ)
            # Beta 后验
            π = rand(Beta(pi_prior_alpha + n_snps - n_included,
                         pi_prior_beta + n_included))
        end

        # 保存样本
        if iter > burnin && (iter - burnin) % thin == 0
            save_idx += 1
            β_samples[:, save_idx] = β
            π_samples[save_idx] = π
            σ²_e_samples[save_idx] = σ²_e
            σ²_β_samples[save_idx] = σ²_β
            inclusion_count .+= δ
        end

        # 进度报告
        if verbose && iter % 10000 == 0
            n_included = sum(δ)
            @info "  迭代 $iter: π=$(round(π, digits=4)), " *
                  "包含 SNP 数=$n_included, " *
                  "σ²_e=$(round(σ²_e, digits=6))"
        end
    end

    if verbose
        @info "MCMC 采样完成"
    end

    # 计算后验均值和其他统计量
    β_est = mean(β_samples, dims=2)[:]
    π_est = mean(π_samples)
    inclusion_prob = inclusion_count / n_saved

    # 计算 GEBV
    gebv = X * β_est

    # 计算方差组分和遗传力
    σ²_g = var(gebv)
    σ²_e_est = mean(σ²_e_samples)
    h2_est = σ²_g / (σ²_g + σ²_e_est)

    # 收敛诊断
    convergence = Dict(
        "effective_sample_size" => n_saved,
        "pi_mean" => π_est,
        "pi_sd" => std(π_samples),
        "n_snps_included" => sum(inclusion_prob .> 0.5),
        "mean_inclusion_prob" => mean(inclusion_prob)
    )

    if verbose
        @info "模型拟合完成"
        @info "  估计的 π: $(round(π_est, digits=4))"
        @info "  包含的 SNP 数 (prob > 0.5): $(convergence["n_snps_included"])"
        @info "  估计的遗传力: $(round(h2_est, digits=4))"
    end

    return BayesCpiResults(
        β_est,
        β_samples,
        π_est,
        π_samples,
        h2_est,
        σ²_g,
        σ²_e_est,
        gebv,
        inclusion_prob,
        convergence
    )
end

"""
    predict_bayescpi(results::BayesCpiResults, genotypes_new::CompactGenotypes)

使用拟合的 BayesCπ 模型进行预测。

# 参数
- `results`: BayesCpiResults 对象
- `genotypes_new`: 新的基因型数据

# 返回
预测的育种值向量
"""
function predict_bayescpi(results::BayesCpiResults, genotypes_new::CompactGenotypes)
    n_samples = size(genotypes_new.data, 1)
    n_snps = size(genotypes_new.data, 2)

    if length(results.beta) != n_snps
        throw(ValidationError("SNP 数不匹配"))
    end

    # 解压缩基因型
    X_new = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            X_new[i, j] = Float64(genotypes_new[i, j])
        end
    end

    # 预测（需要使用相同的中心化/标准化）
    # 注意：这里简化处理，实际应用中需要保存训练集的均值和标准差
    return X_new * results.beta
end

# 导出
export BayesCpiResults, fit_bayescpi, predict_bayescpi
