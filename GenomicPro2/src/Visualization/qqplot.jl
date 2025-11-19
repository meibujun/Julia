"""
# QQ 图（Quantile-Quantile Plot）

用于检验 GWAS P 值是否符合预期分布。

## 用途
1. 检测群体分层
2. 识别系统性偏差
3. 评估模型拟合质量

## 解释
- 点沿对角线：P 值分布符合预期
- 早期偏离：系统性偏差或群体分层
- 尾部偏离：真实关联信号
"""

using Statistics

"""
    prepare_qq_plot(pvalues::Vector{Float64};
                    confidence_interval::Float64=0.95,
                    lambda::Union{Float64, Nothing}=nothing)

准备 QQ 图的数据。

# 参数
- `pvalues`: P 值向量
- `confidence_interval`: 置信区间（默认 0.95）
- `lambda`: 基因组控制因子（可选，如果提供则显示）

# 返回
命名元组，包含绘图所需的数据：
- `expected`: 期望的 -log10(P) 值
- `observed`: 观察的 -log10(P) 值
- `ci_lower`: 置信区间下界
- `ci_upper`: 置信区间上界
- `lambda`: λ 值（基因组膨胀因子）

# 示例
```julia
qq_data = prepare_qq_plot(gwas_results.pvalues)

# 检查 λ 值
if qq_data.lambda > 1.1
    @warn "检测到检验统计量膨胀 (λ = $(qq_data.lambda))"
end
```
"""
function prepare_qq_plot(pvalues::Vector{Float64};
                         confidence_interval::Float64=0.95,
                         lambda::Union{Float64, Nothing}=nothing)

    # 移除无效 P 值
    valid_pvalues = filter(p -> !isnan(p) && p > 0 && p <= 1, pvalues)
    n = length(valid_pvalues)

    if n == 0
        throw(ValidationError("没有有效的 P 值"))
    end

    # 排序（升序）
    sorted_pvalues = sort(valid_pvalues)

    # 观察的 -log10(P)
    observed = -log10.(sorted_pvalues)

    # 期望的 -log10(P)（在零假设下）
    # P_expected = (i - 0.5) / n
    expected = -log10.((collect(1:n) .- 0.5) / n)

    # 计算置信区间
    # 使用 Beta 分布的分位数
    α = (1 - confidence_interval) / 2

    ci_lower = zeros(n)
    ci_upper = zeros(n)

    for i in 1:n
        # Beta 分布参数
        # P_i ~ Beta(i, n - i + 1)
        # 使用简化近似
        p_lower = max(1e-10, (i - 0.5) / n - 1.96 * sqrt((i - 0.5) * (n - i + 0.5) / n^3))
        p_upper = min(1 - 1e-10, (i - 0.5) / n + 1.96 * sqrt((i - 0.5) * (n - i + 0.5) / n^3))

        ci_lower[i] = -log10(p_upper)
        ci_upper[i] = -log10(p_lower)
    end

    # 计算 λ（如果未提供）
    if lambda === nothing
        # λ = median(χ²_observed) / median(χ²_expected)
        # 使用 P 值计算
        median_p = median(valid_pvalues)

        # 将中位数 P 值转换为卡方统计量
        # χ² = -2 * ln(P) for exponential distribution
        # 更准确的方法：
        z_scores = [quantile_normal_qq(1 - p/2) for p in valid_pvalues]
        chi2_obs = z_scores .^ 2
        median_chi2_obs = median(chi2_obs)
        median_chi2_exp = 0.4549  # χ²(1) 的中位数

        lambda = median_chi2_obs / median_chi2_exp
    end

    return (
        expected = expected,
        observed = observed,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        lambda = lambda,
        n_points = n,
        max_expected = maximum(expected),
        max_observed = maximum(observed)
    )
end

"""
    qq_plot(pvalues::Vector{Float64}; kwargs...)

生成 QQ 图（prepare_qq_plot 的别名）。
"""
qq_plot = prepare_qq_plot

"""
正态分布分位数函数（用于 QQ 图）
"""
function quantile_normal_qq(p::Float64)
    # Beasley-Springer-Moro 算法
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    elseif p == 0.5
        return 0.0
    elseif p < 0.5
        return -quantile_normal_qq(1 - p)
    end

    t = sqrt(-2 * log(1 - p))

    # 系数
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    numerator = c[1] + c[2]*t + c[3]*t^2
    denominator = 1 + d[1]*t + d[2]*t^2 + d[3]*t^3

    z = t - numerator / denominator

    return z
end

"""
    qqplot_by_chromosome(gwas::GWASResult)

为每条染色体生成独立的 QQ 图。

# 返回
字典，键为染色体编号，值为 QQ 图数据
"""
function qqplot_by_chromosome(gwas::GWASResult)
    unique_chroms = sort(unique(gwas.chromosomes))
    qq_data = Dict{Int, NamedTuple}()

    for chrom in unique_chroms
        chrom_idx = findall(gwas.chromosomes .== chrom)
        chrom_pvalues = gwas.pvalues[chrom_idx]

        if length(chrom_pvalues) > 0
            qq_data[chrom] = prepare_qq_plot(chrom_pvalues)
        end
    end

    return qq_data
end

"""
    check_inflation(lambda::Float64; threshold::Float64=1.1)

检查基因组膨胀因子是否超过阈值。

# 参数
- `lambda`: λ 值
- `threshold`: 阈值（默认 1.1）

# 返回
(is_inflated, severity, recommendation)
"""
function check_inflation(lambda::Float64; threshold::Float64=1.1)
    is_inflated = lambda > threshold

    if lambda < 1.0
        severity = "正常（无膨胀）"
        recommendation = "P 值分布正常"
    elseif lambda <= 1.05
        severity = "轻微"
        recommendation = "可以接受，无需调整"
    elseif lambda <= 1.1
        severity = "中等"
        recommendation = "考虑使用基因组控制或添加协变量"
    else
        severity = "严重"
        recommendation = "强烈建议：1) 检查群体分层 2) 添加主成分作为协变量 3) 使用混合线性模型"
    end

    return (
        is_inflated = is_inflated,
        lambda = lambda,
        severity = severity,
        recommendation = recommendation
    )
end

"""
    calculate_expected_pvalues(n::Int)

计算 n 个独立检验的期望 P 值分布。

# 参数
- `n`: 检验数量

# 返回
期望的 P 值向量（已排序）
"""
function calculate_expected_pvalues(n::Int)
    return (collect(1:n) .- 0.5) / n
end

# 导出
export prepare_qq_plot, qq_plot
export qqplot_by_chromosome, check_inflation, calculate_expected_pvalues
