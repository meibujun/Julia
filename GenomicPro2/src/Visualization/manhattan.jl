"""
# Manhattan 图

用于 GWAS 结果可视化的 Manhattan 图。

## 功能
- 按染色体着色
- 显著性阈值线
- 基因注释
- 交互式标记

## 数据结构
返回包含以下字段的命名元组：
- `positions`: SNP 在基因组上的累积位置
- `pvalues`: -log10(P 值)
- `chromosomes`: 染色体编号
- `snp_ids`: SNP 标识符
- `significant_threshold`: 显著性阈值
- `suggestive_threshold`: 提示性阈值
"""

using Statistics

"""
    GWASResult

GWAS 分析结果结构。

# 字段
- `snp_ids`: SNP 标识符
- `chromosomes`: 染色体编号
- `positions`: SNP 位置（碱基对）
- `pvalues`: P 值
- `effect_sizes`: 效应大小（可选）
"""
struct GWASResult
    snp_ids::Vector{String}
    chromosomes::Vector{Int}
    positions::Vector{Int}
    pvalues::Vector{Float64}
    effect_sizes::Union{Vector{Float64}, Nothing}
end

"""
    prepare_manhattan_plot(gwas::GWASResult;
                           significant_threshold::Float64=5e-8,
                           suggestive_threshold::Float64=1e-5,
                           highlight_snps::Union{Vector{String}, Nothing}=nothing,
                           chromosome_colors::Union{Vector{String}, Nothing}=nothing)

准备 Manhattan 图的数据。

# 参数
- `gwas`: GWASResult 对象
- `significant_threshold`: 全基因组显著性阈值（默认 5e-8）
- `suggestive_threshold`: 提示性阈值（默认 1e-5）
- `highlight_snps`: 需要高亮的 SNP 列表
- `chromosome_colors`: 染色体颜色列表（默认交替使用两种颜色）

# 返回
命名元组，包含绘图所需的所有数据

# 示例
```julia
gwas = GWASResult(
    snp_ids = ["rs1", "rs2", ...],
    chromosomes = [1, 1, 2, ...],
    positions = [1000, 2000, ...],
    pvalues = [0.05, 0.001, ...]
)

plot_data = prepare_manhattan_plot(gwas)

# 访问数据
println("显著 SNP 数: ", sum(plot_data.is_significant))
```
"""
function prepare_manhattan_plot(gwas::GWASResult;
                                significant_threshold::Float64=5e-8,
                                suggestive_threshold::Float64=1e-5,
                                highlight_snps::Union{Vector{String}, Nothing}=nothing,
                                chromosome_colors::Union{Vector{String}, Nothing}=nothing)

    n_snps = length(gwas.snp_ids)

    # 验证数据一致性
    if !(length(gwas.chromosomes) == length(gwas.positions) == length(gwas.pvalues) == n_snps)
        throw(ValidationError("数据长度不一致"))
    end

    # 移除缺失或无效的 P 值
    valid_idx = findall(x -> !isnan(x) && x > 0 && x <= 1, gwas.pvalues)

    snp_ids = gwas.snp_ids[valid_idx]
    chromosomes = gwas.chromosomes[valid_idx]
    positions = gwas.positions[valid_idx]
    pvalues = gwas.pvalues[valid_idx]

    # 计算 -log10(P)
    log_pvalues = -log10.(pvalues)

    # 按染色体和位置排序
    sort_idx = sortperm(collect(zip(chromosomes, positions)))
    snp_ids = snp_ids[sort_idx]
    chromosomes = chromosomes[sort_idx]
    positions = positions[sort_idx]
    pvalues = pvalues[sort_idx]
    log_pvalues = log_pvalues[sort_idx]

    # 计算累积位置（用于 X 轴）
    unique_chroms = sort(unique(chromosomes))
    cumulative_positions = zeros(Int, length(chromosomes))
    chromosome_centers = Dict{Int, Float64}()
    chromosome_boundaries = Dict{Int, Int}()

    cumulative_length = 0

    for chrom in unique_chroms
        chrom_idx = findall(chromosomes .== chrom)

        if !isempty(chrom_idx)
            # 该染色体的最大位置
            max_pos = maximum(positions[chrom_idx])

            # 更新累积位置
            cumulative_positions[chrom_idx] .= positions[chrom_idx] .+ cumulative_length

            # 染色体中心（用于标签）
            center = cumulative_length + max_pos / 2
            chromosome_centers[chrom] = center

            # 染色体边界
            chromosome_boundaries[chrom] = cumulative_length + max_pos

            # 更新累积长度
            cumulative_length += max_pos
        end
    end

    # 阈值线
    significant_line = -log10(significant_threshold)
    suggestive_line = -log10(suggestive_threshold)

    # 标记显著 SNP
    is_significant = log_pvalues .>= significant_line
    is_suggestive = log_pvalues .>= suggestive_line

    # 高亮 SNP
    is_highlighted = falses(length(snp_ids))
    if highlight_snps !== nothing
        for (i, snp) in enumerate(snp_ids)
            if snp in highlight_snps
                is_highlighted[i] = true
            end
        end
    end

    # 染色体颜色
    if chromosome_colors === nothing
        # 默认：交替使用两种颜色
        chromosome_colors = ["#1f77b4", "#ff7f0e"]
    end

    colors = [chromosome_colors[mod(chrom - 1, length(chromosome_colors)) + 1] for chrom in chromosomes]

    # 返回绘图数据
    return (
        snp_ids = snp_ids,
        chromosomes = chromosomes,
        positions = cumulative_positions,
        log_pvalues = log_pvalues,
        pvalues = pvalues,
        is_significant = is_significant,
        is_suggestive = is_suggestive,
        is_highlighted = is_highlighted,
        colors = colors,
        chromosome_centers = chromosome_centers,
        chromosome_boundaries = chromosome_boundaries,
        significant_threshold = significant_line,
        suggestive_threshold = suggestive_line,
        max_log_p = maximum(log_pvalues),
        n_significant = sum(is_significant),
        n_suggestive = sum(is_suggestive)
    )
end

"""
    manhattan_plot(gwas::GWASResult; kwargs...)

生成 Manhattan 图（返回绘图数据）。

这是 `prepare_manhattan_plot` 的别名。
"""
manhattan_plot = prepare_manhattan_plot

"""
    find_top_snps(gwas::GWASResult; n::Int=20)

查找 P 值最显著的 SNP。

# 参数
- `gwas`: GWASResult 对象
- `n`: 返回的 SNP 数量

# 返回
命名元组 (snp_ids, pvalues, chromosomes, positions)
"""
function find_top_snps(gwas::GWASResult; n::Int=20)
    # 找到最小的 n 个 P 值
    top_idx = partialsortperm(gwas.pvalues, 1:min(n, length(gwas.pvalues)))

    return (
        snp_ids = gwas.snp_ids[top_idx],
        pvalues = gwas.pvalues[top_idx],
        chromosomes = gwas.chromosomes[top_idx],
        positions = gwas.positions[top_idx],
        effect_sizes = gwas.effect_sizes !== nothing ? gwas.effect_sizes[top_idx] : nothing
    )
end

"""
    genomic_control(gwas::GWASResult)

计算基因组控制因子（Genomic Inflation Factor, λ）。

λ = median(χ²_observed) / median(χ²_expected)

用于检测群体分层或其他混杂因素导致的检验统计量膨胀。

# 返回
λ 值（期望为 1.0，> 1.0 表示膨胀）
"""
function genomic_control(gwas::GWASResult)
    # 将 P 值转换为卡方统计量
    # χ² = qchisq(1 - p, df=1)
    # 近似：对于小 p，χ² ≈ Φ⁻¹(1 - p/2)²

    valid_pvalues = filter(p -> !isnan(p) && p > 0 && p < 1, gwas.pvalues)

    # 计算 Z 分数
    z_scores = [quantile_normal(1 - p/2) for p in valid_pvalues]

    # 卡方统计量
    chi2_obs = z_scores .^ 2

    # 中位数
    median_chi2_obs = median(chi2_obs)
    median_chi2_exp = 0.4549  # χ²(1) 分布的中位数

    λ = median_chi2_obs / median_chi2_exp

    return λ
end

"""
正态分布的分位数函数（简化版本）
"""
function quantile_normal(p::Float64)
    # 使用 Beasley-Springer-Moro 近似
    # 这里使用简化版本
    if p <= 0.5
        return -quantile_normal(1 - p)
    end

    t = sqrt(-2 * log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    z = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)

    return z
end

# 导出
export GWASResult, prepare_manhattan_plot, manhattan_plot
export find_top_snps, genomic_control
