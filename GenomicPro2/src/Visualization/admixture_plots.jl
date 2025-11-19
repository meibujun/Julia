"""
# ADMIXTURE 可视化

ADMIXTURE 分析结果的可视化辅助函数。
"""

"""
    prepare_admixture_plot(admix_results;
                           sample_ids::Union{Vector{String}, Nothing}=nothing,
                           sort_by_cluster::Bool=true,
                           colors::Union{Vector{String}, Nothing}=nothing)

准备 ADMIXTURE 条形图数据。

# 参数
- `admix_results`: ADMIXTUREResults 对象
- `sample_ids`: 样本标识符
- `sort_by_cluster`: 是否按主要群体排序
- `colors`: 祖先群体颜色

# 返回
绘图数据
"""
function prepare_admixture_plot(admix_results;
                                sample_ids::Union{Vector{String}, Nothing}=nothing,
                                sort_by_cluster::Bool=true,
                                colors::Union{Vector{String}, Nothing}=nothing)

    n_samples = size(admix_results.Q, 1)
    K = admix_results.K

    # 样本 ID
    if sample_ids === nothing
        sample_ids = string.(1:n_samples)
    end

    # 排序
    if sort_by_cluster
        # 按主要祖先群体排序
        primary_cluster = [argmax(admix_results.Q[i, :]) for i in 1:n_samples]
        sort_idx = sortperm(primary_cluster)
    else
        sort_idx = 1:n_samples
    end

    Q_sorted = admix_results.Q[sort_idx, :]
    ids_sorted = sample_ids[sort_idx]

    # 默认颜色
    if colors === nothing
        # 使用预定义的调色板
        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        colors = default_colors[1:min(K, length(default_colors))]
    end

    return (
        Q = Q_sorted,
        sample_ids = ids_sorted,
        K = K,
        colors = colors,
        ancestry_labels = ["Ancestry $k" for k in 1:K]
    )
end

admixture_barplot = prepare_admixture_plot

# 导出
export prepare_admixture_plot, admixture_barplot
