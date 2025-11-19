"""
# PCA 可视化

PCA 分析结果的可视化辅助函数。
"""

"""
    prepare_pca_plot(pca_results;
                     pc_x::Int=1,
                     pc_y::Int=2,
                     labels::Union{Vector{String}, Nothing}=nothing,
                     groups::Union{Vector{Int}, Nothing}=nothing)

准备 PCA 散点图数据。

# 参数
- `pca_results`: PCAResults 对象
- `pc_x`: X 轴主成分编号
- `pc_y`: Y 轴主成分编号
- `labels`: 样本标签
- `groups`: 样本分组（用于着色）

# 返回
绘图数据
"""
function prepare_pca_plot(pca_results;
                          pc_x::Int=1,
                          pc_y::Int=2,
                          labels::Union{Vector{String}, Nothing}=nothing,
                          groups::Union{Vector{Int}, Nothing}=nothing)

    n_samples = size(pca_results.scores, 1)

    x = pca_results.scores[:, pc_x]
    y = pca_results.scores[:, pc_y]

    var_x = pca_results.explained_variance[pc_x] * 100
    var_y = pca_results.explained_variance[pc_y] * 100

    return (
        x = x,
        y = y,
        labels = labels !== nothing ? labels : string.(1:n_samples),
        groups = groups,
        pc_x = pc_x,
        pc_y = pc_y,
        variance_x = var_x,
        variance_y = var_y,
        xlabel = "PC$pc_x ($(round(var_x, digits=2))%)",
        ylabel = "PC$pc_y ($(round(var_y, digits=2))%)"
    )
end

pca_scatter_plot = prepare_pca_plot

# 导出
export prepare_pca_plot, pca_scatter_plot
