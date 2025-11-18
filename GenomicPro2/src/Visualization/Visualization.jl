"""
# 可视化工具模块

提供基因组分析的各种可视化功能。

## 主要功能
- Manhattan 图：GWAS 结果可视化
- QQ 图：P 值分布检验
- PCA 图：群体结构可视化
- ADMIXTURE 条形图：混合比例可视化
- LD 热图：连锁不平衡可视化

## 使用示例
```julia
using GenomicPro2.Visualization

# Manhattan 图
manhattan_data = prepare_manhattan_plot(gwas_results)

# QQ 图
qq_data = prepare_qq_plot(pvalues)

# 导出为 JSON（用于 Web 可视化）
export_plot_data("manhattan.json", manhattan_data)
```

## 输出格式
所有可视化函数返回结构化数据，可用于：
1. Julia 原生绘图（Plots.jl, Makie.jl）
2. Web 可视化（导出为 JSON，用于 D3.js/Plotly.js）
3. R 语言绘图（通过 RCall.jl）
"""
module Visualization

using Statistics
using LinearAlgebra
using ..Core: ValidationError

# 包含子模块
include("manhattan.jl")
include("qqplot.jl")
include("pca_plots.jl")
include("admixture_plots.jl")

# 导出
export prepare_manhattan_plot, manhattan_plot
export prepare_qq_plot, qq_plot
export prepare_pca_plot, pca_scatter_plot
export prepare_admixture_plot, admixture_barplot
export export_plot_data

"""
    export_plot_data(filename::String, data; format::Symbol=:json)

将绘图数据导出为文件。

# 参数
- `filename`: 输出文件名
- `data`: 绘图数据（命名元组或字典）
- `format`: 输出格式（:json, :csv, :tsv）

# 示例
```julia
manhattan_data = prepare_manhattan_plot(gwas_results)
export_plot_data("manhattan.json", manhattan_data, format=:json)
```
"""
function export_plot_data(filename::String, data; format::Symbol=:json)
    if format == :json
        # 简单的 JSON 序列化（实际应用中建议使用 JSON3.jl）
        open(filename, "w") do io
            write(io, to_json(data))
        end
    elseif format == :csv || format == :tsv
        delimiter = format == :csv ? "," : "\t"
        # 实现 CSV/TSV 导出
        # ...
    else
        throw(ValidationError("不支持的格式: $format"))
    end

    @info "绘图数据已导出到: $filename"
end

"""
简单的 JSON 序列化（生产环境建议使用 JSON3.jl）
"""
function to_json(data)
    if isa(data, Dict) || isa(data, NamedTuple)
        items = []
        for (k, v) in pairs(data)
            push!(items, "\"$k\": $(to_json(v))")
        end
        return "{" * join(items, ", ") * "}"
    elseif isa(data, Array)
        items = [to_json(x) for x in data]
        return "[" * join(items, ", ") * "]"
    elseif isa(data, AbstractString)
        return "\"$data\""
    elseif isa(data, Number)
        return string(data)
    elseif isa(data, Nothing)
        return "null"
    else
        return "\"$(string(data))\""
    end
end

end # module Visualization
