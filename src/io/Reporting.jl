module Reporting

using DataFrames
using TOML
using LinearAlgebra
using Statistics

"""
    save_report(path, metadata, tables)

生成结构化分析报告，包含元数据和各类统计表。
"""
function save_report(path::AbstractString, metadata::Dict, tables::Dict{Symbol,DataFrame})
    io = open(path, "w")
    try
        println(io, "# 稀有变异上位性 Meta 分析报告")
        println(io, "\n## 元数据")
        for (k, v) in metadata
            println(io, "- $k: $v")
        end
        println(io, "\n## 结果表格")
        for (name, df) in tables
            println(io, "\n### $(String(name))")
            show(io, MIME("text/plain"), df)
            println(io, "\n")
        end
    finally
        close(io)
    end
end

end # module
