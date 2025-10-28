# FAIRModeling.jl - FAIR 数据与模型管理模块
# ==========================================================
# 遵循 FAIR 原则，提供模型的保存、加载和元数据管理功能。
# ==========================================================

module FAIRModeling

using BSON
using Dates
using Pkg
using ..GenomicPrediction: AbstractModel

export save_model, load_model, view_model_metadata

@doc raw"""
    save_model(model::AbstractModel, filepath::String)

将模型及其元数据保存到 BSON 文件。
"""
function save_model(model::AbstractModel, filepath::String)
    println("正在将模型保存至 $filepath ...")

    metadata = Dict(
        :model_type => string(typeof(model)),
        :save_timestamp => now(),
        :julia_version => string(VERSION),
        :package_version => coalesce(Pkg.project().version, "dev")
    )

    BSON.bson(filepath, Dict(:model => model, :metadata => metadata))
    println("模型及元数据保存成功。")
end

@doc raw"""
    load_model(filepath::String) -> AbstractModel

从 BSON 文件加载模型。
"""
function load_model(filepath::String)
    println("正在从 $filepath 加载模型...")
    data = BSON.load(filepath)
    println("模型加载成功。")
    return data[:model]
end

@doc raw"""
    view_model_metadata(filepath::String) -> Dict

查看存储在 BSON 文件中的模型的元数据。
"""
function view_model_metadata(filepath::String)
    println("正在从 $filepath 读取元数据...")
    data = BSON.load(filepath)
    if haskey(data, :metadata)
        return data[:metadata]
    else
        @warn "该模型文件不包含元数据。"
        return Dict()
    end
end

end # module FAIRModeling
