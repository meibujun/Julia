#=###############################################################################
# FAIR 元数据与模型持久化
# 负责记录模型训练过程的核心信息, 支持 JSON/BSON 双格式输出以满足可重用性要求。
###############################################################################=#

using BSON
using Dates
using JSON3
using UUIDs

"""
    mutable struct ModelMetadata

封装遵循 FAIR 原则的元数据, 提供唯一标识、数据来源、训练配置等信息。
"""
Base.@kwdef mutable struct ModelMetadata
    uuid::String = string(uuid4())
    model_type::String
    created_at::DateTime = Dates.now()
    metrics::Dict{String,Float64} = Dict{String,Float64}()
    hyperparameters::Dict{String,Any} = Dict{String,Any}()
    data_sources::Vector{String} = String[]
    notes::String = ""
    software_version::String = "GenomicPrediction-0.1.0"
    julia_version::String = string(VERSION)
    random_seed::Union{Nothing,Int} = nothing
end

"""
    create_metadata(model_type; kwargs...) -> ModelMetadata

快速构造元数据对象, 自动生成 UUID 与时间戳。
"""
function create_metadata(model_type::AbstractString;
                         metrics::Dict{String,Float64} = Dict{String,Float64}(),
                         hyperparameters::Dict{String,Any} = Dict{String,Any}(),
                         data_sources::Vector{String} = String[],
                         notes::AbstractString = "",
                         random_seed::Union{Nothing,Int} = nothing)
    return ModelMetadata(model_type = String(model_type),
                         metrics = copy(metrics),
                         hyperparameters = copy(hyperparameters),
                         data_sources = copy(data_sources),
                         notes = String(notes),
                         random_seed = random_seed)
end

"""
    enrich_metadata!(metadata; kwargs...) -> ModelMetadata

动态更新元数据字段, 支持增量追加数据来源、指标等。
"""
function enrich_metadata!(metadata::ModelMetadata;
                          metrics::Union{Nothing,Dict{String,Float64}} = nothing,
                          hyperparameters::Union{Nothing,Dict{String,Any}} = nothing,
                          data_sources::Union{Nothing,Vector{String}} = nothing,
                          notes::Union{Nothing,String} = nothing,
                          random_seed::Union{Nothing,Int} = nothing)
    isnothing(metrics) || merge!(metadata.metrics, metrics)
    isnothing(hyperparameters) || merge!(metadata.hyperparameters, hyperparameters)
    if !isnothing(data_sources)
        append!(metadata.data_sources, data_sources)
        metadata.data_sources = unique(metadata.data_sources)
    end
    isnothing(notes) || (metadata.notes = notes)
    isnothing(random_seed) || (metadata.random_seed = random_seed)
    return metadata
end

"""
    metadata_dict(metadata) -> Dict

转换为可 JSON 序列化的字典, 方便持久化与日志。
"""
function metadata_dict(metadata::ModelMetadata)
    return Dict(
        "uuid" => metadata.uuid,
        "model_type" => metadata.model_type,
        "created_at" => string(metadata.created_at),
        "metrics" => metadata.metrics,
        "hyperparameters" => metadata.hyperparameters,
        "data_sources" => metadata.data_sources,
        "notes" => metadata.notes,
        "software_version" => metadata.software_version,
        "julia_version" => metadata.julia_version,
        "random_seed" => metadata.random_seed
    )
end

"""
    save_model(model, path; metadata, extra=nothing)

将模型、元数据与附加信息保存为 BSON, 并同步输出 JSON 元数据副本。
"""
function save_model(model, path::AbstractString; metadata::ModelMetadata, extra = nothing)
    open(path, "w") do io
        BSON.@save io model metadata extra
    end
    json_path = endswith(path, ".bson") ? replace(path, ".bson" => "_metadata.json") : string(path, "_metadata.json")
    open(json_path, "w") do io
        JSON3.write(io, metadata_dict(metadata); indent = 2)
    end
    return path
end

"""
    load_model(path) -> (model, metadata, extra)

读取保存的 BSON 文件, 返回模型、元数据与附加对象。
"""
function load_model(path::AbstractString)
    data = BSON.load(path)
    model = data[:model]
    metadata = get(data, :metadata, nothing)
    extra = get(data, :extra, nothing)
    return model, metadata, extra
end

