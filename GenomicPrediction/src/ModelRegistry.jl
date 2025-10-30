# ModelRegistry.jl - 模型注册系统
# ==========================================================
# 提供一个动态的模型注册和创建机制，以增强软件包的可扩展性。
# 用户和插件可以通过此系统注册自定义模型，使其能被工作流 API 无缝调用。
# ==========================================================

module ModelRegistry

using ..GenomicPrediction: AbstractModel

# --- 1. 全局模型注册表 ---
const MODEL_REGISTRY = Dict{String, Function}()

# --- 2. 模块接口 ---
export register_model, create_model, list_models

# --- 3. 功能实现 ---

@doc raw"""
    register_model(name::String, constructor)
将一个新模型及其构造函数注册到全局注册表中。

# Arguments
- `name::String`: 模型的唯一名称（例如， "gblup", "bayes_a"）。
- `constructor`: 一个接受参数字典并返回 `AbstractModel` 实例的函数。
"""
function register_model(name::String, constructor)
    if haskey(MODEL_REGISTRY, name)
        @warn "模型 '$name' 已被注册，将被覆盖。"
    end
    MODEL_REGISTRY[name] = constructor
    println("模型 '$name' 已成功注册。")
end

@doc raw"""
    create_model(name::String, params::Dict) -> AbstractModel
根据名称和参数从注册表中创建一个模型实例。
"""
function create_model(name::String, params::Dict)
    if !haskey(MODEL_REGISTRY, name)
        error("模型 '$name' 未在注册表中找到。可用模型: $(join(keys(MODEL_REGISTRY), ", "))")
    end

    constructor = MODEL_REGISTRY[name]
    try
        return constructor(params)
    catch e
        error("创建模型 '$name' 失败，参数: $params. 错误: $e")
    end
end

@doc raw"""
    list_models() -> Vector{String}
返回所有已注册模型的名称列表。
"""
function list_models()
    return sort(collect(keys(MODEL_REGISTRY)))
end

end # module ModelRegistry
