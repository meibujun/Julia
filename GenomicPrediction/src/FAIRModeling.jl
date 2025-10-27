# FAIRModeling.jl - FAIR 数据与模型管理模块
# ----------------------------------------------------------------
# 遵循 FAIR 原则 (Findable, Accessible, Interoperable, Reusable)
# 提供模型和数据的保存、加载、元数据管理功能，确保可复现性。
# ----------------------------------------------------------------

module FAIRModeling

using BSON
using ..DataProcessing
using ..CoreAlgorithm

export save_model, load_model, view_model_metadata

using Dates
using Pkg

"""
    save_model(model, filepath::String)

将一个训练好的模型对象及其元数据序列化并保存到 BSON 文件。

元数据包括模型类型、参数、保存时间以及软件版本，以遵循 FAIR 原则。

# 参数
- `model`: 训练好的模型实例。
- `filepath::String`: 目标文件的路径。
"""
function save_model(model, filepath::String)
    println("正在将模型保存至 $filepath ...")

    # 1. 提取模型参数
    model_params = Dict(fn => getfield(model, fn) for fn in fieldnames(typeof(model)) if fn != :effects && fn != :history && fn != :chain && fn != :optimizer && fn != :path)

    # 2. 生成元数据
    metadata = Dict(
        :model_type => string(typeof(model)),
        :model_parameters => model_params,
        :save_timestamp => now(),
        :julia_version => string(VERSION),
        :package_version => coalesce(Pkg.project().version, "dev")
    )

    # 3. 将模型和元数据一起保存
    BSON.bson(filepath, Dict(:model => model, :metadata => metadata))
    println("模型及元数据保存成功。")
end

"""
    load_model(filepath::String)

从 BSON 文件中反序列化并加载一个模型对象。

# 参数
- `filepath::String`: BSON 文件的路径。

# 返回
- 加载的模型对象。

# 示例
```julia
# loaded_model = load_model("my_gblup_model.bson")
```
"""
function load_model(filepath::String)
    println("正在从 $filepath 加载模型...")
    data = BSON.load(filepath)
    model = data[:model]
    println("模型加载成功。")
    return model # 确保返回加载的模型
end


"""
    view_model_metadata(filepath::String) -> Dict

加载并显示 BSON 文件中存储的模型的元数据，而不加载整个模型。

# 参数
- `filepath::String`: BSON 文件的路径。

# 返回
- `Dict`: 包含模型元数据的字典。
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
