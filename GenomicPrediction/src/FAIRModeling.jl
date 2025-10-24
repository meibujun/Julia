# FAIRModeling.jl: FAIR 建模模块
# ---------------------------------
#
# 本模块旨在遵循 FAIR (Findable, Accessible, Interoperable, Reusable) 原则，
# 对基因组预测模型及其相关元数据进行管理。
#
# 主要功能:
# - **模型持久化**: 提供统一的接口，用于将训练好的模型对象保存到文件，
#   以及从文件中加载模型，确保模型的可重用性。
# - **元数据管理**: (未来实现) 自动记录和保存模型训练的元数据，
#   例如训练数据来源、超参数、软件版本、性能指标等，以保证分析过程的可追溯性。
# - **模型共享**: (未来实现) 支持将模型和元数据发布到公共仓库，
#   提高模型的可发现性和可获取性。

module FAIRModeling

using BSON

export save_model, load_model

"""
    save_model(model, filepath::String)

将一个训练好的模型对象序列化并保存到指定的文件。

该函数使用 BSON (Binary JSON) 格式，该格式能够高效地存储 Julia 的任意对象，
包括复杂的神经网络模型。

# 参数
- `model`: 任何可序列化的 Julia 对象，通常是一个训练好的模型实例 (例如 `GBLUPModel`)。
- `filepath::String`: 目标文件的路径。推荐使用 `.bson` 作为文件扩展名。

# 示例
```julia
using GenomicPrediction

# (假设 model 是一个已训练的模型)
save_model(model, "my_gblup_model.bson")
```
"""
function save_model(model, filepath::String)
    println("正在将模型保存到: $filepath ...")
    bson(filepath, Dict(:model => model))
    println("模型保存成功。")
    return nothing
end

"""
    load_model(filepath::String)

从 BSON 文件中反序列化并加载一个模型对象。

# 参数
- `filepath::String`: 包含模型对象的 BSON 文件的路径。

# 返回
- 加载的模型对象。返回对象的类型取决于文件中存储的内容。

# 示例
```julia
loaded_model = load_model("my_gblup_model.bson")
predict(loaded_model, new_data)
```
"""
function load_model(filepath::String)
    println("正在从 $filepath 加载模型...")
    data = BSON.load(filepath)
    println("模型加载成功。")
    return data[:model]
end

end # module FAIRModeling
