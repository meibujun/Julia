# ============================================================================
# 核心模块 - 模型规格定义
# AnimalBreeding.jl
# ============================================================================

"""
    RandomEffect

定义一个随机效应的结构体。

# 字段
- `name::String`: 效应的名称，例如 "animal" 或 "herd_year"。
- `type::Symbol`: 效应的类型。常见的类型有:
    - `:additive`: 加性遗传效应。
    - `:maternal`: 母系遗传效应。
    - `:permanent_env`: 永久环境效应。
    - `:iid`: 独立同分布的随机效应。
"""
struct RandomEffect
    name::String
    type::Symbol

    function RandomEffect(name::String, type::Symbol)
        new(name, type)
    end
end

"""
    ModelSpec

定义一个完整的混合线性模型规格。

# 字段
- `traits::Vector{String}`: 一个或多个待分析的性状名称。
- `fixed_effects::Vector{String}`: 模型中的固定效应列表。
- `random_effects::Vector{RandomEffect}`: 模型中的随机效应列表，每个元素是一个`RandomEffect`对象。
"""
mutable struct ModelSpec
    traits::Vector{String}
    fixed_effects::Vector{String}
    random_effects::Vector{RandomEffect}

    function ModelSpec(;
                      traits::Vector{String}=String[],
                      fixed::Vector{String}=String[],
                      random::Vector{RandomEffect}=RandomEffect[])
        new(traits, fixed, random)
    end
end

"""
    define_model(; kwargs...) -> ModelSpec

一个用户友好的函数，用于快速定义和创建一个`ModelSpec`对象。

# 参数
- `traits::Vector{String}`: 性状名称列表。
- `fixed::Vector{String}`: 固定效应列表。
- `random::Vector`: 随机效应列表。每个元素可以是一个 `(name, type)` 的元组或一个预先创建的 `RandomEffect` 对象。

# 返回
- `ModelSpec`: 一个配置好的模型规格对象。
"""
function define_model(;
                     traits::Vector{String}=String[],
                     fixed::Vector{String}=String[],
                     random::Vector=[])

    @info "定义混合线性模型..."

    # 将用户输入的随机效应列表转换为RandomEffect对象列表
    random_effects = RandomEffect[]
    for r in random
        if r isa Tuple && length(r) == 2 && r[1] isa String && r[2] isa Symbol
            push!(random_effects, RandomEffect(r[1], r[2]))
        elseif r isa RandomEffect
            push!(random_effects, r)
        else
            error("无效的随机效应定义: $r。应为 (名称::String, 类型::Symbol) 元组或 RandomEffect 对象。")
        end
    end

    model = ModelSpec(
        traits=traits,
        fixed_effects=fixed,
        random=random_effects
    )

    @info "模型定义完成: $(length(traits))个相干, $(length(fixed))个固定效应, $(length(random_effects))个随机效应"

    return model
end

"""
    Base.show(io::IO, model::ModelSpec)

重载`show`函数，以更友好的格式打印模型信息。
"""
function Base.show(io::IO, model::ModelSpec)
    println(io, "--- 模型规格 ---")
    println(io, "  性状: ", join(model.traits, ", "))
    println(io, "  固定效应: ", join(model.fixed_effects, ", "))
    println(io, "  随机效应:")
    for re in model.random_effects
        println(io, "    - $(re.name) (类型: $(re.type))")
    end
    println(io, "----------------")
end
