# --- 模型定义模块 ---
# 该模块提供了一个灵活的接口，允许用户根据研究需求定义统计模型。
# 用户可以通过类似R语言的风格指定性状、固定效应、随机效应等。

"""
    ModelSpec

一个结构体(struct)，用于存储用户定义的线性混合模型的完整规格。
它像一个蓝图，详细描述了模型的结构，供后续的计算模块使用。

# 字段
- `traits::Vector{String}`: 一个字符串向量，包含所有待分析性状的名称。
- `fixed_effects::Vector{String}`: 一个字符串向量，包含所有固定效应的名称。
- `random_effects::Vector{Tuple{String, Symbol}}`: 一个元组向量，定义随机效应。
    - 每个元组的第一个元素是效应的名称（如 "animal"）。
    - 第二个元素是效应的类型（如 `:additive` 表示加性遗传效应）。
"""
struct ModelSpec
    traits::Vector{String}
    fixed_effects::Vector{String}
    random_effects::Vector{Tuple{String, Symbol}} # 例如: ("animal", :additive)
end

"""
    define_model(;traits::Vector{String}, fixed::Vector{String}, random::Vector{Tuple{String, Symbol}}) -> ModelSpec

一个用户友好的高级函数，用于创建 `ModelSpec` 对象。
它通过关键字参数接收模型定义，使其易于阅读和编写。

# 参数
- `traits::Vector{String}`: 指定一个或多个性状。
- `fixed::Vector{String}`: 指定模型中的固定效应。
- `random::Vector{Tuple{String, Symbol}}`: 指定模型中的随机效应及其类型。

# 返回
- `ModelSpec`: 一个根据输入参数构建的`ModelSpec`对象。

# 示例
```julia
# 定义一个单性状模型，包含herd和year作为固定效应，动物加性遗传效应作为随机效应
model = define_model(
    traits = ["milk_yield"],
    fixed = ["herd", "year"],
    random = [("animal", :additive)]
)
```
"""
function define_model(;traits::Vector{String}, fixed::Vector{String}, random::Vector{Tuple{String, Symbol}})
    # --- 输入验证 ---
    if isempty(traits)
        error("模型定义错误：必须至少指定一个性状。")
    end
    if isempty(random)
        error("模型定义错误：混合模型必须至少包含一个随机效应。")
    end

    # 检查随机效应的类型是否为当前支持的类型
    valid_random_types = [:additive, :permanent_env, :iid]
    for (effect, type) in random
        if type ∉ valid_random_types
            @warn "随机效应类型 '$type' 不是标准类型。当前支持的类型有: $valid_random_types"
        end
    end

    println("--- 模型定义成功 ---")
    println("性状: ", join(traits, ", "))
    println("固定效应: ", join(fixed, ", "))
    println("随机效应: ", join([string(r) for r in random], ", "))
    println("----------------------")

    return ModelSpec(traits, fixed, random)
end

"""
    (spec::ModelSpec)(dm::DataManager) -> Tuple{Vector, Matrix, SparseMatrixCSC}

这是一个函数对象 (functor)，它将 `ModelSpec` 应用于 `DataManager`，
从而构建出混合模型方程 (MME) 所需的核心矩阵和向量：`y`, `X`, 和 `Z`。

这是将符号化的模型定义转换为具体数值计算的关键步骤。

# 参数
- `spec::ModelSpec`: 定义模型结构的对象。
- `dm::DataManager`: 包含所有数据的对象。

# 返回
- `y::Vector`: 响应向量 (表型值)。
- `X::Matrix`: 固定效应的设计矩阵。
- `Z::SparseMatrixCSC`: 随机效应的设计矩阵 (关联矩阵)，通常是稀疏的。
"""
function (spec::ModelSpec)(dm::DataManager)
    # 当前版本暂时只支持单性状模型
    if length(spec.traits) > 1
        error("当前版本仅支持单性状模型。多性状模型功能正在开发中。")
    end
    trait = spec.traits[1]

    pheno = dm.phenotypes

    # --- 1. 构建响应向量 y ---
    if trait ∉ names(pheno)
        error("指定的性状 '$trait' 在表型数据中不存在。")
    end
    y = pheno[!, trait]

    # --- 2. 构建固定效应设计矩阵 X ---
    # 默认包含一个截距项 (intercept)
    X_cols = [ones(Float64, nrow(pheno))]
    # col_names = ["intercept"] # 用于调试和结果解释

    for effect in spec.fixed_effects
        if effect in names(pheno)
            # 对分类变量 (如 herd) 创建哑变量 (dummy variables)
            # eltype检查确保我们只对非数值类型或者被明确标记为分类的列进行哑变量处理
            if eltype(pheno[!, effect]) <: AbstractString || !(eltype(pheno[!, effect]) <: Number)
                dummies = create_dummy_variables(pheno[!, effect])
                X_cols = hcat(X_cols..., dummies)
            else # 对连续变量 (协变量，如 age) 直接使用其数值
                push!(X_cols, pheno[!, effect])
            end
        else
            @warn "固定效应 '$effect' 在表型数据中未找到，将被忽略。"
        end
    end
    X = hcat(X_cols...)

    # --- 3. 构建随机效应设计矩阵 Z ---
    # Z 矩阵将每个表型记录关联到相应的随机效应水平上（例如，将产奶记录关联到具体的某头牛）。
    # 对于简单的动物模型，Z是一个关联矩阵。

    # 当前仅支持 "animal" 作为随机效应
    if length(spec.random_effects) > 1 || spec.random_effects[1][1] != "animal"
        error("当前版本仅支持将 'animal' 作为唯一的随机效应。")
    end

    # Z矩阵的行数等于表型记录数，列数等于谱系中所有动物的总数
    animal_ids_pheno = pheno.animal
    all_animals_ped = sort(unique(dm.pedigree.animal)) # 确保顺序一致
    animal_map = Dict(id => i for (i, id) in enumerate(all_animals_ped))

    n_records = nrow(pheno)
    n_animals = length(all_animals_ped)

    # 使用COO格式 (I, J, V) 构建稀疏矩阵，效率最高
    I = Int[] # 行索引
    J = Int[] # 列索引
    V = Float64[] # 值 (通常为1.0)

    for i in 1:n_records
        animal_id = animal_ids_pheno[i]
        if haskey(animal_map, animal_id)
            j = animal_map[animal_id] # 获取该动物在Z矩阵中的列索引
            push!(I, i)
            push!(J, j)
            push!(V, 1.0)
        end
    end

    Z = sparse(I, J, V, n_records, n_animals)

    return y, X, Z
end

"""
    create_dummy_variables(v::AbstractVector) -> Matrix{Float64}

一个辅助函数，用于为分类向量创建哑变量矩阵。

为了避免多重共线性，如果一个分类变量有 k 个水平，该函数会生成 k-1 个哑变量。
第一个水平被用作基准（reference level）。

# 参数
- `v::AbstractVector`: 一个包含分类数据的向量。

# 返回
- `Matrix{Float64}`: 生成的哑变量矩阵。如果分类少于2个水平，返回一个空矩阵。
"""
function create_dummy_variables(v::AbstractVector)
    levels = unique(skipmissing(v))
    n_levels = length(levels)

    # 如果只有一个或零个水平，则无需创建哑变量
    if n_levels <= 1
        return zeros(length(v), 0)
    end

    # 创建 k-1 个哑变量
    dummies = zeros(length(v), n_levels - 1)
    for i in 2:n_levels
        # 将向量v中等于当前水平的元素位置标记为1
        dummies[:, i-1] = (v .== levels[i])
    end
    return dummies
end