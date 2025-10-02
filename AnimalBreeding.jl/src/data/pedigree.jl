# ============================================================================
# 数据模块 - 谱系数据处理
# AnimalBreeding.jl
# ============================================================================

"""
    load_pedigree(filepath::String; validate::Bool=true) -> DataFrame

从CSV文件加载谱系数据。

# 参数
- `filepath::String`: 谱系文件的路径。
- `validate::Bool`: 是否在加载后立即进行验证。

# 返回
- `DataFrame`: 一个包含 `animal`, `sire`, `dam` 列的谱系数据框。

# 示例
```julia
pedigree_df = load_pedigree("data/pedigree.csv")
```
"""
function load_pedigree(filepath::String; validate::Bool=true)
    @info "加载谱系数据: $filepath"

    # 从CSV文件读取数据
    pedigree = CSV.read(filepath, DataFrame)

    # 确保必需的列存在
    required_cols = ["animal", "sire", "dam"]
    for col in required_cols
        if !(col in names(pedigree))
            error("谱系文件 '$filepath' 缺少必需列: $col")
        end
    end

    # 将缺失值（如NA）统一处理为0，代表未知亲本
    for col in ["sire", "dam"]
        pedigree[!, col] = coalesce.(pedigree[!, col], 0)
    end

    if validate
        validate_pedigree(pedigree)
    end

    @info "成功加载 $(nrow(pedigree)) 个个体的谱系数据。"
    return pedigree
end

"""
    validate_pedigree(pedigree::DataFrame)

验证谱系数据的内部一致性和有效性。

# 检查项
- 是否有重复的动物ID。
- 父母ID是否存在于动物列表中（未知亲本除外）。
- 是否存在环路（即个体是自己的祖先）。
"""
function validate_pedigree(pedigree::DataFrame)
    @info "验证谱系数据..."

    if !all(x -> x in names(pedigree), ("animal", "sire", "dam"))
        error("谱系数据必须包含 'animal', 'sire', 'dam' 三个字段。")
    end

    animals = collect(skipmissing(pedigree.animal))
    n_unique = length(unique(animals))
    if n_unique != nrow(pedigree)
        @warn "谱系中存在重复的动物ID。重复的行将在排序时被保留最先出现的记录。"
    end

    # 构建父母索引映射
    animal_set = Set(animals)
    unknown_parent_ids = Int(0)
    for parent_col in ("sire", "dam")
        for parent in pedigree[!, parent_col]
            if ismissing(parent) || parent == 0
                continue
            elseif !(parent in animal_set)
                unknown_parent_ids += 1
            end
        end
    end
    if unknown_parent_ids > 0
        @warn "发现 $unknown_parent_ids 个未在动物列表中的父母ID，这些个体将视为基础动物。"
    end

    # 使用深度优先搜索检测环路
    parent_lookup = Dict{Any, Tuple{Any,Any}}()
    for row in eachrow(pedigree)
        parent_lookup[row.animal] = (row.sire, row.dam)
    end

    state = Dict{Any, Symbol}()
    for animal in animals
        state[animal] = :white
    end

    function dfs(node)
        current_state = get(state, node, :black)
        current_state === :black && return false
        if current_state === :gray
            return true
        end
        state[node] = :gray
        sire, dam = get(parent_lookup, node, (0, 0))
        for parent in (sire, dam)
            if ismissing(parent) || parent == 0
                continue
            end
            if dfs(parent)
                return true
            end
        end
        state[node] = :black
        return false
    end

    for animal in animals
        if dfs(animal)
            error("谱系验证失败：检测到环路 (个体 $animal 是其自身的祖先)。")
        end
    end

    @info "谱系验证完成。"
    return true
end
