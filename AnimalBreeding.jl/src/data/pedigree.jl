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

    animals = Set(pedigree.animal)

    # 检查重复ID
    if length(unique(pedigree.animal)) != nrow(pedigree)
        @warn "警告: 谱系中发现重复的动物ID。"
    end

    # 检查父母ID的有效性
    missing_parents = 0
    for row in eachrow(pedigree)
        if row.sire != 0 && !(row.sire in animals)
            missing_parents += 1
        end
        if row.dam != 0 && !(row.dam in animals)
            missing_parents += 1
        end
    end
    if missing_parents > 0
        @warn "警告: 发现 $missing_parents 个父母ID不在动物列表中。这些ID将被视为基础动物。"
    end

    # 检查环路
    if has_pedigree_loops(pedigree)
        error("错误: 谱系中检测到环路（即有动物是自身的祖先）。")
    end

    @info "谱系验证完成。"
end

"""
    has_pedigree_loops(pedigree::DataFrame) -> Bool

一个辅助函数，用于检测谱系中是否存在环路。
"""
function has_pedigree_loops(pedigree::DataFrame)
    # 这是一个简化的检查，更鲁棒的方法是进行拓扑排序
    # 这里我们假设一个动物的ID总是大于其父母的ID
    # 实际项目中应使用更复杂的图算法
    for row in eachrow(pedigree)
        if row.sire != 0 && row.sire >= row.animal
            @debug "潜在环路: sire $(row.sire) >= animal $(row.animal)"
            return true
        end
        if row.dam != 0 && row.dam >= row.animal
            @debug "潜在环路: dam $(row.dam) >= animal $(row.animal)"
            return true
        end
    end
    return false
end