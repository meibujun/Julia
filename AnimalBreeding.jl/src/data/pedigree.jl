# ============================================================================
# 数据模块 - 谱系数据处理
# AnimalBreeding.jl
# ============================================================================

"""
    load_pedigree(filepath::String; validate::Bool=true) -> DataFrame

从CSV文件加载谱系数据，并按照标准格式整理父母信息。

# 参数
- `filepath::String`: 谱系文件的路径。
- `validate::Bool`: 是否在加载后立即进行验证。

# 返回
- `DataFrame`: 一个包含 `animal`, `sire`, `dam` 列的谱系数据框，父母未知以0表示。

# 示例
```julia
pedigree_df = load_pedigree("data/pedigree.csv")
```
"""
function load_pedigree(filepath::String; validate::Bool=true)
    @info "加载谱系数据: $filepath"

    pedigree = CSV.read(filepath, DataFrame)

    # 确保必需的列存在，同时兼容常见命名风格
    required_cols = Dict(
        :animal => ["animal", "id", "animal_id"],
        :sire   => ["sire", "father", "sire_id"],
        :dam    => ["dam", "mother", "dam_id"],
    )
    rename_map = Dict{String,String}()
    for (std_name, aliases) in required_cols
        col_found = findfirst(name -> name in names(pedigree), aliases)
        if isnothing(col_found)
            error("谱系文件 '$filepath' 缺少必需列: $(first(aliases))")
        end
        rename_map[aliases[col_found]] = String(std_name)
    end
    rename!(pedigree, rename_map)

    # 将缺失值（如NA）统一处理为0，代表未知亲本，同时确保列为整型
    for col in [:sire, :dam]
        pedigree[!, col] = coalesce.(pedigree[!, col], 0)
        pedigree[!, col] = convert.(Int, pedigree[!, col])
    end
    pedigree[!, :animal] = convert.(Int, pedigree[!, :animal])

    if validate
        validate_pedigree(pedigree)
    end

    @info "成功加载 $(nrow(pedigree)) 个个体的谱系数据。"
    return pedigree
end

"""
    _complete_pedigree(pedigree::DataFrame) -> DataFrame

内部辅助函数：补全仅在父母列出现但缺少记录的基础动物，并按ID升序排列。
"""
function _complete_pedigree(pedigree::DataFrame)
    all_ids = unique(vcat(pedigree.animal, pedigree.sire, pedigree.dam))
    filter!(id -> id != 0, all_ids)
    sort!(all_ids)

    sire_lookup = Dict(pedigree.animal .=> pedigree.sire)
    dam_lookup  = Dict(pedigree.animal .=> pedigree.dam)

    complete = DataFrame(animal=all_ids, sire=zeros(Int, length(all_ids)), dam=zeros(Int, length(all_ids)))
    for row in eachrow(complete)
        row.sire = get(sire_lookup, row.animal, 0)
        row.dam  = get(dam_lookup, row.animal, 0)
    end
    return complete
end

"""
    _topological_sort_pedigree(pedigree::DataFrame) -> Vector{Int}

使用Kahn算法对谱系进行拓扑排序，保证任何个体的父母均在其之前出现。
若检测到环路，则抛出异常提示。
"""
function _topological_sort_pedigree(pedigree::DataFrame)
    child_map = Dict{Int, Vector{Int}}()
    indegree = Dict(id => 0 for id in pedigree.animal)

    for row in eachrow(pedigree)
        for parent in (row.sire, row.dam)
            if parent == 0
                continue
            end
            push!(get!(child_map, parent, Int[]), row.animal)
            indegree[row.animal] = get(indegree, row.animal, 0) + 1
            if !haskey(indegree, parent)
                indegree[parent] = 0
            end
        end
    end

    queue = collect(id for id in keys(indegree) if indegree[id] == 0)
    sort!(queue)
    order = Int[]

    while !isempty(queue)
        current = popfirst!(queue)
        push!(order, current)
        for child in get(child_map, current, Int[])
            indegree[child] -= 1
            if indegree[child] == 0
                insert!(queue, searchsortedfirst(queue, child), child)
            end
        end
    end

    if length(order) != length(indegree)
        error("谱系中检测到环路或缺失父母信息，无法完成拓扑排序。")
    end

    return order
end

"""
    validate_pedigree(pedigree::DataFrame)

验证谱系数据的内部一致性和有效性，并在发现潜在问题时提供详细警告。
"""
function validate_pedigree(pedigree::DataFrame)
    @info "验证谱系数据..."

    animals = Set(pedigree.animal)

    # 检查重复ID
    if length(animals) != nrow(pedigree)
        counts = Dict{Int,Int}()
        for id in pedigree.animal
            counts[id] = get(counts, id, 0) + 1
        end
        duplicates = [id for (id, c) in counts if c > 1]
        @warn "谱系中发现重复的动物ID: $(duplicates)"
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
        @warn "发现 $missing_parents 个父母ID不在动物列表中。将在矩阵构建时自动补全。"
    end

    # 检查环路
    try
        _ = _topological_sort_pedigree(_complete_pedigree(pedigree))
    catch err
        error("错误: 谱系中检测到环路（即有动物是自身的祖先）。\n" * sprint(showerror, err))
    end

    @info "谱系验证完成。"
end

"""
    has_pedigree_loops(pedigree::DataFrame) -> Bool

保留向后兼容的API。若检测到环路返回`true`，否则返回`false`。
"""
function has_pedigree_loops(pedigree::DataFrame)
    try
        _ = _topological_sort_pedigree(_complete_pedigree(pedigree))
        return false
    catch
        return true
    end
end
