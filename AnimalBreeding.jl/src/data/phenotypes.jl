# ============================================================================
# 数据模块 - 表型数据处理
# AnimalBreeding.jl
# ============================================================================

"""
    load_phenotypes(filepath::String;
                    trait_cols::Vector{String}=String[],
                    fixed_cols::Vector{String}=String[],
                    na_strings::Vector{String}=["NA", "na", "."]) -> DataFrame

从CSV文件加载表型数据。

# 参数
- `filepath::String`: 表型数据文件的路径。
- `trait_cols::Vector{String}`: 一个字符串向量，指定哪些列是性状。如果为空，则除ID和固定效应外的所有数值列都将被视为性状。
- `fixed_cols::Vector{String}`: 一个字符串向量，指定哪些列是固定效应。
- `na_strings::Vector{String}`: 一个字符串向量，定义在文件中应被视作缺失值的内容。

# 返回
- `DataFrame`: 一个包含表型数据的数据框。

# 示例
```julia
phenotypes = load_phenotypes(
    "data/phenotypes.csv",
    trait_cols=["milk_yield", "fat_percentage"],
    fixed_cols=["herd", "year_season"]
)
```
"""
function load_phenotypes(filepath::String;
                        trait_cols::Vector{String}=String[],
                        fixed_cols::Vector{String}=String[],
                        na_strings::Vector{String}=["NA", "na", "."])

    @info "加载表型数据: $filepath"

    phenotypes = CSV.read(filepath, DataFrame, missingstring=na_strings)

    # 标准化动物ID列名
    if "animal_id" in names(phenotypes)
        rename!(phenotypes, "animal_id" => "animal")
    elseif "id" in names(phenotypes)
        rename!(phenotypes, "id" => "animal")
    elseif !("animal" in names(phenotypes))
        error("表型文件必须包含一个动物标识列 (如 'animal', 'id', 或 'animal_id')。")
    end

    # 自动检测性状列（如果未指定）
    if isempty(trait_cols)
        @info "未指定性状列，将自动检测数值型列作为性状。"
        potential_trait_cols = String[]
        excluded_cols = vcat("animal", fixed_cols)
        for col_name in names(phenotypes)
            if !(col_name in excluded_cols) && eltype(skipmissing(phenotypes[!, col_name])) <: Number
                push!(potential_trait_cols, col_name)
            end
        end
        trait_cols = potential_trait_cols
        @info "自动识别的性状: " * join(trait_cols, ", ")
    end

    # 验证所有指定的列是否存在
    all_specified_cols = vcat("animal", trait_cols, fixed_cols)
    for col in all_specified_cols
        if !(col in names(phenotypes))
            error("指定的列 '$col' 在表型文件中不存在。")
        end
    end

    n_records = nrow(phenotypes)
    n_traits = length(trait_cols)

    # 打印缺失值摘要
    println("  表型数据摘要:")
    for trait in trait_cols
        n_missing = sum(ismissing.(phenotypes[!, trait]))
        missing_pct = round(n_missing / n_records * 100, digits=1)
        println("    - 性状 '$trait': $n_missing / $n_records ($missing_pct%) 条记录缺失。")
    end

    @info "成功加载 $n_records 条记录, 包含 $n_traits 个性状。"
    return phenotypes
end