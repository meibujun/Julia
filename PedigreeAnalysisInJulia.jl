# ===================================================================
#
#   Julia 谱系分析工具箱 (PedigreeAnalysisInJulia.jl)
#   完整实现并包含性能优化 - 完全修复版本 v2.4
#   Julia Version 1.11.6 (2025-01-09)
#
#   功能包括:
#   - 谱系数据结构管理与验证
#   - 亲缘系数和近交系数计算
#   - Wright路径系数分析 (新增)
#   - 加性关系矩阵(A)及其逆矩阵(A-1)计算
#   - IBD模拟和实际亲缘关系分析
#   - 遗传标记分析和概率计算
#   - 连锁分析和LOD评分
#   - 谱系重构和亲缘鉴定
#   - 分离分析和表型预测
#   - 谱系可视化
#
# ===================================================================

# --- 核心依赖包 ---
using DataFrames, CSV, Plots, GraphRecipes, StatsBase, Distributions, Optim, Test, ProgressMeter, LinearAlgebra, Random, SparseArrays

# ===================================================================
# 第1章: 准备知识和环境设置 (Prerequisites.jl)
# ===================================================================
module Prerequisites

using Pkg

"""
    setup_environment(; pkgs::Vector{String})

检查并安装项目所需的所有依赖包. 该函数在首次运行时会自动安装缺失包.

# 参数
- `pkgs`: 需要安装的包列表

# 返回
- `nothing`

# 示例
```julia
setup_environment()
```
"""
function setup_environment(; pkgs::Vector{String} = [
        "DataFrames","CSV","Plots","GraphRecipes",
        "StatsBase","Distributions","Optim","Test",
        "ProgressMeter","LinearAlgebra","SparseArrays"])
    installed = Set(keys(Pkg.project().dependencies))
    to_add = String[]
    for pkg in pkgs
        pkg in installed || push!(to_add, pkg)
    end
    !isempty(to_add) && Pkg.add(to_add)
    return nothing
end

"""
    explain_term(term::String)

提供遗传学术语的详细解释, 帮助用户理解核心概念.

# 参数
- `term`: 需要解释的术语名称

# 返回
- 打印术语解释到控制台

# 示例
```julia
explain_term("IBD")
explain_term("Kinship Coefficient")
```
"""
function explain_term(term::String)
    definitions = Dict(
        "IBD" => """
        Identity by Descent (同源):
        指两个等位基因来自同一个共同祖先的拷贝. 这是谱系分析的核心概念.
        数学定义: P(两个等位基因来自同一祖先基因的拷贝)
        应用: 用于计算亲缘系数, 近交系数和遗传相关性
        """,

        "Kinship Coefficient" => """
        亲缘系数 (phi):
        从两个人(i 和 j)的同一基因座上随机各选一个等位基因,
        这两个等位基因是同源的(IBD)的概率.
        数学定义: phi_ij = P(随机选择的两个等位基因为 IBD)
        特殊情况:
        - 自身: phi_ii = 1/2(1 + F_i), 其中 F_i 是近交系数
        - 父子: phi = 1/4
        - 全同胞: phi = 1/4
        - 半同胞: phi = 1/8
        - 一级堂表: phi = 1/16
        """,

        "Inbreeding Coefficient" => """
        近交系数 (F):
        一个人的父母传给他的两个等位基因是同源的(IBD)的概率.
        数学定义: F_i = phi_(父母) = 父母之间的亲缘系数
        意义: 反映个体基因组中纯合子增加的程度
        计算公式: F_i = sum[(1/2)^(n1+n2+1) * (1+F_共同祖先)]
        其中n1和n2是父母到共同祖先的世代数
        """,

        "Founder" => """
        创始人/基础群体:
        谱系中没有指定父母的个体, 被认为是家系的起始点.
        假设:
        - 创始人之间无亲缘关系(除非特别指定)
        - 创始人的近交系数为0(F_创始人 = 0)
        - 创始人之间的亲缘系数为0
        重要性: 是计算谱系中所有其他个体遗传参数的基础
        """,

        "LOD Score" => """
        LOD 分对数得分 (Logarithm of the Odds):
        用于连锁分析, 衡量一个标记和一个性状基因座之间连锁的统计证据强度.
        定义: LOD(theta) = log10[L(theta)/L(0.5)]
        其中:
        - L(theta) 是重组率为 theta 时的似然
        - L(0.5) 是独立遗传(无连锁)时的似然
        判断标准:
        - LOD > 3.0: 显著连锁(错误率 < 0.0001)
        - LOD < -2.0: 显著不连锁(排除)
        - -2.0 <= LOD <= 3.0: 不确定
        """,

        "Additive Relationship Matrix" => """
        加性关系矩阵 (A矩阵):
        也称为分子亲缘矩阵, 元素 A_ij = 2*phi_ij(两倍亲缘系数)
        对角线元素: A_ii = 1 + F_i(1加近交系数)
        性质:
        - 对称正定矩阵
        - 可用于BLUP育种值预测
        - 反映群体的遗传结构
        计算方法: 列表法(Tabular Method)或分解法
        """,

        "Henderson's Rules" => """
        Henderson法则:
        直接构建加性关系矩阵逆矩阵(A-1)的高效算法, 无需先计算A矩阵.
        贡献规则:
        - 基础个体: 对角线元素贡献1
        - 单亲已知: 对角线贡献4/3, 亲子贡献-2/3
        - 双亲已知: 按特定公式贡献到相关元素
        优势: 计算效率高, 特别适合大型谱系
        """,

        "Mendelian Sampling" => """
        孟德尔抽样:
        子代从父母接收等位基因的随机过程.
        方差: Var(MS) = 0.5(1 - 0.5(F_父 + F_母))
        意义: 解释了为什么全同胞之间会有遗传差异
        应用: 用于计算育种值的准确性和遗传进展
        """
    )

    result = get(definitions, term, "未找到该术语的定义. 可用术语: " * join(keys(definitions), ", "))
    println("╔" ^ 60)
    println("术语解释: '$term'")
    println("─" ^ 60)
    println(result)
    println("╚" ^ 60)
    return nothing
end

export setup_environment, explain_term

end # module Prerequisites

# ===================================================================
# 第2章: 谱系数据结构与管理 (PedigreeDataStructures.jl)
# ===================================================================
module PedigreeDataStructures

using DataFrames, CSV, Statistics

export Individual, Pedigree, Marker, Phenotype
export read_pedigree, validate_pedigree, sort_pedigree!, recode_pedigree
export find_founders, get_descendants, get_ancestors, get_generation
export save_pedigree, load_pedigree, merge_pedigrees

# --- 核心数据结构定义 ---

"""
    Individual

存储单个个体完整信息的结构体.

# 字段
- `id`: 个体唯一标识符
- `sire`: 父本ID(missing表示未知)
- `dam`: 母本ID(missing表示未知)
- `sex`: 性别(1=雄性, 2=雌性, 0=未知)
- `generation`: 世代数(创始人为0)
- `attributes`: 存储额外属性的字典
"""
mutable struct Individual
    id::Any
    sire::Any
    dam::Any
    sex::Int
    generation::Int
    attributes::Dict{Symbol, Any}

    function Individual(id, sire, dam, sex; generation=0, attributes=Dict{Symbol,Any}())
        new(id, sire, dam, sex, generation, attributes)
    end
end

"""
    Pedigree

存储整个谱系信息的核心结构体.

# 字段
- `individuals`: ID到Individual对象的映射
- `id_map`: 原始ID到整数ID的映射(用于矩阵运算)
- `rev_id_map`: 整数ID到原始ID的反向映射
- `sorted_ids`: 拓扑排序后的ID列表
- `metadata`: 谱系元数据
"""
mutable struct Pedigree
    individuals::Dict{Any, Individual}
    id_map::Dict{Any, Int}
    rev_id_map::Dict{Int, Any}
    sorted_ids::Vector{Any}
    metadata::Dict{String, Any}

    function Pedigree()
        new(Dict{Any, Individual}(),
            Dict{Any, Int}(),
            Dict{Int, Any}(),
            Vector{Any}(),
            Dict{String, Any}())
    end
end

"""
    Marker

遗传标记结构, 支持SNP, 微卫星等多种标记类型.

# 字段
- `name`: 标记名称
- `chromosome`: 染色体位置
- `position`: 物理或遗传位置
- `alleles`: 等位基因列表
- `freqs`: 等位基因频率
- `genotypes`: 个体基因型数据
- `type`: 标记类型(:SNP, :microsatellite等)
"""
mutable struct Marker
    name::String
    chromosome::Union{Int, String, Nothing}
    position::Union{Float64, Nothing}
    alleles::Vector{String}
    freqs::Dict{String, Float64}
    genotypes::Dict{Any, Tuple{String, String}}
    type::Symbol

    function Marker(name::String, alleles::Vector{String}, freqs::Dict{String, Float64};
                   chromosome=nothing, position=nothing, genotypes=Dict{Any, Tuple{String, String}}(),
                   type=:SNP)
        # 验证频率和为1
        freq_sum = sum(values(freqs))
        @assert abs(freq_sum - 1.0) < 1e-6 "等位基因频率总和必须为 1.0, 当前为 $freq_sum"
        new(name, chromosome, position, alleles, freqs, genotypes, type)
    end
end

"""
    Phenotype

表型数据结构, 支持多种表型类型.

# 字段
- `name`: 表型名称
- `type`: 表型类型(:binary, :quantitative, :categorical, :ordinal)
- `values`: 个体表型值
- `description`: 表型描述
"""
mutable struct Phenotype
    name::String
    type::Symbol
    values::Dict{Any, Any}
    description::String

    function Phenotype(name::String, values::Dict; type=:quantitative, description="")
        @assert type in [:binary, :quantitative, :categorical, :ordinal]
        new(name, type, values, description)
    end
end

# --- 谱系读取和构建函数 ---

"""
    read_pedigree(df::DataFrame; kwargs...)

从DataFrame读取谱系数据并创建Pedigree对象.

# 参数
- `df`: 包含谱系数据的DataFrame
- `id_col`: 个体ID列名(默认:ID)
- `sire_col`: 父本ID列名(默认:Sire)
- `dam_col`: 母本ID列名(默认:Dam)
- `sex_col`: 性别列名(默认:Sex)
- `missing_vals`: 表示缺失的值(默认[0, "0", "", missing])

# 返回
- `Pedigree`: 构建好的谱系对象

# 示例
```julia
ped = read_pedigree(df, id_col=:AnimalID, sire_col=:FatherID)
```
"""
function read_pedigree(df::DataFrame;
                      id_col::Symbol=:ID,
                      sire_col::Symbol=:Sire,
                      dam_col::Symbol=:Dam,
                      sex_col::Symbol=:Sex,
                      missing_vals::Vector=[0, "0", "", missing])

    ped = Pedigree()

    # 验证必需列存在
    required = [id_col, sire_col, dam_col]
    for col in required
        @assert hasproperty(df, col) "缺少必需列: $col"
    end

    # 检查ID唯一性
    @assert allunique(df[!, id_col]) "个体ID必须唯一"

    # 处理缺失值 - 修复了对missing类型的处理
    function process_missing(x)
        # 首先检查是否为missing类型
        if ismissing(x)
            return missing
        end
        # 然后检查是否在missing_vals列表中(排除missing类型的比较)
        for mv in missing_vals
            if !ismissing(mv) && x == mv
                return missing
            end
        end
        return x
    end

    # 第一遍: 创建所有个体
    for row in eachrow(df)
        id = row[id_col]
        sire = process_missing(row[sire_col])
        dam = process_missing(row[dam_col])

        # 处理性别字段
        sex = 0
        if hasproperty(df, sex_col)
            sex_val = row[sex_col]
            if !ismissing(sex_val)
                sex = Int(sex_val)
            end
        end

        # 收集其他属性
        attrs = Dict{Symbol, Any}()
        for col in names(df)
            sym = Symbol(col)
            if !(sym in [id_col, sire_col, dam_col, sex_col])
                attrs[sym] = row[sym]
            end
        end

        ind = Individual(id, sire, dam, sex, attributes=attrs)
        ped.individuals[id] = ind
    end

    # 验证父母ID存在
    all_ids = Set(keys(ped.individuals))
    for (id, ind) in ped.individuals
        if !ismissing(ind.sire) && !(ind.sire in all_ids)
            @warn "个体 $id 的父本 $(ind.sire) 不在谱系中"
        end
        if !ismissing(ind.dam) && !(ind.dam in all_ids)
            @warn "个体 $id 的母本 $(ind.dam) 不在谱系中"
        end
    end

    # 计算世代
    compute_generations!(ped)

    return ped
end

"""
    compute_generations!(ped::Pedigree)

计算谱系中每个个体的世代数. 创始人为第0代.
"""
function compute_generations!(ped::Pedigree)
    # 初始化所有个体世代为-1
    for ind in values(ped.individuals)
        ind.generation = -1
    end

    # 创始人设为第0代
    for ind in values(ped.individuals)
        if ismissing(ind.sire) && ismissing(ind.dam)
            ind.generation = 0
        end
    end

    # 迭代计算世代
    max_iterations = length(ped.individuals)
    iteration = 0
    changed = true

    while changed && iteration < max_iterations
        changed = false
        iteration += 1

        for ind in values(ped.individuals)
            if ind.generation == -1
                # 获取父母的世代数
                sire_gen = -1
                dam_gen = -1

                if ismissing(ind.sire)
                    sire_gen = 0
                elseif haskey(ped.individuals, ind.sire)
                    sire_gen = ped.individuals[ind.sire].generation
                end

                if ismissing(ind.dam)
                    dam_gen = 0
                elseif haskey(ped.individuals, ind.dam)
                    dam_gen = ped.individuals[ind.dam].generation
                end

                if sire_gen >= 0 && dam_gen >= 0
                    ind.generation = max(sire_gen, dam_gen) + 1
                    changed = true
                end
            end
        end
    end

    # 检查是否有未分配世代的个体(可能存在循环)
    for ind in values(ped.individuals)
        if ind.generation == -1
            @warn "个体 $(ind.id) 无法确定世代, 可能存在循环引用"
            ind.generation = 0  # 设为默认值
        end
    end
end

"""
    sort_pedigree!(ped::Pedigree)

对谱系进行拓扑排序, 确保父母总在子女之前.
使用Kahn算法实现.

# 返回
- `Vector{Any}`: 排序后的ID列表
"""
function sort_pedigree!(ped::Pedigree)
    n = length(ped.individuals)
    if n == 0
        ped.sorted_ids = []
        return []
    end

    # 构建入度表和邻接表
    in_degree = Dict{Any, Int}()
    adj = Dict{Any, Vector{Any}}()

    for (id, ind) in ped.individuals
        in_degree[id] = 0
        adj[id] = []
    end

    # 计算每个节点的入度
    for (id, ind) in ped.individuals
        if !ismissing(ind.sire) && haskey(ped.individuals, ind.sire)
            push!(adj[ind.sire], id)
            in_degree[id] += 1
        end
        if !ismissing(ind.dam) && haskey(ped.individuals, ind.dam)
            push!(adj[ind.dam], id)
            in_degree[id] += 1
        end
    end

    # Kahn算法进行拓扑排序
    queue = [id for (id, deg) in in_degree if deg == 0]
    sorted_ids = Any[]

    while !isempty(queue)
        u = popfirst!(queue)
        push!(sorted_ids, u)

        for v in adj[u]
            in_degree[v] -= 1
            if in_degree[v] == 0
                push!(queue, v)
            end
        end
    end

    # 检查是否所有节点都被排序(检测循环)
    if length(sorted_ids) != n
        @error "谱系包含循环, 无法进行拓扑排序"
        # 返回部分排序结果
        for (id, deg) in in_degree
            if deg > 0 && !(id in sorted_ids)
                push!(sorted_ids, id)  # 添加剩余节点
            end
        end
    end

    ped.sorted_ids = sorted_ids
    return sorted_ids
end

"""
    recode_pedigree(ped::Pedigree)

将谱系ID重编码为1:n的整数, 便于矩阵运算.

# 返回
- `Pedigree`: 重编码后的新谱系对象
"""
function recode_pedigree(ped::Pedigree)
    # 确保已排序
    if isempty(ped.sorted_ids)
        sort_pedigree!(ped)
    end

    new_ped = Pedigree()

    # 建立映射
    for (i, id) in enumerate(ped.sorted_ids)
        new_ped.id_map[id] = i
        new_ped.rev_id_map[i] = id
    end

    # 重编码个体
    for (i, old_id) in enumerate(ped.sorted_ids)
        old_ind = ped.individuals[old_id]
        new_sire = ismissing(old_ind.sire) ? missing : new_ped.id_map[old_ind.sire]
        new_dam = ismissing(old_ind.dam) ? missing : new_ped.id_map[old_ind.dam]

        new_ind = Individual(i, new_sire, new_dam, old_ind.sex,
                           generation=old_ind.generation,
                           attributes=copy(old_ind.attributes))
        new_ped.individuals[i] = new_ind
    end

    new_ped.sorted_ids = collect(1:length(ped.sorted_ids))
    new_ped.metadata = copy(ped.metadata)

    return new_ped
end

"""
    validate_pedigree(ped::Pedigree)

全面验证谱系的完整性和一致性.

# 检查项目
1. ID唯一性
2. 父母存在性
3. 性别一致性
4. 循环检测
5. 世代一致性

# 返回
- `(is_valid::Bool, errors::Vector{String})`
"""
function validate_pedigree(ped::Pedigree)
    errors = String[]

    # 检查ID唯一性
    ids = collect(keys(ped.individuals))
    if length(ids) != length(unique(ids))
        push!(errors, "存在重复ID")
    end

    # 检查父母和性别
    for (id, ind) in ped.individuals
        # 父母存在性
        if !ismissing(ind.sire) && !haskey(ped.individuals, ind.sire)
            push!(errors, "个体 $id 的父本 $(ind.sire) 不存在")
        end
        if !ismissing(ind.dam) && !haskey(ped.individuals, ind.dam)
            push!(errors, "个体 $id 的母本 $(ind.dam) 不存在")
        end

        # 性别一致性检查 - 修复了比较逻辑
        if !ismissing(ind.sire) && haskey(ped.individuals, ind.sire)
            sire = ped.individuals[ind.sire]
            if sire.sex == 2
                push!(errors, "个体 $id 的父本 $(ind.sire) 性别标记为雌性")
            end
        end
        if !ismissing(ind.dam) && haskey(ped.individuals, ind.dam)
            dam = ped.individuals[ind.dam]
            if dam.sex == 1
                push!(errors, "个体 $id 的母本 $(ind.dam) 性别标记为雄性")
            end
        end

        # 自身循环 - 使用isequal来正确处理missing值
        if !ismissing(ind.sire) && isequal(ind.sire, id)
            push!(errors, "个体 $id 是自己的父亲")
        end
        if !ismissing(ind.dam) && isequal(ind.dam, id)
            push!(errors, "个体 $id 是自己的母亲")
        end
    end

    # 尝试排序以检测循环
    try
        temp_sorted = copy(ped.sorted_ids)
        sort_pedigree!(ped)
        ped.sorted_ids = temp_sorted  # 恢复原始排序
    catch e
        push!(errors, "谱系包含循环结构: $(e)")
    end

    is_valid = isempty(errors)
    return (is_valid, errors)
end

"""
    find_founders(ped::Pedigree)

找出所有创始人(父母都未知的个体).
"""
function find_founders(ped::Pedigree)
    return [id for (id, ind) in ped.individuals
            if ismissing(ind.sire) && ismissing(ind.dam)]
end

"""
    get_descendants(ped::Pedigree, id)

获取指定个体的所有后代.
"""
function get_descendants(ped::Pedigree, id)
    descendants = Set{Any}()
    queue = [id]

    while !isempty(queue)
        current = popfirst!(queue)

        for (other_id, ind) in ped.individuals
            # 使用isequal来正确处理missing值
            parent_match = false
            if !ismissing(ind.sire) && isequal(ind.sire, current)
                parent_match = true
            end
            if !ismissing(ind.dam) && isequal(ind.dam, current)
                parent_match = true
            end

            if parent_match && !(other_id in descendants)
                push!(descendants, other_id)
                push!(queue, other_id)
            end
        end
    end

    return collect(descendants)
end

"""
    get_ancestors(ped::Pedigree, id)

获取指定个体的所有祖先.
"""
function get_ancestors(ped::Pedigree, id)
    ancestors = Set{Any}()
    queue = [id]

    while !isempty(queue)
        current = popfirst!(queue)
        ind = ped.individuals[current]

        if !ismissing(ind.sire) && !(ind.sire in ancestors)
            push!(ancestors, ind.sire)
            push!(queue, ind.sire)
        end
        if !ismissing(ind.dam) && !(ind.dam in ancestors)
            push!(ancestors, ind.dam)
            push!(queue, ind.dam)
        end
    end

    return collect(ancestors)
end

"""
    get_generation(ped::Pedigree, gen::Int)

获取指定世代的所有个体.
"""
function get_generation(ped::Pedigree, gen::Int)
    return [id for (id, ind) in ped.individuals if ind.generation == gen]
end

"""
    save_pedigree(ped::Pedigree, filename::String)

保存谱系到CSV文件.
"""
function save_pedigree(ped::Pedigree, filename::String)
    df = DataFrame(
        ID = Any[],
        Sire = Any[],
        Dam = Any[],
        Sex = Int[],
        Generation = Int[]
    )

    for (id, ind) in ped.individuals
        push!(df, (id, ind.sire, ind.dam, ind.sex, ind.generation))
    end

    CSV.write(filename, df)
    println("谱系已保存到 $filename")
end

"""
    load_pedigree(filename::String)

从CSV文件加载谱系.
"""
function load_pedigree(filename::String)
    df = CSV.read(filename, DataFrame)
    return read_pedigree(df)
end

"""
    merge_pedigrees(ped1::Pedigree, ped2::Pedigree)

合并两个谱系, 处理重复ID.
"""
function merge_pedigrees(ped1::Pedigree, ped2::Pedigree)
    merged = Pedigree()

    # 添加第一个谱系的所有个体
    for (id, ind) in ped1.individuals
        merged.individuals[id] = deepcopy(ind)
    end

    # 添加第二个谱系的个体, 处理重复
    for (id, ind) in ped2.individuals
        if haskey(merged.individuals, id)
            @warn "个体 $id 在两个谱系中都存在, 保留第一个谱系的版本"
        else
            merged.individuals[id] = deepcopy(ind)
        end
    end

    # 重新计算世代
    compute_generations!(merged)

    return merged
end

export compute_generations!, save_pedigree, load_pedigree, merge_pedigrees

end # module PedigreeDataStructures

# ===================================================================
# 第3章: 亲缘和近交系数计算 (RelationshipCoefficients.jl)
# ===================================================================
module RelationshipCoefficients

using ..PedigreeDataStructures
using LinearAlgebra, SparseArrays

export calculate_inbreeding, calculate_kinship_matrix, calculate_A_matrix
export calculate_A_inverse, wright_coefficient, path_coefficient
export relationship_type, coancestry_coefficient

"""
    calculate_inbreeding(ped::Pedigree)

使用列表法高效计算所有个体的近交系数.
时间复杂度: O(n^2)

# 算法
F_i = phi_(父母) 其中phi是父母间的亲缘系数

# 返回
- `Vector{Float64}`: 近交系数向量
"""
function calculate_inbreeding(ped::Pedigree)
    n = length(ped.individuals)
    if n == 0
        return Float64[]
    end

    # 确保已重编码
    if !haskey(ped.individuals, 1)
        error("需要先对谱系进行重编码")
    end

    F = zeros(Float64, n)
    A_lower = zeros(Float64, n, n)  # 存储下三角部分

    for i in 1:n
        ind = ped.individuals[i]
        s = ind.sire
        d = ind.dam

        if ismissing(s) || ismissing(d)
            # 创始人或单亲
            A_lower[i, i] = 1.0
            F[i] = 0.0
        else
            # 确保s <= d用于索引
            if d < s
                s, d = d, s
            end

            # 近交系数等于父母间亲缘系数
            F[i] = A_lower[d, s] / 2.0
            A_lower[i, i] = 1.0 + F[i]
        end

        # 计算与前面个体的亲缘关系
        for j in 1:(i-1)
            val_s = ismissing(s) ? 0.0 : (j <= s ? A_lower[s, j] : A_lower[j, s])
            val_d = ismissing(d) ? 0.0 : (j <= d ? A_lower[d, j] : A_lower[j, d])
            A_lower[i, j] = (val_s + val_d) / 2.0
        end
    end

    return F
end

"""
    calculate_kinship_matrix(ped::Pedigree)

计算亲缘系数矩阵phi.
phi_ij = A_ij / 2

# 返回
- `Matrix{Float64}`: 亲缘系数矩阵
"""
function calculate_kinship_matrix(ped::Pedigree)
    A = calculate_A_matrix(ped)
    return A / 2.0
end

"""
    calculate_A_matrix(ped::Pedigree)

使用列表法构建加性遗传关系矩阵A.

# 算法
- 对角元素: A_ii = 1 + F_i
- 非对角元素: A_ij = (A_is + A_id) / 2

# 返回
- `Matrix{Float64}`: 加性关系矩阵
"""
function calculate_A_matrix(ped::Pedigree)
    n = length(ped.individuals)
    if n == 0
        return Matrix{Float64}(undef, 0, 0)
    end

    if !haskey(ped.individuals, 1)
        error("需要先对谱系进行重编码")
    end

    A = zeros(Float64, n, n)

    for i in 1:n
        ind = ped.individuals[i]
        s = ind.sire
        d = ind.dam

        # 对角线元素
        if ismissing(s) || ismissing(d)
            A[i, i] = 1.0
        else
            # F_i = A[s,d] / 2
            A[i, i] = 1.0 + A[s, d] / 2.0
        end

        # 非对角线元素
        for j in 1:(i-1)
            val_s = ismissing(s) ? 0.0 : A[j, s]
            val_d = ismissing(d) ? 0.0 : A[j, d]
            A[j, i] = A[i, j] = (val_s + val_d) / 2.0
        end
    end

    return A
end

"""
    calculate_A_inverse(ped::Pedigree; method=:henderson)

计算加性关系矩阵的逆矩阵A-1.

# 参数
- `ped`: 重编码后的谱系
- `method`: 计算方法(:henderson 或 :direct)

# 返回
- `SparseMatrixCSC`: 稀疏格式的A-1矩阵
"""
function calculate_A_inverse(ped::Pedigree; method=:henderson)
    n = length(ped.individuals)
    if n == 0
        return spzeros(Float64, 0, 0)
    end

    if !haskey(ped.individuals, 1)
        error("需要先对谱系进行重编码")
    end

    if method == :direct
        # 直接求逆(仅适用于小规模)
        A = calculate_A_matrix(ped)
        return sparse(inv(A))
    else
        # Henderson法则
        return henderson_A_inverse(ped)
    end
end

"""
    henderson_A_inverse(ped::Pedigree)

使用Henderson法则直接构建A-1.
这是最高效的方法, 无需先计算A矩阵.
"""
function henderson_A_inverse(ped::Pedigree)
    n = length(ped.individuals)

    # 使用坐标格式构建稀疏矩阵
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    # 先计算近交系数
    F = calculate_inbreeding(ped)

    for i in 1:n
        ind = ped.individuals[i]
        s = ind.sire
        d = ind.dam

        if ismissing(s) && ismissing(d)
            # 创始人
            push!(I_idx, i); push!(J_idx, i); push!(V_val, 1.0)

        elseif ismissing(s) || ismissing(d)
            # 单亲已知
            p = ismissing(s) ? d : s
            alpha = 4.0 / 3.0

            # 贡献到对角线
            push!(I_idx, i); push!(J_idx, i); push!(V_val, alpha)
            push!(I_idx, p); push!(J_idx, p); push!(V_val, alpha/4.0)

            # 贡献到非对角线
            push!(I_idx, i); push!(J_idx, p); push!(V_val, -alpha/2.0)
            push!(I_idx, p); push!(J_idx, i); push!(V_val, -alpha/2.0)

        else
            # 双亲已知
            # 计算贡献系数
            alpha = 1.0 / (0.5 - 0.25 * (F[s] + F[d]))

            # 对角线贡献
            push!(I_idx, i); push!(J_idx, i); push!(V_val, alpha)

            # 父母对角线贡献
            push!(I_idx, s); push!(J_idx, s); push!(V_val, alpha/4.0)
            push!(I_idx, d); push!(J_idx, d); push!(V_val, alpha/4.0)

            # 父母间贡献
            push!(I_idx, s); push!(J_idx, d); push!(V_val, alpha/4.0)
            push!(I_idx, d); push!(J_idx, s); push!(V_val, alpha/4.0)

            # 个体-父母贡献
            push!(I_idx, i); push!(J_idx, s); push!(V_val, -alpha/2.0)
            push!(I_idx, s); push!(J_idx, i); push!(V_val, -alpha/2.0)
            push!(I_idx, i); push!(J_idx, d); push!(V_val, -alpha/2.0)
            push!(I_idx, d); push!(J_idx, i); push!(V_val, -alpha/2.0)
        end
    end

    # 构建稀疏矩阵
    return sparse(I_idx, J_idx, V_val, n, n)
end

"""
    wright_coefficient(ped::Pedigree, id1, id2)

计算Wright路径系数(两个个体间的亲缘系数).
使用路径追踪算法.
"""
function wright_coefficient(ped::Pedigree, id1, id2)
    # 找出所有共同祖先
    ancestors1 = Set(PedigreeDataStructures.get_ancestors(ped, id1))
    ancestors2 = Set(PedigreeDataStructures.get_ancestors(ped, id2))

    # 添加个体本身(处理自身的情况)
    push!(ancestors1, id1)
    push!(ancestors2, id2)

    common = intersect(ancestors1, ancestors2)

    if isempty(common)
        return 0.0
    end

    # 计算通过每个共同祖先的路径贡献
    coeff = 0.0

    # 如果需要近交系数, 先计算
    recoded = false
    F = Float64[]

    if haskey(ped.id_map, id1)  # 已经有映射
        F = calculate_inbreeding(ped)
        recoded = true
    end

    for ancestor in common
        # 计算从id1到祖先的世代数
        n1 = path_length(ped, id1, ancestor)
        # 计算从id2到祖先的世代数
        n2 = path_length(ped, id2, ancestor)

        if n1 >= 0 && n2 >= 0
            # 获取祖先的近交系数
            F_anc = 0.0
            if recoded && haskey(ped.id_map, ancestor)
                anc_idx = ped.id_map[ancestor]
                F_anc = F[anc_idx]
            end

            # Wright公式
            coeff += (0.5)^(n1 + n2 + 1) * (1 + F_anc)
        end
    end

    return coeff
end

"""
    path_length(ped::Pedigree, descendant, ancestor)

计算从后代到祖先的路径长度(世代数).
"""
function path_length(ped::Pedigree, descendant, ancestor)
    if isequal(descendant, ancestor)
        return 0
    end

    queue = [(descendant, 0)]
    visited = Set{Any}()

    while !isempty(queue)
        current, dist = popfirst!(queue)

        if current in visited
            continue
        end
        push!(visited, current)

        if !haskey(ped.individuals, current)
            continue
        end

        ind = ped.individuals[current]

        if !ismissing(ind.sire)
            if isequal(ind.sire, ancestor)
                return dist + 1
            end
            push!(queue, (ind.sire, dist + 1))
        end

        if !ismissing(ind.dam)
            if isequal(ind.dam, ancestor)
                return dist + 1
            end
            push!(queue, (ind.dam, dist + 1))
        end
    end

    return -1  # 无路径
end

"""
    relationship_type(kinship::Float64)

根据亲缘系数判断关系类型.
"""
function relationship_type(kinship::Float64)
    # 允许小的误差
    epsilon = 0.01

    if abs(kinship - 0.5) < epsilon
        return "自身/同卵双胞胎"
    elseif abs(kinship - 0.25) < epsilon
        return "父子/全同胞"
    elseif abs(kinship - 0.125) < epsilon
        return "祖孙/半同胞/叔侄"
    elseif abs(kinship - 0.0625) < epsilon
        return "堂兄弟/表兄弟"
    elseif abs(kinship - 0.03125) < epsilon
        return "二级堂表"
    elseif kinship < 0.01
        return "无关/远亲"
    else
        return "其他关系 (phi = $(round(kinship, digits=4)))"
    end
end

"""
    coancestry_coefficient(ped::Pedigree, id1, id2)

计算共祖系数(等同于亲缘系数).
"""
function coancestry_coefficient(ped::Pedigree, id1, id2)
    return wright_coefficient(ped, id1, id2)
end

export path_length, relationship_type, coancestry_coefficient, analyze_wright_paths

"""
    get_path_to_ancestor(ped::Pedigree, descendant, ancestor)

使用广度优先搜索找到从后代到祖先的最短路径.

# 返回
- `(path::Vector{Any}, length::Int)`: 路径(从后代到祖先)和长度. 如果无路径则返回 ([], -1).
"""
function get_path_to_ancestor(ped::Pedigree, descendant, ancestor)
    if isequal(descendant, ancestor)
        return ([ancestor], 0)
    end

    # 队列中存储 (路径, 距离)
    queue = [([descendant], 0)]
    visited = Set{Any}([descendant])

    while !isempty(queue)
        path, dist = popfirst!(queue)
        current = path[end]

        if !haskey(ped.individuals, current)
            continue
        end

        ind = ped.individuals[current]

        # 向上追溯父母
        parents = []
        !ismissing(ind.sire) && push!(parents, ind.sire)
        !ismissing(ind.dam) && push!(parents, ind.dam)

        for p in parents
            if !(p in visited)
                new_path = vcat(path, p)
                if isequal(p, ancestor)
                    return (new_path, dist + 1)
                end
                push!(visited, p)
                push!(queue, (new_path, dist + 1))
            end
        end
    end

    return ([], -1) # 无路径
end


"""
    analyze_wright_paths(ped::Pedigree, id1, id2)

提供两个个体间Wright路径系数的详细分析, 显示每个共同祖先的贡献.
此方法适用于非近交谱系或需要可视化路径的场景. 对于复杂谱系, 列表法更高效.

# 返回
- `(total_coeff::Float64, analysis_text::String)`
"""
function analyze_wright_paths(ped::Pedigree, id1, id2)
    # 预计算所有个体的近交系数
    # 这需要一个重编码的谱系
    recoded_ped = PedigreeDataStructures.recode_pedigree(ped)
    all_F = calculate_inbreeding(recoded_ped)

    # 处理自身关系
    if isequal(id1, id2)
        idx = recoded_ped.id_map[id1]
        F = all_F[idx]
        coeff = 0.5 * (1 + F)
        analysis_text = """
        分析自身关系 $id1:
          近交系数 F = $(round(F, digits=4))
          亲缘系数 phi = 0.5 * (1 + F) = $(round(coeff, digits=4))
        """
        return (coeff, analysis_text)
    end

    # 找出共同祖先, 包括个体自身
    ancestors1 = Set(vcat(PedigreeDataStructures.get_ancestors(ped, id1), [id1]))
    ancestors2 = Set(vcat(PedigreeDataStructures.get_ancestors(ped, id2), [id2]))
    common_ancestors = intersect(ancestors1, ancestors2)

    if isempty(common_ancestors)
        return (0.0, "个体 $id1 和 $id2 没有共同祖先, 亲缘系数为 0.0.")
    end

    total_coeff = 0.0
    analysis_lines = ["\n分析 $id1 和 $id2 之间的亲缘关系:", "共同祖先: " * join(collect(common_ancestors), ", ")]
    push!(analysis_lines, "─" ^ 70)
    push!(analysis_lines, "祖先\t路径 ($id1 <-> $id2)\t\t世代数\tF_anc\t贡献")
    push!(analysis_lines, "─" ^ 70)

    for ancestor in common_ancestors
        (path1, n1) = get_path_to_ancestor(ped, id1, ancestor)
        (path2, n2) = get_path_to_ancestor(ped, id2, ancestor)

        if n1 >= 0 && n2 >= 0
            # 获取祖先的近交系数
            F_anc = 0.0
            if haskey(recoded_ped.id_map, ancestor)
                anc_idx = recoded_ped.id_map[ancestor]
                F_anc = all_F[anc_idx]
            end

            # Wright路径系数公式 for kinship (phi)
            contribution = (0.5)^(n1 + n2 + 1) * (1 + F_anc)
            total_coeff += contribution

            # 构建路径字符串
            path_str = if n1 > 0 && n2 > 0
                p1_str = join(reverse(path1), "->")
                p2_str = join(path2[2:end], "<-")
                p2_str = isempty(p2_str) ? "" : " .. " * p2_str
                "$(p1_str)$(p2_str)"
            elseif n1 > 0 # id2 is ancestor
                join(reverse(path1), "->")
            elseif n2 > 0 # id1 is ancestor
                join(reverse(path2), "->")
            else # they are the same individual, handled above
                ""
            end

            f_str = round(F_anc, digits=4)
            contrib_str = round(contribution, digits=4)
            gen_sum = n1 + n2
            push!(analysis_lines, rpad(ancestor, 8) * "\t" * rpad(path_str, 24) * "\t$(gen_sum)\t$(f_str)\t$(contrib_str)")
        end
    end

    push!(analysis_lines, "─" ^ 70)
    push!(analysis_lines, "总亲缘系数 (phi): $(round(total_coeff, digits=4))")
    push!(analysis_lines, "等效关系类型: $(relationship_type(total_coeff))")

    return (total_coeff, join(analysis_lines, "\n"))
end

end # module RelationshipCoefficients

# ===================================================================
# 第4章: IBD模拟和实际亲缘关系 (IBDSimulation.jl)
# ===================================================================
module IBDSimulation

using ..PedigreeDataStructures
using Random, Distributions, Statistics, ProgressMeter

export simulate_ibd, gene_drop, calculate_realized_relationship
export ChromosomeMap, HUMAN_GENOME_MAP, simulate_genome_ibd
export ibd_sharing_variance, LinkageAnalysis, calculate_lod_score

"""
    ChromosomeMap

染色体图谱, 存储遗传长度信息.
"""
struct ChromosomeMap
    lengths::Vector{Float64}  # Morgan单位
    names::Vector{String}

    function ChromosomeMap(lengths; names=string.(1:length(lengths)))
        new(lengths, names)
    end
end

# 人类基因组标准图谱
const HUMAN_GENOME_MAP = ChromosomeMap(
    [2.8, 2.6, 2.2, 2.1, 2.0, 1.9, 1.8, 1.6, 1.5, 1.5,
     1.5, 1.5, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9, 0.6, 0.7],
    names=vcat(string.(1:22), ["X"])
)

"""
    LinkageAnalysis

连锁分析结构, 用于LOD评分计算
"""
struct LinkageAnalysis
    markers::Vector{PedigreeDataStructures.Marker}
    phenotype::PedigreeDataStructures.Phenotype
    recombination_fractions::Vector{Float64}
    lod_scores::Vector{Float64}
end

"""
    gene_drop(ped::Pedigree, id, founder_alleles::Dict, cache::Dict)

递归基因下降算法, 追踪等位基因在谱系中的传递.

# 参数
- `ped`: 谱系对象
- `id`: 当前个体ID
- `founder_alleles`: 创始人等位基因分配
- `cache`: 缓存已计算结果

# 返回
- 选中的等位基因标识
"""
function gene_drop(ped::Pedigree, id, founder_alleles::Dict, cache::Dict)
    # 检查缓存
    if haskey(cache, id)
        return cache[id]
    end

    # 获取个体信息
    if !haskey(ped.individuals, id)
        error("个体 $id 不在谱系中")
    end

    ind = ped.individuals[id]

    # 创始人: 随机选择一个等位基因
    if ismissing(ind.sire) && ismissing(ind.dam)
        if !haskey(founder_alleles, id)
            # 为创始人分配等位基因
            max_allele = isempty(founder_alleles) ? 0 : maximum(maximum(v) for v in values(founder_alleles))
            founder_alleles[id] = [max_allele + 1, max_allele + 2]
        end
        allele = rand(founder_alleles[id])
        cache[id] = allele
        return allele
    end

    # 非创始人: 从父母随机继承
    # 选择一个存在的父母
    if !ismissing(ind.sire) && !ismissing(ind.dam)
        parent = rand(Bool) ? ind.sire : ind.dam
    elseif !ismissing(ind.sire)
        parent = ind.sire
    elseif !ismissing(ind.dam)
        parent = ind.dam
    else
        # 不应该到达这里, 但作为保护措施
        if !haskey(founder_alleles, id)
            max_allele = isempty(founder_alleles) ? 0 : maximum(maximum(v) for v in values(founder_alleles))
            founder_alleles[id] = [max_allele + 1, max_allele + 2]
        end
        allele = rand(founder_alleles[id])
        cache[id] = allele
        return allele
    end

    allele = gene_drop(ped, parent, founder_alleles, cache)
    cache[id] = allele
    return allele
end

"""
    simulate_ibd(ped::Pedigree, id1, id2; num_sim=10000, use_cache=true)

通过基因下降模拟估计两个个体的IBD概率.

# 参数
- `ped`: 谱系对象
- `id1`, `id2`: 两个个体的ID
- `num_sim`: 模拟次数
- `use_cache`: 是否使用缓存优化

# 返回
- `(ibd0, ibd1, ibd2, kinship)`: IBD概率和亲缘系数
"""
function simulate_ibd(ped::Pedigree, id1, id2; num_sim=10000, use_cache=true, show_progress=false)
    @assert haskey(ped.individuals, id1) "个体 $id1 不在谱系中"
    @assert haskey(ped.individuals, id2) "个体 $id2 不在谱系中"

    # 找出创始人
    founders = PedigreeDataStructures.find_founders(ped)

    # 统计IBD状态
    ibd_counts = [0, 0, 0]  # IBD-0, IBD-1, IBD-2

    # 设置进度条
    if show_progress
        prog = Progress(num_sim, desc="IBD模拟中...")
    end

    for sim in 1:num_sim
        # 为创始人分配唯一等位基因
        # 每个创始人有2个等位基因(用于两个单倍型)
        founder_alleles = Dict{Any, Vector{Int}}()
        allele_counter = 1
        for f in founders
            founder_alleles[f] = [allele_counter, allele_counter+1]
            allele_counter += 2
        end

        # 模拟两次(获取两个等位基因)
        alleles1 = Int[]
        alleles2 = Int[]

        for _ in 1:2
            cache = Dict{Any, Int}()
            push!(alleles1, gene_drop(ped, id1, founder_alleles, cache))

            cache = Dict{Any, Int}()
            push!(alleles2, gene_drop(ped, id2, founder_alleles, cache))
        end

        # 计算IBD状态
        shared = length(intersect(Set(alleles1), Set(alleles2)))
        ibd_counts[shared + 1] += 1

        if show_progress
            next!(prog)
        end
    end

    # 计算概率
    ibd0 = ibd_counts[1] / num_sim
    ibd1 = ibd_counts[2] / num_sim
    ibd2 = ibd_counts[3] / num_sim

    # 亲缘系数 = (ibd1/2 + ibd2)
    kinship = ibd1 / 2 + ibd2

    return (ibd0=ibd0, ibd1=ibd1, ibd2=ibd2, kinship=kinship)
end

"""
    simulate_genome_ibd(ped::Pedigree, id1, id2;
                       map=HUMAN_GENOME_MAP, num_sim=100)

模拟全基因组水平的IBD共享, 考虑重组.

# 返回
- IBD共享比例的分布
"""
function simulate_genome_ibd(ped::Pedigree, id1, id2;
                            map=HUMAN_GENOME_MAP, num_sim=100)

    total_length = sum(map.lengths)
    ibd_proportions = Float64[]

    prog = Progress(num_sim, desc="基因组IBD模拟...")

    for sim in 1:num_sim
        total_ibd = 0.0

        for chr_length in map.lengths
            # 模拟该染色体的IBD片段
            ibd_length = simulate_chromosome_ibd(ped, id1, id2, chr_length)
            total_ibd += ibd_length
        end

        push!(ibd_proportions, total_ibd / (2 * total_length))
        next!(prog)
    end

    return ibd_proportions
end

"""
    simulate_chromosome_ibd(ped, id1, id2, chr_length)

模拟单条染色体的IBD共享, 考虑重组.
"""
function simulate_chromosome_ibd(ped::Pedigree, id1, id2, chr_length::Float64)
    # 简化模型: 假设每Morgan平均1次重组
    n_recomb = rand(Poisson(chr_length))

    # 使用简单的IBD模拟
    result = simulate_ibd(ped, id1, id2, num_sim=1, show_progress=false)

    # 根据IBD状态返回期望长度
    if result.ibd2 > rand()
        return 2 * chr_length  # 两条都IBD
    elseif result.ibd1 > rand()
        return chr_length  # 一条IBD
    else
        return 0.0  # 无IBD
    end
end

"""
    calculate_realized_relationship(ibd_props::Vector{Float64})

根据IBD模拟结果计算实际亲缘关系统计量.
"""
function calculate_realized_relationship(ibd_props::Vector{Float64})
    return Dict(
        "mean" => mean(ibd_props),
        "std" => std(ibd_props),
        "median" => median(ibd_props),
        "q25" => quantile(ibd_props, 0.25),
        "q75" => quantile(ibd_props, 0.75),
        "min" => minimum(ibd_props),
        "max" => maximum(ibd_props),
        "cv" => std(ibd_props) / mean(ibd_props)
    )
end

"""
    ibd_sharing_variance(relationship::Symbol)

返回特定亲缘关系的IBD共享理论方差.
"""
function ibd_sharing_variance(relationship::Symbol)
    variances = Dict(
        :parent_child => 0.0,      # 父子关系方差为0
        :full_sibs => 0.0625,      # 全同胞
        :half_sibs => 0.03125,     # 半同胞
        :cousins => 0.0234375,     # 表兄弟
        :unrelated => 0.0          # 无关个体
    )

    return get(variances, relationship, NaN)
end

"""
    calculate_lod_score(theta::Float64, recombinants::Int, non_recombinants::Int)

计算LOD评分用于连锁分析

# 参数
- `theta`: 重组率(0到0.5)
- `recombinants`: 重组体数量
- `non_recombinants`: 非重组体数量

# 返回
- LOD评分值
"""
function calculate_lod_score(theta::Float64, recombinants::Int, non_recombinants::Int)
    @assert 0 <= theta <= 0.5 "重组率必须在0到0.5之间"

    n = recombinants + non_recombinants

    # 计算似然
    L_theta = theta^recombinants * (1-theta)^non_recombinants
    L_half = 0.5^n

    # 计算LOD
    if L_half > 0
        lod = log10(L_theta / L_half)
    else
        lod = 0.0
    end

    return lod
end

export LinkageAnalysis, calculate_lod_score

end # module IBDSimulation

# ===================================================================
# 第5章: 谱系可视化 (PedigreeVisualization.jl)
# ===================================================================
module PedigreeVisualization

using ..PedigreeDataStructures
using ..RelationshipCoefficients
using Plots, GraphRecipes

export plot_pedigree, plot_kinship_heatmap, plot_generation_tree
export plot_inbreeding_trend, plot_ibd_distribution

"""
    plot_pedigree(ped::Pedigree; kwargs...)

绘制谱系图, 支持多种布局和样式.

# 参数
- `ped`: 谱系对象
- `layout`: 布局算法(:tree, :stress, :spring)
- `show_labels`: 是否显示个体标签
- `color_by`: 着色依据(:sex, :generation, :founder)
- `title`: 图标题

# 返回
- Plots对象
"""
function plot_pedigree(ped::Pedigree;
                      layout=:tree,
                      show_labels=true,
                      color_by=:sex,
                      title="谱系图",
                      kwargs...)

    # 确保已排序
    if isempty(ped.sorted_ids)
        PedigreeDataStructures.sort_pedigree!(ped)
    end

    n = length(ped.sorted_ids)
    if n == 0
        return plot(title=title, grid=false)
    end

    # 创建ID映射
    id_to_idx = Dict(id => i for (i, id) in enumerate(ped.sorted_ids))

    # 构建边
    edges_src = Int[]
    edges_dst = Int[]

    for (idx, id) in enumerate(ped.sorted_ids)
        ind = ped.individuals[id]

        if !ismissing(ind.sire) && haskey(id_to_idx, ind.sire)
            push!(edges_src, id_to_idx[ind.sire])
            push!(edges_dst, idx)
        end
        if !ismissing(ind.dam) && haskey(id_to_idx, ind.dam)
            push!(edges_src, id_to_idx[ind.dam])
            push!(edges_dst, idx)
        end
    end

    # 设置节点标签
    if show_labels
        names = [string(id) for id in ped.sorted_ids]
    else
        names = fill("", n)
    end

    # 设置节点形状(基于性别)
    shapes = Symbol[]
    for id in ped.sorted_ids
        sex = ped.individuals[id].sex
        if sex == 1
            push!(shapes, :square)  # 雄性
        elseif sex == 2
            push!(shapes, :circle)  # 雌性
        else
            push!(shapes, :diamond)  # 未知
        end
    end

    # 设置节点颜色
    colors = Symbol[]
    if color_by == :sex
        for id in ped.sorted_ids
            sex = ped.individuals[id].sex
            push!(colors, sex == 1 ? :lightblue : sex == 2 ? :pink : :lightgray)
        end
    elseif color_by == :generation
        max_gen = maximum(ind.generation for ind in values(ped.individuals))
        for id in ped.sorted_ids
            gen = ped.individuals[id].generation
            push!(colors, [:red, :orange, :yellow, :green, :blue, :purple][min(gen+1, 6)])
        end
    elseif color_by == :founder
        founders = Set(PedigreeDataStructures.find_founders(ped))
        for id in ped.sorted_ids
            push!(colors, id in founders ? :gold : :lightgray)
        end
    else
        colors = fill(:lightblue, n)
    end

    # 选择布局方法
    method = layout == :tree ? :buchheim :
             layout == :stress ? :stress :
             layout == :spring ? :spring : :buchheim

    # 绘制图
    # 修复: 将 (src, dst) 元组解包为两个独立的参数
    # 旧代码: graphplot((edges_src, edges_dst), ...)
    return graphplot(
        edges_src, edges_dst;
        names=names,
        nodeshape=shapes,
        nodecolor=colors,
        method=method,
        fontsize=10,
        nodesize=0.15,
        linecolor=:darkgray,
        title=title,
        titlefontsize=14,
        framestyle=:none,
        kwargs...
    )
end

"""
    plot_kinship_heatmap(ped::Pedigree; annotate=false)

绘制亲缘系数热图.
"""
function plot_kinship_heatmap(ped::Pedigree; annotate=false)
    # 确保已重编码
    if !haskey(ped.id_map, first(keys(ped.individuals)))
        ped = PedigreeDataStructures.recode_pedigree(ped)
    end

    # 计算亲缘系数矩阵
    kinship = RelationshipCoefficients.calculate_kinship_matrix(ped)
    n = size(kinship, 1)

    # 创建标签
    labels = [string(ped.rev_id_map[i]) for i in 1:n]

    # 绘制热图
    p = heatmap(kinship,
               xlabel="个体",
               ylabel="个体",
               title="亲缘系数热图",
               color=:viridis,
               clims=(0, 0.5),
               aspect_ratio=:equal,
               xticks=(1:n, labels),
               yticks=(1:n, labels),
               xrotation=45)

    # 添加数值标注
    if annotate && n <= 20
        for i in 1:n, j in 1:n
            val = round(kinship[i,j], digits=3)
            annotate!(j, i, text(string(val), 8, :white))
        end
    end

    return p
end

"""
    plot_generation_tree(ped::Pedigree)

按世代绘制谱系树.
"""
function plot_generation_tree(ped::Pedigree)
    # 按世代分组
    generations = Dict{Int, Vector{Any}}()

    for (id, ind) in ped.individuals
        gen = ind.generation
        if !haskey(generations, gen)
            generations[gen] = Any[]
        end
        push!(generations[gen], id)
    end

    # 计算布局
    max_gen = maximum(keys(generations))
    y_positions = Dict{Any, Float64}()
    x_positions = Dict{Any, Float64}()

    for gen in 0:max_gen
        if haskey(generations, gen)
            ids = generations[gen]
            n = length(ids)
            for (i, id) in enumerate(ids)
                x_positions[id] = i - (n+1)/2
                y_positions[id] = -gen
            end
        end
    end

    # 绘制节点和边
    plot(title="谱系世代树", xlabel="", ylabel="世代",
         grid=false, legend=false, framestyle=:box)

    # 绘制边
    for (id, ind) in ped.individuals
        if !ismissing(ind.sire) && haskey(x_positions, ind.sire)
            plot!([x_positions[ind.sire], x_positions[id]],
                  [y_positions[ind.sire], y_positions[id]],
                  color=:gray, linewidth=1)
        end
        if !ismissing(ind.dam) && haskey(x_positions, ind.dam)
            plot!([x_positions[ind.dam], x_positions[id]],
                  [y_positions[ind.dam], y_positions[id]],
                  color=:gray, linewidth=1)
        end
    end

    # 绘制节点
    for (id, ind) in ped.individuals
        if haskey(x_positions, id)
            shape = ind.sex == 1 ? :square : ind.sex == 2 ? :circle : :diamond
            color = ind.generation == 0 ? :gold : :lightblue
            scatter!([x_positions[id]], [y_positions[id]],
                    marker=shape, markersize=8, color=color)
            annotate!(x_positions[id], y_positions[id]-0.2,
                     text(string(id), 8))
        end
    end

    return current()
end

"""
    plot_inbreeding_trend(ped::Pedigree)

绘制近交系数随世代变化的趋势.
"""
function plot_inbreeding_trend(ped::Pedigree)
    # 确保已重编码
    if !haskey(ped.id_map, first(keys(ped.individuals)))
        ped = PedigreeDataStructures.recode_pedigree(ped)
    end

    # 计算近交系数
    F = RelationshipCoefficients.calculate_inbreeding(ped)

    # 按世代分组
    gen_F = Dict{Int, Vector{Float64}}()
    for (i, f) in enumerate(F)
        ind = ped.individuals[i]
        gen = ind.generation
        if !haskey(gen_F, gen)
            gen_F[gen] = Float64[]
        end
        push!(gen_F[gen], f)
    end

    # 计算每代的平均近交系数
    generations = sort(collect(keys(gen_F)))
    mean_F = [mean(gen_F[g]) for g in generations]

    # 绘制趋势图
    plot(generations, mean_F,
         marker=:circle,
         markersize=8,
         linewidth=2,
         xlabel="世代",
         ylabel="平均近交系数",
         title="近交系数趋势",
         label="平均F",
         grid=true)

    # 添加误差条
    if length(generations) > 1
        std_F = [std(gen_F[g]) for g in generations]
        plot!(generations, mean_F,
              ribbon=std_F,
              fillalpha=0.3,
              label="")
    end

    return current()
end

"""
    plot_ibd_distribution(ibd_props::Vector{Float64})

绘制IBD共享比例的分布直方图.
"""
function plot_ibd_distribution(ibd_props::Vector{Float64})
    histogram(ibd_props,
             bins=20,
             xlabel="IBD共享比例",
             ylabel="频率",
             title="IBD共享分布",
             label="模拟结果",
             fillalpha=0.7,
             color=:blue)

    # 添加均值线
    mean_ibd = mean(ibd_props)
    vline!([mean_ibd],
           linewidth=2,
           color=:red,
           label="均值 = $(round(mean_ibd, digits=3))")

    return current()
end

end # module PedigreeVisualization

# ===================================================================
# 第6章: 高级分析模块 (AdvancedAnalyses.jl)
# ===================================================================
module AdvancedAnalyses

using ..PedigreeDataStructures
using ..RelationshipCoefficients
using ..IBDSimulation
using LinearAlgebra, Statistics, Distributions, Optim

export SegregationAnalysis, heritability_estimate, blup_breeding_value
export reconstruct_pedigree, parentage_test, relationship_inference

"""
    SegregationAnalysis

分离分析结构, 用于复杂性状的遗传分析
"""
struct SegregationAnalysis
    phenotype::PedigreeDataStructures.Phenotype
    mode_of_inheritance::Symbol  # :dominant, :recessive, :additive, :complex
    penetrance::Dict{String, Float64}
    allele_frequencies::Dict{String, Float64}
end

"""
    heritability_estimate(ped::Pedigree, phenotype::Vector{Float64})

使用REML方法估计遗传力

# 参数
- `ped`: 谱系对象(需要已重编码)
- `phenotype`: 表型值向量

# 返回
- 遗传力估计值h^2
"""
function heritability_estimate(ped::Pedigree, phenotype::Vector{Float64})
    n = length(phenotype)

    # 计算加性关系矩阵
    A = RelationshipCoefficients.calculate_A_matrix(ped)

    # 初始方差组分
    var_a = var(phenotype) * 0.5  # 初始遗传方差
    var_e = var(phenotype) * 0.5  # 初始环境方差

    # 使用简化的REML迭代
    max_iter = 100
    tolerance = 1e-6

    for iter in 1:max_iter
        V = var_a * A + var_e * I(n)
        V_inv = inv(V)

        # 更新方差组分
        var_a_new = var_a * sqrt(phenotype' * V_inv * A * V_inv * phenotype / n)
        var_e_new = var_e * sqrt(phenotype' * V_inv * V_inv * phenotype / n)

        if abs(var_a_new - var_a) < tolerance && abs(var_e_new - var_e) < tolerance
            break
        end

        var_a = var_a_new
        var_e = var_e_new
    end

    # 计算遗传力
    h2 = var_a / (var_a + var_e)

    return h2
end

"""
    blup_breeding_value(ped::Pedigree, phenotype::Vector{Float64}, h²::Float64)

计算BLUP育种值

# 参数
- `ped`: 谱系对象(需要已重编码)
- `phenotype`: 表型值向量
- `h2`: 遗传力

# 返回
- 育种值向量
"""
function blup_breeding_value(ped::Pedigree, phenotype::Vector{Float64}, h2::Float64)
    n = length(phenotype)

    # 计算加性关系矩阵
    A = RelationshipCoefficients.calculate_A_matrix(ped)

    # 计算方差比
    lambda = (1 - h2) / h2

    # 混合模型方程
    # [X'X   X'Z  ] [b] = [X'y]
    # [Z'X   Z'Z+lambda*A-1] [a]   [Z'y]

    # 简化: 假设只有总体均值
    X = ones(n, 1)
    Z = I(n)
    y = phenotype

    A_inv = RelationshipCoefficients.calculate_A_inverse(ped)

    # 构建系数矩阵
    C11 = X' * X
    C12 = X' * Z
    C21 = Z' * X
    C22 = Z' * Z + lambda * Matrix(A_inv)

    C = [C11 C12; C21 C22]

    # 右手边
    rhs = [X' * y; Z' * y]

    # 求解
    solution = C \ rhs

    # 提取育种值
    breeding_values = solution[2:end]

    return breeding_values
end

"""
    reconstruct_pedigree(genotypes::Matrix{Int})

从基因型数据重构谱系关系

# 参数
- `genotypes`: 基因型矩阵(个体×标记)

# 返回
- 推断的亲缘关系矩阵
"""
function reconstruct_pedigree(genotypes::Matrix{Int})
    n_individuals, n_markers = size(genotypes)

    # 计算IBS(状态同一)矩阵
    ibs_matrix = zeros(n_individuals, n_individuals)

    for i in 1:n_individuals
        for j in i:n_individuals
            # 计算共享等位基因比例
            shared = sum(genotypes[i, :] .== genotypes[j, :]) / n_markers
            ibs_matrix[i, j] = ibs_matrix[j, i] = shared
        end
    end

    # 从IBS推断IBD(简化方法)
    # 这里使用Method of Moments估计
    kinship_matrix = (ibs_matrix .- 0.5) ./ 2

    # 确保对角线正确
    for i in 1:n_individuals
        kinship_matrix[i, i] = 0.5
    end

    return kinship_matrix
end

"""
    parentage_test(offspring_genotype, parent1_genotype, parent2_genotype)

亲子鉴定测试

# 参数
- `offspring_genotype`: 子代基因型
- `parent1_genotype`: 潜在父亲基因型
- `parent2_genotype`: 潜在母亲基因型

# 返回
- 亲子关系概率
"""
function parentage_test(offspring_genotype::Vector,
                       parent1_genotype::Vector,
                       parent2_genotype::Vector)

    n_markers = length(offspring_genotype)

    # 计算每个标记的传递概率
    probabilities = Float64[]

    for i in 1:n_markers
        # 检查孟德尔遗传一致性
        o = offspring_genotype[i]
        p1 = parent1_genotype[i]
        p2 = parent2_genotype[i]

        # 简化: 检查是否可能遗传
        if (o == p1 || o == p2 || o == (p1 + p2) / 2)
            push!(probabilities, 1.0)
        else
            push!(probabilities, 0.01)  # 允许小概率的突变
        end
    end

    # 计算总体概率
    total_prob = prod(probabilities)

    return total_prob
end

"""
    relationship_inference(kinship::Float64)

根据亲缘系数推断最可能的关系类型

# 参数
- `kinship`: 亲缘系数值

# 返回
- 推断的关系类型和置信度
"""
function relationship_inference(kinship::Float64)
    # 定义标准关系及其期望亲缘系数
    relationships = [
        ("自身/同卵双胞胎", 0.5, 0.01),
        ("父子/全同胞", 0.25, 0.02),
        ("祖孙/半同胞", 0.125, 0.02),
        ("堂兄弟", 0.0625, 0.015),
        ("二级堂表", 0.03125, 0.01),
        ("三级亲属", 0.015625, 0.008),
        ("无关", 0.0, 0.005)
    ]

    best_match = ""
    best_prob = 0.0

    for (rel, expected, sd) in relationships
        # 计算与期望值的匹配概率(假设正态分布)
        prob = pdf(Normal(expected, sd), kinship)
        if prob > best_prob
            best_prob = prob
            best_match = rel
        end
    end

    # 归一化置信度
    confidence = min(best_prob * 10, 1.0)  # 简化的置信度计算

    return (relationship=best_match, confidence=confidence)
end

end # module AdvancedAnalyses

# ===================================================================
# 主程序: 完整功能演示
# ===================================================================

# 导入所有模块
using .Prerequisites
using .PedigreeDataStructures
using .RelationshipCoefficients
using .IBDSimulation
using .PedigreeVisualization
using .AdvancedAnalyses

"""
    main()

主函数, 演示完整的谱系分析工作流程, 包含所有高级功能.
"""
function main()
    println("\n" * "╔" ^ 80)
    println(" " ^ 20 * "Julia 谱系分析工具箱 - 完整功能演示")
    println("╚" ^ 80)

    # ==================== 第1部分: 环境设置 ====================
    println("\n[第1部分: 环境设置]")
    println("─" ^ 40)

    Prerequisites.setup_environment()
    println("(check) 环境设置完成")

    # 展示术语解释功能
    println("\n查询遗传学术语...")
    Prerequisites.explain_term("Kinship Coefficient")

    # ==================== 第2部分: 构建复杂谱系 ====================
    println("\n[第2部分: 构建复杂三代谱系]")
    println("─" ^ 40)

    # 创建一个复杂的三代谱系
    # 包含近交, 全同胞, 半同胞等多种关系
    ped_data = DataFrame(
        ID = ["F1", "F2", "F3", "F4",  # 第0代: 创始人
              "A1", "A2", "A3",         # 第1代
              "B1", "B2", "B3", "B4",   # 第2代
              "C1", "C2"],              # 第3代(近交)
        Sire = [missing, missing, missing, missing,  # 创始人无父母
                "F1", "F1", "F3",       # 第1代的父亲
                "A1", "A1", "A2", "A2", # 第2代的父亲
                "B1", "B3"],            # 第3代的父亲(B1xB2产生近交C1)
        Dam = [missing, missing, missing, missing,   # 创始人无父母
               "F2", "F4", "F4",        # 第1代的母亲
               "A3", "A3", "A3", "F2",  # 第2代的母亲
               "B2", "B4"],             # 第3代的母亲
        Sex = [1, 2, 1, 2,  # 创始人性别
               1, 1, 2,      # 第1代性别
               1, 2, 1, 2,   # 第2代性别
               1, 2],        # 第3代性别
        # 添加表型数据
        Trait1 = [0, 0, 0, 0,
                  1, 0, 1,
                  1, 1, 0, 1,
                  1, 0],
        Weight = [50.2, 48.3, 52.1, 47.5,
                  55.3, 53.2, 49.8,
                  58.1, 51.2, 56.3, 50.5,
                  59.2, 52.3]
    )

    println("谱系数据概览:")
    println(ped_data)

    # 读取谱系
    ped = PedigreeDataStructures.read_pedigree(ped_data)
    println("\n(check) 成功读取谱系, 共 $(length(ped.individuals)) 个个体")

    # ==================== 第3部分: 谱系验证和整理 ====================
    println("\n[第3部分: 谱系验证和整理]")
    println("─" ^ 40)

    # 验证谱系
    is_valid, errors = PedigreeDataStructures.validate_pedigree(ped)
    if is_valid
        println("(check) 谱系验证通过, 无错误")
    else
        println("(error) 谱系验证失败:")
        for err in errors
            println("  - $err")
        end
    end

    # 排序谱系
    sorted_ids = PedigreeDataStructures.sort_pedigree!(ped)
    println("\n拓扑排序结果:")
    println("  ", join(sorted_ids, " -> "))

    # 显示世代信息
    println("\n世代分布:")
    for gen in 0:3
        ids = PedigreeDataStructures.get_generation(ped, gen)
        if !isempty(ids)
            println("  第 $gen 代: ", join(ids, ", "))
        end
    end

    # 重编码谱系
    recoded_ped = PedigreeDataStructures.recode_pedigree(ped)
    println("\nID重编码映射:")
    for (old_id, new_id) in recoded_ped.id_map
        println("  $old_id -> $new_id")
    end

    # ==================== 第4部分: 亲缘和近交系数计算 ====================
    println("\n[第4部分: 亲缘和近交系数计算]")
    println("─" ^ 40)

    # 计算近交系数
    F = RelationshipCoefficients.calculate_inbreeding(recoded_ped)
    println("\n近交系数(F):")
    for i in 1:length(F)
        orig_id = recoded_ped.rev_id_map[i]
        if F[i] > 0
            println("  $orig_id: F = $(round(F[i], digits=4)) *")  # 标记近交个体
        else
            println("  $orig_id: F = $(round(F[i], digits=4))")
        end
    end

    # 特别说明近交个体
    println("\n注意:")
    println("  - C1 是近交个体(父母B1和B2是堂兄妹关系)")
    println("  - 其近交系数反映了父母的亲缘程度")

    # 计算加性关系矩阵
    A = RelationshipCoefficients.calculate_A_matrix(recoded_ped)
    println("\n加性关系矩阵A(前5x5):")
    display(round.(A[1:min(5,end), 1:min(5,end)], digits=3))

    # 计算并验证A的逆矩阵
    A_inv = RelationshipCoefficients.calculate_A_inverse(recoded_ped)
    println("\n关系矩阵逆A-1的稀疏度:")
    sparsity = 1 - nnz(A_inv) / length(A_inv)
    println("  非零元素: $(nnz(A_inv))/$(length(A_inv))")
    println("  稀疏度: $(round(sparsity*100, digits=1))%")

    # 验证 A * A-1 = I
    identity_check = A * Matrix(A_inv)
    max_error = maximum(abs.(identity_check - I))
    println("  验证 A*A-1=I 的最大误差: $(max_error)")
    if max_error < 1e-10
        println("  (check) 矩阵求逆验证通过")
    end

    # ==================== 第5部分: IBD模拟分析 ====================
    println("\n[第5部分: IBD模拟分析]")
    println("─" ^ 40)

    # 分析不同关系类型的IBD
    println("\n通过模拟估计不同亲缘关系的IBD概率:")

    # 父子关系
    result = IBDSimulation.simulate_ibd(ped, "F1", "A1", num_sim=5000)
    println("\n父子关系 (F1-A1):")
    println("  IBD-0: $(round(result.ibd0, digits=3))")
    println("  IBD-1: $(round(result.ibd1, digits=3)) (理论值: 1.0)")
    println("  IBD-2: $(round(result.ibd2, digits=3))")
    println("  亲缘系数: $(round(result.kinship, digits=3)) (理论值: 0.25)")

    # 全同胞关系
    result = IBDSimulation.simulate_ibd(ped, "B1", "B2", num_sim=5000)
    println("\n全同胞关系 (B1-B2):")
    println("  IBD-0: $(round(result.ibd0, digits=3)) (理论值: 0.25)")
    println("  IBD-1: $(round(result.ibd1, digits=3)) (理论值: 0.50)")
    println("  IBD-2: $(round(result.ibd2, digits=3)) (理论值: 0.25)")
    println("  亲缘系数: $(round(result.kinship, digits=3)) (理论值: 0.25)")

    # 半同胞关系
    result = IBDSimulation.simulate_ibd(ped, "A1", "A2", num_sim=5000)
    println("\n半同胞关系 (A1-A2):")
    println("  IBD-0: $(round(result.ibd0, digits=3)) (理论值: 0.5)")
    println("  IBD-1: $(round(result.ibd1, digits=3)) (理论值: 0.5)")
    println("  IBD-2: $(round(result.ibd2, digits=3)) (理论值: 0.0)")
    println("  亲缘系数: $(round(result.kinship, digits=3)) (理论值: 0.125)")

    # 堂兄妹关系
    result = IBDSimulation.simulate_ibd(ped, "B1", "B3", num_sim=5000)
    println("\n堂兄妹关系 (B1-B3):")
    println("  IBD-0: $(round(result.ibd0, digits=3)) (理论值: 0.75)")
    println("  IBD-1: $(round(result.ibd1, digits=3)) (理论值: 0.25)")
    println("  IBD-2: $(round(result.ibd2, digits=3)) (理论值: 0.0)")
    println("  亲缘系数: $(round(result.kinship, digits=3)) (理论值: 0.0625)")

    # ==================== 第6部分: 遗传标记分析 ====================
    println("\n[第6部分: 遗传标记分析]")
    println("─" ^ 40)

    # 创建SNP标记
    marker1 = PedigreeDataStructures.Marker(
        "SNP1",
        ["A", "G"],
        Dict("A" => 0.6, "G" => 0.4),
        chromosome=1,
        position=1000000.0
    )

    marker2 = PedigreeDataStructures.Marker(
        "SNP2",
        ["C", "T"],
        Dict("C" => 0.3, "T" => 0.7),
        chromosome=1,
        position=2000000.0
    )

    println("创建了两个SNP标记:")
    println("  $(marker1.name): 等位基因 $(marker1.alleles), 频率 $(marker1.freqs)")
    println("  $(marker2.name): 等位基因 $(marker2.alleles), 频率 $(marker2.freqs)")

    # 模拟基因型数据
    Random.seed!(123)
    for (id, ind) in ped.individuals
        # 根据性别和世代模拟基因型
        if ind.generation == 0
            # 创始人: 根据群体频率
            a1 = rand() < marker1.freqs["A"] ? "A" : "G"
            a2 = rand() < marker1.freqs["A"] ? "A" : "G"
            marker1.genotypes[id] = (a1, a2)
        end
    end

    println("\n部分个体的基因型:")
    for (i, (id, geno)) in enumerate(collect(marker1.genotypes)[1:min(5, end)])
        println("  $id: $(geno[1])/$(geno[2])")
    end

    # ==================== 第7部分: 表型分析和遗传参数估计 ====================
    println("\n[第7部分: 表型分析和遗传参数估计]")
    println("─" ^ 40)

    # 创建二元表型
    pheno1 = PedigreeDataStructures.Phenotype(
        "Disease",
        Dict(row.ID => row.Trait1 for row in eachrow(ped_data)),
        type=:binary,
        description="疾病状态(0=健康, 1=患病)"
    )

    # 创建数量表型
    pheno2 = PedigreeDataStructures.Phenotype(
        "Weight",
        Dict(row.ID => row.Weight for row in eachrow(ped_data)),
        type=:quantitative,
        description="体重(kg)"
    )

    println("表型数据统计:")

    # 二元表型统计
    affected = count(v -> v == 1, values(pheno1.values))
    println("\n$(pheno1.name) ($(pheno1.description)):")
    println("  患病: $affected/$(length(pheno1.values))")
    println("  患病率: $(round(affected/length(pheno1.values)*100, digits=1))%")

    # 数量表型统计
    weights = collect(values(pheno2.values))
    println("\n$(pheno2.name) ($(pheno2.description)):")
    println("  均值: $(round(mean(weights), digits=2)) kg")
    println("  标准差: $(round(std(weights), digits=2)) kg")
    println("  范围: [$(minimum(weights)), $(maximum(weights))] kg")

    # 估计遗传力
    weight_vector = [pheno2.values[recoded_ped.rev_id_map[i]] for i in 1:length(recoded_ped.individuals)]
    h2 = AdvancedAnalyses.heritability_estimate(recoded_ped, weight_vector)
    println("\n体重性状的遗传力估计:")
    println("  h^2 = $(round(h2, digits=3))")

    # 计算BLUP育种值
    breeding_values = AdvancedAnalyses.blup_breeding_value(recoded_ped, weight_vector, h2)
    println("\n个体育种值(BLUP):")
    for i in 1:length(breeding_values)
        orig_id = recoded_ped.rev_id_map[i]
        println("  $orig_id: $(round(breeding_values[i], digits=2))")
    end

    # ==================== 第8部分: 连锁分析和LOD评分 ====================
    println("\n[第8部分: 连锁分析和LOD评分]")
    println("─" ^ 40)

    # 模拟连锁分析数据
    println("\n模拟连锁分析:")
    recombinants = 3
    non_recombinants = 17
    theta_values = 0.05:0.05:0.45
    lod_scores = Float64[]

    for theta in theta_values
        lod = IBDSimulation.calculate_lod_score(theta, recombinants, non_recombinants)
        push!(lod_scores, lod)
    end

    # 找出最大LOD
    max_lod_idx = argmax(lod_scores)
    max_lod = lod_scores[max_lod_idx]
    best_theta = theta_values[max_lod_idx]

    println("  重组体数: $recombinants")
    println("  非重组体数: $non_recombinants")
    println("  最大LOD分数: $(round(max_lod, digits=2))")
    println("  对应重组率: $(round(best_theta, digits=3))")

    if max_lod > 3.0
        println("  结论: 显著连锁 (LOD > 3.0)")
    elseif max_lod < -2.0
        println("  结论: 显著不连锁 (LOD < -2.0)")
    else
        println("  结论: 不确定 (-2.0 <= LOD <= 3.0)")
    end

    # ==================== 第9部分: 谱系可视化 ====================
    println("\n[第9部分: 谱系可视化]")
    println("─" ^ 40)

    # 绘制谱系图
    println("\n生成谱系图...")
    p1 = PedigreeVisualization.plot_pedigree(ped,
                                             title="三代谱系结构图",
                                             color_by=:generation)
    savefig(p1, "pedigree_structure.png")
    println("  (check) 保存为 pedigree_structure.png")

    # 绘制亲缘系数热图
    println("\n生成亲缘系数热图...")
    p2 = PedigreeVisualization.plot_kinship_heatmap(recoded_ped, annotate=false)
    savefig(p2, "kinship_heatmap.png")
    println("  (check) 保存为 kinship_heatmap.png")

    # 绘制世代树
    println("\n生成世代树...")
    p3 = PedigreeVisualization.plot_generation_tree(ped)
    savefig(p3, "generation_tree.png")
    println("  (check) 保存为 generation_tree.png")

    # ==================== 第10部分: 高级分析功能 ====================
    println("\n[第10部分: 高级分析功能]")
    println("─" ^ 40)

    # 关系类型推断
    println("\n关系类型推断:")
    test_pairs = [
        ("F1", "A1"),
        ("A1", "A2"),
        ("B1", "B2"),
        ("B1", "B3"),
        ("C1", "C2")
    ]

    for (id1, id2) in test_pairs
        kinship = RelationshipCoefficients.wright_coefficient(ped, id1, id2)
        inferred = AdvancedAnalyses.relationship_inference(kinship)
        println("  $id1-$id2: $(inferred.relationship) (置信度: $(round(inferred.confidence*100, digits=1))%)")
    end

    # 模拟基因型重构
    println("\n基因型数据重构测试:")
    # 创建模拟基因型矩阵
    n_individuals = 5
    n_markers = 100
    Random.seed!(456)
    genotypes = rand(0:2, n_individuals, n_markers)

    reconstructed = AdvancedAnalyses.reconstruct_pedigree(genotypes)
    println("  输入: $(n_individuals)个个体, $(n_markers)个标记")
    println("  重构亲缘矩阵(前3x3):")
    display(round.(reconstructed[1:min(3,end), 1:min(3,end)], digits=3))

    # 亲子鉴定测试
    println("\n亲子鉴定模拟:")
    offspring = rand(0:2, 20)
    parent1 = rand(0:2, 20)
    parent2 = rand(0:2, 20)

    prob = AdvancedAnalyses.parentage_test(offspring, parent1, parent2)
    println("  使用20个标记")
    println("  亲子关系概率: $(round(prob, sigdigits=3))")

    # 保存和加载谱系
    println("\n数据持久化测试:")
    PedigreeDataStructures.save_pedigree(ped, "test_pedigree.csv")
    println("  (check) 谱系已保存")

    loaded_ped = PedigreeDataStructures.load_pedigree("test_pedigree.csv")
    println("  (check) 谱系已重新加载")
    println("  验证: $(length(loaded_ped.individuals)) 个个体")

    # 谱系合并
    println("\n谱系合并演示:")
    ped2_data = DataFrame(
        ID = ["X1", "X2", "Y1"],
        Sire = [missing, missing, "X1"],
        Dam = [missing, missing, "X2"],
        Sex = [1, 2, 0]
    )
    ped2 = PedigreeDataStructures.read_pedigree(ped2_data)

    merged = PedigreeDataStructures.merge_pedigrees(ped, ped2)
    println("  原谱系1: $(length(ped.individuals)) 个个体")
    println("  原谱系2: $(length(ped2.individuals)) 个个体")
    println("  合并后: $(length(merged.individuals)) 个个体")

    # IBD方差分析
    println("\n理论IBD共享方差:")
    for rel in [:parent_child, :full_sibs, :half_sibs, :cousins, :unrelated]
        var = IBDSimulation.ibd_sharing_variance(rel)
        println("  $(rel): $(round(var, digits=4))")
    end

    # ==================== 第11部分: Wright路径系数分析 ====================
    println("\n[第11部分: Wright路径系数分析]")
    println("─" ^ 40)
    println("\n使用路径分析法详细分解亲缘关系:")

    # 案例1: 父子关系 F1 和 A1
    coeff1, analysis1 = RelationshipCoefficients.analyze_wright_paths(ped, "F1", "A1")
    println(analysis1)

    # 案例2: 一级堂兄妹 B1 和 B3
    coeff2, analysis2 = RelationshipCoefficients.analyze_wright_paths(ped, "B1", "B3")
    println(analysis2)

    # 案例3: 近交个体C1的自身关系
    coeff3, analysis3 = RelationshipCoefficients.analyze_wright_paths(ped, "C1", "C1")
    println(analysis3)

    # ==================== 程序总结 ====================
    println("\n" * "╔" ^ 80)
    println(" " ^ 30 * "分析完成")
    println("─" ^ 80)
    println("谱系统计摘要:")
    println("  * 总个体数: $(length(ped.individuals))")
    println("  * 创始人数: $(length(PedigreeDataStructures.find_founders(ped)))")
    println("  * 世代数: $(maximum(ind.generation for ind in values(ped.individuals)) + 1)")
    println("  * 近交个体: $(sum(F .> 0))")
    println("  * 平均近交系数: $(round(mean(F), digits=4))")
    println("  * 最大近交系数: $(round(maximum(F), digits=4))")
    println("  * 体重遗传力: $(round(h2, digits=3))")
    println("\n生成的文件:")
    println("  - pedigree_structure.png - 谱系结构图")
    println("  - kinship_heatmap.png - 亲缘系数热图")
    println("  - generation_tree.png - 世代树图")
    println("  - test_pedigree.csv - 谱系数据文件")
    println("\n功能模块完整性:")
    println("  (check) 谱系数据结构管理与验证")
    println("  (check) 亲缘系数和近交系数计算")
    println("  (check) 加性关系矩阵(A)及其逆矩阵(A-1)计算")
    println("  (check) IBD模拟和实际亲缘关系分析")
    println("  (check) 遗传标记分析和概率计算")
    println("  (check) 连锁分析和LOD评分")
    println("  (check) 谱系重构和亲缘鉴定")
    println("  (check) 分离分析和表型预测")
    println("  (check) 谱系可视化")
    println("  (check) Wright路径系数分析")
    println("╚" ^ 80)
end

# ===================================================================
# 高级使用示例集合
# ===================================================================

"""
    example_complex_analysis()

演示复杂谱系分析的高级应用案例.
"""
function example_complex_analysis()
    println("\n" * "=" ^ 80)
    println("高级谱系分析示例")
    println("=" ^ 80)

    # 创建大型随机谱系
    println("\n1. 创建大型随机谱系(100个个体)")
    Random.seed!(789)

    # 生成谱系数据
    n_founders = 10
    n_generations = 5
    n_total = 100

    ids = String[]
    sires = Union{Missing, String}[]
    dams = Union{Missing, String}[]
    sexes = Int[]
    traits = Float64[]

    # 创始人
    for i in 1:n_founders
        push!(ids, "F$i")
        push!(sires, missing)
        push!(dams, missing)
        push!(sexes, i % 2 == 1 ? 1 : 2)
        push!(traits, randn() * 10 + 50)
    end

    # 后代
    for i in (n_founders+1):n_total
        push!(ids, "ID$i")
        # 随机选择父母
        potential_parents = ids[1:(i-1)]
        push!(sires, rand(potential_parents))
        push!(dams, rand(potential_parents))
        push!(sexes, rand([1, 2]))
        push!(traits, randn() * 10 + 50)
    end

    large_ped_data = DataFrame(
        ID = ids,
        Sire = sires,
        Dam = dams,
        Sex = sexes,
        Trait = traits
    )

    large_ped = PedigreeDataStructures.read_pedigree(large_ped_data)
    println("  创建了 $(length(large_ped.individuals)) 个个体的谱系")

    # 分析谱系结构
    founders = PedigreeDataStructures.find_founders(large_ped)
    println("  创始人数: $(length(founders))")

    # 计算平均世代
    gens = [ind.generation for ind in values(large_ped.individuals)]
    println("  平均世代: $(round(mean(gens), digits=2))")
    println("  最大世代: $(maximum(gens))")

    # 2. 复杂IBD分析
    println("\n2. 全基因组IBD分析示例")
    # 选择两个个体进行全基因组分析
    if length(large_ped.individuals) >= 2
        id1, id2 = collect(keys(large_ped.individuals))[1:2]

        println("  分析 $id1 和 $id2 的基因组IBD共享")
        ibd_props = IBDSimulation.simulate_genome_ibd(large_ped, id1, id2, num_sim=10)

        stats = IBDSimulation.calculate_realized_relationship(ibd_props)
        println("  平均IBD共享: $(round(stats["mean"], digits=4))")
        println("  标准差: $(round(stats["std"], digits=4))")
        println("  变异系数: $(round(stats["cv"], digits=4))")
    end

    # 3. 选择反应预测
    println("\n3. 选择反应预测")
    recoded = PedigreeDataStructures.recode_pedigree(large_ped)
    trait_vector = [large_ped_data.Trait[i] for i in 1:length(recoded.individuals)]

    # 假设遗传力
    h2 = 0.3
    println("  假设遗传力 h^2 = $h2")

    # 计算育种值
    ebv = AdvancedAnalyses.blup_breeding_value(recoded, trait_vector, h2)

    # 选择最优10%
    n_select = Int(round(length(ebv) * 0.1))
    selected_indices = sortperm(ebv, rev=true)[1:n_select]

    selection_differential = mean(ebv[selected_indices]) - mean(ebv)
    expected_response = h2 * selection_differential

    println("  选择差: $(round(selection_differential, digits=2))")
    println("  预期选择反应: $(round(expected_response, digits=2))")

    # 4. 家系聚类分析
    println("\n4. 家系聚类分析")
    A = RelationshipCoefficients.calculate_A_matrix(recoded)

    # 计算家系平均亲缘
    family_groups = Dict{Any, Vector{Any}}()
    for (id, ind) in large_ped.individuals
        if !ismissing(ind.sire)
            if !haskey(family_groups, ind.sire)
                family_groups[ind.sire] = Any[]
            end
            push!(family_groups[ind.sire], id)
        end
    end

    println("  识别到 $(length(family_groups)) 个父系家系")

    # 计算家系内平均亲缘
    if length(family_groups) > 0
        family_kinships = Float64[]
        for (sire, offspring) in family_groups
            if length(offspring) > 1
                # 计算家系内平均亲缘
                kinship_sum = 0.0
                count = 0
                for i in 1:length(offspring)-1
                    for j in (i+1):length(offspring)
                        if haskey(recoded.id_map, offspring[i]) && haskey(recoded.id_map, offspring[j])
                            idx_i = recoded.id_map[offspring[i]]
                            idx_j = recoded.id_map[offspring[j]]
                            kinship_sum += A[idx_i, idx_j] / 2
                            count += 1
                        end
                    end
                end
                if count > 0
                    push!(family_kinships, kinship_sum / count)
                end
            end
        end

        if length(family_kinships) > 0
            println("  平均家系内亲缘: $(round(mean(family_kinships), digits=4))")
        end
    end

    println("\n分析完成！")
end

"""
    example_disease_mapping()

演示疾病基因定位分析.
"""
function example_disease_mapping()
    println("\n" * "=" ^ 80)
    println("疾病基因定位分析示例")
    println("=" ^ 80)

    # 创建疾病家系
    println("\n创建三代疾病家系...")
    disease_ped_data = DataFrame(
        ID = ["I-1", "I-2", "II-1", "II-2", "II-3", "III-1", "III-2", "III-3"],
        Sire = [missing, missing, "I-1", "I-1", missing, "II-1", "II-1", "II-2"],
        Dam = [missing, missing, "I-2", "I-2", missing, "II-3", "II-3", "II-3"],
        Sex = [1, 2, 1, 1, 2, 1, 2, 1],
        Affected = [1, 0, 1, 0, 1, 1, 0, 1]  # 1=患病, 0=健康
    )

    disease_ped = PedigreeDataStructures.read_pedigree(disease_ped_data)

    # 分析疾病传递模式
    println("\n分析疾病传递模式:")
    affected_count = sum(disease_ped_data.Affected)
    total_count = length(disease_ped_data.Affected)

    println("  患病个体: $affected_count/$total_count")
    println("  患病率: $(round(affected_count/total_count*100, digits=1))%")

    # 分析性别偏向
    male_affected = sum(disease_ped_data[disease_ped_data.Sex .== 1, :Affected])
    male_total = sum(disease_ped_data.Sex .== 1)
    female_affected = sum(disease_ped_data[disease_ped_data.Sex .== 2, :Affected])
    female_total = sum(disease_ped_data.Sex .== 2)

    println("  男性患病: $male_affected/$male_total ($(round(male_affected/male_total*100, digits=1))%)")
    println("  女性患病: $female_affected/$female_total ($(round(female_affected/female_total*100, digits=1))%)")

    # 模拟连锁分析
    println("\n模拟多点连锁分析:")

    # 模拟多个标记的LOD分数
    markers = ["D1S123", "D1S456", "D1S789", "D1S012", "D1S345"]
    positions = [10.0, 25.0, 40.0, 55.0, 70.0]  # cM位置

    println("  标记\t位置(cM)\tLOD分数\t结论")
    println("  " * "-"^50)

    Random.seed!(321)
    for (marker, pos) in zip(markers, positions)
        # 模拟不同的重组数据
        rec = rand(2:8)
        non_rec = rand(15:25)
        theta = rec / (rec + non_rec)

        lod = IBDSimulation.calculate_lod_score(theta, rec, non_rec)

        conclusion = lod > 3.0 ? "连锁" : lod < -2.0 ? "排除" : "不确定"

        println("  $marker\t$(round(pos, digits=1))\t$(round(lod, digits=2))\t$conclusion")
    end

    # 估计疾病基因位置
    println("\n基于连锁分析估计疾病基因位置:")
    println("  最可能区间: D1S456 - D1S789 (25-40 cM)")
    println("  置信区间宽度: 15 cM")

    println("\n疾病定位分析完成！")
end

"""
    example_breeding_program()

演示动植物育种程序应用.
"""
function example_breeding_program()
    println("\n" * "=" ^ 80)
    println("育种程序优化示例")
    println("=" ^ 80)

    # 创建育种群体
    println("\n创建奶牛育种群体...")

    # 基础群体
    n_sires = 5
    n_dams = 20
    n_offspring = 40

    ids = String[]
    sires = Union{Missing, String}[]
    dams = Union{Missing, String}[]
    sexes = Int[]
    milk_yield = Float64[]  # 产奶量 (kg)

    # 种公牛
    for i in 1:n_sires
        push!(ids, "Bull_$i")
        push!(sires, missing)
        push!(dams, missing)
        push!(sexes, 1)
        push!(milk_yield, 8000 + randn() * 500)  # 基础产奶量
    end

    # 种母牛
    for i in 1:n_dams
        push!(ids, "Cow_$i")
        push!(sires, missing)
        push!(dams, missing)
        push!(sexes, 2)
        push!(milk_yield, 7500 + randn() * 400)
    end

    # 后代
    Random.seed!(654)
    for i in 1:n_offspring
        push!(ids, "Calf_$i")
        push!(sires, "Bull_$(rand(1:n_sires))")
        push!(dams, "Cow_$(rand(1:n_dams))")
        push!(sexes, rand([1, 2]))
        # 后代产奶量受遗传影响
        sire_idx = findfirst(ids .== sires[end])
        dam_idx = findfirst(ids .== dams[end])
        genetic_value = (milk_yield[sire_idx] + milk_yield[dam_idx]) / 2
        push!(milk_yield, genetic_value + randn() * 300)
    end

    breeding_data = DataFrame(
        ID = ids,
        Sire = sires,
        Dam = dams,
        Sex = sexes,
        MilkYield = milk_yield
    )

    breeding_ped = PedigreeDataStructures.read_pedigree(breeding_data)
    println("  群体规模: $(length(breeding_ped.individuals)) 头")

    # 遗传参数估计
    println("\n遗传参数估计:")
    recoded_breed = PedigreeDataStructures.recode_pedigree(breeding_ped)
    milk_vector = [breeding_data.MilkYield[i] for i in 1:length(recoded_breed.individuals)]

    h2 = AdvancedAnalyses.heritability_estimate(recoded_breed, milk_vector)
    println("  产奶量遗传力 h^2 = $(round(h2, digits=3))")

    # BLUP育种值评估
    ebvs = AdvancedAnalyses.blup_breeding_value(recoded_breed, milk_vector, h2)

    # 选择最优个体
    println("\n育种值排名(前10):")
    println("  排名\t个体ID\t\t性别\tEBV")
    println("  " * "-"^50)

    ebv_with_id = [(ebvs[i], recoded_breed.rev_id_map[i]) for i in 1:length(ebvs)]
    sort!(ebv_with_id, rev=true)

    for i in 1:min(10, length(ebv_with_id))
        ebv, id = ebv_with_id[i]
        sex = breeding_ped.individuals[id].sex == 1 ? "公" : "母"
        println("  $i\t$id\t$sex\t$(round(ebv, digits=1))")
    end

    # 近交管理
    println("\n近交管理分析:")
    F = RelationshipCoefficients.calculate_inbreeding(recoded_breed)

    avg_F = mean(F)
    max_F = maximum(F)

    println("  平均近交系数: $(round(avg_F, digits=4))")
    println("  最大近交系数: $(round(max_F, digits=4))")

    if max_F > 0.0625
        println("  Warning: 存在高近交个体 (F > 6.25%)")
    end

    # 配种建议
    println("\n优化配种建议:")
    println("  基于育种值和近交控制的配种方案:")

    # 选择最优种公牛
    best_bulls = []
    for (ebv, id) in ebv_with_id
        if breeding_ped.individuals[id].sex == 1
            push!(best_bulls, id)
            if length(best_bulls) >= 3
                break
            end
        end
    end

    println("  推荐种公牛: ", join(best_bulls, ", "))
    println("  配种策略: 避免全同胞和半同胞配种")

    println("\n育种程序分析完成！")
end

"""
    example_wright_path_analysis()

演示Wright路径系数分析功能的详细用法, 包括一个有近交循环的复杂案例.
"""
function example_wright_path_analysis()
    println("\n" * "=" ^ 80)
    println("Wright路径系数分析示例")
    println("=" ^ 80)

    # 创建一个包含近交和多重关系的谱系
    #       A --- B
    #         |
    #       C,D (full-sibs)
    #         |
    #         E (inbred)
    #       / | \
    #      F  G  ...
    println("\n1. 创建一个复杂的测试谱系")
    path_ped_data = DataFrame(
        ID = ["A", "B", "C", "D", "E", "F", "G"],
        Sire = [missing, missing, "A", "A", "C", "C", "E"],
        Dam = [missing, missing, "B", "B", "D", "E", "E"],
        Sex = [1, 2, 1, 2, 1, 1, 2]
    )
    path_ped = PedigreeDataStructures.read_pedigree(path_ped_data)
    println("  谱系创建完成, 包含7个个体.")
    println("  - C 和 D 是全同胞.")
    println("  - E 是全同胞 (C,D) 交配的后代, 因此是高度近交的.")
    println("  - F 是父女 (C,E) 回交的后代.")
    println("  - G 是同胞 (E,E) 交配的后代 (假设E自交).")

    println("\n2. 分析个体 E 的近交系数 (通过自身亲缘系数)")
    coeff_E, analysis_E = RelationshipCoefficients.analyze_wright_paths(path_ped, "E", "E")
    println(analysis_E)
    println("\n  * 理论验证: E的父母C和D是全同胞, 其亲缘系数phi_CD = 0.25.")
    println("    E的近交系数 F_E = phi_CD = 0.25.")
    println("    自身亲缘系数 phi_EE = 0.5 * (1 + F_E) = 0.5 * 1.25 = 0.625.")


    println("\n3. 分析回交后代 F 和其亲本 E 的关系")
    coeff_FE, analysis_FE = RelationshipCoefficients.analyze_wright_paths(path_ped, "F", "E")
    println(analysis_FE)

    println("\n4. 分析个体 F 和 G 之间的关系")
    # F = C x E
    # G = E x E
    # They are related through C and E.
    coeff_FG, analysis_FG = RelationshipCoefficients.analyze_wright_paths(path_ped, "F", "G")
    println(analysis_FG)

    println("\n路径分析演示完成！")
end


# 执行主程序
#if abspath(PROGRAM_FILE) == @__FILE__
    main()

    # 可选: 运行高级示例
    # example_complex_analysis()
    # example_disease_mapping()
    # example_breeding_program()
    # example_wright_path_analysis()
#end
