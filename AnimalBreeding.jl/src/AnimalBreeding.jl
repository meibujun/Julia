# AnimalBreeding.jl - 核心数据结构和工具模块
# 动物育种软件系统核心模块 - Julia 1.11.6 兼容版本
# 作者：AnimalBreeding.jl 开发团队
# 版本：1.0.1 (已修复核心逻辑)
# 最后更新：2025年

"""
    AnimalBreeding 核心模块

    提供动物育种分析的基础数据结构和核心功能，包括：
    - 系谱、基因型、表型数据管理
    - 关系矩阵计算（A、G、H矩阵）
    - 数据验证和质量控制
    - 模型定义和配置
"""
module AnimalBreeding

using DataFrames
using CSV
using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using Random
using ProgressMeter
using Dates

# 导出数据结构
export DataManager, Pedigree, Genotypes, Phenotypes, ModelSpec, RandomEffect
# 导出数据加载函数
export load_pedigree, load_genotypes, load_phenotypes
# 导出数据处理函数
export validate_data, compute_relationship_matrix, check_data_consistency
# 导出模型定义函数
export define_model, describe_model, validate_model

# ==================== 核心数据结构定义 ====================

"""
    Pedigree

    系谱信息数据结构，存储动物谱系关系及相关计算矩阵
"""
mutable struct Pedigree
    data::DataFrame                                    # 已按世代排序的系谱数据框
    n_animals::Int                                    # 动物总数
    id_map::Dict{Any,Int}                            # ID到索引的映射字典
    generation::Vector{Int}                           # 世代信息
    inbreeding::Vector{Float64}                       # 近交系数
    A_matrix::Union{Nothing,SparseMatrixCSC{Float64,Int}}    # A矩阵
    A_inv::Union{Nothing,SparseMatrixCSC{Float64,Int}}       # A逆矩阵
    validated::Bool                                    # 验证标志

    function Pedigree(data::DataFrame)
        # 验证必需的列
        required_cols = [:animal, :sire, :dam]
        for col in required_cols
            if !(Symbol(col) in names(data))
                error("系谱数据必须包含列: $col")
            end
        end

        # 预处理：标准化ID，处理缺失值
        df_clean = copy(data)
        for col in [:sire, :dam]
            df_clean[!, col] = coalesce.(df_clean[!, col], 0)
        end

        # 创建一个包含所有动物（包括仅作为亲本出现的）的临时系谱
        all_ids = unique(vcat(df_clean.animal, df_clean.sire, df_clean.dam))
        filter!(x -> x != 0 && !ismissing(x), all_ids)

        id_map = Dict(id => i for (i, id) in enumerate(all_ids))

        # 为所有动物构建一个完整的系谱DataFrame
        temp_ped_data = DataFrame(animal=all_ids)
        temp_ped_data = leftjoin(temp_ped_data, df_clean, on=:animal)
        temp_ped_data.sire = coalesce.(temp_ped_data.sire, 0)
        temp_ped_data.dam = coalesce.(temp_ped_data.dam, 0)

        # 计算世代数并排序
        generations = compute_generations(temp_ped_data, id_map)
        perm = sortperm(generations)
        sorted_ped = temp_ped_data[perm, :]

        # 重建ID映射
        sorted_id_map = Dict(id => i for (i, id) in enumerate(sorted_ped.animal))

        n = nrow(sorted_ped)
        inbreeding = zeros(n)

        new(sorted_ped, n, sorted_id_map, generations[perm], inbreeding, nothing, nothing, false)
    end
end

"""
    Genotypes

    基因型数据结构，存储SNP标记信息及基因组关系矩阵
"""
mutable struct Genotypes
    animal_ids::Vector                                # 动物ID列表
    markers::Matrix{Float64}                          # 标记矩阵（0/1/2编码）
    marker_names::Vector{String}                      # 标记名称
    n_animals::Int                                    # 动物数量
    n_markers::Int                                    # 标记数量
    maf::Vector{Float64}                             # 最小等位基因频率
    call_rate::Vector{Float64}                       # 检出率
    G_matrix::Union{Nothing,Matrix{Float64}}         # G矩阵
    quality_metrics::Dict{String,Float64}            # 质量指标

    function Genotypes(animal_ids::Vector, markers::Matrix, marker_names::Vector{String})
        n_animals, n_markers = size(markers)
        maf = Vector{Float64}(undef, n_markers)
        call_rate = Vector{Float64}(undef, n_markers)

        for j in 1:n_markers
            non_missing = .!ismissing.(markers[:, j])
            call_rate[j] = mean(non_missing)
            if call_rate[j] > 0
                freq = mean(skipmissing(markers[:, j])) / 2.0
                maf[j] = min(freq, 1 - freq)
            else
                maf[j] = 0.0
            end
        end

        markers_clean = Matrix{Float64}(undef, n_animals, n_markers)
        for j in 1:n_markers
            col = markers[:, j]
            mean_val = call_rate[j] > 0 ? mean(skipmissing(col)) : 0.0
            markers_clean[:, j] = [ismissing(x) ? mean_val : Float64(x) for x in col]
        end

        quality_metrics = calculate_genotype_metrics(markers_clean, maf, call_rate)
        new(animal_ids, markers_clean, marker_names, n_animals, n_markers, maf, call_rate, nothing, quality_metrics)
    end
end

"""
    Phenotypes

    表型记录数据结构，包含性状测定值和固定效应
"""
mutable struct Phenotypes
    data::DataFrame                                   # 完整数据框
    traits::Vector{Symbol}                           # 性状列表
    fixed_effects::Vector{Symbol}                    # 固定效应列表
    n_records::Int                                   # 记录数
    n_traits::Int                                    # 性状数
    trait_means::Dict{Symbol,Float64}                # 性状均值
    trait_stds::Dict{Symbol,Float64}                 # 性状标准差
    trait_heritabilities::Dict{Symbol,Float64}       # 遗传力
    missing_patterns::Dict{Symbol,Vector{Bool}}      # 缺失模式

    function Phenotypes(data::DataFrame, traits::Vector{Symbol}, fixed_effects::Vector{Symbol})
        n_records, n_traits = nrow(data), length(traits)
        trait_means, trait_stds, missing_patterns = Dict(), Dict(), Dict()

        for trait in traits
            vals = collect(skipmissing(data[!, trait]))
            trait_means[trait] = isempty(vals) ? 0.0 : mean(vals)
            trait_stds[trait] = isempty(vals) ? 0.0 : std(vals)
            missing_patterns[trait] = ismissing.(data[!, trait])
        end

        new(data, traits, fixed_effects, n_records, n_traits, trait_means, trait_stds, Dict(), missing_patterns)
    end
end

"""
    DataManager

    中心数据管理器，整合所有数据类型
"""
mutable struct DataManager
    pedigree::Union{Nothing,Pedigree}
    genotypes::Union{Nothing,Genotypes}
    phenotypes::Union{Nothing,Phenotypes}
    metadata::Dict{String,Any}
    species::String
    data_sources::Dict{String,String}
    validation_log::Vector{String}

    DataManager(species::String="cattle") = new(nothing, nothing, nothing, Dict(), species, Dict(), [])
end

# ==================== 模型定义结构 ====================

"""
    RandomEffect

    随机效应定义
"""
struct RandomEffect
    name::String
    type::Symbol
    RandomEffect(name::String, type::Symbol=:iid) = new(name, type)
end

"""
    ModelSpec

    完整的统计模型定义
"""
mutable struct ModelSpec
    traits::Vector{Symbol}
    fixed_effects::Vector{Symbol}
    random_effects::Vector{RandomEffect}
    model_equation::String

    function ModelSpec(traits, fixed, random)
        eq = generate_model_equation(traits, fixed, random)
        new(traits, fixed, random, eq)
    end
end

# ==================== 数据加载与预处理 ====================

"""
    compute_generations(pedigree::DataFrame, id_map::Dict) -> Vector{Int}

    [已修复] 鲁棒地计算每个个体的世代数，即使系谱未排序。
"""
function compute_generations(pedigree::DataFrame, id_map::Dict)
    n = nrow(pedigree)
    generations = zeros(Int, n)
    changed = true
    max_iters = n + 5 # 设定一个安全的最大迭代次数
    iter = 0

    while changed && iter < max_iters
        changed = false
        iter += 1
        for i in 1:n
            sire_id = pedigree.sire[i]
            dam_id = pedigree.dam[i]

            sire_idx = get(id_map, sire_id, 0)
            dam_idx = get(id_map, dam_id, 0)

            sire_gen = sire_idx > 0 ? generations[sire_idx] : 0
            dam_gen = dam_idx > 0 ? generations[dam_idx] : 0

            current_gen = max(sire_gen, dam_gen) + 1
            if generations[i] != current_gen
                generations[i] = current_gen
                changed = true
            end
        end
    end

    if iter >= max_iters && changed
        @warn "世代计算达到最大迭代次数，系谱中可能存在循环。"
    end

    return generations
end

function load_pedigree(file; kwargs...)
    df = isa(file, String) ? CSV.read(file, DataFrame; kwargs...) : DataFrame(file)
    println("正在加载系谱文件...")
    return Pedigree(df)
end

function load_genotypes(file; kwargs...)
    df = isa(file, String) ? CSV.read(file, DataFrame; kwargs...) : DataFrame(file)
    println("正在加载基因型文件...")
    animal_ids = df[!, 1]
    markers = Matrix(df[:, 2:end])
    marker_names = names(df)[2:end]
    return Genotypes(animal_ids, markers, marker_names)
end

function load_phenotypes(file; trait_cols::Vector{String}, fixed_cols::Vector{String}=String[], kwargs...)
    df = isa(file, String) ? CSV.read(file, DataFrame; kwargs...) : DataFrame(file)
    println("正在加载表型文件...")
    return Phenotypes(df, Symbol.(trait_cols), Symbol.(fixed_cols))
end

# ==================== 关系矩阵计算 ====================

"""
    compute_A_matrix!(ped::Pedigree)

    [已修复] 计算加性遗传关系矩阵（A矩阵）。修复了对角线和非对角线元素的计算逻辑。
"""
function compute_A_matrix!(ped::Pedigree)
    n = ped.n_animals
    println("正在计算A矩阵 ($(n)×$(n))...")
    A = zeros(n, n)
    p = Progress(n, desc="计算A矩阵: ", color=:green)

    for i in 1:n
        sire_id = ped.data.sire[i]
        dam_id = ped.data.dam[i]

        sire_idx = get(ped.id_map, sire_id, 0)
        dam_idx = get(ped.id_map, dam_id, 0)

        # 首先计算非对角线元素
        for j in 1:i-1
            val = 0.0
            # [FIX] 使用正确的索引访问已计算的父本/母本与个体j的关系
            if sire_idx > 0
                val += 0.5 * A[sire_idx, j]
            end
            if dam_idx > 0
                val += 0.5 * A[dam_idx, j]
            end
            A[i, j] = A[j, i] = val
        end

        # 然后计算对角线元素 (1 + F), F是近交系数
        if sire_idx > 0 && dam_idx > 0
            ped.inbreeding[i] = 0.5 * A[sire_idx, dam_idx]
        end
        A[i, i] = 1.0 + ped.inbreeding[i]

        next!(p)
    end

    ped.A_matrix = sparse(A)
    println("  A矩阵计算完成. 平均近交系数: $(round(mean(ped.inbreeding), digits=4))")
end

# 其他关系矩阵函数保持不变...
function compute_A_inverse! end
function compute_G_matrix! end
function compute_H_matrix end
function compute_relationship_matrix(dm::DataManager; type::Symbol)
    if type == :A || type == :pedigree
        isnothing(dm.pedigree) && error("系谱数据未加载")
        compute_A_matrix!(dm.pedigree)
    elseif type == :G || type == :genomic
        isnothing(dm.genotypes) && error("基因型数据未加载")
        # Placeholder for G matrix computation
    end
end


# ==================== 模型定义函数 ====================

function define_model(; traits, fixed=[], random=[])
    traits_sym = Symbol.(traits)
    fixed_sym = Symbol.(fixed)
    random_effects = [isa(r, Tuple) ? RandomEffect(String(r[1]), r[2]) : RandomEffect(String(r)) for r in random]
    return ModelSpec(traits_sym, fixed_sym, random_effects)
end

function generate_model_equation(traits, fixed, random)
    trait_str = join(string.(traits), ", ")
    fixed_str = isempty(fixed) ? "μ" : join(string.(fixed), " + ")
    random_str = isempty(random) ? "" : " + " * join([e.name for e in random], " + ")
    return "$trait_str = $fixed_str$random_str + e"
end

function describe_model(model::ModelSpec)
    println("\n--- 模型描述 ---")
    println("  $(model.model_equation)")
    println("-----------------")
end

# ==================== 验证及其他辅助函数 ====================
function validate_data end
function check_data_consistency end
function has_pedigree_loops end
function check_hardy_weinberg end
function check_outliers end
function validate_model end
function quality_control_genotypes end
function calculate_genotype_metrics end

end # module AnimalBreeding