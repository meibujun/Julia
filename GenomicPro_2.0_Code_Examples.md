# GenomicPro 2.0 核心代码实现示例

本文档提供 GenomicPro 2.0 关键模块的具体代码实现示例。

---

## 目录

1. [核心类型定义](#1-核心类型定义)
2. [数据层实现](#2-数据层实现)
3. [GRM 计算优化实现](#3-grm-计算优化实现)
4. [GBLUP 求解器](#4-gblup-求解器)
5. [BayesR 实现](#5-bayesr-实现)
6. [Pipeline 系统](#6-pipeline-系统)
7. [使用示例](#7-使用示例)

---

## 1. 核心类型定义

### src/Core/types.jl

```julia
module Core

using LinearAlgebra

# ============================================================================
# 抽象类型层次
# ============================================================================

"""
所有基因组数据的根类型
"""
abstract type AbstractGenomicData{T} end

"""
基因型数据抽象类型
"""
abstract type AbstractGenotypeData{T} <: AbstractGenomicData{T} end

"""
表型数据抽象类型
"""
abstract type AbstractPhenotypeData{T} <: AbstractGenomicData{T} end

"""
系谱数据抽象类型
"""
abstract type AbstractPedigreeData{T} <: AbstractGenomicData{T} end

# ============================================================================
# 通用接口（所有数据类型必须实现）
# ============================================================================

"""
    n_samples(data::AbstractGenomicData) -> Int

返回样本数量
"""
function n_samples end

"""
    sample_ids(data::AbstractGenomicData) -> Vector{String}

返回样本 ID 列表
"""
function sample_ids end

"""
    validate(data::AbstractGenomicData) -> ValidationResult

验证数据完整性和一致性
"""
function validate end

# ============================================================================
# 基因型数据特定接口
# ============================================================================

"""
    n_markers(geno::AbstractGenotypeData) -> Int

返回标记数量
"""
function n_markers end

"""
    marker_ids(geno::AbstractGenotypeData) -> Vector{String}

返回标记 ID 列表
"""
function marker_ids end

"""
    allele_frequencies(geno::AbstractGenotypeData) -> Vector{Float64}

返回等位基因频率
"""
function allele_frequencies end

"""
    missing_rate(geno::AbstractGenotypeData; dim=0) -> Union{Float64, Vector{Float64}}

计算缺失率
- dim=0: 总体缺失率
- dim=1: 每个样本的缺失率
- dim=2: 每个标记的缺失率
"""
function missing_rate end

# ============================================================================
# 验证结果
# ============================================================================

struct ValidationResult
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
    metadata::Dict{Symbol, Any}

    function ValidationResult()
        new(true, String[], String[], Dict{Symbol, Any}())
    end
end

function Base.show(io::IO, vr::ValidationResult)
    if vr.valid
        printstyled(io, "✓ 验证通过\n"; color=:green, bold=true)
    else
        printstyled(io, "✗ 验证失败\n"; color=:red, bold=true)
        println(io, "错误 ($(length(vr.errors))):")
        for err in vr.errors
            println(io, "  • ", err)
        end
    end

    if !isempty(vr.warnings)
        printstyled(io, "⚠ 警告 ($(length(vr.warnings))):\n"; color=:yellow)
        for warn in vr.warnings
            println(io, "  • ", warn)
        end
    end
end

function merge(results::Vector{ValidationResult})
    merged = ValidationResult()
    merged.valid = all(r -> r.valid, results)
    merged.errors = vcat([r.errors for r in results]...)
    merged.warnings = vcat([r.warnings for r in results]...)
    merge!(merged.metadata, [r.metadata for r in results]...)
    return merged
end

# ============================================================================
# 自定义异常
# ============================================================================

abstract type GenomicProException <: Exception end

struct DataValidationError <: GenomicProException
    msg::String
    field::Symbol
    value::Any
end

struct DimensionMismatchError <: GenomicProException
    expected::Tuple
    actual::Tuple
end

struct ConvergenceError <: GenomicProException
    msg::String
    iterations::Int
    residual::Float64
end

Base.showerror(io::IO, e::DataValidationError) =
    print(io, "DataValidationError: $(e.msg)\n  Field: $(e.field)\n  Value: $(e.value)")

Base.showerror(io::IO, e::DimensionMismatchError) =
    print(io, "DimensionMismatchError: expected $(e.expected), got $(e.actual)")

Base.showerror(io::IO, e::ConvergenceError) =
    print(io, "ConvergenceError: $(e.msg)\n  Iterations: $(e.iterations)\n  Residual: $(e.residual)")

export AbstractGenomicData, AbstractGenotypeData, AbstractPhenotypeData, AbstractPedigreeData
export ValidationResult, GenomicProException, DataValidationError, DimensionMismatchError, ConvergenceError
export n_samples, sample_ids, validate
export n_markers, marker_ids, allele_frequencies, missing_rate

end  # module Core
```

---

## 2. 数据层实现

### src/Data/genotypes.jl

```julia
module Genotypes

using ..Core
using LinearAlgebra, Statistics, SparseArrays

# ============================================================================
# CompactGenotypes - 高效 2-bit 编码存储
# ============================================================================

"""
    CompactGenotypes{T<:Integer}

使用 2-bit 编码的紧凑基因型存储

# 编码方案
- 00: 纯合参考 (0 个替代等位基因)
- 01: 杂合 (1 个替代等位基因)
- 10: 纯合替代 (2 个替代等位基因)
- 11: 缺失

# 内存使用
对于 n 个样本和 m 个标记：
- 原始存储: n × m × 8 bytes (Float64)
- 紧凑存储: n × m × 2 bits = n × m / 4 bytes
- 节省: 96.875%
"""
struct CompactGenotypes{T<:Integer} <: AbstractGenotypeData{T}
    # 核心数据（2-bit 编码）
    data::Vector{UInt8}

    # 维度信息
    n_samples::Int
    n_markers::Int

    # 标识符
    sample_ids::Vector{String}
    marker_ids::Vector{String}

    # 缺失值掩码
    missing_mask::BitMatrix

    # 元数据
    chromosome::Vector{String}
    position::Vector{Int}
    ref_allele::Vector{String}
    alt_allele::Vector{String}

    # 统计信息（延迟计算）
    allele_freqs::Vector{Float64}

    function CompactGenotypes(
        data::AbstractMatrix{T},
        sample_ids::Vector{String},
        marker_ids::Vector{String};
        chromosome::Union{Vector{String}, Nothing} = nothing,
        position::Union{Vector{Int}, Nothing} = nothing,
        ref_allele::Union{Vector{String}, Nothing} = nothing,
        alt_allele::Union{Vector{String}, Nothing} = nothing
    ) where T<:Integer
        n_samples, n_markers = size(data)

        # 验证输入
        @assert length(sample_ids) == n_samples "样本 ID 数量不匹配"
        @assert length(marker_ids) == n_markers "标记 ID 数量不匹配"
        @assert all(x -> x in [0, 1, 2] || ismissing(x), data) "基因型必须是 0, 1, 2 或 missing"

        # 编码为 2-bit
        encoded, missing_mask = encode_genotypes(data)

        # 计算等位基因频率
        freqs = compute_allele_frequencies_kahan(data, missing_mask)

        # 默认元数据
        chrom = isnothing(chromosome) ? fill("0", n_markers) : chromosome
        pos = isnothing(position) ? collect(1:n_markers) : position
        ref = isnothing(ref_allele) ? fill("A", n_markers) : ref_allele
        alt = isnothing(alt_allele) ? fill("T", n_markers) : alt_allele

        new{T}(
            encoded,
            n_samples,
            n_markers,
            sample_ids,
            marker_ids,
            missing_mask,
            chrom,
            pos,
            ref,
            alt,
            freqs
        )
    end
end

# ============================================================================
# 编码/解码
# ============================================================================

"""
    encode_genotypes(data::AbstractMatrix) -> (Vector{UInt8}, BitMatrix)

将基因型矩阵编码为 2-bit 格式
"""
function encode_genotypes(data::AbstractMatrix{T}) where T
    n_samples, n_markers = size(data)

    # 每个 UInt8 存储 4 个基因型（2 bits each）
    n_bytes = ceil(Int, n_samples * n_markers / 4)
    encoded = zeros(UInt8, n_bytes)
    missing_mask = falses(n_samples, n_markers)

    idx = 1
    byte_idx = 1
    shift = 0

    for j in 1:n_markers
        for i in 1:n_samples
            val = data[i, j]

            # 编码
            code = if ismissing(val)
                missing_mask[i, j] = true
                0b11
            elseif val == 0
                0b00
            elseif val == 1
                0b01
            elseif val == 2
                0b10
            else
                error("无效的基因型值: $val")
            end

            # 写入
            encoded[byte_idx] |= (code << shift)

            # 更新位置
            shift += 2
            if shift == 8
                shift = 0
                byte_idx += 1
            end
        end
    end

    return encoded, missing_mask
end

"""
    decode_genotypes(cg::CompactGenotypes) -> Matrix{Int}

解码基因型数据
"""
function decode_genotypes(cg::CompactGenotypes)
    data = Matrix{Union{Int, Missing}}(undef, cg.n_samples, cg.n_markers)

    byte_idx = 1
    shift = 0

    for j in 1:cg.n_markers
        for i in 1:cg.n_samples
            # 读取 2 bits
            code = (cg.data[byte_idx] >> shift) & 0b11

            # 解码
            if cg.missing_mask[i, j]
                data[i, j] = missing
            else
                data[i, j] = Int(code)
            end

            # 更新位置
            shift += 2
            if shift == 8
                shift = 0
                byte_idx += 1
            end
        end
    end

    return data
end

# ============================================================================
# 等位基因频率计算（Kahan 求和）
# ============================================================================

"""
    compute_allele_frequencies_kahan(data, missing_mask) -> Vector{Float64}

使用 Kahan 求和算法计算等位基因频率，提高数值稳定性
"""
function compute_allele_frequencies_kahan(data::AbstractMatrix, missing_mask::BitMatrix)
    n_samples, n_markers = size(data)
    freqs = zeros(Float64, n_markers)

    for j in 1:n_markers
        sum_val = 0.0
        c = 0.0  # Kahan 补偿
        n_valid = 0

        for i in 1:n_samples
            if !missing_mask[i, j]
                val = Float64(data[i, j])

                # Kahan 求和
                y = val - c
                t = sum_val + y
                c = (t - sum_val) - y
                sum_val = t

                n_valid += 1
            end
        end

        freqs[j] = n_valid > 0 ? sum_val / (2 * n_valid) : 0.0
    end

    return freqs
end

# ============================================================================
# 基本接口实现
# ============================================================================

Core.n_samples(cg::CompactGenotypes) = cg.n_samples
Core.n_markers(cg::CompactGenotypes) = cg.n_markers
Core.sample_ids(cg::CompactGenotypes) = cg.sample_ids
Core.marker_ids(cg::CompactGenotypes) = cg.marker_ids
Core.allele_frequencies(cg::CompactGenotypes) = cg.allele_freqs

Base.size(cg::CompactGenotypes) = (cg.n_samples, cg.n_markers)

function Base.getindex(cg::CompactGenotypes, i::Int, j::Int)
    @boundscheck checkbounds(cg, i, j)

    # 计算位置
    linear_idx = (j - 1) * cg.n_samples + i
    byte_idx = div(linear_idx - 1, 4) + 1
    shift = 2 * mod(linear_idx - 1, 4)

    # 读取
    code = (cg.data[byte_idx] >> shift) & 0b11

    return cg.missing_mask[i, j] ? missing : Int(code)
end

function Core.missing_rate(cg::CompactGenotypes; dim::Int=0)
    if dim == 0
        # 总体缺失率
        return sum(cg.missing_mask) / length(cg.missing_mask)
    elseif dim == 1
        # 每个样本
        return vec(sum(cg.missing_mask, dims=2)) ./ cg.n_markers
    elseif dim == 2
        # 每个标记
        return vec(sum(cg.missing_mask, dims=1)) ./ cg.n_samples
    else
        error("dim 必须是 0, 1, 或 2")
    end
end

# ============================================================================
# 数据验证
# ============================================================================

function Core.validate(cg::CompactGenotypes)
    result = ValidationResult()

    # 检查维度
    if cg.n_samples <= 0
        push!(result.errors, "样本数必须 > 0")
        result.valid = false
    end

    if cg.n_markers <= 0
        push!(result.errors, "标记数必须 > 0")
        result.valid = false
    end

    # 检查 ID 唯一性
    if length(unique(cg.sample_ids)) != cg.n_samples
        push!(result.errors, "样本 ID 不唯一")
        result.valid = false
    end

    if length(unique(cg.marker_ids)) != cg.n_markers
        push!(result.errors, "标记 ID 不唯一")
        result.valid = false
    end

    # 检查缺失率
    overall_missing = missing_rate(cg; dim=0)
    if overall_missing > 0.5
        push!(result.warnings, "总体缺失率较高: $(round(overall_missing*100, digits=2))%")
    end

    result.metadata[:overall_missing_rate] = overall_missing
    result.metadata[:n_samples] = cg.n_samples
    result.metadata[:n_markers] = cg.n_markers

    return result
end

# ============================================================================
# 工具函数
# ============================================================================

"""
    to_matrix(cg::CompactGenotypes; impute=false) -> Matrix{Float64}

转换为普通矩阵，可选插补缺失值为平均值
"""
function to_matrix(cg::CompactGenotypes; impute::Bool=false)
    data = decode_genotypes(cg)

    if impute
        for j in 1:cg.n_markers
            # 用等位基因频率插补
            impute_val = 2 * cg.allele_freqs[j]
            for i in 1:cg.n_samples
                if cg.missing_mask[i, j]
                    data[i, j] = impute_val
                end
            end
        end
    end

    return Float64.(data)
end

"""
    memory_usage(cg::CompactGenotypes) -> NamedTuple

返回内存使用情况
"""
function memory_usage(cg::CompactGenotypes)
    data_bytes = sizeof(cg.data)
    mask_bytes = sizeof(cg.missing_mask)
    metadata_bytes = sum([
        sizeof(cg.sample_ids),
        sizeof(cg.marker_ids),
        sizeof(cg.chromosome),
        sizeof(cg.position),
        sizeof(cg.ref_allele),
        sizeof(cg.alt_allele),
        sizeof(cg.allele_freqs)
    ])

    total_bytes = data_bytes + mask_bytes + metadata_bytes

    # 原始存储（Float64）
    naive_bytes = cg.n_samples * cg.n_markers * 8

    return (
        data = data_bytes,
        missing_mask = mask_bytes,
        metadata = metadata_bytes,
        total = total_bytes,
        savings = 1 - total_bytes / naive_bytes
    )
end

export CompactGenotypes
export encode_genotypes, decode_genotypes, to_matrix, memory_usage

end  # module Genotypes
```

---

## 3. GRM 计算优化实现

### src/LinearAlgebra/grm.jl

```julia
module GRM

using LinearAlgebra, Statistics
using ..Core, ..Data.Genotypes
using ProgressMeter

# ============================================================================
# GRM 类型
# ============================================================================

struct GenomicRelationshipMatrix{T<:AbstractFloat}
    matrix::Matrix{T}
    method::Symbol
    scaled::Bool
    n_markers_used::Int

    function GenomicRelationshipMatrix(
        matrix::Matrix{T},
        method::Symbol,
        scaled::Bool,
        n_markers_used::Int
    ) where T
        @assert issymmetric(matrix) "GRM 必须是对称矩阵"
        @assert method in [:vanraden, :astle_balding, :robust] "未知的方法: $method"

        new{T}(matrix, method, scaled, n_markers_used)
    end
end

Base.size(grm::GenomicRelationshipMatrix) = size(grm.matrix)
Base.getindex(grm::GenomicRelationshipMatrix, i...) = getindex(grm.matrix, i...)

# ============================================================================
# 主计算函数
# ============================================================================

"""
    compute_grm(geno; kwargs...) -> GenomicRelationshipMatrix

计算基因组关系矩阵

# 参数
- `geno::AbstractGenotypeData`: 基因型数据
- `method::Symbol=:vanraden`: 计算方法 (:vanraden, :astle_balding, :robust)
- `min_maf::Float64=0.01`: 最小等位基因频率
- `max_missing::Float64=0.1`: 每个标记最大缺失率
- `center::Bool=true`: 是否中心化
- `scale::Bool=true`: 是否缩放
- `chunk_size::Int=1000`: 分块大小
- `progress::Bool=true`: 显示进度条

# 算法
VanRaden 方法:
    G = ZZ' / (2 Σ pᵢ(1-pᵢ))
其中 Z = M - 2P, M 是基因型矩阵, P 是等位基因频率
"""
function compute_grm(
    geno::AbstractGenotypeData;
    method::Symbol = :vanraden,
    min_maf::Float64 = 0.01,
    max_missing::Float64 = 0.1,
    center::Bool = true,
    scale::Bool = true,
    chunk_size::Int = 1000,
    progress::Bool = true
)
    # 验证参数
    @assert 0 < min_maf < 0.5 "min_maf 必须在 (0, 0.5) 范围内"
    @assert 0 <= max_missing <= 1 "max_missing 必须在 [0, 1] 范围内"
    @assert chunk_size > 0 "chunk_size 必须 > 0"

    n = n_samples(geno)
    m = n_markers(geno)

    # 过滤标记
    freqs = allele_frequencies(geno)
    marker_missing = missing_rate(geno; dim=2)

    valid_markers = (freqs .>= min_maf) .&
                    (freqs .<= (1 - min_maf)) .&
                    (marker_missing .<= max_missing)

    n_valid = sum(valid_markers)

    if n_valid == 0
        error("没有通过过滤的标记")
    end

    @info "使用 $n_valid / $m 个标记计算 GRM"

    # 初始化
    G = zeros(Float64, n, n)
    scaling_factor = 0.0

    # 分块计算
    prog = Progress(n_valid; enabled=progress, desc="计算 GRM: ")

    marker_idx = findall(valid_markers)
    n_chunks = ceil(Int, n_valid / chunk_size)

    for chunk_id in 1:n_chunks
        chunk_start = (chunk_id - 1) * chunk_size + 1
        chunk_end = min(chunk_id * chunk_size, n_valid)
        chunk_markers = marker_idx[chunk_start:chunk_end]

        # 提取并标准化
        Z = extract_and_standardize(
            geno,
            chunk_markers,
            freqs[chunk_markers];
            center = center,
            scale = scale
        )

        # 累加 G = G + ZZ'
        BLAS.syrk!('U', 'N', 1.0, Z, 1.0, G)

        # 更新缩放因子
        if method == :vanraden
            for j in chunk_markers
                p = freqs[j]
                scaling_factor += 2 * p * (1 - p)
            end
        end

        next!(prog, step=length(chunk_markers))
    end

    # 对称化（只计算了上三角）
    copytri!(G, 'U')

    # 缩放
    if method == :vanraden && scale
        G ./= scaling_factor
    elseif method == :astle_balding
        # G_AB = (M - E[M])(M - E[M])' / tr((M - E[M])(M - E[M])')
        G ./= tr(G)
    end

    return GenomicRelationshipMatrix(G, method, scale, n_valid)
end

# ============================================================================
# 辅助函数
# ============================================================================

"""
    extract_and_standardize(geno, markers, freqs; kwargs...)

提取并标准化基因型数据
"""
function extract_and_standardize(
    geno::CompactGenotypes,
    markers::AbstractVector{Int},
    freqs::AbstractVector{Float64};
    center::Bool = true,
    scale::Bool = true
)
    n = n_samples(geno)
    m = length(markers)

    Z = Matrix{Float64}(undef, n, m)

    for (j_new, j_old) in enumerate(markers)
        p = freqs[j_new]

        for i in 1:n
            val = geno[i, j_old]

            # 缺失值插补为平均值
            geno_val = ismissing(val) ? 2*p : Float64(val)

            # 中心化: M - 2p
            if center
                geno_val -= 2*p
            end

            # 标准化: (M - 2p) / sqrt(2p(1-p))
            if scale
                std_dev = sqrt(2 * p * (1 - p))
                if std_dev > 1e-10
                    geno_val /= std_dev
                end
            end

            Z[i, j_new] = geno_val
        end
    end

    return Z
end

"""
    validate_grm(G::GenomicRelationshipMatrix) -> ValidationResult

验证 GRM 的数值属性
"""
function validate_grm(G::GenomicRelationshipMatrix)
    result = ValidationResult()

    # 对称性
    if !issymmetric(G.matrix)
        push!(result.errors, "矩阵不对称")
        result.valid = false
    end

    # 对角线检查
    diag_values = diag(G.matrix)
    if any(diag_values .< 0)
        push!(result.warnings, "存在负的对角元素")
    end

    # 正定性检查（计算最小特征值）
    try
        min_eigenvalue = eigmin(Symmetric(G.matrix))
        if min_eigenvalue < -1e-6
            push!(result.warnings, "矩阵可能非正定 (最小特征值: $min_eigenvalue)")
        end
        result.metadata[:min_eigenvalue] = min_eigenvalue
    catch e
        push!(result.errors, "特征值计算失败: $e")
        result.valid = false
    end

    # 条件数检查
    try
        cond_num = cond(G.matrix)
        result.metadata[:condition_number] = cond_num
        if cond_num > 1e10
            push!(result.warnings, "条件数很大 ($cond_num)，可能数值不稳定")
        end
    catch
    end

    return result
end

export GenomicRelationshipMatrix, compute_grm, validate_grm

end  # module GRM
```

---

## 4. GBLUP 求解器

### src/Models/gblup.jl

```julia
module GBLUP

using LinearAlgebra, Statistics
using ..Core, ..Data, ..GRM
using ..LinearAlgebra.Solvers

# ============================================================================
# GBLUP 模型
# ============================================================================

"""
GBLUP 模型

# 模型方程
y = Xβ + Zu + e

其中:
- y: 表型向量
- X: 固定效应设计矩阵
- β: 固定效应
- Z: 随机效应设计矩阵
- u ~ N(0, Gσ²_g): 基因组育种值
- e ~ N(0, Iσ²_e): 残差

# 混合模型方程（MME）
[X'X    X'Z  ] [β̂]   [X'y]
[Z'X  Z'Z+G⁻¹λ] [û] = [Z'y]

其中 λ = σ²_e / σ²_g
"""
mutable struct GBLUPModel{T<:AbstractFloat} <: AbstractGenomicModel
    # 模型参数
    fixed_effects::Union{Vector{T}, Nothing}
    breeding_values::Union{Vector{T}, Nothing}

    # 方差组分
    σ²_g::Union{T, Nothing}  # 遗传方差
    σ²_e::Union{T, Nothing}  # 残差方差

    # 数据
    G::Union{GenomicRelationshipMatrix, Nothing}

    # 配置
    config::GBLUPConfig{T}

    # 训练信息
    training_info::Union{TrainingInfo, Nothing}

    function GBLUPModel(config::GBLUPConfig{T} = GBLUPConfig{Float64}()) where T
        new{T}(nothing, nothing, nothing, nothing, nothing, config, nothing)
    end
end

struct GBLUPConfig{T<:AbstractFloat}
    # 求解器
    solver::Symbol  # :cholesky, :pcg
    max_iterations::Int
    tolerance::T

    # 方差组分估计
    variance_method::Symbol  # :reml, :ml, :provided
    provided_heritability::Union{T, Nothing}

    # 其他
    compute_se::Bool
    verbose::Bool

    function GBLUPConfig{T}(;
        solver::Symbol = :pcg,
        max_iterations::Int = 1000,
        tolerance::T = 1e-6,
        variance_method::Symbol = :reml,
        provided_heritability::Union{T, Nothing} = nothing,
        compute_se::Bool = false,
        verbose::Bool = true
    ) where T
        new{T}(
            solver,
            max_iterations,
            tolerance,
            variance_method,
            provided_heritability,
            compute_se,
            verbose
        )
    end
end

struct TrainingInfo
    converged::Bool
    iterations::Int
    final_residual::Float64
    solve_time::Float64
    variance_estimation_time::Float64
end

# ============================================================================
# 训练
# ============================================================================

"""
    fit!(model::GBLUPModel, geno, pheno; kwargs...)

训练 GBLUP 模型
"""
function fit!(
    model::GBLUPModel{T},
    geno::AbstractGenotypeData,
    pheno::AbstractPhenotypeData;
    G::Union{GenomicRelationshipMatrix, Nothing} = nothing,
    σ²_g::Union{T, Nothing} = nothing,
    σ²_e::Union{T, Nothing} = nothing
) where T
    cfg = model.config

    # 1. 验证数据
    validate_data_consistency(geno, pheno)

    # 2. 计算或使用提供的 G 矩阵
    if isnothing(G)
        cfg.verbose && @info "计算 GRM..."
        t_grm = @elapsed begin
            G = compute_grm(geno; progress=cfg.verbose)
        end
        cfg.verbose && @info "GRM 计算完成 ($t_grm 秒)"
    end
    model.G = G

    # 3. 提取表型
    y = get_phenotype_vector(pheno)
    n = length(y)

    # 4. 估计方差组分（如果未提供）
    t_var = 0.0
    if isnothing(σ²_g) || isnothing(σ²_e)
        cfg.verbose && @info "估计方差组分..."
        t_var = @elapsed begin
            σ²_g, σ²_e = estimate_variance_components(
                y,
                G.matrix,
                cfg.variance_method
            )
        end
        cfg.verbose && @info "方差组分: σ²_g = $σ²_g, σ²_e = $σ²_e"
    end

    model.σ²_g = σ²_g
    model.σ²_e = σ²_e

    # 5. 求解 MME
    λ = σ²_e / σ²_g

    cfg.verbose && @info "求解混合模型方程 (λ = $λ)..."
    t_solve = @elapsed begin
        if cfg.solver == :cholesky
            u = solve_mme_direct(y, G.matrix, λ)
            converged = true
            iterations = 0
            residual = 0.0
        elseif cfg.solver == :pcg
            u, converged, iterations, residual = solve_mme_pcg(
                y,
                G.matrix,
                λ;
                max_iter = cfg.max_iterations,
                tol = cfg.tolerance,
                verbose = cfg.verbose
            )
        else
            error("未知的求解器: $(cfg.solver)")
        end
    end

    model.breeding_values = u

    # 6. 保存训练信息
    model.training_info = TrainingInfo(
        converged,
        iterations,
        residual,
        t_solve,
        t_var
    )

    cfg.verbose && @info "训练完成 ($(t_solve) 秒)"

    return model
end

# ============================================================================
# 求解器实现
# ============================================================================

"""
    solve_mme_direct(y, G, λ) -> Vector

使用 Cholesky 分解直接求解

MME: (G + Iλ)u = y
"""
function solve_mme_direct(
    y::AbstractVector{T},
    G::AbstractMatrix{T},
    λ::T
) where T
    n = length(y)

    # 构建系数矩阵 C = G + λI
    C = copy(G)
    for i in 1:n
        C[i, i] += λ
    end

    # Cholesky 分解
    try
        chol = cholesky!(Symmetric(C))
        u = chol \ y
        return u
    catch e
        @warn "Cholesky 分解失败，使用 LU 分解"
        u = C \ y
        return u
    end
end

"""
    solve_mme_pcg(y, G, λ; kwargs...) -> (Vector, Bool, Int, Float64)

使用预条件共轭梯度法求解

返回: (解, 是否收敛, 迭代次数, 最终残差)
"""
function solve_mme_pcg(
    y::AbstractVector{T},
    G::AbstractMatrix{T},
    λ::T;
    max_iter::Int = 1000,
    tol::T = 1e-6,
    verbose::Bool = true
) where T
    n = length(y)

    # 初始化
    u = zeros(T, n)
    r = copy(y)  # 残差 r = y - (G + λI)u = y (因为 u=0)

    # 预条件器: 对角元素的倒数
    precond = ones(T, n)
    for i in 1:n
        precond[i] = 1 / (G[i, i] + λ)
    end

    z = r .* precond  # 预条件残差
    p = copy(z)       # 搜索方向

    rz_old = dot(r, z)

    # 迭代
    for iter in 1:max_iter
        # Ap = (G + λI)p
        Ap = G * p + λ * p

        # 步长
        pAp = dot(p, Ap)
        α = rz_old / pAp

        # 更新解
        u .+= α * p

        # 更新残差
        r .-= α * Ap

        # 检查收敛
        residual_norm = norm(r)
        if residual_norm < tol
            verbose && @info "PCG 收敛 (iter=$iter, residual=$residual_norm)"
            return u, true, iter, residual_norm
        end

        # 预条件
        z .= r .* precond

        # 更新搜索方向
        rz_new = dot(r, z)
        β = rz_new / rz_old
        p .= z + β * p

        rz_old = rz_new

        if verbose && iter % 100 == 0
            @info "PCG iter $iter, residual = $residual_norm"
        end
    end

    @warn "PCG 未收敛 (max_iter=$max_iter)"
    return u, false, max_iter, norm(r)
end

# ============================================================================
# 方差组分估计
# ============================================================================

"""
    estimate_variance_components(y, G, method) -> (σ²_g, σ²_e)

估计遗传方差和残差方差

# 方法
- :reml - REML (restricted maximum likelihood)
- :ml - ML (maximum likelihood)
- :simple - 简单估计 (快速但不准确)
"""
function estimate_variance_components(
    y::AbstractVector{T},
    G::AbstractMatrix{T},
    method::Symbol
) where T
    n = length(y)

    if method == :simple
        # 简单估计: 假设 h² = 0.5
        var_y = var(y)
        σ²_g = 0.5 * var_y
        σ²_e = 0.5 * var_y
        return σ²_g, σ²_e

    elseif method == :reml
        # EM-REML
        return em_reml(y, G)

    else
        error("未知的方差组分估计方法: $method")
    end
end

"""
    em_reml(y, G; max_iter=100, tol=1e-6)

EM-REML 算法估计方差组分
"""
function em_reml(
    y::AbstractVector{T},
    G::AbstractMatrix{T};
    max_iter::Int = 100,
    tol::T = 1e-6
) where T
    n = length(y)

    # 初始值
    σ²_g = var(y) * 0.5
    σ²_e = var(y) * 0.5

    for iter in 1:max_iter
        # E 步: 计算 V = Gσ²_g + Iσ²_e
        V = G * σ²_g
        for i in 1:n
            V[i, i] += σ²_e
        end

        # 求逆
        V_inv = inv(V)

        # M 步: 更新方差
        u = G * V_inv * y
        σ²_g_new = (dot(u, u) + tr(G - G * V_inv * G * σ²_g)) / n

        e = y - u
        σ²_e_new = (dot(e, e) + tr(I - V_inv * σ²_e)) / n

        # 检查收敛
        diff = abs(σ²_g_new - σ²_g) + abs(σ²_e_new - σ²_e)
        if diff < tol
            return σ²_g_new, σ²_e_new
        end

        σ²_g = σ²_g_new
        σ²_e = σ²_e_new
    end

    @warn "EM-REML 未收敛"
    return σ²_g, σ²_e
end

# ============================================================================
# 预测
# ============================================================================

"""
    predict(model::GBLUPModel, geno)

预测基因组育种值
"""
function predict(model::GBLUPModel, geno::AbstractGenotypeData)
    if isnothing(model.breeding_values)
        error("模型尚未训练")
    end

    # 对于训练集样本，直接返回育种值
    return model.breeding_values
end

# ============================================================================
# 辅助函数
# ============================================================================

function validate_data_consistency(geno, pheno)
    geno_ids = Set(sample_ids(geno))
    pheno_ids = Set(sample_ids(pheno))

    if geno_ids != pheno_ids
        error("基因型和表型的样本 ID 不一致")
    end
end

function get_phenotype_vector(pheno::PhenotypeData)
    # 简化版本：假设只有一个性状
    return pheno.data[:, 2]  # 第一列是 ID
end

export GBLUPModel, GBLUPConfig, TrainingInfo
export fit!, predict

end  # module GBLUP
```

---

## 5. BayesR 实现

### src/Models/bayesian/bayesr.jl

```julia
module BayesR

using Distributions, Random, Statistics, LinearAlgebra
using ProgressMeter
using ..Core

# ============================================================================
# BayesR 模型
# ============================================================================

"""
BayesR: 4-组分混合模型

标记效应的先验:
βⱼ ~ π₀δ₀ + π₁N(0, σ₁²) + π₂N(0, σ₂²) + π₃N(0, σ₃²)

其中:
- π₀: 零效应概率
- π₁, π₂, π₃: 非零效应的混合比例
- σ₁² = 0.0001σ²_g, σ₂² = 0.001σ²_g, σ₃² = 0.01σ²_g
"""
mutable struct BayesRModel{T<:AbstractFloat} <: AbstractGenomicModel
    # 标记效应
    marker_effects::Union{Vector{T}, Nothing}

    # 混合比例
    π::Union{Vector{T}, Nothing}  # [π₀, π₁, π₂, π₃]

    # 方差组分
    σ²::Union{Vector{T}, Nothing}  # [0, σ₁², σ₂², σ₃²]
    σ²_e::Union{T, Nothing}

    # MCMC 样本
    samples::Union{MCMCSamples{T}, Nothing}

    # 配置
    config::BayesRConfig{T}

    function BayesRModel(config::BayesRConfig{T} = BayesRConfig{Float64}()) where T
        new{T}(nothing, nothing, nothing, nothing, nothing, config)
    end
end

struct BayesRConfig{T<:AbstractFloat}
    # MCMC 参数
    num_iterations::Int
    burn_in::Int
    thin::Int

    # 方差组分
    variance_ratios::Vector{T}  # [0.0, 0.0001, 0.001, 0.01]

    # 先验
    π_prior::Vector{T}  # Dirichlet 先验参数
    ν::T                # 自由度 (逆卡方分布)
    S::T                # 尺度参数

    # 其他
    update_variance::Bool
    seed::Union{Int, Nothing}
    verbose::Bool

    function BayesRConfig{T}(;
        num_iterations::Int = 50000,
        burn_in::Int = 10000,
        thin::Int = 10,
        variance_ratios::Vector{T} = T[0.0, 0.0001, 0.001, 0.01],
        π_prior::Vector{T} = T[1.0, 1.0, 1.0, 1.0],
        ν::T = 4.0,
        S::T = 0.0,
        update_variance::Bool = true,
        seed::Union{Int, Nothing} = nothing,
        verbose::Bool = true
    ) where T
        new{T}(
            num_iterations,
            burn_in,
            thin,
            variance_ratios,
            π_prior,
            ν,
            S,
            update_variance,
            seed,
            verbose
        )
    end
end

struct MCMCSamples{T}
    marker_effects::Matrix{T}  # samples × markers
    π::Matrix{T}              # samples × 4
    σ²::Matrix{T}             # samples × 4
    σ²_e::Vector{T}           # samples

    # 诊断
    acceptance_rate::Vector{T}
    ess::Vector{T}  # Effective sample size
end

# ============================================================================
# Gibbs 采样
# ============================================================================

"""
    fit!(model::BayesRModel, geno, pheno)

使用 Gibbs 采样训练 BayesR 模型
"""
function fit!(
    model::BayesRModel{T},
    geno::AbstractGenotypeData,
    pheno::AbstractPhenotypeData
) where T
    cfg = model.config

    # 设置随机种子
    if !isnothing(cfg.seed)
        Random.seed!(cfg.seed)
    end

    # 提取数据
    X = to_matrix(geno; impute=true)  # n × m
    y = get_phenotype_vector(pheno)   # n
    n, m = size(X)

    cfg.verbose && @info "BayesR 训练开始"
    cfg.verbose && @info "样本: $n, 标记: $m"
    cfg.verbose && @info "MCMC: $(cfg.num_iterations) 迭代, burn-in: $(cfg.burn_in), thin: $(cfg.thin)"

    # 初始化
    β = zeros(T, m)                    # 标记效应
    π = T[0.5, 0.167, 0.167, 0.166]    # 混合比例
    δ = ones(Int, m)                   # 组分指示器 (1,2,3,4)
    μ = mean(y)                        # 总平均
    σ²_g = var(y) * 0.5                # 遗传方差
    σ²_e = var(y) * 0.5                # 残差方差
    σ² = σ²_g * cfg.variance_ratios    # 各组分方差

    # 预计算 X'X 对角线
    x_sqsum = vec(sum(X.^2, dims=1))

    # 残差
    ŷ = X * β .+ μ
    e = y - ŷ

    # 存储样本
    n_save = div(cfg.num_iterations - cfg.burn_in, cfg.thin)
    β_samples = zeros(T, n_save, m)
    π_samples = zeros(T, n_save, 4)
    σ²_samples = zeros(T, n_save, 4)
    σ²_e_samples = zeros(T, n_save)

    save_idx = 1

    # Gibbs 采样
    prog = Progress(cfg.num_iterations; enabled=cfg.verbose)

    for iter in 1:cfg.num_iterations
        # 1. 采样标记效应和组分
        for j in 1:m
            # 移除当前标记的贡献
            e .+= X[:, j] * β[j]

            # 计算后验
            rhs = dot(X[:, j], e)

            # 对每个组分计算概率
            log_probs = zeros(T, 4)

            for k in 1:4
                if k == 1
                    # 零效应
                    log_probs[k] = log(π[k])
                else
                    # 非零效应
                    v = 1 / (x_sqsum[j] / σ²_e + 1 / σ²[k])
                    m_post = v * rhs / σ²_e

                    log_probs[k] = log(π[k]) - 0.5 * log(σ²[k]) +
                                   0.5 * m_post^2 / v
                end
            end

            # 归一化
            max_log = maximum(log_probs)
            probs = exp.(log_probs .- max_log)
            probs ./= sum(probs)

            # 采样组分
            δ[j] = rand(Categorical(probs))

            # 采样效应
            if δ[j] == 1
                β[j] = 0
            else
                v = 1 / (x_sqsum[j] / σ²_e + 1 / σ²[δ[j]])
                m_post = v * rhs / σ²_e
                β[j] = rand(Normal(m_post, sqrt(v)))
            end

            # 更新残差
            e .-= X[:, j] * β[j]
        end

        # 2. 采样混合比例
        counts = [sum(δ .== k) for k in 1:4]
        π = rand(Dirichlet(cfg.π_prior + counts))

        # 3. 采样方差组分
        if cfg.update_variance
            for k in 2:4
                # 只更新非零组分
                idx = δ .== k
                n_k = sum(idx)

                if n_k > 0
                    SS = sum(β[idx].^2)
                    ν_post = cfg.ν + n_k
                    S_post = cfg.S + SS

                    σ²[k] = rand(InverseGamma(ν_post/2, S_post/2))
                end
            end
        end

        # 4. 采样残差方差
        SS_e = sum(e.^2)
        ν_e = cfg.ν + n
        S_e = cfg.S + SS_e
        σ²_e = rand(InverseGamma(ν_e/2, S_e/2))

        # 5. 采样总平均
        μ = rand(Normal(mean(y - X * β), sqrt(σ²_e / n)))

        # 保存样本
        if iter > cfg.burn_in && (iter - cfg.burn_in) % cfg.thin == 0
            β_samples[save_idx, :] = β
            π_samples[save_idx, :] = π
            σ²_samples[save_idx, :] = σ²
            σ²_e_samples[save_idx] = σ²_e
            save_idx += 1
        end

        next!(prog)
    end

    # 后处理
    model.marker_effects = vec(mean(β_samples, dims=1))
    model.π = vec(mean(π_samples, dims=1))
    model.σ² = vec(mean(σ²_samples, dims=1))
    model.σ²_e = mean(σ²_e_samples)

    model.samples = MCMCSamples(
        β_samples,
        π_samples,
        σ²_samples,
        σ²_e_samples,
        T[],  # acceptance_rate (placeholder)
        T[]   # ess (placeholder)
    )

    cfg.verbose && @info "BayesR 训练完成"
    cfg.verbose && @info "混合比例: $(model.π)"

    return model
end

# ============================================================================
# 预测
# ============================================================================

function predict(model::BayesRModel, geno::AbstractGenotypeData)
    if isnothing(model.marker_effects)
        error("模型尚未训练")
    end

    X = to_matrix(geno; impute=true)
    gebv = X * model.marker_effects

    return gebv
end

# ============================================================================
# 后验分析
# ============================================================================

"""
    posterior_inclusion_probability(model::BayesRModel) -> Vector

计算每个标记的后验包含概率（PIP）
"""
function posterior_inclusion_probability(model::BayesRModel)
    if isnothing(model.samples)
        error("没有 MCMC 样本")
    end

    # PIP = 1 - P(βⱼ = 0)
    pip = vec(mean(model.samples.marker_effects .!= 0, dims=1))

    return pip
end

export BayesRModel, BayesRConfig, MCMCSamples
export fit!, predict, posterior_inclusion_probability

end  # module BayesR
```

---

## 6. Pipeline 系统

### src/Workflows/pipeline.jl

```julia
module Pipelines

using ..Core, ..Data, ..Models
using Dates

# ============================================================================
# Pipeline 定义
# ============================================================================

struct PipelineStage
    name::String
    func::Function
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    dependencies::Vector{String}
end

struct Pipeline
    name::String
    stages::Vector{PipelineStage}
    state::Dict{Symbol, Any}
    completed_stages::Set{String}

    function Pipeline(name::String, stages::Vector{PipelineStage})
        new(name, stages, Dict{Symbol, Any}(), Set{String}())
    end
end

# ============================================================================
# Pipeline 执行
# ============================================================================

"""
    run!(pipeline::Pipeline; resume=false)

执行 pipeline
"""
function run!(pipeline::Pipeline; resume::Bool=false)
    @info "开始执行 Pipeline: $(pipeline.name)"

    if !resume
        empty!(pipeline.completed_stages)
    end

    for stage in pipeline.stages
        # 检查依赖
        if !all(dep -> dep in pipeline.completed_stages, stage.dependencies)
            error("Stage $(stage.name) 的依赖未满足")
        end

        # 跳过已完成的 stage
        if resume && stage.name in pipeline.completed_stages
            @info "跳过已完成的 stage: $(stage.name)"
            continue
        end

        @info "执行 stage: $(stage.name)"
        t_start = now()

        # 准备输入
        inputs = [pipeline.state[sym] for sym in stage.inputs]

        # 执行
        try
            outputs = stage.func(inputs...)

            # 保存输出
            for (sym, val) in zip(stage.outputs, outputs)
                pipeline.state[sym] = val
            end

            push!(pipeline.completed_stages, stage.name)

            t_end = now()
            @info "Stage $(stage.name) 完成 ($(t_end - t_start))"

        catch e
            @error "Stage $(stage.name) 失败" exception=(e, catch_backtrace())
            rethrow(e)
        end
    end

    @info "Pipeline 完成!"
    return pipeline
end

# ============================================================================
# 预定义 Pipeline
# ============================================================================

"""
    create_genomic_prediction_pipeline(config) -> Pipeline

创建标准的基因组预测 pipeline
"""
function create_genomic_prediction_pipeline(config::Dict{Symbol, Any})
    stages = [
        # Stage 1: 加载数据
        PipelineStage(
            "load_data",
            (geno_file, pheno_file) -> (
                read_genotypes(geno_file),
                read_phenotypes(pheno_file)
            ),
            [:genotype_file, :phenotype_file],
            [:genotypes, :phenotypes],
            String[]
        ),

        # Stage 2: QC
        PipelineStage(
            "quality_control",
            (geno, pheno, qc_params) -> quality_control(geno, pheno, qc_params),
            [:genotypes, :phenotypes, :qc_params],
            [:genotypes_qc, :phenotypes_qc, :qc_report],
            ["load_data"]
        ),

        # Stage 3: 计算 GRM
        PipelineStage(
            "compute_grm",
            (geno,) -> (compute_grm(geno),),
            [:genotypes_qc],
            [:G],
            ["quality_control"]
        ),

        # Stage 4: 训练模型
        PipelineStage(
            "train_model",
            (geno, pheno, G, model_type, model_params) ->
                train_model(geno, pheno, G, model_type, model_params),
            [:genotypes_qc, :phenotypes_qc, :G, :model_type, :model_params],
            [:model, :predictions],
            ["compute_grm"]
        ),

        # Stage 5: 交叉验证
        PipelineStage(
            "cross_validate",
            (geno, pheno, model_type, model_params, cv_folds) ->
                cross_validate(geno, pheno, model_type, model_params, cv_folds),
            [:genotypes_qc, :phenotypes_qc, :model_type, :model_params, :cv_folds],
            [:cv_results],
            ["quality_control"]
        ),

        # Stage 6: 生成报告
        PipelineStage(
            "generate_report",
            (model, cv_results, qc_report, output_dir) ->
                generate_report(model, cv_results, qc_report, output_dir),
            [:model, :cv_results, :qc_report, :output_dir],
            [:report_path],
            ["train_model", "cross_validate"]
        )
    ]

    pipeline = Pipeline("genomic_prediction", stages)

    # 设置初始状态
    merge!(pipeline.state, config)

    return pipeline
end

export Pipeline, PipelineStage, run!, create_genomic_prediction_pipeline

end  # module Pipelines
```

---

## 7. 使用示例

### 基础使用

```julia
using GenomicPro2

# 1. 加载数据
geno = read_genotypes("data.vcf.gz")
pheno = read_phenotypes("phenotypes.csv")

# 2. 数据验证
validate(geno)
validate(pheno)

# 3. QC
geno_qc = apply_quality_control(
    geno;
    max_missing_marker = 0.1,
    max_missing_sample = 0.1,
    min_maf = 0.01
)

# 4. 计算 GRM
G = compute_grm(geno_qc; method=:vanraden, progress=true)

# 5. 训练 GBLUP
model = GBLUPModel(GBLUPConfig(
    solver = :pcg,
    variance_method = :reml,
    verbose = true
))

fit!(model, geno_qc, pheno; G=G)

# 6. 预测
gebv = predict(model, geno_qc)

# 7. 评估
accuracy = cross_validate(model, geno_qc, pheno; folds=5)
println("预测准确性: $accuracy")
```

### 使用 Pipeline

```julia
using GenomicPro2.Pipelines

# 配置
config = Dict{Symbol, Any}(
    :genotype_file => "data.vcf.gz",
    :phenotype_file => "phenotypes.csv",
    :output_dir => "results/",

    :qc_params => QCParams(
        max_missing_marker = 0.1,
        min_maf = 0.01
    ),

    :model_type => :bayesr,
    :model_params => Dict(
        :num_iterations => 50000,
        :burn_in => 10000
    ),

    :cv_folds => 5
)

# 创建并执行
pipeline = create_genomic_prediction_pipeline(config)
run!(pipeline)

# 查看结果
model = pipeline.state[:model]
cv_results = pipeline.state[:cv_results]
```

### BayesR 使用

```julia
using GenomicPro2.Models.BayesR

# 配置
config = BayesRConfig(
    num_iterations = 50000,
    burn_in = 10000,
    thin = 10,
    update_variance = true,
    verbose = true
)

# 训练
model = BayesRModel(config)
fit!(model, geno_qc, pheno)

# 查看结果
println("混合比例: $(model.π)")
println("方差组分: $(model.σ²)")

# PIP
pip = posterior_inclusion_probability(model)
top_markers = sortperm(pip, rev=true)[1:100]

println("Top 100 markers:")
for i in top_markers[1:10]
    println("  $(marker_ids(geno)[i]): PIP = $(pip[i])")
end
```

---

**文档版本**: 1.0
**最后更新**: 2025-11-15

这些代码示例展示了 GenomicPro 2.0 的核心实现，包括：
- 类型安全的数据结构
- 数值稳定的算法
- 清晰的接口设计
- 完善的错误处理
- 高效的内存管理

可作为实际实现的参考模板。
