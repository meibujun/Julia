#=###############################################################################
# 数据处理模块
# 负责数据加载、质量控制、缺失值填补、关系矩阵构建与模拟数据生成。
# 全部函数均配有中文文档字符串, 方便 Documenter 自动抓取。
###############################################################################=#

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distances: pairwise, Euclidean
using Base.Threads

"""
    struct QualityReport

质量控制输出的汇总结构, 用于记录筛选阈值、标记保留情况与逐标记统计指标。
"""
Base.@kwdef struct QualityReport
    maf_threshold::Float64
    missing_rate_threshold::Float64
    variance_floor::Float64
    initial_markers::Int
    retained_markers::Int
    removed_markers::Vector{String} = String[]
    removal_reasons::Dict{Symbol,Int} = Dict{Symbol,Int}()
    per_marker_stats::DataFrame = DataFrame()
end

# 内部常量: 记录质量控制剔除原因代码, 以便在线程循环中仅写入整数避免竞争。
const _QC_REASON_LABEL = Dict(
    UInt8(0) => :kept,
    UInt8(1) => :missing_rate,
    UInt8(2) => :maf,
    UInt8(3) => :variance
)

"""
    struct GenomicDataset

封装后的基因组数据容器, 统一保存基因型矩阵、表型向量及元信息。

# 字段说明
- `genotype::Matrix{Float64}`: 标准化前的基因型矩阵 (个体 × 标记)。
- `phenotype::Vector{Float64}`: 表型响应向量, 支持缺失值用 `NaN` 表示以便后续筛除。
- `sample_ids::Vector{String}`: 个体唯一标识。
- `marker_names::Vector{String}`: 基因标记名称。
- `metadata::Dict{String,Any}`: 附加元数据, 例如数据来源、批次、模拟参数等。
"""
Base.@kwdef mutable struct GenomicDataset
    genotype::Matrix{Float64}
    phenotype::Vector{Float64}
    sample_ids::Vector{String}
    marker_names::Vector{String}
    metadata::Dict{String,Any} = Dict{String,Any}()
end

"""
    Base.getproperty(dataset::GenomicDataset, name::Symbol)

为常用字段提供友好别名, 例如 `dataset.ids` 即 `dataset.sample_ids`, `dataset.markers` 即 `dataset.marker_names`。
"""
function Base.getproperty(dataset::GenomicDataset, name::Symbol)
    name === :ids && return getfield(dataset, :sample_ids)
    name === :markers && return getfield(dataset, :marker_names)
    return getfield(dataset, name)
end

"""
    Base.setproperty!(dataset::GenomicDataset, name::Symbol, value)

保证别名赋值与实际字段保持同步, 避免出现不同步的风险。
"""
function Base.setproperty!(dataset::GenomicDataset, name::Symbol, value)
    if name === :ids
        return setfield!(dataset, :sample_ids, value)
    elseif name === :markers
        return setfield!(dataset, :marker_names, value)
    else
        return setfield!(dataset, name, value)
    end
end

"""
    load_genomic_table(path; format=:csv, kwargs...) -> DataFrame

读取原始基因型表格, 默认按 CSV 解析。对于大型矩阵, 推荐使用 Arrow/Parquet, 此处示例支持 CSV/TSV。
"""
function load_genomic_table(path::AbstractString; format::Symbol = :csv, kwargs...)
    format === :csv && return CSV.read(path, DataFrame; kwargs...)
    format === :tsv && return CSV.read(path, DataFrame; delim='\t', kwargs...)
    throw(ArgumentError("暂不支持的格式: $(format)"))
end

"""
    load_phenotype_table(path; kwargs...) -> DataFrame

读取表型文件, 默认 CSV。支持用户通过 `kwargs` 自定义缺失值符号、日期解析等参数。
"""
load_phenotype_table(path::AbstractString; kwargs...) = CSV.read(path, DataFrame; kwargs...)

"""
    merge_genomic_phenotype(geno_df, pheno_df; id_col=:id, phenotype_col=:phenotype,
                            covariate_cols=Symbol[]) -> GenomicDataset

根据样本 ID 合并基因型与表型数据表, 自动将非数值列保存在元数据中。
"""
function merge_genomic_phenotype(geno_df::DataFrame, pheno_df::DataFrame;
                                 id_col::Symbol = :id,
                                 phenotype_col::Symbol = :phenotype,
                                 covariate_cols::Vector{Symbol} = Symbol[])
    in(id_col, names(geno_df)) || throw(ArgumentError("基因型表缺少 ID 列"))
    in(id_col, names(pheno_df)) || throw(ArgumentError("表型表缺少 ID 列"))
    in(phenotype_col, names(pheno_df)) || throw(ArgumentError("表型列不存在"))

    merged = innerjoin(geno_df, pheno_df; on=id_col, makeunique=true)
    ids = String.(merged[:, id_col])

    # 将基因型列转换为 Float64, 缺失值使用 NaN 表示, 方便后续数值运算。
    marker_columns = filter(!=(id_col), names(geno_df))
    marker_names = String.(marker_columns)
    geno_matrix = Matrix{Float64}(undef, size(merged, 1), length(marker_columns))
    for (j, colname) in enumerate(marker_columns)
        column = merged[:, colname]
        if eltype(column) <: Number
            geno_matrix[:, j] .= Float64.(coalesce.(column, NaN))
        else
            parsed = tryparse.(Float64, String.(column))
            geno_matrix[:, j] .= replace(parsed, nothing=>NaN)
        end
    end

    phenotype = Float64.(coalesce.(merged[:, phenotype_col], NaN))

    metadata = Dict(
        "source" => "merge",
        "timestamp" => Dates.now(),
        "covariates" => Dict(col => merged[:, col] for col in covariate_cols if col in names(merged))
    )

    return GenomicDataset(geno_matrix, phenotype, ids, marker_names, metadata)
end

"""
    quality_control!(dataset; maf_threshold=0.01, missing_rate=0.1,
                      variance_floor=1e-8, return_report=false,
                      collect_stats=true) -> GenomicDataset 或 (GenomicDataset, QualityReport)

执行常见质量控制: 过滤缺失率过高或次等位基因频率过低的标记, 并剔除方差不足的列。
当 `return_report=true` 时返回 `(dataset, report)` 元组, 便于在流水线中复用筛选统计信息。
`collect_stats=false` 可关闭逐标记统计, 适用于超大规模数据以节省内存。
"""
function quality_control!(dataset::GenomicDataset;
                          maf_threshold::Real = 0.01,
                          missing_rate::Real = 0.1,
                          variance_floor::Real = 1e-8,
                          return_report::Bool = false,
                          collect_stats::Bool = true)
    X = dataset.genotype
    n, p = size(X)
    p == 0 && throw(ArgumentError("基因型矩阵无列, 无需执行质量控制"))

    keep = trues(p)
    missing_rates = zeros(Float64, p)
    mafs = zeros(Float64, p)
    variances = zeros(Float64, p)
    reason_codes = fill(UInt8(0), p)

    @inbounds Threads.@threads for j in 1:p
        column = view(X, :, j)
        finite_mask = .!isnan.(column)
        miss_ratio = 1.0 - (count(finite_mask) / n)
        missing_rates[j] = miss_ratio
        if miss_ratio > missing_rate
            keep[j] = false
            reason_codes[j] = UInt8(1)
            continue
        end
        valid = column[finite_mask]
        if isempty(valid)
            keep[j] = false
            reason_codes[j] = UInt8(1)
            continue
        end
        mean_allele = mean(valid) / 2
        maf = min(mean_allele, 1 - mean_allele)
        mafs[j] = maf
        variance_val = var(valid)
        variances[j] = variance_val
        if maf < maf_threshold
            keep[j] = false
            reason_codes[j] = UInt8(2)
        elseif variance_val < variance_floor
            keep[j] = false
            reason_codes[j] = UInt8(3)
        end
    end

    original_names = copy(dataset.marker_names)
    dataset.genotype = dataset.genotype[:, keep]
    dataset.marker_names = dataset.marker_names[keep]

    removed = original_names[.!keep]
    reason_symbols = [_QC_REASON_LABEL[code] for code in reason_codes]
    removal_counts = Dict{Symbol,Int}()
    for sym in reason_symbols
        removal_counts[sym] = get(removal_counts, sym, 0) + 1
    end
    stats_df = collect_stats ? DataFrame(
        marker = original_names,
        missing_rate = missing_rates,
        maf = mafs,
        variance = variances,
        keep = keep,
        reason = reason_symbols
    ) : DataFrame()

    report = QualityReport(
        maf_threshold = float(maf_threshold),
        missing_rate_threshold = float(missing_rate),
        variance_floor = float(variance_floor),
        initial_markers = p,
        retained_markers = count(keep),
        removed_markers = removed,
        removal_reasons = removal_counts,
        per_marker_stats = stats_df
    )

    dataset.metadata["quality_control"] = Dict(
        "timestamp" => Dates.now(),
        "maf_threshold" => report.maf_threshold,
        "missing_rate" => report.missing_rate_threshold,
        "variance_floor" => report.variance_floor,
        "initial_markers" => report.initial_markers,
        "retained_markers" => report.retained_markers,
        "removed_markers" => removed,
        "removal_reasons" => report.removal_reasons
    )

    return return_report ? (dataset, report) : dataset
end

"""
    standardize_genotypes!(dataset; center=true, scale=true) -> Tuple{Vector,Vector}

对基因型矩阵按列执行中心化/标准化, 返回列均值与标准差, 方便模型在预测时复原。
"""
function standardize_genotypes!(dataset::GenomicDataset; center::Bool = true, scale::Bool = true)
    X = dataset.genotype
    p = size(X, 2)
    means = zeros(Float64, p)
    stds = ones(Float64, p)

    @inbounds Threads.@threads for j in 1:p
        column = view(X, :, j)
        mask = .!isnan.(column)
        if any(mask)
            μ = center ? mean(column[mask]) : 0.0
            σ = scale ? std(column[mask]; corrected=false) : 1.0
            σ = σ == 0.0 ? 1.0 : σ
            means[j] = μ
            stds[j] = σ
            column[mask] .-= μ
            column[mask] ./= σ
        else
            means[j] = 0.0
            stds[j] = 1.0
        end
        column[.!mask] .= 0.0
    end

    dataset.metadata["scaling"] = Dict(
        "timestamp" => Dates.now(),
        "center" => center,
        "scale" => scale,
        "means" => means,
        "stds" => stds
    )

    return means, stds
end

"""
    _impute_statistic!(X, reducer)

内部辅助函数, 以列为单位使用给定统计量填补缺失值, 并采用多线程提升速度。
"""
function _impute_statistic!(X::AbstractMatrix, reducer::Function)
    p = size(X, 2)
    @inbounds Threads.@threads for j in 1:p
        column = view(X, :, j)
        mask = .!isnan.(column)
        fill_value = any(mask) ? reducer(column[mask]) : 0.0
        column[.!mask] .= fill_value
    end
    return X
end

"""
    impute_missing!(dataset; method=:mean, knn_k=5, rng=Random.default_rng(),
                    max_knn_samples=2000)

对缺失基因型执行填补。支持:
- `:mean` / `:median`: 使用列均值或中位数;
- `:knn`: 采用 KNN (欧氏距离) 按个体相似度加权平均; 当样本数超过 `max_knn_samples` 时自动回退到均值填补, 避免 O(n^2) 距离计算导致内存爆炸。
"""
function impute_missing!(dataset::GenomicDataset;
                         method::Symbol = :mean,
                         knn_k::Integer = 5,
                         rng::AbstractRNG = Random.default_rng(),
                         max_knn_samples::Integer = 2000)
    X = dataset.genotype
    n, p = size(X)
    fallback = false
    _ = rng  # 明确使用 RNG 参数, 以便调用者可传入自定义随机源参与未来扩展

    if method == :mean || method == :median
        reducer = method == :median ? median : mean
        _impute_statistic!(X, reducer)
    elseif method == :knn
        if n > max_knn_samples
            fallback = true
            _impute_statistic!(X, mean)
        else
            row_missing = [any(isnan, view(X, i, :)) for i in 1:n]
            if !any(row_missing)
                dataset.metadata["imputation"] = Dict(
                    "timestamp" => Dates.now(),
                    "method" => String(method),
                    "knn_k" => knn_k,
                    "max_knn_samples" => max_knn_samples,
                    "fallback" => fallback
                )
                return dataset
            end
            X_imputed = copy(X)
            _impute_statistic!(X_imputed, mean)
            distances = pairwise(Euclidean(), X_imputed; dims = 1)
            for i in 1:n
                row_missing[i] || continue
                miss_cols = findall(isnan, view(X, i, :))
                isempty(miss_cols) && continue
                neighbor_order = sortperm(view(distances, i, :))
                neighbor_order = filter(!=(i), neighbor_order)
                k = min(knn_k, length(neighbor_order))
                k == 0 && continue
                neighbors = neighbor_order[1:k]
                weights = distances[i, neighbors]
                if all(iszero, weights)
                    weights .= 1.0
                end
                weights = max.(weights, eps())
                inv_w = 1 ./ weights
                inv_sum = sum(inv_w)
                inv_sum == 0.0 && (inv_sum = eps())
                inv_w ./= inv_sum
                for col in miss_cols
                    neighbor_vals = X_imputed[neighbors, col]
                    X[i, col] = dot(inv_w, neighbor_vals)
                end
            end
        end
    else
        throw(ArgumentError("未知填补方法: $(method)"))
    end

    dataset.metadata["imputation"] = Dict(
        "timestamp" => Dates.now(),
        "method" => String(method),
        "knn_k" => knn_k,
        "max_knn_samples" => max_knn_samples,
        "fallback" => fallback
    )
    return dataset
end

"""
    build_grm(dataset; method=:vanraden) -> Matrix{Float64}

基于标准化后的基因型矩阵构建基因关系矩阵 (Genomic Relationship Matrix)。
目前实现 VanRaden I 方法, 即 `G = ZZ' / (2 Σ p_i (1-p_i))`。
"""
function build_grm(dataset::GenomicDataset; method::Symbol = :vanraden)
    X = dataset.genotype
    n, p = size(X)
    p == 0 && throw(ArgumentError("基因型矩阵为空, 无法构建 GRM"))
    method == :vanraden || throw(ArgumentError("暂不支持的方法: $(method)"))

    # 还原原始 0/1/2 计数以计算等位基因频率。
    Z = copy(X)
    denom = 0.0
    @inbounds for j in 1:p
        col = view(Z, :, j)
        μ = mean(col)
        col .-= μ
        freq = (μ + 1.0) / 2  # 假设之前标准化后的均值约为 0
        denom += 2 * freq * (1 - freq)
    end
    denom = denom ≈ 0 ? eps() : denom
    return (Z * transpose(Z)) / denom
end

"""
    build_grm(X::AbstractMatrix)

兼容旧接口, 直接根据矩阵构建 GRM (假设数据已中心化)。
"""
function build_grm(X::AbstractMatrix)
    temp_dataset = GenomicDataset(Matrix{Float64}(X), zeros(size(X, 1)),
                                  ["sample_$(i)" for i in 1:size(X, 1)],
                                  ["marker_$(j)" for j in 1:size(X, 2)],
                                  Dict("source" => "matrix_only"))
    return build_grm(temp_dataset)
end

"""
    kfold_split(dataset, k; rng=Random.default_rng(), stratified=false) -> Vector{Tuple}

生成 k 折交叉验证划分。若 `stratified=true`, 将按表型高于/低于中位数分层, 适合二分类或连续表型的均衡划分。
"""
function kfold_split(dataset::GenomicDataset, k::Integer; rng::AbstractRNG = Random.default_rng(), stratified::Bool = false)
    n = size(dataset.genotype, 1)
    k > 1 || throw(ArgumentError("k 必须大于 1"))

    if stratified
        target = dataset.phenotype
        finite_mask = .!isnan.(target)
        finite_values = target[finite_mask]
        cutoff = median(finite_values)
        high = findall(i -> finite_mask[i] && target[i] >= cutoff, 1:length(target))
        low = findall(i -> finite_mask[i] && target[i] < cutoff, 1:length(target))
        folds = [Int[] for _ in 1:k]
        for group in (high, low)
            shuffled = copy(group)
            Random.shuffle!(rng, shuffled)
            for (i, idx) in enumerate(shuffled)
                push!(folds[mod1(i, k)], idx)
            end
        end
    else
        indices = collect(1:n)
        Random.shuffle!(rng, indices)
        folds = [indices[i:k:n] for i in 1:k]
    end

    splits = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, k)
    for i in 1:k
        test_idx = folds[i]
        train_idx = Int[]
        for j in 1:k
            j == i && continue
            append!(train_idx, folds[j])
        end
        splits[i] = (train_idx, test_idx)
    end
    return splits
end

"""
    make_holdout_split(dataset; test_ratio=0.2, rng=Random.default_rng()) -> Tuple

生成固定比例的训练/测试索引划分。
"""
function make_holdout_split(dataset::GenomicDataset; test_ratio::Real = 0.2, rng::AbstractRNG = Random.default_rng())
    n = size(dataset.genotype, 1)
    test_size = clamp(round(Int, n * test_ratio), 1, n - 1)
    indices = collect(1:n)
    Random.shuffle!(rng, indices)
    test_idx = indices[1:test_size]
    train_idx = indices[(test_size + 1):end]
    return train_idx, test_idx
end

"""
    simulate_genomic_data(n_individuals, n_markers; h2=0.5, maf_beta=(1.5,3.0),
                          effect_dist=Normal(), seed=42) -> GenomicDataset

生成可控的模拟数据。MAF 服从 Beta 分布, 标记效应可指定任意连续分布。
"""
function simulate_genomic_data(n_individuals::Integer, n_markers::Integer;
                               h2::Real = 0.5,
                               maf_beta::Tuple{Real,Real} = (1.5, 3.0),
                               effect_dist::Distribution = Normal(),
                               seed::Integer = 42)
    rng = MersenneTwister(seed)
    maf_dist = Beta(maf_beta[1], maf_beta[2])
    allele_freqs = rand(rng, maf_dist, n_markers)
    genotype = zeros(Float64, n_individuals, n_markers)
    for j in 1:n_markers
        p = allele_freqs[j]
        genotype[:, j] .= rand(rng, Binomial(2, p), n_individuals)
    end
    true_effects = rand(rng, effect_dist, n_markers)
    genetic_values = genotype * true_effects
    vg = var(genetic_values)
    ve = vg * (1 - h2) / max(h2, eps())
    phenotype = genetic_values .+ randn(rng, n_individuals) .* sqrt(ve)

    sample_ids = ["ind_$(i)" for i in 1:n_individuals]
    marker_names = ["snp_$(j)" for j in 1:n_markers]
    metadata = Dict(
        "source" => "simulation",
        "heritability" => h2,
        "allele_freqs" => allele_freqs,
        "effect_distribution" => effect_dist,
        "seed" => seed
    )
    return GenomicDataset(genotype, phenotype, sample_ids, marker_names, metadata)
end

