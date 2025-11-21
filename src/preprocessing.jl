function impute_missing!(dataset::OmicsDataset, strategy::Symbol)
    for row in eachrow(dataset.matrix)
        if strategy == :mean
            value = _nanmean(row)
            isnan(value) && (value = 0.0)
            _replace_nan!(row, value)
        elseif strategy == :median
            value = _nanmedian(row)
            isnan(value) && (value = 0.0)
            _replace_nan!(row, value)
        elseif strategy == :zero
            _replace_nan!(row, 0.0)
        else
            throw(ArgumentError("Unsupported imputation strategy: $(strategy)"))
        end
    end
    dataset.metadata["imputation"] = String(strategy)
    return dataset
end

function log_transform!(dataset::OmicsDataset, base::Float64)
    base > 0 || throw(ArgumentError("Log base must be positive"))
    offset = 0.0
    minimum_value = minimum(dataset.matrix)
    if minimum_value <= 0
        offset = abs(minimum_value) + 1e-6
    end
    dataset.matrix .= log.(dataset.matrix .+ offset .+ 1e-9) ./ log(base)
    dataset.metadata["log_base"] = base
    dataset.metadata["log_offset"] = offset
    return dataset
end

function zscore_normalize!(dataset::OmicsDataset)
    for row in eachrow(dataset.matrix)
        μ = mean(row)
        σ = std(row)
        σ ≈ 0 && (σ = 1.0)
        row .-= μ
        row ./= σ
    end
    dataset.metadata["normalization"] = "zscore"
    return dataset
end

function minmax_scale!(dataset::OmicsDataset)
    for row in eachrow(dataset.matrix)
        minv = minimum(row)
        maxv = maximum(row)
        range = maxv - minv
        range ≈ 0 && (range = 1.0)
        row .-= minv
        row ./= range
    end
    dataset.metadata["normalization"] = "minmax"
    return dataset
end

function robust_scale!(dataset::OmicsDataset)
    for row in eachrow(dataset.matrix)
        med = median(row)
        q1 = quantile(row, 0.25)
        q3 = quantile(row, 0.75)
        iqr = q3 - q1
        iqr ≈ 0 && (iqr = 1.0)
        row .-= med
        row ./= iqr
    end
    dataset.metadata["normalization"] = "robust"
    return dataset
end

function normalize!(dataset::OmicsDataset, method::Symbol; log_base::Union{Nothing, Float64}=nothing)
    if log_base !== nothing
        log_transform!(dataset, log_base)
    end
    if method == :zscore
        zscore_normalize!(dataset)
    elseif method == :minmax
        minmax_scale!(dataset)
    elseif method == :robust
        robust_scale!(dataset)
    elseif method == :none
        dataset.metadata["normalization"] = "none"
    else
        throw(ArgumentError("Unsupported normalization method: $(method)"))
    end
    return dataset
end

function filter_low_quality!(dataset::OmicsDataset, config::DatasetConfig, qc::QualityControlConfig)
    apply_quality_filters!(dataset, config, qc)
    dataset.metadata["post_filter_features"] = length(dataset.features)
    return dataset
end

function align_datasets(datasets::Vector{OmicsDataset})
    isempty(datasets) && throw(ArgumentError("No datasets provided for alignment"))
    sample_sets = map(ds -> Set(ds.samples), datasets)
    common_samples = reduce(intersect, sample_sets)
    isempty(common_samples) && throw(ArgumentError("Datasets do not share common samples"))
    ordered_samples = [sample for sample in datasets[1].samples if sample in common_samples]
    aligned = OmicsDataset[]
    for ds in datasets
        sample_index = Dict(sample => idx for (idx, sample) in enumerate(ds.samples))
        matrix = Matrix{Float64}(undef, size(ds.matrix, 1), length(ordered_samples))
        for (j, sample) in enumerate(ordered_samples)
            idx = sample_index[sample]
            matrix[:, j] = ds.matrix[:, idx]
        end
        metadata = copy(ds.metadata)
        metadata["aligned_samples"] = ordered_samples
        push!(aligned, OmicsDataset(ds.name, matrix, copy(ds.features), ordered_samples, metadata))
    end
    return aligned
end

function summarise_dataset(dataset::OmicsDataset)
    Dict(
        :name => dataset.name,
        :samples => length(dataset.samples),
        :features => length(dataset.features),
        :missing_ratio => dataset_missing_ratio(dataset),
        :metadata => dataset.metadata
    )
end
