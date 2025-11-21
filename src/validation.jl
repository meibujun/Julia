function validate_dataset(dataset::OmicsDataset, qc::QualityControlConfig)
    nsamples = length(dataset.samples)
    nfeatures = length(dataset.features)
    nsamples >= qc.min_samples || throw(ArgumentError("Dataset $(dataset.name) does not have enough samples"))
    nfeatures >= qc.min_features || throw(ArgumentError("Dataset $(dataset.name) does not have enough features"))
    length(unique(dataset.samples)) == nsamples || throw(ArgumentError("Dataset $(dataset.name) has duplicate sample identifiers"))
    length(unique(dataset.features)) == nfeatures || throw(ArgumentError("Dataset $(dataset.name) has duplicate feature identifiers"))
    return dataset
end

function dataset_missing_ratio(dataset::OmicsDataset)
    isempty(dataset.features) && return 0.0
    ratios = map(row -> _nan_ratio(row), eachrow(dataset.matrix))
    return mean(ratios)
end

function dataset_variances(dataset::OmicsDataset)
    return map(row -> _variance(row), eachrow(dataset.matrix))
end

function apply_quality_filters!(dataset::OmicsDataset, config::DatasetConfig, qc::QualityControlConfig)
    mask = trues(length(dataset.features))
    filtered = 0
    for (i, row) in enumerate(eachrow(dataset.matrix))
        missing_ratio = _nan_ratio(row)
        variance = _variance(row)
        if missing_ratio > (1 - config.min_nonmissing_ratio) || variance < max(config.min_variance, qc.min_variance)
            mask[i] = false
            filtered += 1
        end
    end
    if filtered > 0
        dataset.matrix = dataset.matrix[mask, :]
        dataset.features = dataset.features[mask]
    end
    dataset.metadata["filtered_features"] = filtered
    return dataset
end
