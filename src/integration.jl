function integrate_concatenate(datasets::Vector{OmicsDataset})
    combined_matrix = vcat([ds.matrix for ds in datasets]...)
    combined_features = String[]
    for ds in datasets
        append!(combined_features, [string(ds.name, "::", feature) for feature in ds.features])
    end
    metadata = Dict{String, Any}(
        "strategy" => "concatenate",
        "source_datasets" => [ds.name for ds in datasets]
    )
    return IntegratedDataset("Integrated", combined_matrix, combined_features, copy(datasets[1].samples), metadata)
end

function integrate_weighted_sum(datasets::Vector{OmicsDataset}, weights::Dict{String, Float64})
    reference_features = datasets[1].features
    for ds in datasets[2:end]
        ds.features == reference_features || throw(ArgumentError("Weighted sum integration requires identical feature ordering"))
    end
    weight_vec = [weights[ds.name] for ds in datasets]
    weight_sum = sum(weight_vec)
    weight_sum == 0 && throw(ArgumentError("Weights must not sum to zero"))
    normalised = weight_vec ./ weight_sum
    combined_matrix = zeros(Float64, size(datasets[1].matrix))
    for (w, ds) in zip(normalised, datasets)
        combined_matrix .+= w .* ds.matrix
    end
    metadata = Dict{String, Any}(
        "strategy" => "weighted_sum",
        "weights" => Dict(ds.name => weights[ds.name] for ds in datasets)
    )
    return IntegratedDataset("Integrated", combined_matrix, copy(reference_features), copy(datasets[1].samples), metadata)
end

function integrate_datasets(datasets::Vector{OmicsDataset}, config::IntegrationConfig)
    isempty(datasets) && throw(ArgumentError("No datasets provided for integration"))
    if config.strategy == :concatenate
        return integrate_concatenate(datasets)
    elseif config.strategy == :weighted_sum
        weights = _ensure_weights(datasets, config.weights)
        return integrate_weighted_sum(datasets, weights)
    else
        throw(ArgumentError("Unsupported integration strategy: $(config.strategy)"))
    end
end
