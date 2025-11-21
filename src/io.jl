function load_omics_dataset(config::DatasetConfig)
    raw_missing = get(config.metadata, "missing_strings", String[])
    missing_strings = raw_missing isa AbstractVector ? [String(v) for v in raw_missing] : String[]
    delim = _to_char(config.delimiter)
    df = CSV.read(config.path, DataFrame; delim=delim, missingstring=missing_strings, ignorerepeated=true)
    feature_idx = _feature_column_index(df, config.feature_column)
    features = String.(df[!, feature_idx])
    sample_cols = Symbol[]
    for (idx, col) in enumerate(names(df))
        idx == feature_idx && continue
        push!(sample_cols, col)
    end
    samples = String.(sample_cols)
    nfeatures = length(features)
    nsamples = length(samples)
    matrix = Matrix{Float64}(undef, nfeatures, nsamples)
    for (j, colname) in enumerate(sample_cols)
        matrix[:, j] = _coerce_to_float_column(df[!, colname])
    end
    metadata = Dict{String, Any}(
        "source_path" => config.path,
        "delimiter" => String(delim),
        "missing_strings" => missing_strings,
        "sample_columns" => samples
    )
    for (k, v) in config.metadata
        metadata[k] = v
    end
    return OmicsDataset(config.name, matrix, features, samples, metadata)
end

function read_pipeline_config(path::AbstractString)
    config_json = JSON3.read(read(path, String))
    haskey(config_json, "datasets") || throw(ArgumentError("Pipeline configuration missing 'datasets' section"))
    dataset_configs = DatasetConfig[]
    for entry in config_json["datasets"]
        metadata = Dict{String, Any}()
        if haskey(entry, "metadata") && entry["metadata"] !== nothing
            for (k, v) in entry["metadata"]
                key = String(k)
                if v isa AbstractVector
                    metadata[key] = [String(x) for x in v]
                elseif v isa AbstractString
                    metadata[key] = String(v)
                else
                    metadata[key] = v
                end
            end
        end
        feature_column = 1
        if haskey(entry, "feature_column") && entry["feature_column"] !== nothing
            value = entry["feature_column"]
            if value isa Integer
                feature_column = Int(value)
            else
                feature_column = Symbol(String(value))
            end
        end
        push!(dataset_configs, DatasetConfig(
            name = String(entry["name"]),
            path = String(entry["path"]),
            delimiter = haskey(entry, "delimiter") && entry["delimiter"] !== nothing ? _to_char(String(entry["delimiter"])) : ',',
            feature_column = feature_column,
            normalization = haskey(entry, "normalization") && entry["normalization"] !== nothing ? Symbol(String(entry["normalization"])) : :zscore,
            log_base = haskey(entry, "log_base") && entry["log_base"] !== nothing ? Float64(entry["log_base"]) : nothing,
            imputation = haskey(entry, "imputation") && entry["imputation"] !== nothing ? Symbol(String(entry["imputation"])) : :mean,
            min_nonmissing_ratio = haskey(entry, "min_nonmissing_ratio") && entry["min_nonmissing_ratio"] !== nothing ? Float64(entry["min_nonmissing_ratio"]) : 0.7,
            min_variance = haskey(entry, "min_variance") && entry["min_variance"] !== nothing ? Float64(entry["min_variance"]) : 1e-8,
            metadata = metadata
        ))
    end
    integration_section = haskey(config_json, "integration") ? config_json["integration"] : Dict()
    weights_dict = Dict{String, Float64}()
    if haskey(integration_section, "weights") && integration_section["weights"] !== nothing
        for (k, v) in integration_section["weights"]
            weights_dict[String(k)] = Float64(v)
        end
    end
    integration_config = IntegrationConfig(
        strategy = haskey(integration_section, "strategy") && integration_section["strategy"] !== nothing ? Symbol(String(integration_section["strategy"])) : :concatenate,
        weights = weights_dict
    )
    analysis_section = haskey(config_json, "analysis") ? config_json["analysis"] : Dict()
    analysis_config = AnalysisConfig(
        run_pca = Bool(get(analysis_section, "run_pca", true)),
        pca_components = Int(get(analysis_section, "pca_components", 3)),
        clustering = haskey(analysis_section, "clustering") && analysis_section["clustering"] !== nothing ? Symbol(String(analysis_section["clustering"])) : :kmeans,
        cluster_count = Int(get(analysis_section, "cluster_count", 3)),
        random_seed = Int(get(analysis_section, "random_seed", 42))
    )
    report_section = haskey(config_json, "report") ? config_json["report"] : Dict()
    report_config = ReportConfig(
        output_path = haskey(report_section, "output_path") && report_section["output_path"] !== nothing ? String(report_section["output_path"]) : nothing,
        format = haskey(report_section, "format") && report_section["format"] !== nothing ? Symbol(String(report_section["format"])) : :json,
        text_output_path = haskey(report_section, "text_output_path") && report_section["text_output_path"] !== nothing ? String(report_section["text_output_path"]) : nothing,
        include_datasets = Bool(get(report_section, "include_datasets", true)),
        include_analysis = Bool(get(report_section, "include_analysis", true))
    )
    qc_section = haskey(config_json, "quality_control") ? config_json["quality_control"] : Dict()
    quality_control = QualityControlConfig(
        min_samples = Int(get(qc_section, "min_samples", 1)),
        min_features = Int(get(qc_section, "min_features", 1)),
        min_nonmissing_ratio = Float64(get(qc_section, "min_nonmissing_ratio", 0.5)),
        min_variance = Float64(get(qc_section, "min_variance", 1e-8))
    )
    return PipelineConfig(dataset_configs;
        integration_config=integration_config,
        analysis_config=analysis_config,
        report_config=report_config,
        quality_control=quality_control)
end
