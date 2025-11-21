Base.@kwdef struct DatasetConfig
    name::String
    path::String
    delimiter::Char = ','
    feature_column::Union{Int, Symbol, String} = 1
    normalization::Symbol = :zscore
    log_base::Union{Nothing, Float64} = nothing
    imputation::Symbol = :mean
    min_nonmissing_ratio::Float64 = 0.7
    min_variance::Float64 = 1e-8
    metadata::Dict{String, Any} = Dict{String, Any}()
end

Base.@kwdef struct IntegrationConfig
    strategy::Symbol = :concatenate
    weights::Dict{String, Float64} = Dict{String, Float64}()
end

Base.@kwdef struct AnalysisConfig
    run_pca::Bool = true
    pca_components::Int = 3
    clustering::Union{Nothing, Symbol} = :kmeans
    cluster_count::Int = 3
    random_seed::Int = 42
end

Base.@kwdef struct ReportConfig
    output_path::Union{Nothing, String} = nothing
    format::Symbol = :json
    text_output_path::Union{Nothing, String} = nothing
    include_datasets::Bool = true
    include_analysis::Bool = true
end

Base.@kwdef struct QualityControlConfig
    min_samples::Int = 1
    min_features::Int = 1
    min_nonmissing_ratio::Float64 = 0.5
    min_variance::Float64 = 1e-8
end

struct PipelineConfig
    dataset_configs::Vector{DatasetConfig}
    integration_config::IntegrationConfig
    analysis_config::AnalysisConfig
    report_config::ReportConfig
    quality_control::QualityControlConfig
end

Base.@kwdef mutable struct OmicsDataset
    name::String
    matrix::Matrix{Float64}
    features::Vector{String}
    samples::Vector{String}
    metadata::Dict{String, Any} = Dict{String, Any}()
end

Base.@kwdef mutable struct IntegratedDataset
    name::String
    matrix::Matrix{Float64}
    features::Vector{String}
    samples::Vector{String}
    metadata::Dict{String, Any} = Dict{String, Any}()
end

struct PCAResult
    scores::Matrix{Float64}
    loadings::Matrix{Float64}
    explained_variance::Vector{Float64}
    explained_ratio::Vector{Float64}
end

struct ClusteringResult
    assignments::Vector{Int}
    centroids::Matrix{Float64}
    inertia::Float64
end

Base.@kwdef struct AnalysisResult
    pca::Union{Nothing, PCAResult} = nothing
    clustering::Union{Nothing, ClusteringResult} = nothing
end

struct PipelineResult
    datasets::Vector{OmicsDataset}
    integrated::IntegratedDataset
    analysis::AnalysisResult
    report::Dict{Symbol, Any}
end

PipelineConfig(dataset_configs::Vector{DatasetConfig};
               integration_config::IntegrationConfig=IntegrationConfig(),
               analysis_config::AnalysisConfig=AnalysisConfig(),
               report_config::ReportConfig=ReportConfig(),
               quality_control::QualityControlConfig=QualityControlConfig()) =
    PipelineConfig(dataset_configs, integration_config, analysis_config, report_config, quality_control)
