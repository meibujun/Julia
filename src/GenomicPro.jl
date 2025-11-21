module GenomicPro

export DatasetConfig, IntegrationConfig, AnalysisConfig, ReportConfig, QualityControlConfig,
       PipelineConfig, PipelineResult, OmicsDataset, IntegratedDataset,
       load_omics_dataset, read_pipeline_config, run_pipeline, align_datasets,
       integrate_datasets, normalize!, impute_missing!, filter_low_quality!,
       run_pca, run_kmeans, build_report, save_report, render_summary

using CSV
using DataFrames
using JSON3
using LinearAlgebra
using Statistics
using Random
using Dates

include("types.jl")
include("utils.jl")
include("validation.jl")
include("io.jl")
include("preprocessing.jl")
include("integration.jl")
include("analysis.jl")
include("reporting.jl")
include("pipeline.jl")

end
