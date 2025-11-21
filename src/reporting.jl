function dataset_statistics(dataset::OmicsDataset)
    variances = dataset_variances(dataset)
    variance_summary = isempty(variances) ? Dict(:mean => 0.0, :min => 0.0, :max => 0.0) : Dict(
        :mean => mean(variances),
        :min => minimum(variances),
        :max => maximum(variances)
    )
    Dict(
        :name => dataset.name,
        :samples => length(dataset.samples),
        :features => length(dataset.features),
        :missing_ratio => dataset_missing_ratio(dataset),
        :variance_summary => variance_summary,
        :metadata => dataset.metadata
    )
end

function analysis_statistics(result::AnalysisResult)
    stats = Dict{Symbol, Any}()
    if result.pca !== nothing
        stats[:pca] = Dict(
            :explained_ratio => result.pca.explained_ratio,
            :explained_variance => result.pca.explained_variance
        )
    end
    if result.clustering !== nothing
        stats[:clustering] = Dict(
            :assignments => result.clustering.assignments,
            :inertia => result.clustering.inertia
        )
    end
    return stats
end

function build_report(datasets::Vector{OmicsDataset}, integrated::IntegratedDataset, analysis::AnalysisResult, config::PipelineConfig)
    report = Dict{Symbol, Any}(
        :timestamp => Dates.format(Dates.now(), Dates.RFC3339),
        :integrated => Dict(
            :samples => length(integrated.samples),
            :features => length(integrated.features),
            :metadata => integrated.metadata
        ),
        :quality_control => Dict(
            :min_samples => config.quality_control.min_samples,
            :min_features => config.quality_control.min_features,
            :min_nonmissing_ratio => config.quality_control.min_nonmissing_ratio,
            :min_variance => config.quality_control.min_variance
        )
    )
    if config.report_config.include_datasets
        report[:datasets] = [dataset_statistics(ds) for ds in datasets]
    end
    if config.report_config.include_analysis
        report[:analysis] = analysis_statistics(analysis)
    end
    return report
end

function save_report(report::Dict{Symbol, Any}, config::ReportConfig)
    if config.output_path !== nothing
        mkpath(dirname(config.output_path))
        open(config.output_path, "w") do io
            JSON3.write(io, report)
        end
    end
    if config.text_output_path !== nothing
        mkpath(dirname(config.text_output_path))
        open(config.text_output_path, "w") do io
            println(io, render_summary(report))
        end
    end
    return report
end

function render_summary(report::Dict{Symbol, Any})
    io = IOBuffer()
    println(io, "GenomicPro Multi-Omics Report")
    println(io, "Generated at: ", report[:timestamp])
    integrated = report[:integrated]
    println(io, "Integrated dataset: $(integrated[:samples]) samples x $(integrated[:features]) features")
    if haskey(report, :datasets)
        println(io, "Datasets:")
        for ds in report[:datasets]
            println(io, "  - $(ds[:name]): $(ds[:samples]) samples, $(ds[:features]) features, missing=$(round(ds[:missing_ratio], digits=3))")
        end
    end
    if haskey(report, :analysis)
        if haskey(report[:analysis], :pca)
            ratios = report[:analysis][:pca][:explained_ratio]
            println(io, "PCA explained variance ratios: ", join(round.(ratios, digits=3), ", "))
        end
        if haskey(report[:analysis], :clustering)
            inertia = report[:analysis][:clustering][:inertia]
            println(io, "Clustering inertia: ", round(inertia, digits=3))
        end
    end
    return String(take!(io))
end

function render_summary(result::PipelineResult)
    return render_summary(result.report)
end
