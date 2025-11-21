function run_pipeline(config::PipelineConfig)
    datasets = OmicsDataset[]
    for ds_config in config.dataset_configs
        dataset = load_omics_dataset(ds_config)
        validate_dataset(dataset, config.quality_control)
        impute_missing!(dataset, ds_config.imputation)
        normalize!(dataset, ds_config.normalization; log_base=ds_config.log_base)
        filter_low_quality!(dataset, ds_config, config.quality_control)
        push!(datasets, dataset)
    end
    aligned = align_datasets(datasets)
    integrated = integrate_datasets(aligned, config.integration_config)
    analysis = run_analysis(integrated, config.analysis_config)
    report = build_report(aligned, integrated, analysis, config)
    save_report(report, config.report_config)
    return PipelineResult(aligned, integrated, analysis, report)
end
