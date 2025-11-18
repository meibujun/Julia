"""
Data Summary and Statistical Utilities

Provides comprehensive summary statistics and data exploration tools.
"""

using Statistics
using Printf

"""
    GenotypeDataSummary

Comprehensive summary statistics for genotype data.

# Fields
- `n_samples::Int`: Number of samples
- `n_markers::Int`: Number of markers
- `total_genotypes::Int`: Total number of genotyp calls
- `missing_genotypes::Int`: Number of missing genotypes
- `missing_rate::Float64`: Overall missing rate
- `mean_maf::Float64`: Mean minor allele frequency
- `median_maf::Float64`: Median minor allele frequency
- `maf_distribution::NamedTuple`: MAF quantiles
- `marker_missing_stats::NamedTuple`: Marker-level missing statistics
- `sample_missing_stats::NamedTuple`: Sample-level missing statistics
- `genotype_counts::NamedTuple`: Counts of 0, 1, 2 genotypes
- `memory_mb::Float64`: Memory usage in MB
"""
struct GenotypeDataSummary
    n_samples::Int
    n_markers::Int
    total_genotypes::Int
    missing_genotypes::Int
    missing_rate::Float64
    mean_maf::Float64
    median_maf::Float64
    maf_distribution::NamedTuple
    marker_missing_stats::NamedTuple
    sample_missing_stats::NamedTuple
    genotype_counts::NamedTuple
    memory_mb::Float64
    chromosomes::Union{Vector{String},Nothing}
end

function Base.show(io::IO, summary::GenotypeDataSummary)
    println(io, "="^70)
    println(io, "Genotype Data Summary")
    println(io, "="^70)

    println(io, "\nDimensions:")
    @printf(io, "  Samples: %d\n", summary.n_samples)
    @printf(io, "  Markers: %d\n", summary.n_markers)
    @printf(io, "  Total genotypes: %d\n", summary.total_genotypes)

    if !isnothing(summary.chromosomes)
        unique_chrs = unique(summary.chromosomes)
        @printf(io, "  Chromosomes: %d unique (%s)\n",
                length(unique_chrs), join(unique_chrs, ", "))
    end

    println(io, "\nMissing Data:")
    @printf(io, "  Missing genotypes: %d (%.2f%%)\n",
            summary.missing_genotypes, summary.missing_rate * 100)
    @printf(io, "  Per-marker missing: %.4f (mean), %.4f (median)\n",
            summary.marker_missing_stats.mean, summary.marker_missing_stats.median)
    @printf(io, "  Per-sample missing: %.4f (mean), %.4f (median)\n",
            summary.sample_missing_stats.mean, summary.sample_missing_stats.median)

    println(io, "\nMinor Allele Frequency:")
    @printf(io, "  Mean MAF: %.4f\n", summary.mean_maf)
    @printf(io, "  Median MAF: %.4f\n", summary.median_maf)
    @printf(io, "  MAF distribution:\n")
    @printf(io, "    Min:  %.4f\n", summary.maf_distribution.min)
    @printf(io, "    Q25:  %.4f\n", summary.maf_distribution.q25)
    @printf(io, "    Q50:  %.4f\n", summary.maf_distribution.q50)
    @printf(io, "    Q75:  %.4f\n", summary.maf_distribution.q75)
    @printf(io, "    Max:  %.4f\n", summary.maf_distribution.max)

    println(io, "\nGenotype Distribution:")
    total = summary.genotype_counts.n0 + summary.genotype_counts.n1 + summary.genotype_counts.n2
    @printf(io, "  Homozygous ref (0): %d (%.2f%%)\n",
            summary.genotype_counts.n0,
            100 * summary.genotype_counts.n0 / total)
    @printf(io, "  Heterozygous (1):   %d (%.2f%%)\n",
            summary.genotype_counts.n1,
            100 * summary.genotype_counts.n1 / total)
    @printf(io, "  Homozygous alt (2): %d (%.2f%%)\n",
            summary.genotype_counts.n2,
            100 * summary.genotype_counts.n2 / total)

    println(io, "\nMemory Usage:")
    @printf(io, "  Total: %.2f MB\n", summary.memory_mb)
    @printf(io, "  Per sample: %.2f KB\n", summary.memory_mb * 1024 / summary.n_samples)
    @printf(io, "  Per marker: %.2f KB\n", summary.memory_mb * 1024 / summary.n_markers)

    println(io, "="^70)
end

"""
    summarize(geno::CompactGenotypes) -> GenotypeDataSummary

Generate comprehensive summary statistics for genotype data.

# Arguments
- `geno::CompactGenotypes`: Genotype data

# Returns
- `GenotypeDataSummary`: Comprehensive summary statistics

# Example
```julia
geno = read_plink("data")
summary = summarize(geno)
println(summary)
```
"""
function summarize(geno::CompactGenotypes)
    n_samples = geno.n_samples
    n_markers = geno.n_markers
    total_genotypes = n_samples * n_markers

    # Count missing and genotypes
    n_missing = 0
    n0 = 0
    n1 = 0
    n2 = 0

    for i in 1:n_samples
        for j in 1:n_markers
            val = get_genotype(geno, i, j)
            if ismissing(val)
                n_missing += 1
            elseif val == 0
                n0 += 1
            elseif val == 1
                n1 += 1
            elseif val == 2
                n2 += 1
            end
        end
    end

    missing_rate_overall = n_missing / total_genotypes

    # MAF statistics
    mafs = minor_allele_frequency(geno)
    mean_maf = mean(mafs)
    median_maf = median(mafs)

    maf_dist = (
        min = minimum(mafs),
        q25 = quantile(mafs, 0.25),
        q50 = median(mafs),
        q75 = quantile(mafs, 0.75),
        max = maximum(mafs)
    )

    # Marker-level missing statistics
    marker_missing = missing_rate(geno; dim=2)
    marker_stats = (
        mean = mean(marker_missing),
        median = median(marker_missing),
        min = minimum(marker_missing),
        max = maximum(marker_missing)
    )

    # Sample-level missing statistics
    sample_missing = missing_rate(geno; dim=1)
    sample_stats = (
        mean = mean(sample_missing),
        median = median(sample_missing),
        min = minimum(sample_missing),
        max = maximum(sample_missing)
    )

    # Genotype counts
    geno_counts = (n0=n0, n1=n1, n2=n2)

    # Memory usage
    mem_info = memory_usage(geno)
    memory_mb = mem_info.total / 1e6

    # Chromosomes
    chromosomes = isempty(geno.chromosome) ? nothing : geno.chromosome

    return GenotypeDataSummary(
        n_samples, n_markers, total_genotypes,
        n_missing, missing_rate_overall,
        mean_maf, median_maf, maf_dist,
        marker_stats, sample_stats, geno_counts,
        memory_mb, chromosomes
    )
end

"""
    PhenotypeDataSummary

Summary statistics for phenotype data.
"""
struct PhenotypeDataSummary
    n_samples::Int
    n_traits::Int
    trait_summaries::Vector{NamedTuple}
    missing_counts::Vector{Int}
    covariate_info::Union{NamedTuple,Nothing}
end

function Base.show(io::IO, summary::PhenotypeDataSummary)
    println(io, "="^70)
    println(io, "Phenotype Data Summary")
    println(io, "="^70)

    println(io, "\nDimensions:")
    @printf(io, "  Samples: %d\n", summary.n_samples)
    @printf(io, "  Traits: %d\n", summary.n_traits)

    if !isnothing(summary.covariate_info)
        @printf(io, "  Covariates: %d\n", summary.covariate_info.n_covariates)
    end

    println(io, "\nTrait Statistics:")
    println(io, "-"^70)
    @printf(io, "%-15s %10s %10s %10s %10s %10s\n",
            "Trait", "Mean", "SD", "Min", "Max", "Missing")
    println(io, "-"^70)

    for (i, (name, stats, n_miss)) in enumerate(zip(
        1:summary.n_traits,
        summary.trait_summaries,
        summary.missing_counts
    ))
        trait_name = i <= summary.n_traits ? "Trait $i" : "Covariate $(i-summary.n_traits)"
        @printf(io, "%-15s %10.3f %10.3f %10.3f %10.3f %10d\n",
                trait_name, stats.mean, stats.sd, stats.min, stats.max, n_miss)
    end
    println(io, "-"^70)

    println(io, "="^70)
end

"""
    summarize(pheno::PhenotypeData) -> PhenotypeDataSummary

Generate summary statistics for phenotype data.
"""
function summarize(pheno::PhenotypeData)
    n_samples = length(pheno.sample_ids)
    n_traits = length(pheno.trait_names)

    trait_summaries = NamedTuple[]
    missing_counts = Int[]

    for j in 1:n_traits
        values = pheno.values[:, j]
        non_missing = values[.!ismissing.(values)]

        if isempty(non_missing)
            stats = (mean=NaN, sd=NaN, min=NaN, max=NaN)
            n_miss = n_samples
        else
            stats = (
                mean = mean(non_missing),
                sd = std(non_missing),
                min = minimum(non_missing),
                max = maximum(non_missing)
            )
            n_miss = n_samples - length(non_missing)
        end

        push!(trait_summaries, stats)
        push!(missing_counts, n_miss)
    end

    covariate_info = isnothing(pheno.covariates) ? nothing : (
        n_covariates = size(pheno.covariates, 2),
    )

    return PhenotypeDataSummary(
        n_samples, n_traits, trait_summaries, missing_counts, covariate_info
    )
end

"""
    compare_datasets(geno1::CompactGenotypes, geno2::CompactGenotypes; name1::String="Dataset 1", name2::String="Dataset 2")

Compare two genotype datasets side-by-side.
"""
function compare_datasets(geno1::CompactGenotypes, geno2::CompactGenotypes;
                         name1::String="Dataset 1", name2::String="Dataset 2")
    summary1 = summarize(geno1)
    summary2 = summarize(geno2)

    println("="^80)
    println("Dataset Comparison")
    println("="^80)

    println("\nDimensions:")
    println("-"^80)
    @printf("%-30s %20s %20s\n", "Metric", name1, name2)
    println("-"^80)
    @printf("%-30s %20d %20d\n", "Samples", summary1.n_samples, summary2.n_samples)
    @printf("%-30s %20d %20d\n", "Markers", summary1.n_markers, summary2.n_markers)
    @printf("%-30s %20.2f %20.2f\n", "Memory (MB)", summary1.memory_mb, summary2.memory_mb)

    println("\nData Quality:")
    println("-"^80)
    @printf("%-30s %20.4f %20.4f\n", "Missing rate", summary1.missing_rate, summary2.missing_rate)
    @printf("%-30s %20.4f %20.4f\n", "Mean MAF", summary1.mean_maf, summary2.mean_maf)
    @printf("%-30s %20.4f %20.4f\n", "Median MAF", summary1.median_maf, summary2.median_maf)

    println("-"^80)
end

"""
    detect_outliers(geno::CompactGenotypes; method::Symbol=:iqr, threshold::Float64=3.0) -> Vector{Int}

Detect outlier samples based on missing rate or heterozygosity.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `method::Symbol`: Method to use (:iqr for IQR, :sd for standard deviation)
- `threshold::Float64`: Threshold multiplier (default: 3.0 for IQR, 3.0 for SD)

# Returns
- `Vector{Int}`: Indices of outlier samples
"""
function detect_outliers(geno::CompactGenotypes; method::Symbol=:iqr, threshold::Float64=3.0)
    # Calculate heterozygosity rate for each sample
    het_rates = heterozygosity_rate(geno; dim=1)

    outliers = Int[]

    if method == :iqr
        q25 = quantile(het_rates, 0.25)
        q75 = quantile(het_rates, 0.75)
        iqr = q75 - q25

        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr

        for (i, rate) in enumerate(het_rates)
            if rate < lower || rate > upper
                push!(outliers, i)
            end
        end

    elseif method == :sd
        μ = mean(het_rates)
        σ = std(het_rates)

        lower = μ - threshold * σ
        upper = μ + threshold * σ

        for (i, rate) in enumerate(het_rates)
            if rate < lower || rate > upper
                push!(outliers, i)
            end
        end
    else
        throw(ArgumentError("Unknown method: $method. Use :iqr or :sd"))
    end

    return outliers
end

"""
    marker_quality_summary(geno::CompactGenotypes) -> DataFrame

Generate per-marker quality summary (requires DataFrames.jl).
"""
function marker_quality_summary(geno::CompactGenotypes)
    mafs = minor_allele_frequency(geno)
    missing_rates = missing_rate(geno; dim=2)

    # Calculate HWE p-values if possible
    hwe_pvalues = Float64[]
    for j in 1:geno.n_markers
        try
            # Count genotypes
            n0 = n1 = n2 = 0
            for i in 1:geno.n_samples
                val = get_genotype(geno, i, j)
                if !ismissing(val)
                    if val == 0
                        n0 += 1
                    elseif val == 1
                        n1 += 1
                    elseif val == 2
                        n2 += 1
                    end
                end
            end

            pval = hardy_weinberg_test(n0, n1, n2)
            push!(hwe_pvalues, pval)
        catch
            push!(hwe_pvalues, NaN)
        end
    end

    return (
        marker_ids = geno.marker_ids,
        chromosome = geno.chromosome,
        position = geno.position,
        maf = mafs,
        missing_rate = missing_rates,
        hwe_pvalue = hwe_pvalues
    )
end

# Export
export GenotypeDataSummary, PhenotypeDataSummary
export summarize, compare_datasets, detect_outliers, marker_quality_summary
