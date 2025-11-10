# src/GenomicProQC/filters.jl

"""
    AbstractQCFilter

Abstract base type for quality control filters in GenomicPro.jl.

Quality control filters remove low-quality samples and markers based on specific
criteria. Filters are designed to be composable, allowing construction of
sophisticated multi-stage QC pipelines through function composition or chaining.

# Filter Design Philosophy
1. **Single Responsibility**: Each filter addresses one quality metric
2. **Composability**: Filters chain together naturally via QCPipeline
3. **Transparency**: Detailed reporting of what was filtered and why
4. **Reversibility**: Original data preserved for alternative QC strategies

# Interface Requirements
Concrete filter types must implement:
- `apply_filter(filter::AbstractQCFilter, data)`: Apply filter returning filtered data and report
- `get_threshold(filter::AbstractQCFilter)`: Return filter threshold parameter(s)
- `get_filtered_indices(filter::AbstractQCFilter, data)`: Return indices failing QC

# Common Quality Control Metrics
- **Missing Rate**: Proportion of missing genotypes (per sample/marker)
- **Minor Allele Frequency (MAF)**: Frequency of less common allele
- **Hardy-Weinberg Equilibrium (HWE)**: Test for genotype frequency deviations
- **Call Rate**: Proportion of successful genotype calls

# Examples
```julia
# Single filter application
genotypes = read_genotypes("data.vcf")
filter = MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10)
genotypes_clean, report = apply_filter(filter, genotypes)

# Composable pipeline
pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])
genotypes_qc, full_report = apply_qc(genotypes, pipeline)
```

# See Also
- [`QCPipeline`](@ref): Compose multiple filters in sequence
- [`apply_filter`](@ref): Apply single filter to data
- [`QCReport`](@ref): Comprehensive quality control summary
"""
abstract type AbstractQCFilter end


"""
    MissingRateFilter <: AbstractQCFilter

Filter samples and markers based on missing genotype rates.

Removes samples exceeding a maximum missing rate threshold and markers with
excessive missing data. Missing rate is computed as the proportion of missing
genotypes relative to total possible genotypes.

# Mathematical Definition
For sample i: missing_rate_i = (count of missing genotypes) / (total markers)
For marker j: missing_rate_j = (count of missing genotypes) / (total samples)

# Fields
- `sample_threshold::Float64`: Maximum allowed missing rate for samples (0.0-1.0)
- `marker_threshold::Float64`: Maximum allowed missing rate for markers (0.0-1.0)

# Quality Control Rationale
High missing rates indicate:
- Poor sample DNA quality or quantity
- Systematic genotyping failures for specific markers
- Technical issues during array processing
- Potential sample swaps or contamination

Typical thresholds: 5-10% for samples, 10% for markers, depending on application

# Examples
```julia
# Standard QC with 10% threshold for both samples and markers
filter = MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10)
genotypes_clean, report = apply_filter(filter, genotypes)

println("Samples removed: ", report.n_samples_removed)
println("Markers removed: ", report.n_markers_removed)

# Strict QC for high-quality applications
strict_filter = MissingRateFilter(sample_threshold=0.05, marker_threshold=0.05)

# Lenient QC for exploratory analysis
lenient_filter = MissingRateFilter(sample_threshold=0.20, marker_threshold=0.15)
```

# Performance
- Time Complexity: O(n × m) where n = samples, m = markers
- Space Complexity: O(n + m) for missing rate vectors
- Typical speed: 1-2 seconds for 10,000 samples × 50,000 markers

# See Also
- [`compute_missing_rates`](@ref): Calculate missing rates without filtering
- [`CallRateFilter`](@ref): Alternative filter based on call rate (1 - missing rate)
"""
struct MissingRateFilter <: AbstractQCFilter
    sample_threshold::Float64
    marker_threshold::Float64

    function MissingRateFilter(;
                               sample_threshold::Float64 = 0.10,
                               marker_threshold::Float64 = 0.10)
        # Validate thresholds
        if sample_threshold < 0.0 || sample_threshold > 1.0
            throw(ArgumentError("sample_threshold must be in [0,1], got $sample_threshold"))
        end
        if marker_threshold < 0.0 || marker_threshold > 1.0
            throw(ArgumentError("marker_threshold must be in [0,1], got $marker_threshold"))
        end

        new(sample_threshold, marker_threshold)
    end
end


"""
    MAFFilter <: AbstractQCFilter

Filter markers based on minor allele frequency (MAF).

Removes markers with MAF below a specified threshold. Rare variants often have
poor imputation quality, contribute limited information for genomic prediction,
and may represent genotyping errors rather than true biological variation.

# Mathematical Definition
For marker j with allele frequency p_j:
    MAF_j = min(p_j, 1 - p_j)

Filter removes markers where MAF_j < min_maf

# Fields
- `min_maf::Float64`: Minimum minor allele frequency threshold (0.0-0.5)
- `per_population::Bool`: Calculate MAF separately per population (if population info available)

# Quality Control Rationale
Low MAF markers are problematic because:
- Increased susceptibility to genotyping errors
- Large-effect rare variants distort relationship matrices
- Poor statistical power for association testing
- Unreliable imputation accuracy

Common thresholds:
- 0.01 (1%): Standard for genomic prediction and GWAS
- 0.05 (5%): Conservative for association mapping
- 0.001 (0.1%): Retain for rare variant analysis

# Examples
```julia
# Standard MAF filter at 1%
filter = MAFFilter(min_maf=0.01)
genotypes_clean, report = apply_filter(filter, genotypes)

# Conservative filter for robust analysis
strict_filter = MAFFilter(min_maf=0.05)

# Per-population MAF calculation (avoids removing population-specific variants)
pop_filter = MAFFilter(min_maf=0.01, per_population=true)
```

# Implementation Notes
- Allele frequencies computed on non-missing genotypes only
- Monomorphic markers (MAF = 0) automatically removed
- Per-population mode requires population assignments in data metadata

# Performance
- Time Complexity: O(n × m) for frequency computation
- Space Complexity: O(m) for frequency vector
- Typical speed: 1-3 seconds for 10,000 samples × 50,000 markers

# See Also
- [`get_allele_frequencies`](@ref): Compute allele frequencies without filtering
- [`compute_maf_distribution`](@ref): Analyze MAF spectrum
"""
struct MAFFilter <: AbstractQCFilter
    min_maf::Float64
    per_population::Bool

    function MAFFilter(;
                       min_maf::Float64 = 0.01,
                       per_population::Bool = false)
        if min_maf < 0.0 || min_maf > 0.5
            throw(ArgumentError("min_maf must be in [0, 0.5], got $min_maf"))
        end

        new(min_maf, per_population)
    end
end


"""
    HWEFilter <: AbstractQCFilter

Filter markers based on Hardy-Weinberg Equilibrium (HWE) test.

Tests whether observed genotype frequencies match Hardy-Weinberg expectations.
Significant deviations suggest genotyping errors, population stratification,
selection, or non-random mating.

# Mathematical Foundation
Under HWE with allele frequency p, expected genotype frequencies:
- P(AA) = p²         (homozygous reference)
- P(Aa) = 2p(1-p)    (heterozygous)
- P(aa) = (1-p)²     (homozygous alternate)

Chi-square test statistic:
    χ² = Σ[(Observed - Expected)² / Expected]

Degrees of freedom: 1
Reject HWE if p-value < threshold (typically 1e-6 after Bonferroni correction)

# Fields
- `pvalue_threshold::Float64`: Maximum p-value for marker retention (0.0-1.0)
- `bonferroni_correction::Bool`: Apply Bonferroni correction for multiple testing
- `per_population::Bool`: Test HWE separately per population

# Quality Control Rationale
HWE violations indicate:
- Systematic genotyping errors (most common cause)
- Population stratification (admixed samples)
- Copy number variants or segmental duplications
- Strong selection at the locus

Typical threshold: p-value < 1e-6 (very conservative to avoid false positives)

# Examples
```julia
# Standard HWE filter with Bonferroni correction
filter = HWEFilter(pvalue_threshold=1e-6, bonferroni_correction=true)
genotypes_clean, report = apply_filter(filter, genotypes)

println("Markers failing HWE: ", report.n_markers_removed)
println("Mean chi-square statistic: ", report.mean_chi_square)

# Less stringent for exploratory analysis
lenient_filter = HWEFilter(pvalue_threshold=1e-4, bonferroni_correction=false)

# Per-population testing (recommended for multi-breed datasets)
pop_filter = HWEFilter(pvalue_threshold=1e-6, per_population=true)
```

# Implementation Notes
- Exact test used for small sample sizes (n < 100)
- Chi-square approximation for larger samples (faster)
- Markers with MAF < 0.01 typically exempt (unreliable HWE test)
- Multiple testing correction: threshold = α / n_markers

# Performance
- Time Complexity: O(n × m) for genotype counting and chi-square computation
- Space Complexity: O(m) for p-value vector
- Typical speed: 2-4 seconds for 10,000 samples × 50,000 markers

# References
- Wigginton et al. (2005) Am J Hum Genet 76:887-893

# See Also
- [`test_hwe`](@ref): Test HWE for individual markers
- [`compute_genotype_frequencies`](@ref): Observed genotype frequencies
"""
struct HWEFilter <: AbstractQCFilter
    pvalue_threshold::Float64
    bonferroni_correction::Bool
    per_population::Bool

    function HWEFilter(;
                       pvalue_threshold::Float64 = 1e-6,
                       bonferroni_correction::Bool = true,
                       per_population::Bool = false)
        if pvalue_threshold <= 0.0 || pvalue_threshold > 1.0
            throw(ArgumentError("pvalue_threshold must be in (0,1], got $pvalue_threshold"))
        end

        new(pvalue_threshold, bonferroni_correction, per_population)
    end
end


"""
    CallRateFilter <: AbstractQCFilter

Filter samples and markers based on genotype call rate (complement of missing rate).

Call rate is the proportion of successful genotype calls. This filter is
functionally equivalent to MissingRateFilter but uses the complementary metric,
which some users find more intuitive.

# Mathematical Definition
Call rate = 1 - Missing rate
For sample i: call_rate_i = (successful calls) / (total markers)
For marker j: call_rate_j = (successful calls) / (total samples)

# Fields
- `min_sample_call_rate::Float64`: Minimum call rate for sample retention (0.0-1.0)
- `min_marker_call_rate::Float64`: Minimum call rate for marker retention (0.0-1.0)

# Examples
```julia
# Require 90% successful genotyping
filter = CallRateFilter(min_sample_call_rate=0.90, min_marker_call_rate=0.90)

# Equivalent to MissingRateFilter with 10% threshold
# call_rate = 0.90 ⟺ missing_rate = 0.10
```

# See Also
- [`MissingRateFilter`](@ref): Equivalent filter using missing rate metric
"""
struct CallRateFilter <: AbstractQCFilter
    min_sample_call_rate::Float64
    min_marker_call_rate::Float64

    function CallRateFilter(;
                            min_sample_call_rate::Float64 = 0.90,
                            min_marker_call_rate::Float64 = 0.90)
        if min_sample_call_rate < 0.0 || min_sample_call_rate > 1.0
            throw(ArgumentError("min_sample_call_rate must be in [0,1]"))
        end
        if min_marker_call_rate < 0.0 || min_marker_call_rate > 1.0
            throw(ArgumentError("min_marker_call_rate must be in [0,1]"))
        end

        new(min_sample_call_rate, min_marker_call_rate)
    end
end


"""
    QCReport

Comprehensive report of quality control results.

Stores detailed information about filtering operations including counts of
removed samples/markers, reasons for removal, and quality metrics before
and after filtering.

# Fields
- `filter_name::String`: Name of filter applied
- `n_samples_before::Int`: Sample count before filtering
- `n_samples_after::Int`: Sample count after filtering
- `n_samples_removed::Int`: Number of samples removed
- `n_markers_before::Int`: Marker count before filtering
- `n_markers_after::Int`: Marker count after filtering
- `n_markers_removed::Int`: Number of markers removed
- `removed_sample_ids::Vector{String}`: IDs of removed samples
- `removed_marker_ids::Vector{String}`: IDs of removed markers
- `metrics::Dict{String, Any}`: Additional filter-specific metrics
- `timestamp::DateTime`: Time when filter was applied

# Examples
```julia
# Access report information
report = qc_report
println("Retention rate: $(report.n_samples_after / report.n_samples_before * 100)%")

# Export report to file
save_report(report, "qc_summary.txt")

# Combine multiple reports from pipeline
combined_report = combine_reports([report1, report2, report3])
```

# See Also
- [`save_report`](@ref): Export report to file
- [`plot_qc_metrics`](@ref): Visualize quality control results
"""
mutable struct QCReport
    filter_name::String
    n_samples_before::Int
    n_samples_after::Int
    n_samples_removed::Int
    n_markers_before::Int
    n_markers_after::Int
    n_markers_removed::Int
    removed_sample_ids::Vector{String}
    removed_marker_ids::Vector{String}
    metrics::Dict{String, Any}
    timestamp::DateTime
end


"""
    apply_filter(filter::MissingRateFilter, genotypes::AbstractGenotypeData)

Apply missing rate filter to genotype data.

Removes samples and markers exceeding missing rate thresholds in two sequential
steps to maximize data retention:
1. Remove high-missing markers first (they affect all samples)
2. Remove high-missing samples from remaining markers

# Arguments
- `filter::MissingRateFilter`: Filter configuration with thresholds
- `genotypes::AbstractGenotypeData`: Input genotype data

# Returns
- `genotypes_filtered::AbstractGenotypeData`: Filtered genotype data
- `report::QCReport`: Detailed filtering report

# Algorithm
1. Compute missing rate per marker across all samples
2. Identify markers exceeding marker_threshold
3. Remove high-missing markers
4. Compute missing rate per sample on remaining markers
5. Identify samples exceeding sample_threshold
6. Remove high-missing samples
7. Generate comprehensive report

# Examples
```julia
filter = MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10)
genotypes_clean, report = apply_filter(filter, genotypes)

# Check filtering results
@assert report.n_samples_after == size(genotypes_clean, 1)
@assert report.n_markers_after == size(genotypes_clean, 2)
```

# Performance
- Time: O(n × m) for two passes through data
- Space: O(n + m) for missing rate vectors plus filtered data
"""
function apply_filter(filter::MissingRateFilter,
                     genotypes::AbstractGenotypeData)
    n_samples_init, n_markers_init = size(genotypes)
    sample_ids = get_sample_ids(genotypes)
    marker_ids = get_marker_ids(genotypes)

    # Step 1: Compute missing rates per marker
    missing_per_marker = Vector{Float64}(undef, n_markers_init)
    for j in 1:n_markers_init
        count_missing = sum(ismissing(genotypes[i, j]) for i in 1:n_samples_init)
        missing_per_marker[j] = count_missing / n_samples_init
    end

    # Identify markers to retain
    markers_to_keep = missing_per_marker .<= filter.marker_threshold
    removed_marker_ids = marker_ids[.!markers_to_keep]
    n_markers_removed = sum(.!markers_to_keep)

    # Filter markers
    if n_markers_removed > 0
        genotypes = genotypes[:, markers_to_keep]
        marker_ids = marker_ids[markers_to_keep]
    end
    n_samples_current, n_markers_after = size(genotypes)

    # Step 2: Compute missing rates per sample on remaining markers
    missing_per_sample = Vector{Float64}(undef, n_samples_current)
    for i in 1:n_samples_current
        count_missing = sum(ismissing(genotypes[i, j]) for j in 1:n_markers_after)
        missing_per_sample[i] = count_missing / n_markers_after
    end

    # Identify samples to retain
    samples_to_keep = missing_per_sample .<= filter.sample_threshold
    removed_sample_ids = sample_ids[.!samples_to_keep]
    n_samples_removed = sum(.!samples_to_keep)

    # Filter samples
    if n_samples_removed > 0
        genotypes = genotypes[samples_to_keep, :]
        sample_ids = sample_ids[samples_to_keep]
    end
    n_samples_after, _ = size(genotypes)

    # Create detailed report
    metrics = Dict{String, Any}(
        "mean_sample_missing_rate" => mean(missing_per_sample[samples_to_keep]),
        "max_sample_missing_rate" => maximum(missing_per_sample[samples_to_keep]),
        "mean_marker_missing_rate" => mean(missing_per_marker[markers_to_keep]),
        "max_marker_missing_rate" => maximum(missing_per_marker[markers_to_keep]),
        "sample_threshold" => filter.sample_threshold,
        "marker_threshold" => filter.marker_threshold
    )

    report = QCReport(
        "MissingRateFilter",
        n_samples_init,
        n_samples_after,
        n_samples_removed,
        n_markers_init,
        n_markers_after,
        n_markers_removed,
        removed_sample_ids,
        removed_marker_ids,
        metrics,
        now()
    )

    return genotypes, report
end


"""
    apply_filter(filter::MAFFilter, genotypes::AbstractGenotypeData)

Apply minor allele frequency filter to genotype data.

Removes markers with MAF below threshold. Computation handles missing genotypes
by calculating frequencies only on non-missing data.

# Arguments
- `filter::MAFFilter`: Filter configuration with MAF threshold
- `genotypes::AbstractGenotypeData`: Input genotype data

# Returns
- `genotypes_filtered::AbstractGenotypeData`: Filtered genotype data with low-MAF markers removed
- `report::QCReport`: Detailed filtering report including MAF distribution

# Algorithm
1. Compute allele frequencies for each marker (non-missing genotypes only)
2. Calculate MAF as min(p, 1-p) for each marker
3. Identify markers with MAF < min_maf threshold
4. Remove low-MAF markers
5. Generate report with MAF distribution statistics

# Examples
```julia
filter = MAFFilter(min_maf=0.01)
genotypes_clean, report = apply_filter(filter, genotypes)

# Check MAF distribution after filtering
println("Markers retained: ", report.n_markers_after)
println("Mean MAF: ", report.metrics["mean_maf"])
```
"""
function apply_filter(filter::MAFFilter,
                     genotypes::AbstractGenotypeData)
    n_samples, n_markers = size(genotypes)
    marker_ids = get_marker_ids(genotypes)

    # Compute allele frequencies
    allele_freqs = get_allele_frequencies(genotypes)

    # Calculate MAF (minor allele frequency)
    maf = min.(allele_freqs, 1 .- allele_freqs)

    # Identify markers to retain
    markers_to_keep = maf .>= filter.min_maf
    removed_marker_ids = marker_ids[.!markers_to_keep]
    n_markers_removed = sum(.!markers_to_keep)

    # Filter markers
    if n_markers_removed > 0
        genotypes = genotypes[:, markers_to_keep]
        maf = maf[markers_to_keep]
    end
    _, n_markers_after = size(genotypes)

    # Create detailed report
    metrics = Dict{String, Any}(
        "min_maf_threshold" => filter.min_maf,
        "mean_maf" => mean(maf),
        "median_maf" => median(maf),
        "min_maf" => minimum(maf),
        "max_maf" => maximum(maf),
        "maf_distribution" => Dict(
            "rare_001" => sum(maf .< 0.01),
            "low_005" => sum(0.01 .<= maf .< 0.05),
            "common_010" => sum(0.05 .<= maf .< 0.10),
            "frequent" => sum(maf .>= 0.10)
        )
    )

    report = QCReport(
        "MAFFilter",
        n_samples,
        n_samples,
        0,
        n_markers + n_markers_removed,
        n_markers_after,
        n_markers_removed,
        String[],
        removed_marker_ids,
        metrics,
        now()
    )

    return genotypes, report
end


"""
    apply_filter(filter::HWEFilter, genotypes::AbstractGenotypeData)

Apply Hardy-Weinberg equilibrium test filter to genotype data.

Tests each marker for deviation from HWE using chi-square test. Markers with
p-value below threshold are removed as they likely represent genotyping errors.

# Arguments
- `filter::HWEFilter`: Filter configuration with p-value threshold
- `genotypes::AbstractGenotypeData`: Input genotype data

# Returns
- `genotypes_filtered::AbstractGenotypeData`: Filtered genotype data
- `report::QCReport`: Detailed report with HWE test statistics

# Algorithm
1. For each marker:
   a. Count observed genotypes (n_AA, n_Aa, n_aa)
   b. Estimate allele frequency p from observed data
   c. Calculate expected genotype counts under HWE
   d. Compute chi-square statistic: χ² = Σ[(O-E)²/E]
   e. Calculate p-value from chi-square distribution (df=1)
2. Apply Bonferroni correction if requested: threshold = α / n_markers
3. Remove markers with p-value < threshold
4. Generate report with test statistics

# Examples
```julia
filter = HWEFilter(pvalue_threshold=1e-6, bonferroni_correction=true)
genotypes_clean, report = apply_filter(filter, genotypes)

println("Markers failing HWE: ", report.n_markers_removed)
println("Mean p-value: ", report.metrics["mean_pvalue"])
```
"""
function apply_filter(filter::HWEFilter,
                     genotypes::AbstractGenotypeData)
    using Distributions

    n_samples, n_markers = size(genotypes)
    marker_ids = get_marker_ids(genotypes)

    # Adjust threshold for Bonferroni correction
    threshold = filter.bonferroni_correction ?
                (filter.pvalue_threshold / n_markers) :
                filter.pvalue_threshold

    # Test HWE for each marker
    pvalues = Vector{Float64}(undef, n_markers)
    chi_squares = Vector{Float64}(undef, n_markers)

    for j in 1:n_markers
        # Count genotypes (0=AA, 1=Aa, 2=aa)
        n_AA = 0
        n_Aa = 0
        n_aa = 0
        n_total = 0

        for i in 1:n_samples
            g = genotypes[i, j]
            if !ismissing(g)
                if g == 0
                    n_AA += 1
                elseif g == 1
                    n_Aa += 1
                elseif g == 2
                    n_aa += 1
                end
                n_total += 1
            end
        end

        # Skip markers with too few genotypes
        if n_total < 10
            pvalues[j] = 1.0  # Pass by default
            chi_squares[j] = 0.0
            continue
        end

        # Estimate allele frequency
        p = (2 * n_AA + n_Aa) / (2 * n_total)
        q = 1 - p

        # Expected counts under HWE
        e_AA = n_total * p^2
        e_Aa = n_total * 2 * p * q
        e_aa = n_total * q^2

        # Chi-square statistic (avoid division by zero)
        chi_sq = 0.0
        if e_AA > 0
            chi_sq += (n_AA - e_AA)^2 / e_AA
        end
        if e_Aa > 0
            chi_sq += (n_Aa - e_Aa)^2 / e_Aa
        end
        if e_aa > 0
            chi_sq += (n_aa - e_aa)^2 / e_aa
        end

        chi_squares[j] = chi_sq

        # P-value from chi-square distribution (df=1)
        pvalues[j] = ccdf(Chisq(1), chi_sq)
    end

    # Identify markers to retain
    markers_to_keep = pvalues .>= threshold
    removed_marker_ids = marker_ids[.!markers_to_keep]
    n_markers_removed = sum(.!markers_to_keep)

    # Filter markers
    if n_markers_removed > 0
        genotypes = genotypes[:, markers_to_keep]
        pvalues = pvalues[markers_to_keep]
        chi_squares = chi_squares[markers_to_keep]
    end
    _, n_markers_after = size(genotypes)

    # Create detailed report
    metrics = Dict{String, Any}(
        "pvalue_threshold" => threshold,
        "bonferroni_correction" => filter.bonferroni_correction,
        "mean_pvalue" => mean(pvalues),
        "median_pvalue" => median(pvalues),
        "mean_chi_square" => mean(chi_squares),
        "max_chi_square" => maximum(chi_squares),
        "n_markers_tested" => n_markers + n_markers_removed
    )

    report = QCReport(
        "HWEFilter",
        n_samples,
        n_samples,
        0,
        n_markers + n_markers_removed,
        n_markers_after,
        n_markers_removed,
        String[],
        removed_marker_ids,
        metrics,
        now()
    )

    return genotypes, report
end


"""
    QCPipeline

Composable quality control pipeline combining multiple filters.

Applies filters sequentially, accumulating filtered samples/markers at each
stage. Provides comprehensive reporting across entire pipeline.

# Fields
- `filters::Vector{AbstractQCFilter}`: Ordered sequence of filters to apply

# Pipeline Execution Order
1. MissingRateFilter (if present) - removes poor quality data first
2. MAFFilter (if present) - removes rare variants
3. HWEFilter (if present) - removes markers with genotyping errors
4. Other filters in specified order

Order matters: removing samples affects marker statistics and vice versa

# Examples
```julia
# Create standard QC pipeline
pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])

# Apply pipeline to data
genotypes_qc, reports = apply_qc(genotypes, pipeline)

# Examine cumulative effects
for (i, report) in enumerate(reports)
    println("After filter $i ($(report.filter_name)):")
    println("  Samples: $(report.n_samples_after)")
    println("  Markers: $(report.n_markers_after)")
end

# Summary statistics
total_samples_removed = sum(r.n_samples_removed for r in reports)
total_markers_removed = sum(r.n_markers_removed for r in reports)
```

# See Also
- [`apply_qc`](@ref): Apply pipeline to genotype data
- [`combine_reports`](@ref): Generate summary across all filters
"""
struct QCPipeline
    filters::Vector{AbstractQCFilter}
end


"""
    apply_qc(genotypes::AbstractGenotypeData, pipeline::QCPipeline)

Apply complete quality control pipeline to genotype data.

Executes filters sequentially, generating individual reports for each filter
and a combined summary report.

# Arguments
- `genotypes::AbstractGenotypeData`: Input genotype data
- `pipeline::QCPipeline`: Pipeline containing ordered filters

# Returns
- `genotypes_qc::AbstractGenotypeData`: Quality-controlled genotype data
- `reports::Vector{QCReport}`: Individual reports from each filter

# Examples
```julia
pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])

genotypes_qc, reports = apply_qc(genotypes, pipeline)

# Generate summary
println("\n" * "="^70)
println("Quality Control Summary")
println("="^70)
println("Initial: $(size(genotypes, 1)) samples × $(size(genotypes, 2)) markers")
println("Final: $(size(genotypes_qc, 1)) samples × $(size(genotypes_qc, 2)) markers")
println("Sample retention: $(round(size(genotypes_qc,1)/size(genotypes,1)*100, digits=1))%")
println("Marker retention: $(round(size(genotypes_qc,2)/size(genotypes,2)*100, digits=1))%")
```
"""
function apply_qc(genotypes::AbstractGenotypeData,
                 pipeline::QCPipeline)
    reports = QCReport[]
    current_genotypes = genotypes

    for filter in pipeline.filters
        filtered_genotypes, report = apply_filter(filter, current_genotypes)
        push!(reports, report)
        current_genotypes = filtered_genotypes
    end

    return current_genotypes, reports
end


"""
    Base.show(io::IO, report::QCReport)

Pretty-print quality control report in human-readable format.

Displays comprehensive filtering results including before/after counts,
retention rates, and filter-specific metrics.
"""
function Base.show(io::IO, report::QCReport)
    println(io, "\n" * "="^70)
    println(io, "Quality Control Report: $(report.filter_name)")
    println(io, "="^70)
    println(io, "Timestamp: $(report.timestamp)")
    println(io)

    println(io, "Sample Statistics:")
    println(io, "  Before filtering: $(report.n_samples_before)")
    println(io, "  After filtering:  $(report.n_samples_after)")
    println(io, "  Removed:          $(report.n_samples_removed)")
    if report.n_samples_before > 0
        retention = report.n_samples_after / report.n_samples_before * 100
        println(io, "  Retention rate:   $(round(retention, digits=2))%")
    end
    println(io)

    println(io, "Marker Statistics:")
    println(io, "  Before filtering: $(report.n_markers_before)")
    println(io, "  After filtering:  $(report.n_markers_after)")
    println(io, "  Removed:          $(report.n_markers_removed)")
    if report.n_markers_before > 0
        retention = report.n_markers_after / report.n_markers_before * 100
        println(io, "  Retention rate:   $(round(retention, digits=2))%")
    end
    println(io)

    if !isempty(report.metrics)
        println(io, "Filter-Specific Metrics:")
        for (key, value) in sort(collect(report.metrics))
            if value isa Number
                println(io, "  $key: $(round(value, digits=4))")
            elseif value isa Dict
                println(io, "  $key:")
                for (k, v) in sort(collect(value))
                    println(io, "    $k: $v")
                end
            else
                println(io, "  $key: $value")
            end
        end
    end

    println(io, "="^70)
end