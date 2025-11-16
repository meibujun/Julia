"""
Quality control reporting and visualization.
"""

"""
    QCReport

Comprehensive quality control report for genotype data.

# Fields
- `n_samples::Int`: Number of samples
- `n_markers::Int`: Number of markers
- `overall_missing_rate::Float64`: Overall missing rate
- `sample_missing_stats::NamedTuple`: Sample missing rate statistics
- `marker_missing_stats::NamedTuple`: Marker missing rate statistics
- `maf_stats::NamedTuple`: MAF statistics
- `heterozygosity_stats::NamedTuple`: Heterozygosity statistics
- `hwe_stats::NamedTuple`: HWE test statistics
- `duplicates::Vector{Tuple{String,String,Float64}}`: Potential duplicate pairs
- `outlier_samples::Vector{String}`: Samples with unusual characteristics
"""
struct QCReport
    n_samples::Int
    n_markers::Int
    overall_missing_rate::Float64
    sample_missing_stats::NamedTuple
    marker_missing_stats::NamedTuple
    maf_stats::NamedTuple
    heterozygosity_stats::NamedTuple
    hwe_stats::Union{NamedTuple, Nothing}
    duplicates::Vector{Tuple{String,String,Float64}}
    outlier_samples::Vector{String}
end

"""
    qc_report(geno::CompactGenotypes; check_hwe::Bool=true, check_duplicates::Bool=true) -> QCReport

Generate comprehensive quality control report.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `check_hwe::Bool`: Whether to run HWE tests (can be slow) (default: true)
- `check_duplicates::Bool`: Whether to check for duplicates (default: true)

# Returns
QCReport object

# Example
```julia
report = qc_report(geno)
println(report)

# Save to file
open("qc_report.txt", "w") do io
    write(io, string(report))
end
```
"""
function qc_report(geno::CompactGenotypes;
                  check_hwe::Bool = true,
                  check_duplicates::Bool = true)

    println("Generating QC report...")

    # Basic statistics
    n_samples_val = n_samples(geno)
    n_markers_val = n_markers(geno)
    overall_missing = missing_rate(geno; dim=0)

    # Sample missing rates
    println("  - Computing sample statistics...")
    sample_missing = missing_rate(geno; dim=1)
    sample_missing_stats = (
        mean = mean(sample_missing),
        median = median(sample_missing),
        min = minimum(sample_missing),
        max = maximum(sample_missing),
        std = std(sample_missing)
    )

    # Marker missing rates
    println("  - Computing marker statistics...")
    marker_missing = missing_rate(geno; dim=2)
    marker_missing_stats = (
        mean = mean(marker_missing),
        median = median(marker_missing),
        min = minimum(marker_missing),
        max = maximum(marker_missing),
        std = std(marker_missing)
    )

    # MAF statistics
    println("  - Computing MAF statistics...")
    maf = minor_allele_frequency(geno)
    maf_stats = (
        mean = mean(maf),
        median = median(maf),
        min = minimum(maf),
        max = maximum(maf),
        n_rare = count(maf .< 0.01),
        n_common = count(maf .>= 0.05)
    )

    # Heterozygosity statistics
    println("  - Computing heterozygosity statistics...")
    sample_het = heterozygosity_rate(geno; dim=1)
    marker_het = heterozygosity_rate(geno; dim=2)
    exp_het = expected_heterozygosity(geno)
    F_coeffs = inbreeding_coefficient(geno; dim=1)

    heterozygosity_stats = (
        overall = heterozygosity_rate(geno; dim=0),
        sample_mean = mean(sample_het),
        sample_std = std(sample_het),
        marker_mean = mean(marker_het),
        expected_mean = mean(exp_het),
        F_mean = mean(F_coeffs),
        F_std = std(F_coeffs)
    )

    # HWE statistics (optional, can be slow)
    hwe_stats = nothing
    if check_hwe
        println("  - Running HWE tests (this may take a while)...")
        hwe_pvalues = Float64[]

        for j in 1:n_markers_val
            # Count genotypes
            n0, n1, n2 = 0, 0, 0
            for i in 1:n_samples_val
                if !ismissing(geno, i, j)
                    val = geno[i, j]
                    if val == 0
                        n0 += 1
                    elseif val == 1
                        n1 += 1
                    else
                        n2 += 1
                    end
                end
            end

            pval = hardy_weinberg_test(n0, n1, n2)
            push!(hwe_pvalues, pval)
        end

        hwe_stats = (
            mean_pvalue = mean(hwe_pvalues),
            median_pvalue = median(hwe_pvalues),
            n_fail_1e6 = count(hwe_pvalues .< 1e-6),
            n_fail_1e4 = count(hwe_pvalues .< 1e-4),
            min_pvalue = minimum(hwe_pvalues)
        )
    end

    # Check for duplicates (optional)
    duplicates = Tuple{String,String,Float64}[]
    if check_duplicates
        println("  - Checking for duplicate samples...")
        dup_indices = identify_duplicates(geno; threshold=0.95)

        sample_id_list = sample_ids(geno)
        for (i, j, cor_val) in dup_indices
            push!(duplicates, (sample_id_list[i], sample_id_list[j], cor_val))
        end
    end

    # Identify outlier samples
    println("  - Identifying outlier samples...")
    outliers = String[]

    # High missing rate
    high_missing_idx = findall(sample_missing .> 0.1)
    for idx in high_missing_idx
        push!(outliers, "$(sample_ids(geno)[idx]) (high missing rate: $(@sprintf("%.1f%%", sample_missing[idx]*100)))")
    end

    # Extreme heterozygosity
    mean_het = mean(sample_het)
    std_het = std(sample_het)
    extreme_het_idx = findall(abs.(sample_het .- mean_het) .> 3 * std_het)
    for idx in extreme_het_idx
        push!(outliers, "$(sample_ids(geno)[idx]) (extreme heterozygosity: $(@sprintf("%.3f", sample_het[idx])))")
    end

    # Extreme inbreeding coefficient
    extreme_F_idx = findall(abs.(F_coeffs) .> 0.2)
    for idx in extreme_F_idx
        push!(outliers, "$(sample_ids(geno)[idx]) (extreme F: $(@sprintf("%.3f", F_coeffs[idx])))")
    end

    println("✓ QC report generated")

    return QCReport(
        n_samples_val,
        n_markers_val,
        overall_missing,
        sample_missing_stats,
        marker_missing_stats,
        maf_stats,
        heterozygosity_stats,
        hwe_stats,
        duplicates,
        unique(outliers)
    )
end

"""
    Base.show(io::IO, report::QCReport)

Display QC report in human-readable format.
"""
function Base.show(io::IO, report::QCReport)
    println(io, "="^80)
    println(io, "GenomicPro2 Quality Control Report")
    println(io, "="^80)

    println(io, "\n📊 Dataset Overview")
    println(io, "─"^80)
    println(io, "  Samples: $(report.n_samples)")
    println(io, "  Markers: $(report.n_markers)")
    println(io, "  Overall missing rate: $(@sprintf("%.2f%%", report.overall_missing_rate * 100))")

    println(io, "\n👤 Sample Statistics")
    println(io, "─"^80)
    println(io, "  Missing rate (per sample):")
    println(io, "    Mean:   $(@sprintf("%.2f%%", report.sample_missing_stats.mean * 100))")
    println(io, "    Median: $(@sprintf("%.2f%%", report.sample_missing_stats.median * 100))")
    println(io, "    Range:  $(@sprintf("%.2f%%", report.sample_missing_stats.min * 100)) - $(@sprintf("%.2f%%", report.sample_missing_stats.max * 100))")
    println(io, "    SD:     $(@sprintf("%.2f%%", report.sample_missing_stats.std * 100))")

    println(io, "\n🧬 Marker Statistics")
    println(io, "─"^80)
    println(io, "  Missing rate (per marker):")
    println(io, "    Mean:   $(@sprintf("%.2f%%", report.marker_missing_stats.mean * 100))")
    println(io, "    Median: $(@sprintf("%.2f%%", report.marker_missing_stats.median * 100))")
    println(io, "    Range:  $(@sprintf("%.2f%%", report.marker_missing_stats.min * 100)) - $(@sprintf("%.2f%%", report.marker_missing_stats.max * 100))")

    println(io, "\n  Minor allele frequency:")
    println(io, "    Mean:   $(@sprintf("%.4f", report.maf_stats.mean))")
    println(io, "    Median: $(@sprintf("%.4f", report.maf_stats.median))")
    println(io, "    Range:  $(@sprintf("%.4f", report.maf_stats.min)) - $(@sprintf("%.4f", report.maf_stats.max))")
    println(io, "    Rare markers (MAF < 0.01):  $(report.maf_stats.n_rare) ($(@sprintf("%.1f%%", 100 * report.maf_stats.n_rare / report.n_markers)))")
    println(io, "    Common markers (MAF ≥ 0.05): $(report.maf_stats.n_common) ($(@sprintf("%.1f%%", 100 * report.maf_stats.n_common / report.n_markers)))")

    println(io, "\n🧪 Heterozygosity")
    println(io, "─"^80)
    println(io, "  Overall heterozygosity: $(@sprintf("%.4f", report.heterozygosity_stats.overall))")
    println(io, "  Expected (under HWE):   $(@sprintf("%.4f", report.heterozygosity_stats.expected_mean))")
    println(io, "  Per sample (mean ± SD): $(@sprintf("%.4f ± %.4f", report.heterozygosity_stats.sample_mean, report.heterozygosity_stats.sample_std))")
    println(io, "  Inbreeding coefficient (F):")
    println(io, "    Mean: $(@sprintf("%.4f", report.heterozygosity_stats.F_mean))")
    println(io, "    SD:   $(@sprintf("%.4f", report.heterozygosity_stats.F_std))")

    if report.hwe_stats !== nothing
        println(io, "\n⚖️  Hardy-Weinberg Equilibrium")
        println(io, "─"^80)
        println(io, "  Mean p-value:    $(@sprintf("%.2e", report.hwe_stats.mean_pvalue))")
        println(io, "  Median p-value:  $(@sprintf("%.2e", report.hwe_stats.median_pvalue))")
        println(io, "  Min p-value:     $(@sprintf("%.2e", report.hwe_stats.min_pvalue))")
        println(io, "  Markers failing p < 1e-6: $(report.hwe_stats.n_fail_1e6) ($(@sprintf("%.1f%%", 100 * report.hwe_stats.n_fail_1e6 / report.n_markers)))")
        println(io, "  Markers failing p < 1e-4: $(report.hwe_stats.n_fail_1e4) ($(@sprintf("%.1f%%", 100 * report.hwe_stats.n_fail_1e4 / report.n_markers)))")
    end

    if !isempty(report.duplicates)
        println(io, "\n🔍 Potential Duplicates")
        println(io, "─"^80)
        println(io, "  Found $(length(report.duplicates)) potential duplicate pairs:")
        for (id1, id2, cor_val) in report.duplicates[1:min(10, length(report.duplicates))]
            println(io, "    $id1 ↔ $id2 (r = $(@sprintf("%.4f", cor_val)))")
        end
        if length(report.duplicates) > 10
            println(io, "    ... and $(length(report.duplicates) - 10) more")
        end
    end

    if !isempty(report.outlier_samples)
        println(io, "\n⚠️  Outlier Samples")
        println(io, "─"^80)
        println(io, "  Found $(length(report.outlier_samples)) outlier samples:")
        for outlier in report.outlier_samples[1:min(10, length(report.outlier_samples))]
            println(io, "    $outlier")
        end
        if length(report.outlier_samples) > 10
            println(io, "    ... and $(length(report.outlier_samples) - 10) more")
        end
    end

    println(io, "\n" * "="^80)
end
