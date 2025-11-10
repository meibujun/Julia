# examples/02_quality_control.jl

"""
Example 2: Comprehensive Quality Control Workflow

This example demonstrates:
1. Loading raw genotype data
2. Assessing initial data quality
3. Applying multi-stage QC pipeline
4. Visualizing QC results
5. Exporting cleaned data

Author: GenomicPro Development Team
Date: 2025
Julia Version: 1.12.1
"""

using GenomicPro
using Statistics, Printf, Plots

println("="^70)
println("GenomicPro.jl Example 2: Quality Control Pipeline")
println("="^70)
println()

# ============================================================================
# Step 1: Load and Examine Raw Data
# ============================================================================
println("Step 1: Loading raw genotype data...")

# Simulate realistic dataset with quality issues
Random.seed!(123)
n_samples = 5000
n_markers = 100000

println("  Simulating $n_samples samples × $n_markers markers")
println("  Including realistic quality issues:")
println("    - Variable missing rates (0-30%)")
println("    - Rare variants (MAF < 0.01)")
println("    - HWE violations from genotyping errors")
println()

# Simulate genotypes with quality issues
genotypes_raw = Matrix{Union{Int, Missing}}(undef, n_samples, n_markers)

for j in 1:n_markers
    # Simulate allele frequency from beta distribution
    p = rand(Beta(0.5, 0.5))

    # Introduce ~5% markers with HWE violations (genotyping errors)
    hwe_violation = rand() < 0.05
    het_excess = hwe_violation ? 0.3 : 0.0  # Excess heterozygosity

    for i in 1:n_samples
        # Variable missing rate: most markers <5%, some problematic markers ~20-30%
        missing_prob = j < n_markers * 0.95 ? 0.02 : 0.25

        if rand() < missing_prob
            genotypes_raw[i, j] = missing
        else
            # Sample genotype with potential HWE violation
            r = rand()
            if r < (1-p)^2 - het_excess
                genotypes_raw[i, j] = 0  # AA
            elseif r < (1-p)^2 + 2*p*(1-p) + het_excess
                genotypes_raw[i, j] = 1  # Aa (excess heterozygotes)
            else
                genotypes_raw[i, j] = 2  # aa
            end
        end
    end
end

# Add some samples with high missing rate (simulate poor DNA quality)
n_poor_samples = 50
for i in 1:n_poor_samples
    for j in 1:n_markers
        if rand() < 0.25  # 25% missing for these samples
            genotypes_raw[i, j] = missing
        end
    end
end

println("  Raw data generated with quality issues embedded")
println()

# Convert to GenomicPro format
sample_ids = ["Sample_" * lpad(i, 5, '0') for i in 1:n_samples]
marker_ids = ["rs" * string(i) for i in 1:n_markers]

genotypes = TwoBitGenotypes(genotypes_raw,
                            sample_ids=sample_ids,
                            marker_ids=marker_ids)

println("  Data loaded: $(size(genotypes, 1)) samples × $(size(genotypes, 2)) markers")
println("  Memory usage: $(round(Base.summarysize(genotypes) / 1e9, digits=3)) GB")
println()

# ============================================================================
# Step 2: Initial Data Quality Assessment
# ============================================================================
println("Step 2: Assessing initial data quality...")
println()

# Compute missing rates
println("  Computing missing rates...")
missing_per_sample = Vector{Float64}(undef, n_samples)
for i in 1:n_samples
    count_missing = sum(ismissing(genotypes[i, j]) for j in 1:n_markers)
    missing_per_sample[i] = count_missing / n_markers
end

missing_per_marker = Vector{Float64}(undef, n_markers)
for j in 1:n_markers
    count_missing = sum(ismissing(genotypes[i, j]) for i in 1:n_samples)
    missing_per_marker[j] = count_missing / n_samples
end

println("  Sample missing rate statistics:")
println("    Mean:   $(round(mean(missing_per_sample)*100, digits=2))%")
println("    Median: $(round(median(missing_per_sample)*100, digits=2))%")
println("    Max:    $(round(maximum(missing_per_sample)*100, digits=2))%")
println("    Samples >10% missing: $(sum(missing_per_sample .> 0.10))")
println()

println("  Marker missing rate statistics:")
println("    Mean:   $(round(mean(missing_per_marker)*100, digits=2))%")
println("    Median: $(round(median(missing_per_marker)*100, digits=2))%")
println("    Max:    $(round(maximum(missing_per_marker)*100, digits=2))%")
println("    Markers >10% missing: $(sum(missing_per_marker .> 0.10))")
println()

# Compute allele frequencies and MAF
println("  Computing allele frequencies...")
allele_freqs = get_allele_frequencies(genotypes)
maf = min.(allele_freqs, 1 .- allele_freqs)

println("  MAF distribution:")
println("    Rare (MAF < 0.01):     $(sum(maf .< 0.01)) markers ($(round(sum(maf .< 0.01)/n_markers*100, digits=1))%)")
println("    Low (0.01 ≤ MAF < 0.05): $(sum(0.01 .<= maf .< 0.05)) markers ($(round(sum(0.01 .<= maf .< 0.05)/n_markers*100, digits=1))%)")
println("    Common (MAF ≥ 0.05):   $(sum(maf .>= 0.05)) markers ($(round(sum(maf .>= 0.05)/n_markers*100, digits=1))%)")
println()

# ============================================================================
# Step 3: Apply Quality Control Pipeline
# ============================================================================
println("Step 3: Applying comprehensive QC pipeline...")
println()

# Define QC pipeline with standard thresholds
qc_pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6, bonferroni_correction=true)
])

println("  Pipeline configuration:")
println("    1. MissingRateFilter: sample ≤ 10%, marker ≤ 10%")
println("    2. MAFFilter: MAF ≥ 1%")
println("    3. HWEFilter: p-value ≥ 1e-6 (Bonferroni corrected)")
println()

# Apply pipeline
println("  Executing pipeline...")
genotypes_qc, reports = apply_qc(genotypes, qc_pipeline)
println()

# Display individual filter reports
for report in reports
    println(report)
end

# ============================================================================
# Step 4: Summary Statistics
# ============================================================================
println("\n" * "="^70)
println("Overall Quality Control Summary")
println("="^70)
println()

println("Data Retention:")
println("  Samples: $(size(genotypes, 1)) → $(size(genotypes_qc, 1)) ($(round(size(genotypes_qc,1)/size(genotypes,1)*100, digits=1))% retained)")
println("  Markers: $(size(genotypes, 2)) → $(size(genotypes_qc, 2)) ($(round(size(genotypes_qc,2)/size(genotypes,2)*100, digits=1))% retained)")
println()

# Cumulative filtering
total_samples_removed = sum(r.n_samples_removed for r in reports)
total_markers_removed = sum(r.n_markers_removed for r in reports)

println("Cumulative Removal:")
println("  Samples: $total_samples_removed")
println("  Markers: $total_markers_removed")
println()

# Breakdown by filter
println("Removal Breakdown by Filter:")
for report in reports
    println("  $(report.filter_name):")
    println("    Samples: $(report.n_samples_removed)")
    println("    Markers: $(report.n_markers_removed)")
end
println()

# Final data quality
println("Final Data Quality:")
missing_final = [ismissing(genotypes_qc[i,j]) for i in 1:size(genotypes_qc,1), j in 1:size(genotypes_qc,2)]
println("  Overall missing rate: $(round(mean(missing_final)*100, digits=2))%")

afs_final = get_allele_frequencies(genotypes_qc)
maf_final = min.(afs_final, 1 .- afs_final)
println("  Mean MAF: $(round(mean(maf_final), digits=3))")
println("  Median MAF: $(round(median(maf_final), digits=3))")
println()

println("="^70)
println("Quality Control Completed Successfully!")
println("="^70)
println()

println("Next steps:")
println("  - Use cleaned data for genomic prediction")
println("  - Perform imputation if needed")
println("  - Run downstream analyses")
println()

# ============================================================================
# Step 5: Visualization (Optional)
# ============================================================================
println("Step 5: Generating QC visualizations...")

# Plot missing rate distributions
p1 = histogram(missing_per_sample[missing_per_sample .<= 0.10] .* 100,
              xlabel="Sample Missing Rate (%)",
              ylabel="Frequency",
              title="Sample Missing Rates (Before QC)",
              legend=false,
              bins=50)

# Plot MAF distribution
p2 = histogram(maf[maf .>= 0.01],
              xlabel="Minor Allele Frequency",
              ylabel="Frequency",
              title="MAF Distribution (After Filtering)",
              legend=false,
              bins=50)

# Combine plots
plot(p1, p2, layout=(1,2), size=(1200,400))
savefig("qc_summary.png")
println("  Saved visualization: qc_summary.png")
println()