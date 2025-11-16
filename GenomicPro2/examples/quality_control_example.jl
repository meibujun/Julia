"""
Quality Control Workflow Example

This example demonstrates comprehensive quality control for genomic data:
1. Data loading
2. Initial QC report
3. Filter application
4. Post-QC report
5. Outlier identification
6. Duplicate detection

Run with: julia --project examples/quality_control_example.jl
"""

using GenomicPro2
using Statistics
using Printf

println("="^80)
println("GenomicPro2 Quality Control Example")
println("="^80)

# ============================================================================
# 1. Generate Test Data with Known QC Issues
# ============================================================================

println("\n📊 Step 1: Generating test data with QC issues...")

n_samples = 300
n_markers = 2000

# Generate base genotype data
println("  - Creating $n_samples samples × $n_markers markers")
geno_data = rand(0:2, n_samples, n_markers)

# Add various QC issues

# Issue 1: Some markers with high missing rate
println("  - Adding markers with high missing rate...")
high_missing_markers = rand(1:n_markers, 20)
for j in high_missing_markers
    missing_idx = rand(1:n_samples, round(Int, 0.15 * n_samples))
    for i in missing_idx
        geno_data[i, j] = missing
    end
end

# Issue 2: Some samples with high missing rate
println("  - Adding samples with high missing rate...")
high_missing_samples = rand(1:n_samples, 10)
for i in high_missing_samples
    missing_idx = rand(1:n_markers, round(Int, 0.15 * n_markers))
    for j in missing_idx
        geno_data[i, j] = missing
    end
end

# Issue 3: Low MAF markers
println("  - Adding low MAF markers...")
low_maf_markers = rand(1:n_markers, 100)
for j in low_maf_markers
    # Set most genotypes to 0 (rare variant)
    for i in rand(1:n_samples, round(Int, 0.95 * n_samples))
        geno_data[i, j] = 0
    end
end

# Issue 4: Create a duplicate sample (with some noise)
println("  - Adding a duplicate sample...")
dup_idx = rand(1:n_samples)
dup_data = copy(geno_data[dup_idx, :])
# Add some noise (5% different)
noise_idx = rand(1:n_markers, round(Int, 0.05 * n_markers))
for j in noise_idx
    dup_data[j] = rand(0:2)
end
# Add as new sample
geno_data = vcat(geno_data, reshape(dup_data, 1, n_markers))
n_samples += 1

sample_ids = ["Sample_$i" for i in 1:n_samples]
marker_ids = ["SNP_$i" for i in 1:n_markers]

geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

println("\n✓ Test data created with:")
println("    - $(length(high_missing_markers)) markers with high missing rate")
println("    - $(length(high_missing_samples)) samples with high missing rate")
println("    - ~$(length(low_maf_markers)) low MAF markers")
println("    - 1 duplicate sample pair")

# ============================================================================
# 2. Initial QC Report
# ============================================================================

println("\n" * "="^80)
println("Step 2: Generating initial QC report...")
println("="^80)

println("\nInitial Data Summary:")
println("  Samples: $(n_samples(geno))")
println("  Markers: $(n_markers(geno))")
println("  Missing rate: $(@sprintf("%.2f%%", missing_rate(geno) * 100))")

# Generate comprehensive QC report (skip HWE for speed)
initial_report = qc_report(geno; check_hwe=false, check_duplicates=true)

# Display report
println(initial_report)

# Save report to file
open("initial_qc_report.txt", "w") do io
    write(io, string(initial_report))
end
println("\n✓ Initial QC report saved to: initial_qc_report.txt")

# ============================================================================
# 3. Identify Specific Issues
# ============================================================================

println("\n" * "="^80)
println("Step 3: Identifying specific QC issues...")
println("="^80)

# Check heterozygosity outliers
println("\n🔍 Heterozygosity Analysis:")
sample_het = heterozygosity_rate(geno; dim=1)
mean_het = mean(sample_het)
std_het = std(sample_het)

println("  Mean heterozygosity: $(@sprintf("%.4f ± %.4f", mean_het, std_het))")

outlier_het_idx = findall(abs.(sample_het .- mean_het) .> 3 * std_het)
if !isempty(outlier_het_idx)
    println("  ⚠ Found $(length(outlier_het_idx)) heterozygosity outliers:")
    for idx in outlier_het_idx[1:min(5, length(outlier_het_idx))]
        println("    $(sample_ids[idx]): $(@sprintf("%.4f", sample_het[idx]))")
    end
end

# Check inbreeding coefficients
println("\n🔍 Inbreeding Analysis:")
F_coeffs = inbreeding_coefficient(geno)
println("  Mean F coefficient: $(@sprintf("%.4f ± %.4f", mean(F_coeffs), std(F_coeffs)))")

extreme_F_idx = findall(abs.(F_coeffs) .> 0.15)
if !isempty(extreme_F_idx)
    println("  ⚠ Found $(length(extreme_F_idx)) samples with extreme F:")
    for idx in extreme_F_idx[1:min(5, length(extreme_F_idx))]
        println("    $(sample_ids[idx]): F = $(@sprintf("%.4f", F_coeffs[idx]))")
    end
end

# Check for duplicates
println("\n🔍 Duplicate Detection:")
duplicates = identify_duplicates(geno; threshold=0.90)

if !isempty(duplicates)
    println("  ⚠ Found $(length(duplicates)) potential duplicate pairs:")
    for (i, j, cor_val) in duplicates[1:min(5, length(duplicates))]
        println("    $(sample_ids[i]) ↔ $(sample_ids[j]): r = $(@sprintf("%.4f", cor_val))")
    end
end

# ============================================================================
# 4. Apply Quality Control Filters
# ============================================================================

println("\n" * "="^80)
println("Step 4: Applying quality control filters...")
println("="^80)

# Apply comprehensive QC
geno_qc = quality_control(geno;
    min_maf = 0.01,
    max_missing_per_marker = 0.1,
    max_missing_per_sample = 0.1,
    hwe_pvalue = 1e-6,
    apply_hwe = false,  # Skip for speed
    verbose = true
)

# ============================================================================
# 5. Post-QC Report
# ============================================================================

println("\n" * "="^80)
println("Step 5: Post-QC assessment...")
println("="^80)

println("\n📊 Data Reduction Summary:")
println("  Samples: $(n_samples(geno)) → $(n_samples(geno_qc)) ($(@sprintf("%.1f%%", 100 * n_samples(geno_qc) / n_samples(geno))) retained)")
println("  Markers: $(n_markers(geno)) → $(n_markers(geno_qc)) ($(@sprintf("%.1f%%", 100 * n_markers(geno_qc) / n_markers(geno))) retained)")

println("\n📉 Quality Metrics Improvement:")
println("  Missing rate: $(@sprintf("%.2f%%", missing_rate(geno) * 100)) → $(@sprintf("%.2f%%", missing_rate(geno_qc) * 100))")

# MAF distribution before and after
maf_before = minor_allele_frequency(geno)
maf_after = minor_allele_frequency(geno_qc)

println("\n  MAF distribution:")
println("    Before QC: mean = $(@sprintf("%.4f", mean(maf_before))), rare (< 0.01) = $(count(maf_before .< 0.01))")
println("    After QC:  mean = $(@sprintf("%.4f", mean(maf_after))), rare (< 0.01) = $(count(maf_after .< 0.01))")

# Generate post-QC report
println("\nGenerating post-QC report...")
post_report = qc_report(geno_qc; check_hwe=false, check_duplicates=false)

open("post_qc_report.txt", "w") do io
    write(io, string(post_report))
end
println("✓ Post-QC report saved to: post_qc_report.txt")

# ============================================================================
# 6. Verify Data Quality
# ============================================================================

println("\n" * "="^80)
println("Step 6: Final quality verification...")
println("="^80)

# Validate the QC'd data
validation = validate(geno_qc)

if is_valid(validation)
    println("✓ QC'd data passes all validation checks")
else
    println("✗ QC'd data has validation issues:")
    for error in validation.errors
        println("  - $error")
    end
end

# Check that filters were effective
println("\n📋 Filter Effectiveness:")

# All markers should pass MAF threshold
maf_qc = minor_allele_frequency(geno_qc)
@assert all(maf_qc .>= 0.01) "Some markers below MAF threshold!"
println("  ✓ All markers pass MAF ≥ 0.01: $(all(maf_qc .>= 0.01))")

# All samples should pass missing rate threshold
sample_missing_qc = missing_rate(geno_qc; dim=1)
@assert all(sample_missing_qc .<= 0.1) "Some samples above missing threshold!"
println("  ✓ All samples pass missing ≤ 10%: $(all(sample_missing_qc .<= 0.1))")

# All markers should pass missing rate threshold
marker_missing_qc = missing_rate(geno_qc; dim=2)
@assert all(marker_missing_qc .<= 0.1) "Some markers above missing threshold!"
println("  ✓ All markers pass missing ≤ 10%: $(all(marker_missing_qc .<= 0.1))")

# ============================================================================
# 7. Summary Statistics
# ============================================================================

println("\n" * "="^80)
println("Final Summary")
println("="^80)

println("\n✅ Quality Control Complete!")
println("\nFinal Dataset:")
println("  Samples: $(n_samples(geno_qc))")
println("  Markers: $(n_markers(geno_qc))")
println("  Missing rate: $(@sprintf("%.2f%%", missing_rate(geno_qc) * 100))")
println("  Mean MAF: $(@sprintf("%.4f", mean(maf_qc)))")
println("  Mean heterozygosity: $(@sprintf("%.4f", heterozygosity_rate(geno_qc)))")

println("\nData Retention:")
println("  Samples retained: $(@sprintf("%.1f%%", 100 * n_samples(geno_qc) / n_samples(geno)))")
println("  Markers retained: $(@sprintf("%.1f%%", 100 * n_markers(geno_qc) / n_markers(geno)))")

println("\nRecommendations:")
if n_samples(geno_qc) < 0.9 * n_samples(geno)
    println("  ⚠ More than 10% of samples were removed. Review sample quality.")
end

if n_markers(geno_qc) < 0.7 * n_markers(geno)
    println("  ⚠ More than 30% of markers were removed. Consider relaxing filters if needed.")
else
    println("  ✓ Good marker retention rate.")
end

println("\n" * "="^80)
println("QC workflow completed successfully!")
println("="^80)
