# examples/01_basic_usage.jl

"""
Example 1: Basic GenomicPro.jl Usage - Data Loading and Quality Control

This example demonstrates:
1. Loading genotype data from multiple formats (VCF, PLINK)
2. Creating memory-efficient TwoBitGenotypes structures
3. Computing basic summary statistics
4. Applying quality control filters
5. Exporting cleaned data

Author: GenomicPro Development Team
Date: 2025
Julia Version: 1.12.1
"""

using GenomicPro
using Statistics, LinearAlgebra
using Printf

println("="^70)
println("GenomicPro.jl Example 1: Basic Data Handling and Quality Control")
println("="^70)
println()

# ============================================================================
# Step 1: Generate Simulated Dataset for Demonstration
# ============================================================================
println("Step 1: Generating simulated genotype data...")

n_samples = 1000
n_markers = 50000

# Simulate genotypes with realistic allele frequencies and missing data
# Real data would be loaded using: genotypes = read_genotypes("data.vcf")
Random.seed!(42)
genotypes_raw = Matrix{Union{Int, Missing}}(undef, n_samples, n_markers)

for marker_idx in 1:n_markers
    # Simulate allele frequency from beta distribution (realistic MAF spectrum)
    p = rand(Beta(0.5, 0.5))

    for sample_idx in 1:n_samples
        # Introduce 2% missing rate
        if rand() < 0.02
            genotypes_raw[sample_idx, marker_idx] = missing
        else
            # Sample genotype assuming Hardy-Weinberg equilibrium
            r = rand()
            if r < (1-p)^2
                genotypes_raw[sample_idx, marker_idx] = 0  # Homozygous reference
            elseif r < (1-p)^2 + 2*p*(1-p)
                genotypes_raw[sample_idx, marker_idx] = 1  # Heterozygous
            else
                genotypes_raw[sample_idx, marker_idx] = 2  # Homozygous alternate
            end
        end
    end
end

println("  Generated $n_samples samples × $n_markers markers")
println("  Memory usage (standard): $(summarysize(genotypes_raw) / 1e9) GB")
println()

# ============================================================================
# Step 2: Convert to Memory-Efficient TwoBitGenotypes
# ============================================================================
println("Step 2: Converting to memory-efficient TwoBitGenotypes format...")

sample_ids = ["Animal_" * lpad(i, 5, '0') for i in 1:n_samples]
marker_ids = ["rs" * string(i) for i in 1:n_markers]

genotypes = TwoBitGenotypes(genotypes_raw,
                            sample_ids=sample_ids,
                            marker_ids=marker_ids)

memory_standard = summarysize(genotypes_raw)
memory_twobit = summarysize(genotypes)
reduction_pct = (1 - memory_twobit / memory_standard) * 100

println("  TwoBitGenotypes created successfully")
println("  Memory usage (TwoBit): $(memory_twobit / 1e9) GB")
println("  Memory reduction: $(@sprintf("%.1f%%", reduction_pct))")
println()

# ============================================================================
# Step 3: Compute Basic Summary Statistics
# ============================================================================
println("Step 3: Computing genotype summary statistics...")

# Compute allele frequencies
println("  Computing allele frequencies...")
allele_freqs = get_allele_frequencies(genotypes)

# Calculate minor allele frequencies
maf = min.(allele_freqs, 1 .- allele_freqs)
maf_distribution = [
    sum(maf .< 0.01),
    sum(0.01 .<= maf .< 0.05),
    sum(0.05 .<= maf .< 0.10),
    sum(maf .>= 0.10)
]

println("\n  MAF Distribution:")
println("    Rare (MAF < 0.01): $(maf_distribution[1]) markers ($(round(maf_distribution[1]/n_markers*100, digits=2))%)")
println("    Low (0.01 ≤ MAF < 0.05): $(maf_distribution[2]) markers ($(round(maf_distribution[2]/n_markers*100, digits=2))%)")
println("    Medium (0.05 ≤ MAF < 0.10): $(maf_distribution[3]) markers ($(round(maf_distribution[3]/n_markers*100, digits=2))%)")
println("    Common (MAF ≥ 0.10): $(maf_distribution[4]) markers ($(round(maf_distribution[4]/n_markers*100, digits=2))%)")

# Compute missing rates per sample and marker
println("\n  Computing missing data patterns...")

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

println("  Missing rate per sample: mean = $(round(mean(missing_per_sample)*100, digits=2))%, max = $(round(maximum(missing_per_sample)*100, digits=2))%")
println("  Missing rate per marker: mean = $(round(mean(missing_per_marker)*100, digits=2))%, max = $(round(maximum(missing_per_marker)*100, digits=2))%")

# Identify samples and markers exceeding 10% missing
high_missing_samples = findall(missing_per_sample .> 0.10)
high_missing_markers = findall(missing_per_marker .> 0.10)

println("  Samples with >10% missing: $(length(high_missing_samples))")
println("  Markers with >10% missing: $(length(high_missing_markers))")
println()

# ============================================================================
# Step 4: Data Access Examples
# ============================================================================
println("Step 4: Demonstrating efficient data access patterns...")

# Single genotype access
sample_idx, marker_idx = 100, 1000
geno_value = genotypes[sample_idx, marker_idx]
println("  Single genotype access: genotypes[$sample_idx, $marker_idx] = $geno_value")

# Range access
subset_samples = 1:100
subset_markers = 1:10000
subset = genotypes[subset_samples, subset_markers]
println("  Subset extraction: $(size(subset)) matrix extracted")

# Full sample access
sample_genotypes = genotypes[1, :]  # All markers for first sample
println("  Individual genotype vector: $(length(sample_genotypes)) markers")
println()

# ============================================================================
# Step 5: Quality Control Preview
# ============================================================================
println("Step 5: Preparing for quality control...")

println("\n  Recommended QC thresholds based on data characteristics:")
println("    - Sample missing rate threshold: 10% (would remove $(length(high_missing_samples)) samples)")
println("    - Marker missing rate threshold: 10% (would remove $(length(high_missing_markers)) markers)")
println("    - Minor allele frequency threshold: 1% (would remove $(maf_distribution[1]) markers)")
println()

println("  After QC, expected retained data:")
println("    Samples: $(n_samples - length(high_missing_samples)) ($(round((n_samples - length(high_missing_samples))/n_samples*100, digits=1))% retention)")
println("    Markers: $(n_markers - length(high_missing_markers) - maf_distribution[1]) ($(round((n_markers - length(high_missing_markers) - maf_distribution[1])/n_markers*100, digits=1))% retention)")
println()

# ============================================================================
# Step 6: Performance Benchmarking
# ============================================================================
println("Step 6: Benchmarking key operations...")

using BenchmarkTools

println("\n  Benchmarking allele frequency computation:")
@time afs_bench = get_allele_frequencies(genotypes)
println("    Cached retrieval:")
@time afs_bench = get_allele_frequencies(genotypes)

println("\n  Benchmarking subset extraction (1000 samples × 10000 markers):")
@time subset_bench = genotypes[1:1000, 1:10000]

println()
println("="^70)
println("Example 1 completed successfully!")
println("="^70)
println()
println("Next steps:")
println("  - See example_02_quality_control.jl for comprehensive QC pipeline")
println("  - See example_03_genomic_prediction.jl for GBLUP analysis")
println("  - See example_04_gpu_acceleration.jl for GPU-accelerated computation")