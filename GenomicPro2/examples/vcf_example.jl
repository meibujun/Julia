"""
VCF File Format Example

This example demonstrates VCF file I/O functionality:
1. Reading VCF files (compressed and uncompressed)
2. Writing VCF files
3. Filtering variants and samples
4. Converting between VCF and PLINK formats
5. Integration with genomic prediction workflow

Run with: julia --project examples/vcf_example.jl

Note: For reading .vcf.gz files, install CodecZlib:
      using Pkg; Pkg.add("CodecZlib")
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 VCF File Format Example")
println("="^80)

Random.seed!(2024)

# ============================================================================
# 1. Create Sample VCF File
# ============================================================================

println("\n" * "="^80)
println("Step 1: Creating Sample VCF File")
println("="^80)

# Generate sample genotype data
n_samples = 200
n_markers = 1000

println("\nGenerating sample data:")
println("  Samples: $n_samples")
println("  Variants: $n_markers")

geno_data = rand(0:2, n_samples, n_markers)
sample_ids = [string("SAMPLE_", lpad(i, 4, '0')) for i in 1:n_samples]
marker_ids = [string("rs", 1000000 + i) for i in 1:n_markers]

# Create realistic genomic coordinates
# Distribute across 5 chromosomes
markers_per_chr = div(n_markers, 5)
chromosomes = String[]
positions = Int[]

for chr in 1:5
    chr_str = string(chr)
    for pos_idx in 1:markers_per_chr
        push!(chromosomes, chr_str)
        push!(positions, pos_idx * 10000)  # 10kb spacing
    end
end

# Fill remaining markers
while length(chromosomes) < n_markers
    push!(chromosomes, "5")
    push!(positions, length(positions) * 10000)
end

ref_alleles = rand(["A", "C", "G", "T"], n_markers)
alt_alleles = [ref == "A" ? "G" : "A" for ref in ref_alleles]

geno = CompactGenotypes(
    geno_data,
    sample_ids,
    marker_ids;
    chromosome = chromosomes,
    position = positions,
    ref_allele = ref_alleles,
    alt_allele = alt_alleles
)

println("\n✓ Sample data generated")

# ============================================================================
# 2. Write VCF File
# ============================================================================

println("\n" * "="^80)
println("Step 2: Writing VCF File")
println("="^80)

vcf_file = "example_output.vcf"

println("\nWriting to: $vcf_file")

write_vcf(
    vcf_file,
    geno;
    file_format = "VCFv4.2",
    source = "GenomicPro2_Example",
    reference = "Example_Reference_v1",
    verbose = true
)

# Check file size
file_size = filesize(vcf_file)
println("\n✓ VCF file created")
@printf("  File size: %.2f MB\n", file_size / 1e6)

# Show first few lines
println("\nFirst 15 lines of VCF file:")
println("─"^80)
lines = readlines(vcf_file)
for (i, line) in enumerate(lines[1:min(15, length(lines))])
    if length(line) > 80
        println(line[1:77] * "...")
    else
        println(line)
    end
end
println("─"^80)

# ============================================================================
# 3. Read VCF File
# ============================================================================

println("\n" * "="^80)
println("Step 3: Reading VCF File")
println("="^80)

println("\nReading entire VCF file...")
geno_read = read_vcf(vcf_file; verbose = true)

# Verify data matches
println("\nData verification:")
@test geno_read.n_samples == n_samples
@test geno_read.n_markers == n_markers
@test geno_read.sample_ids == sample_ids
@test geno_read.marker_ids == marker_ids

n_mismatches = 0
for i in 1:n_samples
    for j in 1:n_markers
        if get_genotype(geno, i, j) != get_genotype(geno_read, i, j)
            n_mismatches += 1
        end
    end
end

println("  Genotype mismatches: $n_mismatches / $(n_samples * n_markers)")
@test n_mismatches == 0

println("\n✓ VCF file read successfully and data verified")

# ============================================================================
# 4. Read VCF with Filters
# ============================================================================

println("\n" * "="^80)
println("Step 4: Reading VCF with Filters")
println("="^80)

@testset "Region Filtering" begin
    println("\n📊 Filter 1: Reading only chromosome 1")
    geno_chr1 = read_vcf(vcf_file; regions = ["1"], verbose = false)

    expected_chr1 = sum(chromosomes .== "1")
    println("  Expected variants: $expected_chr1")
    println("  Read variants: $(geno_chr1.n_markers)")
    @test geno_chr1.n_markers == expected_chr1
    @test all(geno_chr1.chromosome .== "1")
    println("  ✓ Chromosome filter working correctly")
end

@testset "Multiple Regions" begin
    println("\n📊 Filter 2: Reading chromosomes 1 and 2")
    geno_chr12 = read_vcf(vcf_file; regions = ["1", "2"], verbose = false)

    expected_chr12 = sum((chromosomes .== "1") .| (chromosomes .== "2"))
    println("  Expected variants: $expected_chr12")
    println("  Read variants: $(geno_chr12.n_markers)")
    @test geno_chr12.n_markers == expected_chr12
    println("  ✓ Multi-region filter working correctly")
end

@testset "Max Variants Limit" begin
    println("\n📊 Filter 3: Reading first 100 variants")
    geno_limited = read_vcf(vcf_file; max_variants = 100, verbose = false)

    println("  Read variants: $(geno_limited.n_markers)")
    @test geno_limited.n_markers == 100
    println("  ✓ Variant limit working correctly")
end

# ============================================================================
# 5. Sample Selection
# ============================================================================

println("\n" * "="^80)
println("Step 5: Sample Selection")
println("="^80)

# Select random subset of samples
n_subset = 50
subset_indices = sort(randperm(n_samples)[1:n_subset])
subset_samples = sample_ids[subset_indices]

println("\nReading subset of $n_subset samples...")
geno_subset = read_vcf(vcf_file; samples = subset_samples, verbose = false)

println("  Samples requested: $n_subset")
println("  Samples read: $(geno_subset.n_samples)")
@test geno_subset.n_samples == n_subset
@test geno_subset.n_markers == n_markers

println("\n✓ Sample selection working correctly")

# ============================================================================
# 6. VCF to PLINK Conversion
# ============================================================================

println("\n" * "="^80)
println("Step 6: VCF to PLINK Conversion")
println("="^80)

println("\nConverting VCF to PLINK format...")

# Read VCF
geno_vcf = read_vcf(vcf_file; verbose = false)

# Write as PLINK
plink_prefix = "example_converted"
write_plink(plink_prefix, geno_vcf; verbose = false)

println("  Created files:")
println("    $(plink_prefix).bed")
println("    $(plink_prefix).bim")
println("    $(plink_prefix).fam")

# Read PLINK back
geno_plink = read_plink(plink_prefix; verbose = false)

println("\nVerifying conversion:")
println("  VCF samples: $(geno_vcf.n_samples)")
println("  PLINK samples: $(geno_plink.n_samples)")
println("  VCF variants: $(geno_vcf.n_markers)")
println("  PLINK variants: $(geno_plink.n_markers)")

@test geno_vcf.n_samples == geno_plink.n_samples
@test geno_vcf.n_markers == geno_plink.n_markers

println("\n✓ VCF to PLINK conversion successful")

# ============================================================================
# 7. Integration with Genomic Prediction
# ============================================================================

println("\n" * "="^80)
println("Step 7: Integration with Genomic Prediction")
println("="^80)

println("\n🔬 Complete genomic prediction workflow using VCF data")

# Read VCF
println("\n1. Reading VCF file...")
geno_analysis = read_vcf(vcf_file; regions = ["1", "2"], verbose = false)
println("   Loaded: $(geno_analysis.n_samples) samples × $(geno_analysis.n_markers) variants")

# Quality control
println("\n2. Applying quality control...")
geno_qc = quality_control(
    geno_analysis;
    min_maf = 0.05,
    max_missing_per_marker = 0.1,
    max_missing_per_sample = 0.1,
    hwe_pvalue = 1e-6,
    verbose = false
)
println("   After QC: $(geno_qc.n_samples) samples × $(geno_qc.n_markers) variants")

# LD pruning
println("\n3. LD pruning...")
keep_idx = ld_prune_window(
    geno_qc;
    window_size = 50,
    r2_threshold = 0.8,
    verbose = false
)
geno_pruned = subset_markers(geno_qc, keep_idx)
println("   After LD pruning: $(geno_pruned.n_samples) samples × $(geno_pruned.n_markers) variants")

# Generate phenotypes
println("\n4. Generating phenotypes...")
h2 = 0.65
X = to_matrix(geno_pruned; impute = true)
n_causal = 50
true_effects = zeros(geno_pruned.n_markers)
causal_idx = randperm(geno_pruned.n_markers)[1:n_causal]
true_effects[causal_idx] = randn(n_causal)

genetic_values = X * true_effects
genetic_values = (genetic_values .- mean(genetic_values))
genetic_values = genetic_values .* sqrt(h2 / var(genetic_values))
environmental = randn(geno_pruned.n_samples) * sqrt(1 - h2)
phenotypes = genetic_values .+ environmental

pheno = PhenotypeData(
    geno_pruned.sample_ids,
    ["Trait1"],
    reshape(phenotypes, geno_pruned.n_samples, 1)
)

# GBLUP model
println("\n5. Training GBLUP model...")
G = compute_grm(geno_pruned; method = :vanraden, min_maf = 0.0)
model = GBLUPModel(method = :cholesky, estimate_variances = true)
fit!(model, geno_pruned, pheno; G = G, verbose = false)

println("   Model results:")
@printf("     Heritability: %.4f (true: %.4f)\n", model.result.h2, h2)
@printf("     Genetic variance: %.4f\n", model.result.var_u)
@printf("     Residual variance: %.4f\n", model.result.var_e)

# Predict breeding values
println("\n6. Predicting breeding values...")
gebv = predict(model, geno_pruned)
cor_pred = cor(gebv, genetic_values)
@printf("   Prediction accuracy: %.4f\n", cor_pred)

# Cross-validation
println("\n7. 5-fold cross-validation...")
cv_result = kfold_cv(
    () -> GBLUPModel(method = :cholesky, estimate_variances = false),
    geno_pruned,
    pheno;
    k = 5,
    grm_options = (min_maf = 0.0, method = :vanraden),
    seed = 123,
    verbose = false
)

@printf("   CV correlation: %.4f ± %.4f\n",
        cv_result.metrics.mean_fold_correlation,
        cv_result.metrics.std_fold_correlation)
@printf("   CV R²: %.4f\n", cv_result.metrics.r_squared)

println("\n✓ Complete genomic prediction workflow finished")

# ============================================================================
# 8. Performance Comparison
# ============================================================================

println("\n" * "="^80)
println("Step 8: File Format Performance Comparison")
println("="^80)

using Base: @elapsed

println("\nComparing VCF vs PLINK file sizes and read times:")
println("─"^80)

# File sizes
vcf_size = filesize(vcf_file)
plink_size = filesize("$(plink_prefix).bed") +
             filesize("$(plink_prefix).bim") +
             filesize("$(plink_prefix).fam")

@printf("%-20s %15s\n", "Format", "Size (MB)")
println("─"^80)
@printf("%-20s %15.2f\n", "VCF (text)", vcf_size / 1e6)
@printf("%-20s %15.2f\n", "PLINK (binary)", plink_size / 1e6)
println("─"^80)
@printf("%-20s %15.2fx\n", "Compression ratio", vcf_size / plink_size)
println("─"^80)

# Read times
println("\nRead time comparison:")
println("─"^80)

time_vcf = @elapsed read_vcf(vcf_file; verbose = false)
time_plink = @elapsed read_plink(plink_prefix; verbose = false)

@printf("%-20s %15s\n", "Format", "Time (s)")
println("─"^80)
@printf("%-20s %15.3f\n", "VCF", time_vcf)
@printf("%-20s %15.3f\n", "PLINK", time_plink)
println("─"^80)
@printf("%-20s %15.2fx\n", "Speedup (PLINK)", time_vcf / time_plink)
println("─"^80)

println("\n💡 Format Recommendations:")
println("  • VCF: Standard format, widely compatible, human-readable")
println("  • PLINK: Much smaller file size, faster to read/write")
println("  • Use VCF for data exchange, PLINK for analysis")

# ============================================================================
# 9. Practical Tips
# ============================================================================

println("\n" * "="^80)
println("Practical Tips for VCF Files")
println("="^80)

println("\n💡 Reading VCF Files:")
println("  1. Use regions filter for large files (read by chromosome)")
println("  2. Set max_variants for testing/debugging")
println("  3. Filter by quality score (min_qual) for reliable variants")
println("  4. Use pass_only=true for high-quality datasets")
println("  5. For .vcf.gz files, install: using Pkg; Pkg.add(\"CodecZlib\")")

println("\n💡 Writing VCF Files:")
println("  1. VCF is human-readable but large")
println("  2. Consider compress=true for storage")
println("  3. Include reference genome info for reproducibility")
println("  4. Add source info for traceability")

println("\n💡 Working with Large VCF Files:")
println("  1. Read by chromosome/region to save memory")
println("  2. Apply QC filters during reading (min_qual, pass_only)")
println("  3. Convert to PLINK for faster repeated access")
println("  4. Use LD pruning to reduce dataset size")
println("  5. Consider chunked processing for very large files")

println("\n💡 VCF vs PLINK Comparison:")
println("─"^80)
@printf("%-30s %-20s %-20s\n", "Feature", "VCF", "PLINK")
println("─"^80)
@printf("%-30s %-20s %-20s\n", "File size", "Large", "Small (5-10x)")
@printf("%-30s %-20s %-20s\n", "Read speed", "Slower", "Faster (2-5x)")
@printf("%-30s %-20s %-20s\n", "Human readable", "Yes", "No (.bed)")
@printf("%-30s %-20s %-20s\n", "Standard format", "Yes", "Yes")
@printf("%-30s %-20s %-20s\n", "Multi-allelic support", "Yes", "Limited")
@printf("%-30s %-20s %-20s\n", "Quality scores", "Yes", "No")
@printf("%-30s %-20s %-20s\n", "INFO fields", "Yes", "No")
@printf("%-30s %-20s %-20s\n", "Best use case", "Data exchange", "Analysis")
println("─"^80)

# ============================================================================
# 10. Cleanup and Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ VCF File I/O Example Complete!")

println("\nFiles Created:")
println("  • $vcf_file ($(round(vcf_size/1e6, digits=2)) MB)")
println("  • $(plink_prefix).{bed,bim,fam} ($(round(plink_size/1e6, digits=2)) MB)")

println("\nKey Capabilities Demonstrated:")
println("  ✓ VCF file writing with custom headers")
println("  ✓ VCF file reading with various filters")
println("  ✓ Sample and region selection")
println("  ✓ VCF to PLINK conversion")
println("  ✓ Integration with genomic prediction")
println("  ✓ Quality control pipeline")
println("  ✓ LD pruning")
println("  ✓ Cross-validation")

println("\nCleanup (uncomment to remove files):")
println("  # rm(\"$vcf_file\")")
println("  # rm(\"$(plink_prefix).bed\")")
println("  # rm(\"$(plink_prefix).bim\")")
println("  # rm(\"$(plink_prefix).fam\")")

println("\n" * "="^80)
println("VCF example completed successfully!")
println("="^80)
