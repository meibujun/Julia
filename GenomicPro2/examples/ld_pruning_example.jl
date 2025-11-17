"""
LD Pruning Example

This example demonstrates linkage disequilibrium (LD) pruning:
1. Generating data with LD structure
2. Computing LD statistics (r², D')
3. Window-based LD pruning
4. Pairwise LD pruning
5. LD matrix visualization
6. Impact on genomic prediction

Run with: julia --project examples/ld_pruning_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random
using LinearAlgebra

println("="^80)
println("GenomicPro2 LD Pruning Example")
println("="^80)

# Set seed for reproducibility
Random.seed!(2024)

# ============================================================================
# 1. Generate Data with LD Structure
# ============================================================================

println("\n" * "="^80)
println("Step 1: Generating Data with LD Structure")
println("="^80)

n_samples = 1000
n_markers = 2000
n_ld_blocks = 10
markers_per_block = 50

println("\nDataset properties:")
println("  Samples: $n_samples")
println("  Markers: $n_markers")
println("  LD blocks: $n_ld_blocks")
println("  Markers per block: $markers_per_block")

# Generate genotypes with LD block structure
# This mimics real genomic data where nearby SNPs are correlated
geno_data = zeros(Int, n_samples, n_markers)

current_marker = 1
for block in 1:n_ld_blocks
    # Create base genotype for this block
    base_geno = rand(0:2, n_samples)

    for i in 1:markers_per_block
        if current_marker <= n_markers
            # Add noise to create imperfect LD within block
            noise_level = 0.2  # 20% of genotypes are different
            n_flip = round(Int, n_samples * noise_level)
            flip_idx = randperm(n_samples)[1:n_flip]

            geno_data[:, current_marker] = copy(base_geno)
            geno_data[flip_idx, current_marker] = rand(0:2, n_flip)

            current_marker += 1
        end
    end
end

# Fill remaining markers with independent genotypes
for i in current_marker:n_markers
    geno_data[:, i] = rand(0:2, n_samples)
end

sample_ids = [string("ID", i) for i in 1:n_samples]
marker_ids = [string("SNP", i) for i in 1:n_markers]

# Create realistic chromosome and position info
# Distribute across 5 chromosomes
markers_per_chr = div(n_markers, 5)
chromosome = vcat([fill(string(c), markers_per_chr) for c in 1:5]...)
chromosome = vcat(chromosome, fill("5", n_markers - length(chromosome)))

position = vcat([collect(1:markers_per_chr) .* 10000 for _ in 1:5]...)  # 10kb spacing
position = vcat(position, fill(0, n_markers - length(position)))

geno = CompactGenotypes(
    geno_data,
    sample_ids,
    marker_ids;
    chromosome = chromosome,
    position = position
)

println("\n✓ Data generated with LD block structure")
println("  Expected: High LD within blocks, low LD between blocks")

# ============================================================================
# 2. Compute LD Statistics
# ============================================================================

println("\n" * "="^80)
println("Step 2: Computing LD Statistics")
println("="^80)

println("\n📊 LD within first block (SNPs 1-10):")
println("─"^80)
@printf("%-10s %-10s %12s %12s %12s\n", "SNP1", "SNP2", "r", "r²", "D'")
println("─"^80)

for i in 1:5
    for j in (i+1):min(i+2, 10)
        ld = compute_ld_full(geno, i, j)
        @printf("%-10s %-10s %12.4f %12.4f %12.4f\n",
                marker_ids[i], marker_ids[j],
                ld.r, ld.r2, ld.Dprime)
    end
end
println("─"^80)

println("\n📊 LD across blocks (SNP1 vs SNPs from other blocks):")
println("─"^80)
@printf("%-10s %-10s %12s %12s %12s\n", "SNP1", "SNP2", "r", "r²", "D'")
println("─"^80)

for block in [2, 5, 10]
    idx = block * markers_per_block + 1
    if idx <= n_markers
        ld = compute_ld_full(geno, 1, idx)
        @printf("%-10s %-10s %12.4f %12.4f %12.4f\n",
                marker_ids[1], marker_ids[idx],
                ld.r, ld.r2, ld.Dprime)
    end
end
println("─"^80)

# Compute LD matrix for first 50 markers
println("\n📊 Computing LD matrix for first 50 markers...")
ld_mat = compute_ld_matrix(geno, 1:50)

# Calculate average LD
off_diag_r2 = []
for i in 1:50
    for j in (i+1):50
        push!(off_diag_r2, ld_mat.r2[i, j])
    end
end

@printf("\nLD Matrix Statistics (first 50 markers):\n")
@printf("  Mean r²: %.4f\n", mean(off_diag_r2))
@printf("  Median r²: %.4f\n", median(off_diag_r2))
@printf("  Max r²: %.4f\n", maximum(off_diag_r2))
@printf("  Min r²: %.4f\n", minimum(off_diag_r2))
@printf("  Pairs with r² > 0.8: %d (%.1f%%)\n",
        sum(off_diag_r2 .> 0.8),
        100 * sum(off_diag_r2 .> 0.8) / length(off_diag_r2))

# ============================================================================
# 3. Window-based LD Pruning
# ============================================================================

println("\n" * "="^80)
println("Step 3: Window-based LD Pruning")
println("="^80)

# Try different thresholds
thresholds = [0.99, 0.9, 0.8, 0.5, 0.2]
results_window = Dict()

println("\n🔬 Testing different r² thresholds:")
println("─"^80)
@printf("%-15s %15s %15s %15s\n", "Threshold", "Markers Kept", "% Kept", "% Removed")
println("─"^80)

for threshold in thresholds
    keep_idx = ld_prune_window(
        geno;
        window_size = 50,
        step_size = 10,
        r2_threshold = threshold,
        respect_chromosomes = true,
        verbose = false
    )

    n_kept = length(keep_idx)
    pct_kept = 100 * n_kept / n_markers
    pct_removed = 100 - pct_kept

    results_window[threshold] = keep_idx

    @printf("%-15.2f %15d %15.1f %15.1f\n",
            threshold, n_kept, pct_kept, pct_removed)
end
println("─"^80)

println("\n💡 Interpretation:")
println("  • r² > 0.99: Very strict, removes nearly identical SNPs")
println("  • r² > 0.9: Removes highly redundant SNPs")
println("  • r² > 0.8: Standard threshold, good balance")
println("  • r² > 0.5: Moderate pruning")
println("  • r² > 0.2: Aggressive pruning")

# Use standard threshold (0.8) for downstream analysis
keep_idx_standard = results_window[0.8]
geno_pruned = subset_markers(geno, keep_idx_standard)

println("\n✓ Using r² > 0.8 for downstream analysis")
println("  Markers after pruning: $(geno_pruned.n_markers)")

# ============================================================================
# 4. Chromosome-aware vs Genome-wide Pruning
# ============================================================================

println("\n" * "="^80)
println("Step 4: Chromosome-aware vs Genome-wide Pruning")
println("="^80)

keep_chr_aware = ld_prune_window(
    geno;
    window_size = 50,
    r2_threshold = 0.8,
    respect_chromosomes = true,
    verbose = false
)

keep_genome_wide = ld_prune_window(
    geno;
    window_size = 50,
    r2_threshold = 0.8,
    respect_chromosomes = false,
    verbose = false
)

println("\nComparison:")
println("─"^80)
@printf("%-30s %15s %15s\n", "Method", "Markers Kept", "% Kept")
println("─"^80)
@printf("%-30s %15d %15.1f\n",
        "Chromosome-aware",
        length(keep_chr_aware),
        100 * length(keep_chr_aware) / n_markers)
@printf("%-30s %15d %15.1f\n",
        "Genome-wide",
        length(keep_genome_wide),
        100 * length(keep_genome_wide) / n_markers)
println("─"^80)

println("\n💡 Recommendation: Use chromosome-aware pruning")
println("  • Avoids computing LD across chromosomes (biologically incorrect)")
println("  • Faster computation")
println("  • More interpretable results")

# ============================================================================
# 5. Pairwise LD Pruning (Small Example)
# ============================================================================

println("\n" * "="^80)
println("Step 5: Pairwise LD Pruning")
println("="^80)

# Use subset for demonstration (pairwise is slower)
geno_subset = subset_markers(geno, 1:200)

println("\n🔬 Comparing pruning methods on first 200 markers:")

keep_window_subset = ld_prune_window(
    geno_subset;
    window_size = 50,
    r2_threshold = 0.8,
    verbose = false
)

keep_pairwise_subset = ld_prune_pairwise(
    geno_subset;
    r2_threshold = 0.8,
    verbose = false
)

println("\nPruning Method Comparison:")
println("─"^80)
@printf("%-30s %15s %15s\n", "Method", "Markers Kept", "% Kept")
println("─"^80)
@printf("%-30s %15d %15.1f\n",
        "Window-based",
        length(keep_window_subset),
        100 * length(keep_window_subset) / 200)
@printf("%-30s %15d %15.1f\n",
        "Pairwise",
        length(keep_pairwise_subset),
        100 * length(keep_pairwise_subset) / 200)
println("─"^80)

overlap = length(intersect(keep_window_subset, keep_pairwise_subset))
@printf("\nOverlap: %d markers (%.1f%% of window-based)\n",
        overlap, 100 * overlap / length(keep_window_subset))

println("\n💡 Method Selection:")
println("  • Window-based: Fast, good for large datasets (recommended)")
println("  • Pairwise: More thorough, better for small datasets or fine-tuning")

# ============================================================================
# 6. Impact on Genomic Prediction
# ============================================================================

println("\n" * "="^80)
println("Step 6: Impact on Genomic Prediction")
println("="^80)

# Generate phenotypes
Random.seed!(123)
h2 = 0.6
n_causal = 100

# Simulate genetic effects
true_effects = zeros(n_markers)
causal_idx = randperm(n_markers)[1:n_causal]
true_effects[causal_idx] = randn(n_causal)

X = to_matrix(geno; impute = true)
genetic_values = X * true_effects
genetic_values = (genetic_values .- mean(genetic_values))
genetic_values = genetic_values .* sqrt(h2 / var(genetic_values))

environmental = randn(n_samples) * sqrt(1 - h2)
phenotypes = genetic_values .+ environmental

pheno = PhenotypeData(sample_ids, ["Yield"], reshape(phenotypes, n_samples, 1))

println("\n🔬 Fitting GBLUP models:")

# Model with all markers
println("\n  1. Full marker set ($n_markers markers)...")
G_full = compute_grm(geno; method = :vanraden, min_maf = 0.0)
model_full = GBLUPModel(method = :cholesky, estimate_variances = true)
fit!(model_full, geno, pheno; G = G_full, verbose = false)
gebv_full = predict(model_full, geno)

# Model with pruned markers (r² > 0.8)
println("  2. LD-pruned marker set ($(geno_pruned.n_markers) markers, r² > 0.8)...")
G_pruned = compute_grm(geno_pruned; method = :vanraden, min_maf = 0.0)
model_pruned = GBLUPModel(method = :cholesky, estimate_variances = true)
fit!(model_pruned, geno_pruned, pheno; G = G_pruned, verbose = false)
gebv_pruned = predict(model_pruned, geno_pruned)

# Model with aggressively pruned markers (r² > 0.5)
keep_idx_aggressive = results_window[0.5]
geno_aggressive = subset_markers(geno, keep_idx_aggressive)
println("  3. Aggressively pruned ($(geno_aggressive.n_markers) markers, r² > 0.5)...")
G_aggressive = compute_grm(geno_aggressive; method = :vanraden, min_maf = 0.0)
model_aggressive = GBLUPModel(method = :cholesky, estimate_variances = true)
fit!(model_aggressive, geno_aggressive, pheno; G = G_aggressive, verbose = false)
gebv_aggressive = predict(model_aggressive, geno_aggressive)

# Compare results
println("\n" * "="^80)
println("Prediction Performance Comparison")
println("="^80)

println("\nModel Parameters:")
println("─"^80)
@printf("%-25s %12s %12s %12s %12s\n",
        "Model", "Markers", "h²", "σ²_g", "σ²_e")
println("─"^80)
@printf("%-25s %12d %12.4f %12.4f %12.4f\n",
        "Full", n_markers,
        model_full.result.h2,
        model_full.result.var_u,
        model_full.result.var_e)
@printf("%-25s %12d %12.4f %12.4f %12.4f\n",
        "Pruned (r²>0.8)", geno_pruned.n_markers,
        model_pruned.result.h2,
        model_pruned.result.var_u,
        model_pruned.result.var_e)
@printf("%-25s %12d %12.4f %12.4f %12.4f\n",
        "Aggressive (r²>0.5)", geno_aggressive.n_markers,
        model_aggressive.result.h2,
        model_aggressive.result.var_u,
        model_aggressive.result.var_e)
@printf("%-25s %12s %12.4f %12s %12s\n",
        "True", "-", h2, "-", "-")
println("─"^80)

println("\nPrediction Accuracy (correlation with true breeding values):")
println("─"^80)
@printf("%-25s %15s %15s\n", "Model", "Correlation", "Relative to Full")
println("─"^80)

cor_full = cor(gebv_full, genetic_values)
cor_pruned = cor(gebv_pruned, genetic_values)
cor_aggressive = cor(gebv_aggressive, genetic_values)

@printf("%-25s %15.4f %15s\n", "Full", cor_full, "100.0%")
@printf("%-25s %15.4f %15.1f%%\n",
        "Pruned (r²>0.8)", cor_pruned, 100 * cor_pruned / cor_full)
@printf("%-25s %15.4f %15.1f%%\n",
        "Aggressive (r²>0.5)", cor_aggressive, 100 * cor_aggressive / cor_full)
println("─"^80)

# Agreement between models
cor_full_pruned = cor(gebv_full, gebv_pruned)
cor_full_aggressive = cor(gebv_full, gebv_aggressive)

println("\nGEBV Agreement (correlation between models):")
println("─"^80)
@printf("%-40s %15.4f\n", "Full vs Pruned (r²>0.8)", cor_full_pruned)
@printf("%-40s %15.4f\n", "Full vs Aggressive (r²>0.5)", cor_full_aggressive)
println("─"^80)

# ============================================================================
# 7. Computational Efficiency
# ============================================================================

println("\n" * "="^80)
println("Step 7: Computational Efficiency")
println("="^80)

println("\nGRM Computation Time:")
println("─"^80)

using Base: @elapsed

time_full = @elapsed compute_grm(geno; method = :vanraden, min_maf = 0.0)
time_pruned = @elapsed compute_grm(geno_pruned; method = :vanraden, min_maf = 0.0)
time_aggressive = @elapsed compute_grm(geno_aggressive; method = :vanraden, min_maf = 0.0)

@printf("%-25s %15s %15s\n", "Model", "Time (s)", "Speedup")
println("─"^80)
@printf("%-25s %15.3f %15s\n", "Full ($n_markers)", time_full, "1.0x")
@printf("%-25s %15.3f %15.1fx\n",
        "Pruned ($(geno_pruned.n_markers))",
        time_pruned,
        time_full / time_pruned)
@printf("%-25s %15.3f %15.1fx\n",
        "Aggressive ($(geno_aggressive.n_markers))",
        time_aggressive,
        time_full / time_aggressive)
println("─"^80)

# ============================================================================
# 8. Practical Recommendations
# ============================================================================

println("\n" * "="^80)
println("Practical Recommendations")
println("="^80)

println("\n💡 When to Use LD Pruning:")
println("  ✓ Before computing GRM (reduces computational cost)")
println("  ✓ Before Bayesian models (improves convergence)")
println("  ✓ For PCA and population structure (removes redundancy)")
println("  ✓ When markers are in high LD (e.g., imputed data)")
println("  ✓ For methods assuming SNP independence")

println("\n💡 When NOT to Use LD Pruning:")
println("  ✗ For QTL mapping (might remove causal variants)")
println("  ✗ For fine-mapping (need dense markers)")
println("  ✗ When markers already sparse (e.g., low-density chips)")
println("  ✗ For genomic prediction in well-balanced designs")

println("\n💡 Recommended Settings:")
println("  • Standard pruning: window=50, step=10, r²>0.8")
println("  • Moderate pruning: window=50, step=10, r²>0.5")
println("  • Light pruning: window=50, step=10, r²>0.95")
println("  • Always use chromosome-aware pruning")
println("  • Check impact on prediction accuracy")

println("\n💡 LD Threshold Selection Guide:")
@printf("  • r² > 0.99: Removes only near-duplicates (keep ~95%% SNPs)\n")
@printf("  • r² > 0.90: Removes highly redundant SNPs (keep ~80-90%% SNPs)\n")
@printf("  • r² > 0.80: **Recommended** - Good balance (keep ~60-80%% SNPs)\n")
@printf("  • r² > 0.50: Moderate pruning (keep ~40-60%% SNPs)\n")
@printf("  • r² > 0.20: Aggressive pruning (keep ~20-40%% SNPs)\n")

# ============================================================================
# 9. Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ LD Pruning Analysis Complete!")

println("\nKey Findings:")
@printf("  • Original markers: %d\n", n_markers)
@printf("  • After standard pruning (r²>0.8): %d (%.1f%% kept)\n",
        geno_pruned.n_markers,
        100 * geno_pruned.n_markers / n_markers)
@printf("  • After aggressive pruning (r²>0.5): %d (%.1f%% kept)\n",
        geno_aggressive.n_markers,
        100 * geno_aggressive.n_markers / n_markers)

println("\nPrediction Performance:")
@printf("  • Full model accuracy: %.4f\n", cor_full)
@printf("  • Pruned model accuracy: %.4f (%.1f%% of full)\n",
        cor_pruned, 100 * cor_pruned / cor_full)
@printf("  • GEBV correlation (full vs pruned): %.4f\n", cor_full_pruned)

println("\nComputational Efficiency:")
@printf("  • GRM speedup with pruning: %.1fx\n", time_full / time_pruned)
@printf("  • GRM speedup with aggressive: %.1fx\n", time_full / time_aggressive)

if cor_pruned / cor_full > 0.95
    println("\n✓ Excellent: Pruning retained >95% prediction accuracy")
elseif cor_pruned / cor_full > 0.90
    println("\n✓ Good: Pruning retained >90% prediction accuracy")
else
    println("\n⚠ Warning: Pruning reduced prediction accuracy")
    println("  Consider using less aggressive threshold")
end

println("\n" * "="^80)
println("LD pruning workflow completed successfully!")
println("="^80)
