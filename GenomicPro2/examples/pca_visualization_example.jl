"""
PCA and Visualization Example

This example demonstrates:
1. Population structure analysis with PCA
2. Outlier detection using PCs
3. Sample clustering
4. Data preparation for visualization (Manhattan, QQ, PCA plots)
5. Exporting data for external plotting

Run with: julia --project examples/pca_visualization_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 PCA and Visualization Example")
println("="^80)

Random.seed!(2024)

# ============================================================================
# 1. Generate Simulated Data with Population Structure
# ============================================================================

println("\n" * "="^80)
println("Step 1: Generating Data with Population Structure")
println("="^80)

n_pop1 = 200
n_pop2 = 200
n_pop3 = 100
n_total = n_pop1 + n_pop2 + n_pop3
n_markers = 5000

println("\nSimulating 3 populations:")
println("  Population 1: $n_pop1 samples")
println("  Population 2: $n_pop2 samples")
println("  Population 3: $n_pop3 samples (admixed)")
println("  Total markers: $n_markers")

# Generate genotypes with different allele frequencies per population
geno_data = zeros(Int, n_total, n_markers)

for j in 1:n_markers
    # Different allele frequencies for each population
    p1 = rand() * 0.4 + 0.1  # 0.1-0.5
    p2 = rand() * 0.4 + 0.5  # 0.5-0.9
    p3 = (p1 + p2) / 2        # Admixed

    # Population 1
    for i in 1:n_pop1
        geno_data[i, j] = rand() < p1 ? (rand() < p1 ? 2 : 1) : 0
    end

    # Population 2
    for i in (n_pop1+1):(n_pop1+n_pop2)
        geno_data[i, j] = rand() < p2 ? (rand() < p2 ? 2 : 1) : 0
    end

    # Population 3 (admixed)
    for i in (n_pop1+n_pop2+1):n_total
        geno_data[i, j] = rand() < p3 ? (rand() < p3 ? 2 : 1) : 0
    end
end

sample_ids = vcat(
    [string("Pop1_S", i) for i in 1:n_pop1],
    [string("Pop2_S", i) for i in 1:n_pop2],
    [string("Pop3_S", i) for i in 1:n_pop3]
)

marker_ids = [string("SNP", i) for i in 1:n_markers]
chromosomes = [string(div(i-1, 1000) + 1) for i in 1:n_markers]
positions = [(i-1) % 1000 * 10000 for i in 1:n_markers]

geno = CompactGenotypes(
    geno_data,
    sample_ids,
    marker_ids;
    chromosome = chromosomes,
    position = positions
)

# True population labels
true_labels = vcat(
    fill(1, n_pop1),
    fill(2, n_pop2),
    fill(3, n_pop3)
)

println("\n✓ Data generated with population structure")

# ============================================================================
# 2. Principal Component Analysis
# ============================================================================

println("\n" * "="^80)
println("Step 2: Principal Component Analysis")
println("="^80)

# Perform PCA with LD pruning
pca_result = pca(geno;
    n_pcs = 10,
    min_maf = 0.01,
    ld_prune = true,
    ld_threshold = 0.2,
    method = :grm,
    verbose = true
)

# Analyze variance explained
println("\n📊 Variance Explained by PCs:")
println("-"^80)
@printf("%-5s %15s %20s\n", "PC", "Variance (%)", "Cumulative (%)")
println("-"^80)

for i in 1:min(10, pca_result.n_pcs)
    @printf("%-5d %15.2f %20.2f\n",
            i,
            pca_result.variance_explained[i] * 100,
            pca_result.cumulative_variance[i] * 100)
end

println("-"^80)

# Check if PCA captures population structure
println("\n📈 Population Structure Detection:")
println("  PC1 should separate populations if structure exists")

# Calculate mean PC1 per population
pc1_pop1 = mean(pca_result.pcs[1:n_pop1, 1])
pc1_pop2 = mean(pca_result.pcs[(n_pop1+1):(n_pop1+n_pop2), 1])
pc1_pop3 = mean(pca_result.pcs[(n_pop1+n_pop2+1):end, 1])

@printf("  Population 1 mean PC1: %.4f\n", pc1_pop1)
@printf("  Population 2 mean PC1: %.4f\n", pc1_pop2)
@printf("  Population 3 mean PC1: %.4f\n", pc1_pop3)

separation = abs(pc1_pop1 - pc1_pop2)
@printf("\n  Population separation (|PC1 diff|): %.4f\n", separation)

if separation > 0.5
    println("  ✓ Strong population structure detected!")
elseif separation > 0.2
    println("  ✓ Moderate population structure detected")
else
    println("  ⚠ Weak population structure")
end

# ============================================================================
# 3. Outlier Detection
# ============================================================================

println("\n" * "="^80)
println("Step 3: Outlier Detection Using PCs")
println("="^80)

# Detect outliers using Mahalanobis distance
outliers = detect_outliers_pca(pca_result;
    n_pcs = 4,
    method = :mahalanobis,
    threshold = 6.0
)

println("\n🔍 Outlier Detection Results:")
println("  Method: Mahalanobis distance")
println("  PCs used: 4")
println("  Threshold: 6.0")
println("  Outliers detected: $(length(outliers))")

if !isempty(outliers)
    println("\n  Outlier samples:")
    for (i, idx) in enumerate(outliers[1:min(10, length(outliers))])
        println("    $(i). $(pca_result.sample_ids[idx])")
    end
    if length(outliers) > 10
        println("    ... (showing first 10)")
    end
end

# ============================================================================
# 4. Sample Clustering
# ============================================================================

println("\n" * "="^80)
println("Step 4: Sample Clustering Based on PCs")
println("="^80)

# K-means clustering
n_clusters = 3
clusters = cluster_samples(pca_result;
    n_clusters = n_clusters,
    n_pcs = 2,
    max_iter = 100
)

println("\n📊 Clustering Results:")
println("  Algorithm: K-means")
println("  Number of clusters: $n_clusters")
println("  PCs used: 2")

# Count samples per cluster
println("\n  Cluster sizes:")
for k in 1:n_clusters
    n_in_cluster = sum(clusters .== k)
    @printf("    Cluster %d: %d samples (%.1f%%)\n",
            k, n_in_cluster, 100 * n_in_cluster / n_total)
end

# Compare with true labels (if available)
println("\n  Comparison with true populations:")

# Create confusion matrix
confusion = zeros(Int, 3, 3)
for i in 1:n_total
    true_pop = true_labels[i]
    cluster = clusters[i]
    confusion[true_pop, cluster] += 1
end

println("\n  Confusion Matrix (rows=true, cols=predicted):")
println("  " * "-"^50)
@printf("  %-15s", "")
for k in 1:n_clusters
    @printf("%12s", "Cluster $k")
end
println()
println("  " * "-"^50)

for pop in 1:3
    @printf("  %-15s", "Population $pop")
    for k in 1:n_clusters
        @printf("%12d", confusion[pop, k])
    end
    println()
end
println("  " * "-"^50)

# Calculate clustering accuracy (best match)
max_correct = 0
for perm in [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
    correct = sum([confusion[i, perm[i]] for i in 1:3])
    max_correct = max(max_correct, correct)
end

accuracy = 100 * max_correct / n_total
@printf("\n  Best clustering accuracy: %.1f%%\n", accuracy)

# ============================================================================
# 5. Prepare Visualization Data
# ============================================================================

println("\n" * "="^80)
println("Step 5: Preparing Data for Visualization")
println("="^80)

# PCA scatter plot data
pca_plot = pca_plot_data(pca_result; pc_x=1, pc_y=2)

println("\n📊 PCA Plot Data:")
println("  X-axis: $(pca_plot.xlabel)")
println("  Y-axis: $(pca_plot.ylabel)")
println("  Samples: $(length(pca_plot.pc_x))")

# Save PCA data for plotting
open("pca_data.csv", "w") do io
    println(io, "sample_id,pc1,pc2,population,cluster")
    for i in 1:n_total
        println(io, "$(pca_plot.sample_ids[i]),$(pca_plot.pc_x[i]),$(pca_plot.pc_y[i]),$(true_labels[i]),$(clusters[i])")
    end
end

println("  ✓ Saved to: pca_data.csv")

# Generate mock GWAS results for Manhattan/QQ plot demo
println("\n📊 Generating Mock GWAS Data:")

# Simulate p-values (with some significant SNPs)
n_significant = 50
pvalues = rand(n_markers)

# Make some SNPs significant
sig_indices = randperm(n_markers)[1:n_significant]
pvalues[sig_indices] = rand(n_significant) .* 1e-8

gwas_results = GWASResult(
    marker_ids,
    chromosomes,
    positions,
    pvalues,
    nothing,  # effect_sizes
    nothing   # se
)

# Manhattan plot data
manhattan_data = manhattan_plot_data(gwas_results; significance_threshold=5e-8)

println("  Total SNPs: $(length(manhattan_data.x_positions))")
println("  Significant SNPs: $(sum(manhattan_data.significant))")
println("  Chromosomes: $(length(manhattan_data.chr_labels))")

# Save Manhattan data
open("manhattan_data.csv", "w") do io
    println(io, "chr,pos,x_pos,minus_log10_p,significant")
    for i in 1:n_markers
        println(io, "$(chromosomes[i]),$(positions[i]),$(manhattan_data.x_positions[i]),$(manhattan_data.minus_log10_p[i]),$(manhattan_data.significant[i])")
    end
end

println("  ✓ Saved to: manhattan_data.csv")

# QQ plot data
qq_data = qq_plot_data(pvalues; confidence_level=0.95)

println("\n📊 QQ Plot Data:")
@printf("  Genomic inflation factor (λ): %.3f\n", qq_data.lambda)
println("  Data points: $(qq_data.n)")

if qq_data.lambda > 1.1
    println("  ⚠ Warning: λ > 1.1 suggests population stratification!")
    println("    Consider including PCs as covariates in GWAS")
elseif qq_data.lambda < 0.9
    println("  ⚠ Warning: λ < 0.9 suggests over-correction")
else
    println("  ✓ λ is acceptable (0.9 < λ < 1.1)")
end

# Save QQ data
open("qq_data.csv", "w") do io
    println(io, "expected,observed,ci_lower,ci_upper")
    for i in 1:qq_data.n
        println(io, "$(qq_data.expected[i]),$(qq_data.observed[i]),$(qq_data.ci_lower[i]),$(qq_data.ci_upper[i])")
    end
end

println("  ✓ Saved to: qq_data.csv")

# ============================================================================
# 6. LD Heatmap Data
# ============================================================================

println("\n" * "="^80)
println("Step 6: LD Heatmap Data")
println("="^80)

# Generate LD heatmap for a region
region_markers = 1:min(100, n_markers)
ld_data = ld_heatmap_data(geno, collect(region_markers))

println("\n📊 LD Heatmap:")
println("  Markers: $(ld_data.n_markers)")
println("  Mean r²: $(round(mean(ld_data.r2[ld_data.r2 .!= 1.0]), digits=4))")

# Save LD matrix
open("ld_matrix.csv", "w") do io
    # Header
    print(io, "marker")
    for id in ld_data.marker_ids
        print(io, ",$id")
    end
    println(io)

    # Data
    for i in 1:ld_data.n_markers
        print(io, ld_data.marker_ids[i])
        for j in 1:ld_data.n_markers
            print(io, ",$(ld_data.r2[i,j])")
        end
        println(io)
    end
end

println("  ✓ Saved to: ld_matrix.csv")

# ============================================================================
# 7. Plotting Instructions
# ============================================================================

println("\n" * "="^80)
println("Visualization Instructions")
println("="^80)

println("\n📊 To create plots with Plots.jl:")
println("-"^80)

println("""
using Plots
using CSV
using DataFrames

# 1. PCA Plot
pca_df = CSV.read("pca_data.csv", DataFrame)
scatter(pca_df.pc1, pca_df.pc2;
        group=pca_df.population,
        xlabel="PC1",
        ylabel="PC2",
        title="Population Structure",
        legend=:topright)
savefig("pca_plot.png")

# 2. Manhattan Plot
man_df = CSV.read("manhattan_data.csv", DataFrame)
scatter(man_df.x_pos, man_df.minus_log10_p;
        c=ifelse.(man_df.significant, :red, :blue),
        xlabel="Chromosome",
        ylabel="-log₁₀(p-value)",
        title="Manhattan Plot",
        legend=false,
        markersize=2)
hline!([$(round(-log10(5e-8), digits=2))]; linestyle=:dash, color=:red)
savefig("manhattan_plot.png")

# 3. QQ Plot
qq_df = CSV.read("qq_data.csv", DataFrame)
plot(qq_df.expected, qq_df.observed;
     seriestype=:scatter,
     xlabel="Expected -log₁₀(p)",
     ylabel="Observed -log₁₀(p)",
     title="QQ Plot (λ=$(round($(qq_data.lambda), digits=3)))",
     legend=false)
plot!(qq_df.expected, qq_df.expected; linestyle=:dash, color=:red)
savefig("qq_plot.png")

# 4. LD Heatmap
ld_mat = Matrix(CSV.read("ld_matrix.csv", DataFrame)[:, 2:end])
heatmap(ld_mat;
        c=:RdYlBu,
        xlabel="Marker Index",
        ylabel="Marker Index",
        title="LD Heatmap (r²)")
savefig("ld_heatmap.png")
""")

println("-"^80)

# ============================================================================
# 8. Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ PCA and Visualization Example Complete!")

println("\nKey Findings:")
@printf("  • Samples: %d across %d populations\n", n_total, 3)
@printf("  • PC1 variance explained: %.2f%%\n", pca_result.variance_explained[1] * 100)
@printf("  • PC2 variance explained: %.2f%%\n", pca_result.variance_explained[2] * 100)
@printf("  • Population separation: %.4f\n", separation)
@printf("  • Outliers detected: %d\n", length(outliers))
@printf("  • Clustering accuracy: %.1f%%\n", accuracy)
@printf("  • Genomic inflation (λ): %.3f\n", qq_data.lambda)

println("\nFiles Created:")
println("  • pca_data.csv - PCA scatter plot data")
println("  • manhattan_data.csv - Manhattan plot data")
println("  • qq_data.csv - QQ plot data")
println("  • ld_matrix.csv - LD heatmap data")

println("\nNext Steps:")
println("  1. Use the provided plotting code to visualize results")
println("  2. Include top PCs as covariates in GWAS to control for stratification")
println("  3. Remove outlier samples if necessary")
println("  4. Validate population assignments")

println("\n" * "="^80)
println("PCA and visualization workflow completed successfully!")
println("="^80)

# Cleanup (optional)
println("\nCleanup (uncomment to remove files):")
println("  # rm(\"pca_data.csv\")")
println("  # rm(\"manhattan_data.csv\")")
println("  # rm(\"qq_data.csv\")")
println("  # rm(\"ld_matrix.csv\")")
