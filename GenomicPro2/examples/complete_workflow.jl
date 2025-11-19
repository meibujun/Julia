"""
Complete GenomicPro2 Workflow Example

This example demonstrates a complete genomic prediction workflow:
1. Data generation (simulated)
2. Quality control
3. GRM computation
4. GBLUP model training
5. Cross-validation
6. Prediction

Run with: julia --project examples/complete_workflow.jl
"""

using GenomicPro2
using LinearAlgebra
using Statistics
using Printf

println("="^80)
println("GenomicPro2 Complete Workflow Example")
println("="^80)

# ============================================================================
# 1. Generate Simulated Data
# ============================================================================

println("\n📊 Step 1: Generating simulated data...")

# Simulation parameters
n_samples = 500
n_markers = 5000
n_causal = 100
true_h2 = 0.6

# Generate genotypes
println("  - Creating $n_samples samples × $n_markers SNPs")
geno_data = rand(0:2, n_samples, n_markers)

# Add some missing values (5%)
n_missing = round(Int, 0.05 * n_samples * n_markers)
for _ in 1:n_missing
    i, j = rand(1:n_samples), rand(1:n_markers)
    geno_data[i, j] = missing
end

sample_ids = ["Individual_$i" for i in 1:n_samples]
marker_ids = ["SNP_$i" for i in 1:n_markers]

geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

println("  ✓ Genotype data created")
println("    - Missing rate: $(@sprintf("%.2f%%", missing_rate(geno) * 100))")

mem = memory_usage(geno)
println("    - Memory usage: $(@sprintf("%.2f MB", mem.total / 1e6))")
println("    - Memory savings: $(@sprintf("%.1f%%", mem.savings * 100))")

# Generate phenotypes with genetic signal
println("\n  - Generating phenotypes (h² = $true_h2)")

# Select causal SNPs
causal_snps = sort(randperm(n_markers)[1:n_causal])
effect_sizes = randn(n_causal)

# Compute true breeding values
X = to_matrix(geno; impute=true)
true_u = X[:, causal_snps] * effect_sizes

# Standardize to unit variance
true_u = (true_u .- mean(true_u)) ./ std(true_u)

# Add environmental noise to achieve target h²
var_u = true_h2
var_e = 1 - true_h2
environmental = randn(n_samples) * sqrt(var_e)

phenotypes = true_u .+ environmental

pheno = PhenotypeData(
    sample_ids,
    ["Yield"],
    reshape(phenotypes, n_samples, 1)
)

println("  ✓ Phenotype data created")
println("    - Mean: $(@sprintf("%.3f", mean(phenotypes)))")
println("    - SD: $(@sprintf("%.3f", std(phenotypes)))")

# ============================================================================
# 2. Quality Control
# ============================================================================

println("\n🔍 Step 2: Quality control...")

# Filter by MAF
println("  - Calculating minor allele frequencies...")
maf = minor_allele_frequency(geno)

min_maf = 0.01
low_maf_count = count(maf .< min_maf)
println("    - SNPs with MAF < $min_maf: $low_maf_count (will be filtered in GRM)")

# Check for high missingness
marker_missing = missing_rate(geno; dim=2)
high_missing_threshold = 0.1
high_missing_count = count(marker_missing .> high_missing_threshold)

if high_missing_count > 0
    println("    ⚠ Warning: $high_missing_count SNPs have >$(high_missing_threshold*100)% missing rate")
end

println("  ✓ Quality control complete")

# ============================================================================
# 3. Compute Genomic Relationship Matrix
# ============================================================================

println("\n🧮 Step 3: Computing genomic relationship matrix...")

@time G = compute_grm(geno; method=:vanraden, min_maf=min_maf)

println("  ✓ GRM computed")
println("    - Dimensions: $(size(G))")
println("    - Mean diagonal: $(@sprintf("%.3f", mean(diag(G))))")
println("    - Min/Max off-diagonal: $(@sprintf("%.3f", minimum(G[i,j] for i in 1:n_samples for j in 1:n_samples if i != j))) / $(@sprintf("%.3f", maximum(G[i,j] for i in 1:n_samples for j in 1:n_samples if i != j)))")

# Validate GRM
println("\n  - Validating GRM...")
grm_validation = validate_grm(G)

if is_valid(grm_validation)
    println("    ✓ GRM validation passed")
    if haskey(grm_validation.metadata, :min_eigenvalue)
        println("      - Min eigenvalue: $(@sprintf("%.2e", grm_validation.metadata[:min_eigenvalue]))")
    end
else
    println("    ✗ GRM validation failed:")
    for err in grm_validation.errors
        println("      - $err")
    end
end

if !isempty(grm_validation.warnings)
    println("    Warnings:")
    for warn in grm_validation.warnings
        println("      - $warn")
    end
end

# ============================================================================
# 4. Train GBLUP Model
# ============================================================================

println("\n🎓 Step 4: Training GBLUP model...")

model = GBLUPModel(
    method = :cholesky,
    estimate_variances = true,
    ridge = 1e-5
)

println("  - Solver method: $(model.method)")
println("  - Estimating variance components: $(model.estimate_variances)")

@time result = fit!(model, geno, pheno; G=G, trait_index=1)

println("\n  ✓ Model training complete")
println()
println("  Results:")
println("    - Heritability (h²): $(@sprintf("%.3f", result.heritability)) (true: $true_h2)")
println("    - Genetic variance (σ²ᵤ): $(@sprintf("%.3f", result.var_u))")
println("    - Residual variance (σ²ₑ): $(@sprintf("%.3f", result.var_e))")
println("    - Log-likelihood: $(@sprintf("%.2f", result.log_likelihood))")
println("    - Converged: $(result.converged)")
println("    - Iterations: $(result.iterations)")

# ============================================================================
# 5. Cross-Validation
# ============================================================================

println("\n📈 Step 5: 5-fold cross-validation...")

n_folds = 5
fold_size = n_samples ÷ n_folds
correlations = Float64[]
mse_values = Float64[]

# Shuffle samples
shuffled_idx = randperm(n_samples)

for fold in 1:n_folds
    # Split into training and testing
    test_start = (fold - 1) * fold_size + 1
    test_end = min(fold * fold_size, n_samples)
    test_idx = shuffled_idx[test_start:test_end]
    train_idx = setdiff(shuffled_idx, test_idx)

    # Subset data
    geno_train = subset_samples(geno, train_idx)
    pheno_train_vals = phenotypes[train_idx]
    pheno_train = PhenotypeData(
        sample_ids[train_idx],
        ["Yield"],
        reshape(pheno_train_vals, length(train_idx), 1)
    )

    geno_test = subset_samples(geno, test_idx)
    pheno_test_true = phenotypes[test_idx]

    # Compute GRM for training set
    G_train = compute_grm(geno_train; method=:vanraden, min_maf=min_maf)

    # Train model
    model_cv = GBLUPModel(method=:cholesky, estimate_variances=false)
    fit!(model_cv, geno_train, pheno_train; G=G_train)

    # Predict test set
    predictions = predict(model_cv, geno_test)

    # Evaluate
    valid_pred_idx = findall(predictions .!= 0)  # Only samples in training set
    if !isempty(valid_pred_idx)
        corr = cor(predictions[valid_pred_idx], pheno_test_true[valid_pred_idx])
        mse = mean((predictions[valid_pred_idx] .- pheno_test_true[valid_pred_idx]).^2)

        push!(correlations, corr)
        push!(mse_values, mse)

        println("    Fold $fold: r = $(@sprintf("%.3f", corr)), MSE = $(@sprintf("%.3f", mse))")
    else
        println("    Fold $fold: No predictions (no overlap with training)")
    end
end

if !isempty(correlations)
    println("\n  ✓ Cross-validation complete")
    println("    - Mean correlation: $(@sprintf("%.3f ± %.3f", mean(correlations), std(correlations)))")
    println("    - Mean MSE: $(@sprintf("%.3f ± %.3f", mean(mse_values), std(mse_values)))")

    # Theoretical accuracy for h² and training size
    # Accuracy ≈ √(h² × reliability)
    # where reliability depends on #markers and training size
    println("\n    - Theoretical max accuracy: $(@sprintf("%.3f", sqrt(true_h2)))")
end

# ============================================================================
# 6. Prediction on Full Dataset
# ============================================================================

println("\n🎯 Step 6: Final predictions on full dataset...")

predictions = predict(model, geno)

println("  ✓ Predictions complete")
println("    - Mean GEBV: $(@sprintf("%.3f", mean(predictions)))")
println("    - SD GEBV: $(@sprintf("%.3f", std(predictions)))")

# Correlation with true breeding values
corr_true = cor(predictions, true_u)
println("    - Correlation with true breeding values: $(@sprintf("%.3f", corr_true))")

# Top 10% selection
n_select = round(Int, 0.1 * n_samples)
top_idx = sortperm(predictions, rev=true)[1:n_select]
mean_true_top = mean(true_u[top_idx])
mean_true_overall = mean(true_u)
selection_gain = mean_true_top - mean_true_overall

println("\n  📊 Selection results (top 10%):")
println("    - Mean true breeding value (top 10%): $(@sprintf("%.3f", mean_true_top))")
println("    - Mean true breeding value (overall): $(@sprintf("%.3f", mean_true_overall))")
println("    - Selection gain: $(@sprintf("%.3f", selection_gain))")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("Workflow Summary")
println("="^80)
println("✓ Data: $n_samples samples × $n_markers SNPs")
println("✓ True heritability: $true_h2")
println("✓ Estimated heritability: $(@sprintf("%.3f", result.heritability))")
if !isempty(correlations)
    println("✓ Cross-validation accuracy: $(@sprintf("%.3f", mean(correlations)))")
end
println("✓ Prediction accuracy: $(@sprintf("%.3f", corr_true))")
println("✓ Selection gain: $(@sprintf("%.3f", selection_gain))")
println("="^80)

println("\n✅ Workflow completed successfully!")
