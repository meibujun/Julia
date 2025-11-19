"""
Cross-Validation Example

This example demonstrates how to use cross-validation for model evaluation:
1. k-fold cross-validation
2. Random sub-sampling validation
3. Comparison of different models
4. Model tuning using CV

Run with: julia --project examples/crossvalidation_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 Cross-Validation Example")
println("="^80)

# Set seed for reproducibility
Random.seed!(123)

# ============================================================================
# 1. Generate Simulated Data
# ============================================================================

println("\n📊 Step 1: Generating simulated data...")

n_samples = 500
n_markers = 2000
n_causal = 100
true_h2 = 0.6

# Generate genotypes
geno_data = rand(0:2, n_samples, n_markers)
sample_ids = ["Individual_$i" for i in 1:n_samples]
marker_ids = ["SNP_$i" for i in 1:n_markers]
geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

# Generate phenotypes with genetic signal
causal_snps = sort(randperm(n_markers)[1:n_causal])
effect_sizes = randn(n_causal)

X = to_matrix(geno; impute=true)
true_u = X[:, causal_snps] * effect_sizes
true_u = (true_u .- mean(true_u)) ./ std(true_u)

var_e = (1 - true_h2) / true_h2
environmental = randn(n_samples) * sqrt(var_e)
phenotypes = true_u .+ environmental

pheno = PhenotypeData(sample_ids, ["Yield"], reshape(phenotypes, n_samples, 1))

println("✓ Data generated:")
println("  Samples: $n_samples")
println("  Markers: $n_markers")
println("  True heritability: $true_h2")
println("  Causal SNPs: $n_causal")

# ============================================================================
# 2. Standard k-Fold Cross-Validation
# ============================================================================

println("\n" * "="^80)
println("Step 2: k-Fold Cross-Validation")
println("="^80)

println("\n📊 Testing different k values...")

k_values = [3, 5, 10, 20]
cv_results = Dict()

for k in k_values
    println("\n  Running $k-fold CV...")

    result = kfold_cv(
        () -> GBLUPModel(method=:cholesky, estimate_variances=false),
        geno,
        pheno;
        k = k,
        seed = 123,
        verbose = false
    )

    cv_results[k] = result

    @printf("    Correlation: %.4f ± %.4f\n",
            result.metrics.mean_fold_correlation,
            result.metrics.std_fold_correlation)
    @printf("    R²: %.4f\n", result.metrics.r_squared)
    @printf("    MSE: %.4f\n", result.metrics.mse)
end

# Choose best k (lowest variance in correlations)
best_k = k_values[argmin([cv_results[k].metrics.std_fold_correlation for k in k_values])]
println("\n✓ Best k value (lowest variance): $best_k")

# ============================================================================
# 3. Detailed Analysis of 5-Fold CV
# ============================================================================

println("\n" * "="^80)
println("Step 3: Detailed 5-Fold CV Analysis")
println("="^80)

result_5fold = kfold_cv(
    () -> GBLUPModel(method=:cholesky, estimate_variances=true),
    geno,
    pheno;
    k = 5,
    seed = 123,
    verbose = true
)

# Display full result
println("\nFull results:")
println(result_5fold)

# Per-fold analysis
println("\n📊 Per-fold breakdown:")
println("─"^80)
@printf("%-6s %10s %10s %12s %10s %10s\n",
        "Fold", "N Train", "N Test", "Correlation", "MSE", "R²")
println("─"^80)

for fold_res in result_5fold.fold_results
    @printf("%-6d %10d %10d %12.4f %10.4f %10.4f\n",
            fold_res.fold,
            fold_res.n_train,
            fold_res.n_test,
            fold_res.correlation,
            fold_res.mse,
            fold_res.r_squared)
end
println("─"^80)

# Prediction vs Observed scatter analysis
println("\n📈 Prediction Analysis:")
cor_pred_obs = cor(result_5fold.predictions, result_5fold.observed)
println("  Overall correlation: $(@sprintf("%.4f", cor_pred_obs))")
println("  Mean bias: $(@sprintf("%.4f", result_5fold.metrics.bias))")
println("  Regression slope: $(@sprintf("%.4f", result_5fold.metrics.regression_slope))")

if abs(result_5fold.metrics.regression_slope - 1.0) > 0.2
    println("  ⚠ Regression slope differs from 1.0 - predictions may be biased")
end

# ============================================================================
# 4. Random Sub-sampling Validation
# ============================================================================

println("\n" * "="^80)
println("Step 4: Random Sub-sampling Validation")
println("="^80)

println("\n📊 Testing different test fractions...")

test_fractions = [0.1, 0.2, 0.3, 0.4]
random_results = Dict()

for frac in test_fractions
    println("\n  Running random CV with $(frac*100)% test data...")

    result = random_cv(
        () -> GBLUPModel(method=:cholesky, estimate_variances=false),
        geno,
        pheno;
        n_reps = 20,
        test_fraction = frac,
        seed = 123,
        verbose = false
    )

    random_results[frac] = result

    @printf("    Correlation: %.4f ± %.4f\n",
            result.metrics.mean_fold_correlation,
            result.metrics.std_fold_correlation)
end

# ============================================================================
# 5. Comparison: k-Fold vs Random
# ============================================================================

println("\n" * "="^80)
println("Step 5: k-Fold vs Random Sub-sampling Comparison")
println("="^80)

println("\nMethod Comparison:")
println("─"^80)
@printf("%-30s %12s %12s\n", "Method", "Correlation", "Std Dev")
println("─"^80)

@printf("%-30s %12.4f %12.4f\n",
        "5-fold CV",
        result_5fold.metrics.mean_fold_correlation,
        result_5fold.metrics.std_fold_correlation)

@printf("%-30s %12.4f %12.4f\n",
        "Random (20%, 20 reps)",
        random_results[0.2].metrics.mean_fold_correlation,
        random_results[0.2].metrics.std_fold_correlation)

println("─"^80)

# ============================================================================
# 6. Model Comparison Using CV
# ============================================================================

println("\n" * "="^80)
println("Step 6: Comparing Different Models")
println("="^80)

println("\n📊 Comparing Cholesky vs PCG solvers...")

# Cholesky solver
result_chol = kfold_cv(
    () -> GBLUPModel(method=:cholesky, estimate_variances=false),
    geno,
    pheno;
    k = 5,
    seed = 123,
    verbose = false
)

# PCG solver
result_pcg = kfold_cv(
    () -> GBLUPModel(method=:pcg, max_iter=1000, estimate_variances=false),
    geno,
    pheno;
    k = 5,
    seed = 123,
    verbose = false
)

println("\nSolver Comparison:")
println("─"^80)
@printf("%-20s %12s %10s %10s\n", "Solver", "Correlation", "R²", "MSE")
println("─"^80)
@printf("%-20s %12.4f %10.4f %10.4f\n",
        "Cholesky",
        result_chol.metrics.correlation,
        result_chol.metrics.r_squared,
        result_chol.metrics.mse)
@printf("%-20s %12.4f %10.4f %10.4f\n",
        "PCG",
        result_pcg.metrics.correlation,
        result_pcg.metrics.r_squared,
        result_pcg.metrics.mse)
println("─"^80)

# ============================================================================
# 7. Effect of GRM Computation Options
# ============================================================================

println("\n" * "="^80)
println("Step 7: GRM Options Impact on Prediction")
println("="^80)

println("\n📊 Testing different MAF thresholds...")

maf_thresholds = [0.0, 0.01, 0.05, 0.10]
maf_results = Dict()

for maf in maf_thresholds
    result = kfold_cv(
        () -> GBLUPModel(method=:cholesky, estimate_variances=false),
        geno,
        pheno;
        k = 5,
        grm_options = (min_maf = maf, method = :vanraden),
        seed = 123,
        verbose = false
    )

    maf_results[maf] = result
    @printf("  MAF ≥ %.2f: Correlation = %.4f\n", maf, result.metrics.correlation)
end

best_maf = maf_thresholds[argmax([maf_results[maf].metrics.correlation for maf in maf_thresholds])]
println("\n✓ Best MAF threshold: $best_maf")

# ============================================================================
# 8. Accuracy vs True Heritability
# ============================================================================

println("\n" * "="^80)
println("Step 8: Prediction Accuracy Assessment")
println("="^80)

# Compare CV accuracy to theoretical maximum
theoretical_max = sqrt(true_h2)  # Upper bound on accuracy

println("\nAccuracy Assessment:")
println("  True heritability (h²): $(@sprintf("%.4f", true_h2))")
println("  Theoretical max accuracy: $(@sprintf("%.4f", theoretical_max))")
println("  Observed CV accuracy: $(@sprintf("%.4f", result_5fold.metrics.correlation))")
println("  Accuracy/Max ratio: $(@sprintf("%.2f%%", 100 * result_5fold.metrics.correlation / theoretical_max))")

realized_h2 = result_5fold.metrics.r_squared
println("\n  Realized heritability: $(@sprintf("%.4f", realized_h2))")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("Summary and Recommendations")
println("="^80)

println("\n✅ Cross-Validation Complete!")

println("\nKey Findings:")
println("  • Best k-fold value: $best_k (lowest variance)")
println("  • Best MAF threshold: $best_maf")
println("  • CV Accuracy: $(@sprintf("%.4f", result_5fold.metrics.correlation))")
println("  • Prediction R²: $(@sprintf("%.4f", result_5fold.metrics.r_squared))")
println("  • Model bias: $(@sprintf("%.4f", result_5fold.metrics.bias))")

println("\nRecommendations:")
if result_5fold.metrics.correlation > 0.5
    println("  ✓ Good prediction accuracy achieved")
elseif result_5fold.metrics.correlation > 0.3
    println("  ⚠ Moderate prediction accuracy - consider:")
    println("    - Increasing marker density")
    println("    - Adding more training samples")
    println("    - Trying alternative models (BayesR, etc.)")
else
    println("  ✗ Low prediction accuracy - check:")
    println("    - Data quality (run QC)")
    println("    - Heritability estimation")
    println("    - Population structure")
end

if abs(result_5fold.metrics.regression_slope - 1.0) > 0.2
    println("  ⚠ Prediction bias detected - model may need calibration")
end

println("\n" * "="^80)
println("Cross-validation workflow completed successfully!")
println("="^80)
