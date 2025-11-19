"""
BayesR Example

This example demonstrates BayesR for genomic prediction:
1. Simulating data with sparse genetic architecture
2. Fitting BayesR model
3. Variable selection and effect estimation
4. Comparison with GBLUP
5. Cross-validation for prediction accuracy

Run with: julia --project examples/bayesr_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 BayesR Example")
println("="^80)

# Set seed for reproducibility
Random.seed!(2024)

# ============================================================================
# 1. Generate Simulated Data with Sparse Architecture
# ============================================================================

println("\n" * "="^80)
println("Step 1: Generating Simulated Data")
println("="^80)

n_samples = 1000
n_markers = 5000
n_causal = 100  # Only 2% of SNPs have effects
true_h2 = 0.65

println("\nDataset properties:")
println("  Samples: $n_samples")
println("  Markers: $n_markers")
println("  Causal SNPs: $n_causal ($(100*n_causal/n_markers)%)")
println("  True heritability: $true_h2")

# Generate genotype data
geno_data = rand(0:2, n_samples, n_markers)
sample_ids = [string("ID", i) for i in 1:n_samples]
marker_ids = [string("SNP", i) for i in 1:n_markers]
geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

# Create sparse genetic architecture (mimicking real quantitative traits)
# Following a mixture model:
# - 10 large effect SNPs (explaining 40% of genetic variance)
# - 30 medium effect SNPs (explaining 30% of genetic variance)
# - 60 small effect SNPs (explaining 30% of genetic variance)
# - Rest have zero effect

X = to_matrix(geno; impute=true)
true_effects = zeros(n_markers)

causal_idx = sort(randperm(n_markers)[1:n_causal])

# Large effects (top 10)
large_idx = causal_idx[1:10]
true_effects[large_idx] = randn(10) * 0.15

# Medium effects (next 30)
medium_idx = causal_idx[11:40]
true_effects[medium_idx] = randn(30) * 0.08

# Small effects (next 60)
small_idx = causal_idx[41:100]
true_effects[small_idx] = randn(60) * 0.03

# Generate true breeding values
true_tbv = X * true_effects
true_tbv = (true_tbv .- mean(true_tbv))
true_tbv = true_tbv .* sqrt(true_h2 / var(true_tbv))

# Add environmental noise
environmental = randn(n_samples) * sqrt(1 - true_h2)
phenotypes = true_tbv .+ environmental

pheno = PhenotypeData(sample_ids, ["Yield"], reshape(phenotypes, n_samples, 1))

println("\n✓ Data generated with sparse genetic architecture")
println("  Large effect SNPs (n=10): mean |β| = $(@sprintf("%.4f", mean(abs.(true_effects[large_idx]))))")
println("  Medium effect SNPs (n=30): mean |β| = $(@sprintf("%.4f", mean(abs.(true_effects[medium_idx]))))")
println("  Small effect SNPs (n=60): mean |β| = $(@sprintf("%.4f", mean(abs.(true_effects[small_idx]))))")

# ============================================================================
# 2. Fit BayesR Model
# ============================================================================

println("\n" * "="^80)
println("Step 2: Fitting BayesR Model")
println("="^80)

model_bayesr = BayesRModel(
    n_iter = 50000,
    burn_in = 20000,
    thin = 10,
    mixture_proportions = [0.50, 0.30, 0.15, 0.05],  # Initial guess
    update_pi = true,  # Estimate mixture proportions from data
    seed = 123,
    verbose = true
)

println("\nFitting BayesR...")
fit!(model_bayesr, geno, pheno; min_maf=0.01)

result = model_bayesr.result

# ============================================================================
# 3. Variable Selection Results
# ============================================================================

println("\n" * "="^80)
println("Step 3: Variable Selection Analysis")
println("="^80)

# Identify selected SNPs (PIP > 0.5)
selected_idx = findall(result.marker_pip .> 0.5)
n_selected = length(selected_idx)

println("\n📊 Variable Selection Results:")
println("  Total markers: $n_markers")
println("  Selected (PIP > 0.5): $n_selected ($(100*n_selected/n_markers)%)")
println("  True causal: $n_causal")

# Check overlap with true causal SNPs
causal_set = Set(causal_idx)
selected_set = Set(selected_idx)
true_positives = length(intersect(causal_set, selected_set))
false_positives = n_selected - true_positives
false_negatives = n_causal - true_positives

println("\nConfusion Matrix:")
println("  True Positives: $true_positives")
println("  False Positives: $false_positives")
println("  False Negatives: $false_negatives")

if n_selected > 0
    precision = true_positives / n_selected
    recall = true_positives / n_causal
    f1_score = 2 * precision * recall / (precision + recall)

    @printf("\nPerformance Metrics:\n")
    @printf("  Precision: %.4f\n", precision)
    @printf("  Recall: %.4f\n", recall)
    @printf("  F1-Score: %.4f\n", f1_score)
end

# ============================================================================
# 4. Effect Size Estimation
# ============================================================================

println("\n" * "="^80)
println("Step 4: Effect Size Estimation")
println("="^80)

# Compare estimated vs true effects for causal SNPs
println("\n📈 Effect Size Estimates for Causal SNPs:")

# Correlation between true and estimated effects
cor_effects = cor(true_effects[causal_idx], result.marker_effects[causal_idx])
@printf("  Correlation (true vs estimated): %.4f\n", cor_effects)

# Breakdown by effect size class
println("\nBy Effect Size Class:")

for (name, idx_set) in [("Large", large_idx), ("Medium", medium_idx), ("Small", small_idx)]
    true_mean = mean(abs.(true_effects[idx_set]))
    est_mean = mean(abs.(result.marker_effects[idx_set]))
    pip_mean = mean(result.marker_pip[idx_set])

    @printf("  %s effects:\n", name)
    @printf("    True mean |β|: %.6f\n", true_mean)
    @printf("    Est. mean |β|: %.6f\n", est_mean)
    @printf("    Mean PIP: %.4f\n", pip_mean)
end

# Top 20 SNPs by PIP
println("\n📋 Top 20 SNPs by Posterior Inclusion Probability:")
println("─"^80)
@printf("%-8s %8s %12s %12s %8s\n", "Rank", "SNP", "PIP", "Effect", "True?")
println("─"^80)

top_indices = sortperm(result.marker_pip, rev=true)[1:min(20, n_markers)]
for (rank, idx) in enumerate(top_indices)
    is_causal = idx in causal_idx
    marker = "✓" if is_causal else ""
    @printf("%-8d %8s %12.4f %12.6f %8s\n",
            rank, marker_ids[idx], result.marker_pip[idx],
            result.marker_effects[idx], marker)
end
println("─"^80)

# ============================================================================
# 5. Variance Components
# ============================================================================

println("\n" * "="^80)
println("Step 5: Variance Component Estimation")
println("="^80)

println("\nEstimated vs True Variance Components:")
println("─"^60)
@printf("%-25s %12s %12s\n", "Component", "Estimated", "True")
println("─"^60)

true_var_g = var(true_tbv)
true_var_e = var(environmental)
total_var = var(phenotypes)

@printf("%-25s %12.4f %12.4f\n", "Genetic variance", result.genetic_variance, true_var_g)
@printf("%-25s %12.4f %12.4f\n", "Residual variance", result.residual_variance, true_var_e)
@printf("%-25s %12.4f %12.4f\n", "Total variance",
        result.genetic_variance + result.residual_variance, total_var)
@printf("%-25s %12.4f %12.4f\n", "Heritability", result.heritability, true_h2)
println("─"^60)

h2_error = abs(result.heritability - true_h2)
if h2_error < 0.05
    println("\n✓ Excellent heritability estimation (error < 0.05)")
elseif h2_error < 0.10
    println("\n✓ Good heritability estimation (error < 0.10)")
else
    println("\n⚠ Moderate heritability estimation accuracy")
end

# ============================================================================
# 6. Prediction Accuracy
# ============================================================================

println("\n" * "="^80)
println("Step 6: Prediction Accuracy on Training Data")
println("="^80)

gebv_bayesr = result.gebv_train

# Correlation with true breeding values
cor_bayesr = cor(gebv_bayesr, true_tbv)
r2_bayesr = cor_bayesr^2

println("\nBayesR Prediction Performance:")
@printf("  Correlation with TBV: %.4f\n", cor_bayesr)
@printf("  R² with TBV: %.4f\n", r2_bayesr)
@printf("  Theoretical max: %.4f\n", sqrt(true_h2))
@printf("  Prediction efficiency: %.1f%%\n", 100 * cor_bayesr / sqrt(true_h2))

# ============================================================================
# 7. Comparison with GBLUP
# ============================================================================

println("\n" * "="^80)
println("Step 7: Comparison with GBLUP")
println("="^80)

println("\n🔬 Fitting GBLUP for comparison...")
model_gblup = GBLUPModel(method=:cholesky, estimate_variances=true)
fit!(model_gblup, geno, pheno; min_maf=0.01)

gebv_gblup = predict(model_gblup, geno)

cor_gblup = cor(gebv_gblup, true_tbv)
r2_gblup = cor_gblup^2

println("\nModel Comparison:")
println("─"^80)
@printf("%-20s %15s %15s %15s\n", "Model", "Correlation", "R²", "h² estimate")
println("─"^80)
@printf("%-20s %15.4f %15.4f %15.4f\n", "BayesR", cor_bayesr, r2_bayesr, result.heritability)
@printf("%-20s %15.4f %15.4f %15.4f\n", "GBLUP", cor_gblup, r2_gblup, model_gblup.result.h2)
@printf("%-20s %15.4f %15.4f %15.4f\n", "True", sqrt(true_h2), true_h2, true_h2)
println("─"^80)

improvement = (cor_bayesr - cor_gblup) / cor_gblup * 100
@printf("\nBayesR improvement over GBLUP: %.2f%%\n", improvement)

if improvement > 5
    println("✓ BayesR shows substantial improvement (>5%)")
    println("  This is expected for traits with sparse genetic architecture")
elseif improvement > 0
    println("✓ BayesR shows modest improvement")
else
    println("⚠ GBLUP performs similarly or better")
    println("  This can happen when:")
    println("  - Genetic architecture is highly polygenic")
    println("  - MCMC hasn't fully converged")
    println("  - Dataset is small")
end

# ============================================================================
# 8. Cross-Validation
# ============================================================================

println("\n" * "="^80)
println("Step 8: 5-Fold Cross-Validation")
println("="^80)

println("\n🔬 Running 5-fold CV for BayesR...")
println("  (This may take a few minutes...)\n")

# For CV, use shorter MCMC to save time
cv_result_bayesr = kfold_cv(
    () -> BayesRModel(n_iter=10000, burn_in=5000, thin=10, verbose=false, seed=123),
    geno,
    pheno;
    k = 5,
    grm_options = (min_maf = 0.01,),
    seed = 123,
    verbose = true
)

println("\n🔬 Running 5-fold CV for GBLUP...")
cv_result_gblup = kfold_cv(
    () -> GBLUPModel(method=:cholesky, estimate_variances=false),
    geno,
    pheno;
    k = 5,
    grm_options = (min_maf = 0.01, method = :vanraden),
    seed = 123,
    verbose = true
)

println("\n" * "="^80)
println("Cross-Validation Results")
println("="^80)

println("\n5-Fold CV Performance:")
println("─"^80)
@printf("%-20s %15s %15s %15s\n", "Model", "Correlation", "R²", "MSE")
println("─"^80)
@printf("%-20s %15.4f %15.4f %15.4f\n",
        "BayesR",
        cv_result_bayesr.metrics.correlation,
        cv_result_bayesr.metrics.r_squared,
        cv_result_bayesr.metrics.mse)
@printf("%-20s %15.4f %15.4f %15.4f\n",
        "GBLUP",
        cv_result_gblup.metrics.correlation,
        cv_result_gblup.metrics.r_squared,
        cv_result_gblup.metrics.mse)
println("─"^80)

cv_improvement = (cv_result_bayesr.metrics.correlation - cv_result_gblup.metrics.correlation) /
                 cv_result_gblup.metrics.correlation * 100
@printf("\nCV: BayesR improvement: %.2f%%\n", cv_improvement)

# ============================================================================
# 9. Practical Recommendations
# ============================================================================

println("\n" * "="^80)
println("Practical Recommendations")
println("="^80)

println("\n💡 When to Use BayesR:")
println("  ✓ Traits with sparse genetic architecture (few large effects)")
println("  ✓ When variable selection is important")
println("  ✓ QTL mapping and fine-mapping applications")
println("  ✓ When you need effect size estimates with uncertainty")

println("\n💡 When to Use GBLUP:")
println("  ✓ Highly polygenic traits (many small effects)")
println("  ✓ When computational speed is critical")
println("  ✓ Routine genomic evaluation")
println("  ✓ Very large datasets (>100k samples)")

println("\n💡 BayesR Tuning Tips:")
println("  • MCMC length: 50k-100k iterations for production")
println("  • Burn-in: At least 20-40% of total iterations")
println("  • Thinning: 5-20 to reduce autocorrelation")
println("  • Mixture proportions: Let the model estimate them (update_pi=true)")
println("  • Check convergence: Run multiple chains with different seeds")

println("\n💡 Interpreting Results:")
@printf("  • PIP > 0.9: Strong evidence for QTL\n")
@printf("  • PIP > 0.5: Moderate evidence for QTL\n")
@printf("  • PIP < 0.2: Weak/no evidence\n")
println("  • Focus on top SNPs for downstream validation")

# ============================================================================
# 10. Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ BayesR Analysis Complete!")

println("\nKey Results:")
@printf("  • Heritability: %.4f (true: %.4f)\n", result.heritability, true_h2)
@printf("  • QTL detected (PIP > 0.5): %d (true: %d)\n", n_selected, n_causal)
if n_selected > 0
    @printf("  • Detection precision: %.2f%%\n", 100 * true_positives / n_selected)
    @printf("  • Detection recall: %.2f%%\n", 100 * true_positives / n_causal)
end
@printf("  • Prediction accuracy: %.4f\n", cor_bayesr)
@printf("  • CV prediction: %.4f ± %.4f\n",
        cv_result_bayesr.metrics.mean_fold_correlation,
        cv_result_bayesr.metrics.std_fold_correlation)

println("\nModel Strengths Demonstrated:")
if improvement > 5
    println("  ✓ Superior prediction for sparse architecture")
end
if precision > 0.5
    println("  ✓ Good QTL detection precision")
end
if result.heritability > true_h2 - 0.1 && result.heritability < true_h2 + 0.1
    println("  ✓ Accurate heritability estimation")
end

println("\n" * "="^80)
println("BayesR workflow completed successfully!")
println("="^80)
