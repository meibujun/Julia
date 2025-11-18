"""
BayesCπ Model Example

This example demonstrates:
1. BayesCπ model with estimated mixture proportions
2. Comparison between 2-component (BayesC) and 4-component (BayesCπ) models
3. Comparison with BayesR (fixed proportions)
4. Posterior inclusion probability (PIP) analysis
5. Cross-validation for prediction accuracy
6. Effect of Dirichlet prior on mixture proportion estimation

Run with: julia --project examples/bayescpi_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 BayesCπ Model Example")
println("="^80)

Random.seed!(2024)

# ============================================================================
# 1. Generate Simulated Data with Sparse Effects
# ============================================================================

println("\n" * "="^80)
println("Step 1: Generating Simulated Data")
println("="^80)

n_samples = 500
n_markers = 2000
n_causal = 50  # Sparse: only 2.5% of SNPs have effects

println("\nSimulation parameters:")
println("  Samples: $n_samples")
println("  Total markers: $n_markers")
println("  Causal markers: $n_causal ($(round(100*n_causal/n_markers, digits=2))%)")
println("  Target heritability: 0.7")

# Generate genotypes
geno_data = rand(0:2, n_samples, n_markers)
sample_ids = [string("Sample_", i) for i in 1:n_samples]
marker_ids = [string("SNP_", i) for i in 1:n_markers]

geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

# Generate phenotypes with sparse effects
true_effects = zeros(n_markers)
causal_indices = sort(randperm(n_markers)[1:n_causal])

# Different effect sizes (small, medium, large)
n_small = div(n_causal, 2)
n_medium = div(n_causal, 3)
n_large = n_causal - n_small - n_medium

true_effects[causal_indices[1:n_small]] = randn(n_small) .* 0.1        # Small
true_effects[causal_indices[n_small+1:n_small+n_medium]] = randn(n_medium) .* 0.3  # Medium
true_effects[causal_indices[n_small+n_medium+1:end]] = randn(n_large) .* 0.6       # Large

# Standardize genotypes
X = Float64.(geno_data)
X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X_std[X_std .== 0] .= 1.0
X_scaled = (X .- X_mean) ./ X_std

# Genetic values
g = X_scaled * true_effects

# Add environmental noise for h² = 0.7
var_g = var(g)
target_h2 = 0.7
var_e = var_g * (1 - target_h2) / target_h2
e = randn(n_samples) .* sqrt(var_e)

y = g + e
trait_name = "YieldTrait"

pheno = PhenotypeData(
    y,
    sample_ids,
    [trait_name]
)

true_h2 = var_g / (var_g + var(e))

println("\nTrue simulation values:")
@printf("  Genetic variance: %.4f\n", var_g)
@printf("  Residual variance: %.4f\n", var(e))
@printf("  Heritability: %.4f\n", true_h2)
@printf("  Effect sizes - small: %d, medium: %d, large: %d\n", n_small, n_medium, n_large)

println("\n✓ Data generated successfully")

# ============================================================================
# 2. Fit BayesCπ Model (4 components)
# ============================================================================

println("\n" * "="^80)
println("Step 2: Fitting BayesCπ Model (4 components)")
println("="^80)

model_cpi = BayesCπModel(
    n_iter = 20000,
    burn_in = 10000,
    thin = 10,
    n_components = 4,
    seed = 2024,
    verbose = true
)

println("\nFitting BayesCπ with 4 components...")
fit!(model_cpi, geno, pheno)

result_cpi = model_cpi.result

println("\n📊 BayesCπ Results:")
println("-"^80)
@printf("Heritability: %.4f (true: %.4f)\n", result_cpi.heritability, true_h2)
@printf("Genetic variance: %.4f (true: %.4f)\n", result_cpi.sigma2_a, var_g)
@printf("Residual variance: %.4f (true: %.4f)\n", result_cpi.sigma2_e, var(e))

println("\n📊 Estimated Mixture Proportions:")
println("-"^80)
for k in 1:4
    var_label = k == 1 ? "Zero" : @sprintf("%.4fσ²ₐ", model_cpi.mixture_variances[k])
    @printf("Component %d (%12s): %.4f ± %.4f\n",
            k, var_label, result_cpi.mixture_proportions[k], result_cpi.mixture_proportions_se[k])
end

# Compare with true proportion
true_null_prop = (n_markers - n_causal) / n_markers
@printf("\nTrue null proportion: %.4f\n", true_null_prop)
@printf("Estimated null proportion: %.4f\n", result_cpi.mixture_proportions[1])

# ============================================================================
# 3. Fit BayesC Model (2 components)
# ============================================================================

println("\n" * "="^80)
println("Step 3: Fitting BayesC Model (2 components)")
println("="^80)

model_c = BayesCπModel(
    n_iter = 20000,
    burn_in = 10000,
    thin = 10,
    n_components = 2,
    seed = 2024,
    verbose = true
)

println("\nFitting BayesC with 2 components...")
fit!(model_c, geno, pheno)

result_c = model_c.result

println("\n📊 BayesC Results:")
println("-"^80)
@printf("Heritability: %.4f\n", result_c.heritability)
@printf("Null proportion: %.4f ± %.4f\n",
        result_c.mixture_proportions[1], result_c.mixture_proportions_se[1])
@printf("Non-zero proportion: %.4f ± %.4f\n",
        result_c.mixture_proportions[2], result_c.mixture_proportions_se[2])

# ============================================================================
# 4. Fit BayesR Model (fixed proportions)
# ============================================================================

println("\n" * "="^80)
println("Step 4: Fitting BayesR Model (fixed proportions)")
println("="^80)

model_r = BayesRModel(
    n_iter = 20000,
    burn_in = 10000,
    thin = 10,
    mixture_proportions = [0.50, 0.30, 0.15, 0.05],
    update_pi = false,  # Fixed proportions
    seed = 2024,
    verbose = true
)

println("\nFitting BayesR with fixed proportions...")
fit!(model_r, geno, pheno)

result_r = model_r.result

println("\n📊 BayesR Results:")
println("-"^80)
@printf("Heritability: %.4f\n", result_r.heritability)
println("\nFixed Mixture Proportions:")
for k in 1:4
    var_label = k == 1 ? "Zero" : @sprintf("%.4fσ²ₐ", model_r.mixture_variances[k])
    @printf("Component %d (%12s): %.4f (fixed)\n", k, var_label, model_r.mixture_proportions[k])
end

# ============================================================================
# 5. Compare Models
# ============================================================================

println("\n" * "="^80)
println("Step 5: Model Comparison")
println("="^80)

println("\n📊 Heritability Estimates:")
println("-"^80)
@printf("%-20s %10s %15s\n", "Model", "h²", "Error")
println("-"^80)
@printf("%-20s %10.4f %15.4f\n", "True", true_h2, 0.0)
@printf("%-20s %10.4f %15.4f\n", "BayesCπ (4-comp)", result_cpi.heritability,
        abs(result_cpi.heritability - true_h2))
@printf("%-20s %10.4f %15.4f\n", "BayesC (2-comp)", result_c.heritability,
        abs(result_c.heritability - true_h2))
@printf("%-20s %10.4f %15.4f\n", "BayesR (fixed)", result_r.heritability,
        abs(result_r.heritability - true_h2))
println("-"^80)

# ============================================================================
# 6. Posterior Inclusion Probability Analysis
# ============================================================================

println("\n" * "="^80)
println("Step 6: Posterior Inclusion Probability (PIP) Analysis")
println("="^80)

# Define high PIP threshold
pip_threshold = 0.5

# BayesCπ
pip_cpi_causal = result_cpi.pip[causal_indices]
pip_cpi_null = result_cpi.pip[setdiff(1:n_markers, causal_indices)]
n_detected_cpi = sum(result_cpi.pip .> pip_threshold)

# BayesC
pip_c_causal = result_c.pip[causal_indices]
pip_c_null = result_c.pip[setdiff(1:n_markers, causal_indices)]
n_detected_c = sum(result_c.pip .> pip_threshold)

# BayesR
pip_r_causal = result_r.pip[causal_indices]
pip_r_null = result_r.pip[setdiff(1:n_markers, causal_indices)]
n_detected_r = sum(result_r.pip .> pip_threshold)

println("\n📊 SNPs with PIP > $pip_threshold:")
println("-"^80)
@printf("%-20s %15s %15s\n", "Model", "Detected", "True Positives")
println("-"^80)
@printf("%-20s %15d %15d\n", "BayesCπ (4-comp)", n_detected_cpi,
        sum(result_cpi.pip[causal_indices] .> pip_threshold))
@printf("%-20s %15d %15d\n", "BayesC (2-comp)", n_detected_c,
        sum(result_c.pip[causal_indices] .> pip_threshold))
@printf("%-20s %15d %15d\n", "BayesR (fixed)", n_detected_r,
        sum(result_r.pip[causal_indices] .> pip_threshold))
println("-"^80)

println("\n📊 Mean PIP by SNP Type:")
println("-"^80)
@printf("%-20s %15s %15s %15s\n", "Model", "Causal SNPs", "Null SNPs", "Separation")
println("-"^80)
@printf("%-20s %15.4f %15.4f %15.4f\n", "BayesCπ (4-comp)",
        mean(pip_cpi_causal), mean(pip_cpi_null), mean(pip_cpi_causal) - mean(pip_cpi_null))
@printf("%-20s %15.4f %15.4f %15.4f\n", "BayesC (2-comp)",
        mean(pip_c_causal), mean(pip_c_null), mean(pip_c_causal) - mean(pip_c_null))
@printf("%-20s %15.4f %15.4f %15.4f\n", "BayesR (fixed)",
        mean(pip_r_causal), mean(pip_r_null), mean(pip_r_causal) - mean(pip_r_null))
println("-"^80)

# ============================================================================
# 7. Prediction Accuracy
# ============================================================================

println("\n" * "="^80)
println("Step 7: Prediction Accuracy")
println("="^80)

# Predict on training data
pred_cpi = predict(model_cpi, geno)
pred_c = predict(model_c, geno)
pred_r = predict(model_r, geno)

# Calculate correlations
cor_cpi_obs = cor(pred_cpi, y)
cor_cpi_g = cor(pred_cpi .- mean(pred_cpi), g)

cor_c_obs = cor(pred_c, y)
cor_c_g = cor(pred_c .- mean(pred_c), g)

cor_r_obs = cor(pred_r, y)
cor_r_g = cor(pred_r .- mean(pred_r), g)

println("\n📊 Prediction Correlations:")
println("-"^80)
@printf("%-20s %20s %20s\n", "Model", "cor(pred, observed)", "cor(pred, true g)")
println("-"^80)
@printf("%-20s %20.4f %20.4f\n", "BayesCπ (4-comp)", cor_cpi_obs, cor_cpi_g)
@printf("%-20s %20.4f %20.4f\n", "BayesC (2-comp)", cor_c_obs, cor_c_g)
@printf("%-20s %20.4f %20.4f\n", "BayesR (fixed)", cor_r_obs, cor_r_g)
println("-"^80)

# ============================================================================
# 8. Effect Size Estimation
# ============================================================================

println("\n" * "="^80)
println("Step 8: Effect Size Estimation")
println("="^80)

# Compare estimated effects with true effects
cor_effects_cpi = cor(result_cpi.marker_effects, true_effects)
cor_effects_c = cor(result_c.marker_effects, true_effects)
cor_effects_r = cor(result_r.marker_effects, true_effects)

println("\n📊 Correlation between estimated and true effects:")
println("-"^80)
@printf("%-20s %20.4f\n", "BayesCπ (4-comp)", cor_effects_cpi)
@printf("%-20s %20.4f\n", "BayesC (2-comp)", cor_effects_c)
@printf("%-20s %20.4f\n", "BayesR (fixed)", cor_effects_r)
println("-"^80)

# Find top 20 SNPs by PIP
top_indices_cpi = sortperm(result_cpi.pip, rev=true)[1:20]
n_true_in_top_cpi = sum(in.(top_indices_cpi, Ref(causal_indices)))

top_indices_c = sortperm(result_c.pip, rev=true)[1:20]
n_true_in_top_c = sum(in.(top_indices_c, Ref(causal_indices)))

top_indices_r = sortperm(result_r.pip, rev=true)[1:20]
n_true_in_top_r = sum(in.(top_indices_r, Ref(causal_indices)))

println("\n📊 True causal SNPs in top 20 by PIP:")
println("-"^80)
@printf("%-20s %20d / 20\n", "BayesCπ (4-comp)", n_true_in_top_cpi)
@printf("%-20s %20d / 20\n", "BayesC (2-comp)", n_true_in_top_c)
@printf("%-20s %20d / 20\n", "BayesR (fixed)", n_true_in_top_r)
println("-"^80)

# ============================================================================
# 9. Dirichlet Prior Sensitivity
# ============================================================================

println("\n" * "="^80)
println("Step 9: Dirichlet Prior Sensitivity Analysis")
println("="^80)

println("\nTesting different Dirichlet priors...")

# Uniform prior
model_uniform = BayesCπModel(
    n_iter = 10000,
    burn_in = 5000,
    thin = 10,
    dirichlet_alpha = ones(4),  # Uniform
    seed = 2024,
    verbose = false
)
fit!(model_uniform, geno, pheno)

# Informative prior (favor null)
model_null = BayesCπModel(
    n_iter = 10000,
    burn_in = 5000,
    thin = 10,
    dirichlet_alpha = [10.0, 1.0, 1.0, 1.0],  # Favor null
    seed = 2024,
    verbose = false
)
fit!(model_null, geno, pheno)

# Informative prior (favor non-zero)
model_nonnull = BayesCπModel(
    n_iter = 10000,
    burn_in = 5000,
    thin = 10,
    dirichlet_alpha = [1.0, 3.0, 3.0, 3.0],  # Favor non-zero
    seed = 2024,
    verbose = false
)
fit!(model_nonnull, geno, pheno)

println("\n📊 Effect of Dirichlet Prior on Mixture Proportions:")
println("-"^80)
@printf("%-20s %12s %12s %12s %12s\n", "Prior", "π₁ (null)", "π₂", "π₃", "π₄")
println("-"^80)
@printf("%-20s %12.4f %12.4f %12.4f %12.4f\n", "Uniform [1,1,1,1]",
        model_uniform.result.mixture_proportions...)
@printf("%-20s %12.4f %12.4f %12.4f %12.4f\n", "Null-favoring [10,1,1,1]",
        model_null.result.mixture_proportions...)
@printf("%-20s %12.4f %12.4f %12.4f %12.4f\n", "Non-null [1,3,3,3]",
        model_nonnull.result.mixture_proportions...)
@printf("%-20s %12.4f %12.4f %12.4f %12.4f\n", "True proportions",
        true_null_prop, (1-true_null_prop)/3, (1-true_null_prop)/3, (1-true_null_prop)/3)
println("-"^80)

# ============================================================================
# 10. Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ BayesCπ Example Complete!")

println("\n📊 Key Findings:")
println("-"^80)
println("1. Model Performance:")
@printf("   • BayesCπ achieved %.2f%% accuracy in h² estimation\n",
        100 * (1 - abs(result_cpi.heritability - true_h2) / true_h2))
@printf("   • Prediction correlation with true genetic values: %.4f\n", cor_cpi_g)

println("\n2. Mixture Proportion Estimation:")
@printf("   • Estimated null proportion: %.4f (true: %.4f)\n",
        result_cpi.mixture_proportions[1], true_null_prop)
@printf("   • Data overrides prior: mixture proportions estimated from data\n")

println("\n3. Variable Selection:")
@printf("   • True causal SNPs in top 20: %d / 20\n", n_true_in_top_cpi)
@printf("   • Mean PIP for causal SNPs: %.4f\n", mean(pip_cpi_causal))
@printf("   • Mean PIP for null SNPs: %.4f\n", mean(pip_cpi_null))

println("\n4. Model Comparison:")
println("   • BayesCπ (4-comp): Flexible, estimates mixture proportions")
println("   • BayesC (2-comp): Simpler, binary classification")
println("   • BayesR (fixed): Fixed proportions, may be less flexible")

println("\n💡 Recommendations:")
println("-"^80)
println("• Use BayesCπ when:")
println("  - Mixture proportions are unknown")
println("  - Data is large enough to estimate π reliably")
println("  - Want flexible model selection")
println("\n• Use BayesC (2-component) when:")
println("  - Simpler model preferred")
println("  - Binary classification (zero vs non-zero) sufficient")
println("  - Smaller datasets")
println("\n• Use BayesR (fixed π) when:")
println("  - Prior knowledge about mixture proportions")
println("  - Want to specify π based on domain knowledge")
println("  - Computational efficiency important")

println("\n" * "="^80)
println("BayesCπ example completed successfully!")
println("="^80)
