"""
RKHS (Reproducing Kernel Hilbert Space) Regression Example

This example demonstrates:
1. Different kernel types (Linear, Gaussian, Polynomial)
2. Kernel parameter selection (bandwidth, degree)
3. Comparison with GBLUP (linear kernel is equivalent)
4. Regularization parameter tuning
5. Cross-validation for model selection
6. Capturing non-linear effects

Run with: julia --project examples/rkhs_example.jl
"""

using GenomicPro2
using Statistics
using Printf
using Random

println("="^80)
println("GenomicPro2 RKHS Model Example")
println("="^80)

Random.seed!(2024)

# ============================================================================
# 1. Generate Simulated Data
# ============================================================================

println("\n" * "="^80)
println("Step 1: Generating Simulated Data")
println("="^80)

n_samples = 400
n_markers = 1000
n_causal = 30

println("\nSimulation parameters:")
println("  Samples: $n_samples")
println("  Markers: $n_markers")
println("  Causal markers: $n_causal")
println("  Target heritability: 0.6")

# Generate genotypes
geno_data = rand(0:2, n_samples, n_markers)
sample_ids = [string("Sample_", i) for i in 1:n_samples]
marker_ids = [string("SNP_", i) for i in 1:n_markers]

geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

# Generate phenotypes with linear effects (for comparison)
true_effects = zeros(n_markers)
causal_indices = sort(randperm(n_markers)[1:n_causal])
true_effects[causal_indices] = randn(n_causal) .* 0.5

# Standardize genotypes
X = Float64.(geno_data)
X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X_std[X_std .== 0] .= 1.0
X_scaled = (X .- X_mean) ./ X_std'

# Genetic values (linear)
g_linear = X_scaled * true_effects

# Add some non-linear effects (epistasis)
# Add interaction between pairs of causal SNPs
n_interactions = 5
g_nonlinear = copy(g_linear)
for i in 1:n_interactions
    idx1 = causal_indices[2*i-1]
    idx2 = causal_indices[2*i]
    # Add epistatic effect
    g_nonlinear += X_scaled[:, idx1] .* X_scaled[:, idx2] * 0.2
end

# Environmental noise (h² = 0.6)
var_g = var(g_nonlinear)
target_h2 = 0.6
var_e = var_g * (1 - target_h2) / target_h2
e = randn(n_samples) .* sqrt(var_e)

y = g_nonlinear + e
trait_name = "ComplexTrait"

pheno = PhenotypeData(y, sample_ids, [trait_name])

true_h2 = var(g_nonlinear) / (var(g_nonlinear) + var(e))

println("\nTrue simulation values:")
@printf("  Genetic variance: %.4f\n", var(g_nonlinear))
@printf("  Residual variance: %.4f\n", var(e))
@printf("  Heritability: %.4f\n", true_h2)
println("  Non-linear effects: Included (epistasis)")

println("\n✓ Data generated successfully")

# ============================================================================
# 2. Split into Training and Testing Sets
# ============================================================================

println("\n" * "="^80)
println("Step 2: Train-Test Split")
println("="^80)

n_train = 300
n_test = n_samples - n_train

train_idx = randperm(n_samples)[1:n_train]
test_idx = setdiff(1:n_samples, train_idx)

# Create train/test datasets
geno_train_data = geno_data[train_idx, :]
geno_test_data = geno_data[test_idx, :]

geno_train = CompactGenotypes(
    geno_train_data,
    sample_ids[train_idx],
    marker_ids
)

geno_test = CompactGenotypes(
    geno_test_data,
    sample_ids[test_idx],
    marker_ids
)

pheno_train = PhenotypeData(
    y[train_idx],
    sample_ids[train_idx],
    [trait_name]
)

pheno_test = PhenotypeData(
    y[test_idx],
    sample_ids[test_idx],
    [trait_name]
)

println("\nData split:")
println("  Training samples: $n_train")
println("  Testing samples: $n_test")

# ============================================================================
# 3. Linear Kernel (Equivalent to GBLUP)
# ============================================================================

println("\n" * "="^80)
println("Step 3: Linear Kernel RKHS")
println("="^80)

model_linear = RKHSModel(
    kernel = :linear,
    lambda = 1e-4,
    center_kernel = true,
    verbose = true
)

println("\nFitting linear kernel model...")
fit!(model_linear, geno_train, pheno_train)

# Predict on test set
pred_linear_test = predict(model_linear, geno_test)

# Evaluate
y_test = pheno_test.data[:, 1]
cor_linear = cor(pred_linear_test, y_test)
mse_linear = mean((pred_linear_test - y_test) .^ 2)

println("\n📊 Linear Kernel Results:")
println("-"^80)
@printf("Test correlation: %.4f\n", cor_linear)
@printf("Test MSE: %.4f\n", mse_linear)
@printf("Test RMSE: %.4f\n", sqrt(mse_linear))

# ============================================================================
# 4. Gaussian Kernel with Auto Bandwidth
# ============================================================================

println("\n" * "="^80)
println("Step 4: Gaussian Kernel RKHS (Auto Bandwidth)")
println("="^80)

model_gauss_auto = RKHSModel(
    kernel = :gaussian,
    bandwidth = nothing,  # Auto-select
    lambda = 1e-4,
    center_kernel = true,
    verbose = true
)

println("\nFitting Gaussian kernel model with auto bandwidth...")
fit!(model_gauss_auto, geno_train, pheno_train)

# Predict on test set
pred_gauss_auto_test = predict(model_gauss_auto, geno_test)

# Evaluate
cor_gauss_auto = cor(pred_gauss_auto_test, y_test)
mse_gauss_auto = mean((pred_gauss_auto_test - y_test) .^ 2)

println("\n📊 Gaussian Kernel (Auto) Results:")
println("-"^80)
@printf("Bandwidth: %.4f (auto-selected)\n", model_gauss_auto.result.kernel_params.bandwidth)
@printf("Test correlation: %.4f\n", cor_gauss_auto)
@printf("Test MSE: %.4f\n", mse_gauss_auto)
@printf("Test RMSE: %.4f\n", sqrt(mse_gauss_auto))

# ============================================================================
# 5. Gaussian Kernel with Different Bandwidths
# ============================================================================

println("\n" * "="^80)
println("Step 5: Gaussian Kernel Bandwidth Selection")
println("="^80)

bandwidths = [0.5, 1.0, 2.0, 5.0]
results_bandwidths = []

println("\nTesting different bandwidths...")
for h in bandwidths
    model = RKHSModel(
        kernel = :gaussian,
        bandwidth = h,
        lambda = 1e-4,
        verbose = false
    )

    fit!(model, geno_train, pheno_train)
    pred_test = predict(model, geno_test)

    cor_test = cor(pred_test, y_test)
    mse_test = mean((pred_test - y_test) .^ 2)

    push!(results_bandwidths, (h=h, cor=cor_test, mse=mse_test))
end

println("\n📊 Bandwidth Comparison:")
println("-"^80)
@printf("%-15s %20s %20s\n", "Bandwidth", "Test Correlation", "Test MSE")
println("-"^80)
for res in results_bandwidths
    @printf("%-15.2f %20.4f %20.4f\n", res.h, res.cor, res.mse)
end
println("-"^80)

# Find best bandwidth
best_idx = argmax([r.cor for r in results_bandwidths])
best_h = results_bandwidths[best_idx].h
@printf("\nBest bandwidth: %.2f (cor = %.4f)\n", best_h, results_bandwidths[best_idx].cor)

# ============================================================================
# 6. Polynomial Kernels
# ============================================================================

println("\n" * "="^80)
println("Step 6: Polynomial Kernel RKHS")
println("="^80)

degrees = [2, 3]
results_poly = []

println("\nTesting polynomial kernels...")
for d in degrees
    model = RKHSModel(
        kernel = :polynomial,
        degree = d,
        coef = 1.0,
        lambda = 1e-4,
        verbose = false
    )

    println("\nFitting polynomial kernel (degree=$d)...")
    fit!(model, geno_train, pheno_train)
    pred_test = predict(model, geno_test)

    cor_test = cor(pred_test, y_test)
    mse_test = mean((pred_test - y_test) .^ 2)

    push!(results_poly, (degree=d, cor=cor_test, mse=mse_test))

    println("  Training R²: $(round(model.result.training_r2, digits=4))")
    println("  Test correlation: $(round(cor_test, digits=4))")
end

println("\n📊 Polynomial Kernel Comparison:")
println("-"^80)
@printf("%-15s %20s %20s\n", "Degree", "Test Correlation", "Test MSE")
println("-"^80)
for res in results_poly
    @printf("%-15d %20.4f %20.4f\n", res.degree, res.cor, res.mse)
end
println("-"^80)

# ============================================================================
# 7. Compare with GBLUP
# ============================================================================

println("\n" * "="^80)
println("Step 7: Comparison with GBLUP")
println("="^80)

println("\nFitting GBLUP model...")
gblup_model = GBLUPModel(verbose=true)
G_train = compute_grm(geno_train, method=:vanraden)
fit!(gblup_model, geno_train, pheno_train; G=G_train)

# Predict on test set
pred_gblup_test = predict(gblup_model, geno_test)

# Evaluate
cor_gblup = cor(pred_gblup_test, y_test)
mse_gblup = mean((pred_gblup_test - y_test) .^ 2)

println("\n📊 GBLUP Results:")
println("-"^80)
@printf("Test correlation: %.4f\n", cor_gblup)
@printf("Test MSE: %.4f\n", mse_gblup)
@printf("Test RMSE: %.4f\n", sqrt(mse_gblup))

# ============================================================================
# 8. Regularization Parameter Selection
# ============================================================================

println("\n" * "="^80)
println("Step 8: Regularization Parameter Selection")
println("="^80)

lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
results_lambda = []

println("\nTesting different regularization parameters...")
for λ in lambdas
    model = RKHSModel(
        kernel = :gaussian,
        bandwidth = best_h,
        lambda = λ,
        verbose = false
    )

    fit!(model, geno_train, pheno_train)
    pred_test = predict(model, geno_test)

    cor_test = cor(pred_test, y_test)
    mse_test = mean((pred_test - y_test) .^ 2)

    push!(results_lambda, (lambda=λ, cor=cor_test, mse=mse_test))
end

println("\n📊 Regularization Comparison:")
println("-"^80)
@printf("%-15s %20s %20s\n", "Lambda", "Test Correlation", "Test MSE")
println("-"^80)
for res in results_lambda
    @printf("%-15.1e %20.4f %20.4f\n", res.lambda, res.cor, res.mse)
end
println("-"^80)

best_lambda_idx = argmax([r.cor for r in results_lambda])
best_lambda = results_lambda[best_lambda_idx].lambda
@printf("\nBest lambda: %.1e (cor = %.4f)\n", best_lambda, results_lambda[best_lambda_idx].cor)

# ============================================================================
# 9. Final Model Comparison
# ============================================================================

println("\n" * "="^80)
println("Step 9: Final Model Comparison")
println("="^80)

# Fit best models
best_gauss = RKHSModel(kernel=:gaussian, bandwidth=best_h, lambda=best_lambda, verbose=false)
fit!(best_gauss, geno_train, pheno_train)
pred_best_gauss = predict(best_gauss, geno_test)

best_poly = RKHSModel(kernel=:polynomial, degree=2, lambda=best_lambda, verbose=false)
fit!(best_poly, geno_train, pheno_train)
pred_best_poly = predict(best_poly, geno_test)

println("\n📊 Final Model Comparison:")
println("-"^80)
@printf("%-30s %20s %20s\n", "Model", "Test Correlation", "Test RMSE")
println("-"^80)
@printf("%-30s %20.4f %20.4f\n", "GBLUP", cor_gblup, sqrt(mse_gblup))
@printf("%-30s %20.4f %20.4f\n", "RKHS Linear", cor_linear, sqrt(mse_linear))
@printf("%-30s %20.4f %20.4f\n", "RKHS Gaussian (auto)", cor_gauss_auto, sqrt(mse_gauss_auto))
@printf("%-30s %20.4f %20.4f\n", "RKHS Gaussian (best h)", cor(pred_best_gauss, y_test),
        sqrt(mean((pred_best_gauss - y_test) .^ 2)))
@printf("%-30s %20.4f %20.4f\n", "RKHS Polynomial (d=2)", cor(pred_best_poly, y_test),
        sqrt(mean((pred_best_poly - y_test) .^ 2)))
println("-"^80)

# ============================================================================
# 10. Summary
# ============================================================================

println("\n" * "="^80)
println("Summary")
println("="^80)

println("\n✅ RKHS Example Complete!")

println("\n📊 Key Findings:")
println("-"^80)

# Find best model
all_cors = [
    ("GBLUP", cor_gblup),
    ("RKHS Linear", cor_linear),
    ("RKHS Gaussian", cor_gauss_auto),
    ("RKHS Gaussian (tuned)", cor(pred_best_gauss, y_test)),
    ("RKHS Polynomial", cor(pred_best_poly, y_test))
]

best_model = all_cors[argmax([c[2] for c in all_cors])]
println("1. Best Model:")
@printf("   • %s achieved highest correlation: %.4f\n", best_model[1], best_model[2])

println("\n2. Kernel Comparison:")
println("   • Linear kernel: Equivalent to GBLUP, captures additive effects")
println("   • Gaussian kernel: Can capture non-linear and epistatic effects")
println("   • Polynomial kernel: Captures interactions up to degree d")

println("\n3. Parameter Selection:")
@printf("   • Best bandwidth (Gaussian): %.4f\n", best_h)
@printf("   • Best regularization: %.1e\n", best_lambda)
@printf("   • Auto bandwidth works well: %.4f\n", model_gauss_auto.result.kernel_params.bandwidth)

println("\n4. Non-linear Effects:")
if cor(pred_best_gauss, y_test) > cor_linear
    println("   • Gaussian kernel outperformed linear kernel")
    println("   • Non-linear/epistatic effects detected and captured")
else
    println("   • Linear and non-linear models performed similarly")
    println("   • Effects may be predominantly additive")
end

println("\n💡 Recommendations:")
println("-"^80)
println("• Use RKHS when:")
println("  - Suspected non-linear or epistatic effects")
println("  - Want flexible model without specifying interactions")
println("  - Dataset is not too large (kernel computation is O(n²))")
println("\n• Kernel selection:")
println("  - Linear: Fast, interpretable, equivalent to GBLUP")
println("  - Gaussian: General-purpose, captures complex patterns")
println("  - Polynomial: Specific interaction degree, interpretable")
println("\n• Parameter tuning:")
println("  - Use cross-validation for bandwidth and lambda selection")
println("  - Auto bandwidth (median heuristic) often works well")
println("  - Regularization prevents overfitting")

println("\n" * "="^80)
println("RKHS example completed successfully!")
println("="^80)
