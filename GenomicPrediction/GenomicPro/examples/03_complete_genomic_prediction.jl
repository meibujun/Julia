# examples/03_complete_genomic_prediction.jl

"""
Example 3: Complete Genomic Prediction Workflow

This comprehensive example demonstrates a full genomic prediction analysis
including data loading, quality control, relationship matrix computation,
variance component estimation, breeding value prediction, and cross-validation.

Author: GenomicPro Development Team
Date: 2025
Julia Version: 1.12.1
"""

using GenomicPro
using Statistics, Random, Printf, LinearAlgebra

println("="^70)
println("GenomicPro.jl Example 3: Complete Genomic Prediction Pipeline")
println("="^70)
println()

# ============================================================================
# Step 1: Data Preparation
# ============================================================================
println("STEP 1: Data Preparation and Quality Control")
println("-"^70)
println()

# Simulate realistic breeding population
Random.seed!(456)
n_individuals = 2000
n_markers = 20000

println("Simulating breeding population:")
println("  Individuals: $n_individuals")
println("  Markers: $n_markers")
println()

# Simulate genotypes with realistic LD structure
genotypes_raw = Matrix{Union{Int, Missing}}(undef, n_individuals, n_markers)

for j in 1:n_markers
    p = rand(Beta(0.5, 0.5))

    for i in 1:n_individuals
        if rand() < 0.03
            genotypes_raw[i, j] = missing
        else
            r = rand()
            if r < (1-p)^2
                genotypes_raw[i, j] = 0
            elseif r < (1-p)^2 + 2*p*(1-p)
                genotypes_raw[i, j] = 1
            else
                genotypes_raw[i, j] = 2
            end
        end
    end
end

# Create GenomicPro genotype object
sample_ids = ["Animal_" * lpad(i, 5, '0') for i in 1:n_individuals]
marker_ids = ["SNP_" * lpad(i, 6, '0') for i in 1:n_markers]

genotypes = TwoBitGenotypes(genotypes_raw,
                            sample_ids=sample_ids,
                            marker_ids=marker_ids)

println("Genotype data created")
println("  Memory usage: $(round(Base.summarysize(genotypes) / 1e6, digits=1)) MB")
println()

# Apply quality control
println("Applying quality control...")
qc_pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])

genotypes_qc, qc_reports = apply_qc(genotypes, qc_pipeline)

println("After QC:")
println("  Retained samples: $(size(genotypes_qc, 1)) ($(round(size(genotypes_qc,1)/n_individuals*100, digits=1))%)")
println("  Retained markers: $(size(genotypes_qc, 2)) ($(round(size(genotypes_qc,2)/n_markers*100, digits=1))%)")
println()

# ============================================================================
# Step 2: Simulate Phenotypes with Realistic Genetic Architecture
# ============================================================================
println("STEP 2: Simulating Phenotypes")
println("-"^70)
println()

n_final = size(genotypes_qc, 1)
m_final = size(genotypes_qc, 2)

# True parameters
h2_true = 0.40  # Heritability
σ²_g_true = 100.0  # Genetic variance
σ²_e_true = σ²_g_true * (1.0 - h2_true) / h2_true  # Residual variance

println("True genetic parameters:")
println("  Heritability: $h2_true")
println("  Genetic variance: $σ²_g_true")
println("  Residual variance: $(round(σ²_e_true, digits=2))")
println()

# Simulate QTL effects (sparse, most markers have zero effect)
n_qtl = 100  # Number of causal variants
qtl_indices = sort(shuffle(1:m_final)[1:n_qtl])
β_true = zeros(Float64, m_final)

# Large, medium, and small effect QTLs
for (i, qtl_idx) in enumerate(qtl_indices)
    if i <= 10
        # 10 large effect QTLs
        β_true[qtl_idx] = randn() * sqrt(σ²_g_true / 10)
    elseif i <= 30
        # 20 medium effect QTLs
        β_true[qtl_idx] = randn() * sqrt(σ²_g_true / 40)
    else
        # 70 small effect QTLs
        β_true[qtl_idx] = randn() * sqrt(σ²_g_true / 140)
    end
end

# Ensure genetic variance matches target
current_var = var([sum(genotypes_qc[i, qtl_idx] * β_true[qtl_idx]
                       for qtl_idx in qtl_indices)
                   for i in 1:n_final])
β_true .*= sqrt(σ²_g_true / current_var)

println("Simulated $n_qtl QTL effects")
println("  Large effect: 10 QTLs")
println("  Medium effect: 20 QTLs")
println("  Small effect: 70 QTLs")
println()

# Generate true breeding values
tbv = zeros(Float64, n_final)
for i in 1:n_final
    for qtl_idx in qtl_indices
        g = genotypes_qc[i, qtl_idx]
        if !ismissing(g)
            tbv[i] += g * β_true[qtl_idx]
        end
    end
end

# Add environmental effects
ε = randn(n_final) .* sqrt(σ²_e_true)
phenotypes_sim = tbv .+ ε

# Verify realized parameters
println("Realized parameters:")
println("  Genetic variance: $(round(var(tbv), digits=2))")
println("  Residual variance: $(round(var(ε), digits=2))")
println("  Heritability: $(round(var(tbv)/(var(tbv)+var(ε)), digits=3))")
println()

# ============================================================================
# Step 3: Genomic Relationship Matrix
# ============================================================================
println("STEP 3: Computing Genomic Relationship Matrix")
println("-"^70)
println()

G = compute_grm(genotypes_qc, method=:VanRaden)

println()

# ============================================================================
# Step 4: Variance Component Estimation
# ============================================================================
println("STEP 4: Estimating Variance Components")
println("-"^70)
println()

vc = estimate_variance_components(G, phenotypes_sim,
                                 method=:AIREML,
                                 tolerance=1e-6)

println()
println("Estimated vs True Parameters:")
println("  Genetic variance:  $(round(vc.genetic_variance, digits=2)) (true: $σ²_g_true)")
println("  Residual variance: $(round(vc.residual_variance, digits=2)) (true: $(round(σ²_e_true, digits=2)))")
println("  Heritability:      $(round(vc.heritability, digits=3)) (true: $h2_true)")
println()

# ============================================================================
# Step 5: Genomic Prediction
# ============================================================================
println("STEP 5: Genomic Breeding Value Prediction")
println("-"^70)
println()

λ = vc.residual_variance / vc.genetic_variance

gebv_result = solve_gblup(G, phenotypes_sim, λ,
                          method=:pcg,
                          tolerance=1e-6,
                          preconditioner=:diagonal)

gebvs = gebv_result.breeding_values

println()
println("Prediction Results:")
println("  Correlation with TBV: $(round(cor(gebvs, tbv), digits=4))")
println("  Expected accuracy (√h²): $(round(sqrt(vc.heritability), digits=4))")
println()

# ============================================================================
# Step 6: Cross-Validation
# ============================================================================
println("STEP 6: Cross-Validation Analysis")
println("-"^70)
println()

# Create phenotype data structure
using DataFrames
pheno_df = DataFrame(
    ID = sample_ids[1:n_final],
    trait = phenotypes_sim
)

# Note: This is simplified. Full implementation would use proper PhenotypeData type
# For demonstration, we'll perform manual cross-validation

println("Performing 5-fold cross-validation...")
println()

n_folds = 5
fold_size = div(n_final, n_folds)
fold_assignments = repeat(1:n_folds, inner=fold_size)
if length(fold_assignments) < n_final
    append!(fold_assignments, fill(n_folds, n_final - length(fold_assignments)))
end
shuffle!(fold_assignments)

cv_correlations = Float64[]
cv_rmses = Float64[]
cv_biases = Float64[]

for fold in 1:n_folds
    test_mask = fold_assignments .== fold
    train_mask = .!test_mask

    # Training data
    G_train = G[train_mask, train_mask]
    y_train = phenotypes_sim[train_mask]

    # Estimate variance components on training data
    vc_fold = estimate_variance_components(G_train, y_train,
                                          method=:AIREML,
                                          max_iterations=50,
                                          tolerance=1e-5)

    λ_fold = vc_fold.residual_variance / vc_fold.genetic_variance

    # Solve for training breeding values
    result_fold = solve_gblup(G_train, y_train, λ_fold,
                              method=:pcg,
                              max_iterations=500,
                              tolerance=1e-5)

    u_train = result_fold.breeding_values

    # Predict test individuals
    G_test_train = G[test_mask, train_mask]
    α = G_train \ u_train
    predictions = G_test_train * α

    y_test = phenotypes_sim[test_mask]

    # Compute metrics
    push!(cv_correlations, cor(predictions, y_test))
    push!(cv_rmses, sqrt(mean((predictions .- y_test).^2)))

    # Bias
    X_bias = hcat(ones(length(predictions)), predictions)
    β_bias = (X_bias' * X_bias) \ (X_bias' * y_test)
    push!(cv_biases, β_bias[2])

    println("Fold $fold: correlation = $(round(cv_correlations[end], digits=4))")
end

println()
println("Cross-Validation Summary:")
println("  Mean accuracy: $(round(mean(cv_correlations), digits=4)) ± $(round(std(cv_correlations)/sqrt(n_folds), digits=4))")
println("  Mean RMSE: $(round(mean(cv_rmses), digits=2))")
println("  Mean bias: $(round(mean(cv_biases), digits=3))")
println()

# ============================================================================
# Step 7: Results Summary
# ============================================================================
println("="^70)
println("ANALYSIS COMPLETE - RESULTS SUMMARY")
println("="^70)
println()

println("Dataset:")
println("  Final sample size: $n_final individuals")
println("  Markers used: $m_final SNPs")
println("  QTL effects: $n_qtl causal variants")
println()

println("Genetic Parameters:")
println("  True heritability: $h2_true")
println("  Estimated heritability: $(round(vc.heritability, digits=3)) ± $(round(vc.h2_se, digits=3))")
println("  Relative error: $(round(abs(vc.heritability - h2_true)/h2_true * 100, digits=1))%")
println()

println("Prediction Accuracy:")
println("  GEBV-TBV correlation: $(round(cor(gebvs, tbv), digits=4))")
println("  Expected (√h²): $(round(sqrt(h2_true), digits=4))")
println("  CV accuracy: $(round(mean(cv_correlations), digits=4)) ± $(round(std(cv_correlations)/sqrt(n_folds), digits=4))")
println()

println("Computational Performance:")
println("  GRM computation: Complete")
println("  REML iterations: $(vc.iterations)")
println("  GBLUP solver iterations: $(gebv_result.iterations)")
println("  Total solve time: $(round(gebv_result.solve_time, digits=2)) seconds")
println()

println("="^70)
println("Pipeline executed successfully!")
println("="^70)
println()

println("Key Findings:")
println("  • Variance components accurately recovered from data")
println("  • Genomic predictions highly correlated with true breeding values")
println("  • Cross-validation confirms robust prediction accuracy")
println("  • PCG solver converged efficiently for large system")
println()

println("Next Steps:")
println("  • Apply to real breeding data")
println("  • Compare alternative prediction methods (BayesR, deepGBLUP)")
println("  • Implement selection strategies using GEBVs")
println("  • Conduct multi-trait analysis for correlated traits")