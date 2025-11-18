"""
Test Suite for RKHS (Reproducing Kernel Hilbert Space) Model

Tests for kernel-based semi-parametric regression.
"""

using Test
using GenomicPro2
using Random
using Statistics
using LinearAlgebra

@testset "RKHS Model Tests" begin

    # Set seed for reproducibility
    Random.seed!(2024)

    # Generate test data
    n_samples = 100
    n_markers = 200
    n_causal = 15

    println("\n" * "="^80)
    println("Testing RKHS Model")
    println("="^80)
    println("Generating test data:")
    println("  Samples: $n_samples")
    println("  Markers: $n_markers")
    println("  Causal SNPs: $n_causal")

    # Generate genotypes
    geno_data = rand(0:2, n_samples, n_markers)
    sample_ids = ["S$i" for i in 1:n_samples]
    marker_ids = ["SNP$i" for i in 1:n_markers]

    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    # Generate phenotypes with linear effects (for testing)
    true_effects = zeros(n_markers)
    causal_indices = randperm(n_markers)[1:n_causal]
    true_effects[causal_indices] = randn(n_causal) .* 0.5

    X = Float64.(geno_data)
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    X_scaled = (X .- X_mean) ./ X_std

    g = X_scaled * true_effects
    var_g = var(g)
    var_e = var_g * 0.4 / 0.6  # h² = 0.6
    e = randn(n_samples) .* sqrt(var_e)
    y = g + e

    pheno = PhenotypeData(y, sample_ids, ["TestTrait"])

    println("  True h²: 0.6")

    @testset "Model Construction" begin
        println("\nTest 1: Model Construction")

        # Linear kernel
        model1 = RKHSModel(kernel=:linear)
        @test model1.kernel == :linear
        @test model1.lambda == 1e-5
        @test model1.center_kernel == true

        # Gaussian kernel
        model2 = RKHSModel(kernel=:gaussian)
        @test model2.kernel == :gaussian
        @test isnothing(model2.bandwidth)  # Auto bandwidth

        # Gaussian with fixed bandwidth
        model3 = RKHSModel(kernel=:gaussian, bandwidth=1.0)
        @test model3.bandwidth == 1.0

        # Polynomial kernel
        model4 = RKHSModel(kernel=:polynomial, degree=3, coef=0.5)
        @test model4.kernel == :polynomial
        @test model4.degree == 3
        @test model4.coef == 0.5

        # Invalid constructions
        @test_throws ArgumentError RKHSModel(kernel=:unknown)
        @test_throws ArgumentError RKHSModel(kernel=:gaussian, bandwidth=-1.0)
        @test_throws ArgumentError RKHSModel(kernel=:polynomial, degree=0)
        @test_throws ArgumentError RKHSModel(lambda=-0.1)

        println("  ✓ Model construction tests passed")
    end

    @testset "Kernel Computation" begin
        println("\nTest 2: Kernel Computation")

        X_test = randn(20, 10)

        # Linear kernel
        K_linear = compute_kernel(X_test, X_test, :linear)
        @test size(K_linear) == (20, 20)
        @test issymmetric(K_linear)
        @test all(isfinite.(K_linear))

        # Check linear kernel is equivalent to normalized Gram matrix
        p = size(X_test, 2)
        K_expected = (X_test * X_test') / p
        @test K_linear ≈ K_expected

        # Gaussian kernel
        K_gauss = compute_kernel(X_test, X_test, :gaussian, bandwidth=1.0)
        @test size(K_gauss) == (20, 20)
        @test issymmetric(K_gauss)
        @test all(isfinite.(K_gauss))
        @test all(0 .<= K_gauss .<= 1)  # Gaussian kernel is bounded [0,1]
        @test all(diag(K_gauss) .≈ 1.0)  # Diagonal should be 1

        # Polynomial kernel
        K_poly = compute_kernel(X_test, X_test, :polynomial, degree=2, coef=1.0)
        @test size(K_poly) == (20, 20)
        @test issymmetric(K_poly)
        @test all(isfinite.(K_poly))

        # Test cross-kernel (different samples)
        X_test2 = randn(15, 10)
        K_cross = compute_kernel(X_test, X_test2, :linear)
        @test size(K_cross) == (20, 15)
        @test all(isfinite.(K_cross))

        println("  ✓ Kernel computation tests passed")
    end

    @testset "Kernel Centering" begin
        println("\nTest 3: Kernel Centering")

        X_test = randn(30, 10)
        K = compute_kernel(X_test, X_test, :linear)

        # Center kernel
        K_orig = copy(K)
        center_kernel_matrix!(K)

        # Centered kernel should have zero row/column sums
        @test all(abs.(sum(K, dims=1)) .< 1e-10)
        @test all(abs.(sum(K, dims=2)) .< 1e-10)

        # Centering should preserve trace (approximately)
        # (Not exactly, but should be close)
        @test abs(tr(K) - tr(K_orig)) / abs(tr(K_orig)) < 0.5

        println("  ✓ Kernel centering tests passed")
    end

    @testset "Linear Kernel Model" begin
        println("\nTest 4: Linear Kernel Model")

        model = RKHSModel(
            kernel = :linear,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model, geno, pheno)

        @test !isnothing(model.result)
        @test model.result isa RKHSResult

        result = model.result

        # Check result structure
        @test length(result.alpha) == n_samples
        @test size(result.K_train) == (n_samples, n_samples)
        @test size(result.X_train) == (n_samples, n_markers)
        @test result.n_samples == n_samples
        @test result.n_markers == n_markers

        # Training R² should be reasonable
        @test 0 < result.training_r2 < 1
        @test result.training_r2 > 0.3  # Should have some predictive power

        # Kernel matrix should be symmetric
        @test issymmetric(result.K_train)

        println("  Training R²: $(round(result.training_r2, digits=4))")
        println("  ✓ Linear kernel model tests passed")
    end

    @testset "Gaussian Kernel Model" begin
        println("\nTest 5: Gaussian Kernel Model")

        # Auto bandwidth
        model1 = RKHSModel(
            kernel = :gaussian,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model1, geno, pheno)

        @test !isnothing(model1.result)
        @test !isnothing(model1.result.kernel_params.bandwidth)
        @test model1.result.kernel_params.bandwidth > 0

        # Fixed bandwidth
        model2 = RKHSModel(
            kernel = :gaussian,
            bandwidth = 1.5,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model2, geno, pheno)

        @test model2.result.kernel_params.bandwidth == 1.5

        # Both should have reasonable training R²
        @test model1.result.training_r2 > 0.2
        @test model2.result.training_r2 > 0.2

        println("  Auto bandwidth: $(round(model1.result.kernel_params.bandwidth, digits=4))")
        println("  Training R² (auto): $(round(model1.result.training_r2, digits=4))")
        println("  Training R² (h=1.5): $(round(model2.result.training_r2, digits=4))")
        println("  ✓ Gaussian kernel model tests passed")
    end

    @testset "Polynomial Kernel Model" begin
        println("\nTest 6: Polynomial Kernel Model")

        # Degree 2
        model1 = RKHSModel(
            kernel = :polynomial,
            degree = 2,
            coef = 1.0,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model1, geno, pheno)

        @test model1.result.kernel_params.degree == 2

        # Degree 3
        model2 = RKHSModel(
            kernel = :polynomial,
            degree = 3,
            coef = 1.0,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model2, geno, pheno)

        @test model2.result.kernel_params.degree == 3

        # Both should have reasonable training R²
        @test model1.result.training_r2 > 0.2
        @test model2.result.training_r2 > 0.2

        println("  Training R² (degree 2): $(round(model1.result.training_r2, digits=4))")
        println("  Training R² (degree 3): $(round(model2.result.training_r2, digits=4))")
        println("  ✓ Polynomial kernel model tests passed")
    end

    @testset "Prediction" begin
        println("\nTest 7: Prediction")

        model = RKHSModel(
            kernel = :gaussian,
            lambda = 1e-4,
            verbose = false
        )

        fit!(model, geno, pheno)

        # Predict on training data
        predictions = predict(model, geno)

        @test length(predictions) == n_samples
        @test all(isfinite.(predictions))

        # Calculate correlation
        cor_pred = cor(predictions, y)
        @test cor_pred > 0.4  # Should have predictive power

        println("  Correlation (pred, observed): $(round(cor_pred, digits=4))")
        println("  ✓ Prediction tests passed")
    end

    @testset "Regularization Effect" begin
        println("\nTest 8: Regularization Effect")

        # Strong regularization
        model_strong = RKHSModel(
            kernel = :linear,
            lambda = 1.0,
            verbose = false
        )

        fit!(model_strong, geno, pheno)

        # Weak regularization
        model_weak = RKHSModel(
            kernel = :linear,
            lambda = 1e-6,
            verbose = false
        )

        fit!(model_weak, geno, pheno)

        # Weak regularization should fit training data better
        @test model_weak.result.training_r2 >= model_strong.result.training_r2

        # Dual coefficients should be smaller with strong regularization
        @test mean(abs.(model_strong.result.alpha)) <= mean(abs.(model_weak.result.alpha))

        println("  Training R² (λ=1.0): $(round(model_strong.result.training_r2, digits=4))")
        println("  Training R² (λ=1e-6): $(round(model_weak.result.training_r2, digits=4))")
        println("  ✓ Regularization tests passed")
    end

    @testset "Error Handling" begin
        println("\nTest 9: Error Handling")

        model = RKHSModel(verbose=false)

        # Predict before fitting
        @test_throws ArgumentError predict(model, geno)

        # Mismatched sample sizes
        bad_pheno = PhenotypeData(
            randn(50),
            ["X$i" for i in 1:50],
            ["TestTrait"]
        )
        @test_throws ArgumentError fit!(model, geno, bad_pheno)

        # No common samples
        bad_pheno2 = PhenotypeData(
            randn(n_samples),
            ["Y$i" for i in 1:n_samples],
            ["TestTrait"]
        )
        @test_throws ArgumentError fit!(model, geno, bad_pheno2)

        println("  ✓ Error handling tests passed")
    end

    @testset "Kernel Comparison" begin
        println("\nTest 10: Kernel Comparison")

        # Fit all three kernels
        model_linear = RKHSModel(kernel=:linear, lambda=1e-4, verbose=false)
        model_gauss = RKHSModel(kernel=:gaussian, lambda=1e-4, verbose=false)
        model_poly = RKHSModel(kernel=:polynomial, degree=2, lambda=1e-4, verbose=false)

        fit!(model_linear, geno, pheno)
        fit!(model_gauss, geno, pheno)
        fit!(model_poly, geno, pheno)

        # All should have reasonable R²
        @test model_linear.result.training_r2 > 0.3
        @test model_gauss.result.training_r2 > 0.2
        @test model_poly.result.training_r2 > 0.2

        # All should make valid predictions
        pred_linear = predict(model_linear, geno)
        pred_gauss = predict(model_gauss, geno)
        pred_poly = predict(model_poly, geno)

        @test all(isfinite.(pred_linear))
        @test all(isfinite.(pred_gauss))
        @test all(isfinite.(pred_poly))

        println("\n  Kernel Performance Comparison:")
        println("  " * "-"^60)
        @printf("  %-20s %20s\n", "Kernel", "Training R²")
        println("  " * "-"^60)
        @printf("  %-20s %20.4f\n", "Linear", model_linear.result.training_r2)
        @printf("  %-20s %20.4f\n", "Gaussian", model_gauss.result.training_r2)
        @printf("  %-20s %20.4f\n", "Polynomial (d=2)", model_poly.result.training_r2)
        println("  " * "-"^60)

        println("  ✓ Kernel comparison tests passed")
    end

    @testset "MAF Filtering" begin
        println("\nTest 11: MAF Filtering")

        model = RKHSModel(
            kernel = :linear,
            lambda = 1e-4,
            verbose = false
        )

        # Fit with MAF filter
        fit!(model, geno, pheno; min_maf=0.05)

        @test !isnothing(model.result)

        # Number of markers should be potentially reduced
        n_markers_used = model.result.n_markers
        @test n_markers_used <= n_markers

        println("  Markers after MAF filter: $n_markers_used / $n_markers")
        println("  ✓ MAF filtering tests passed")
    end

    println("\n" * "="^80)
    println("All RKHS Tests Passed!")
    println("="^80)
end
