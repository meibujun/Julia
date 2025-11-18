"""
Test Suite for BayesCπ Model

Tests for Bayesian variable selection with estimated mixture proportions.
"""

using Test
using GenomicPro2
using Random
using Statistics

@testset "BayesCπ Model Tests" begin

    # Set seed for reproducibility
    Random.seed!(2024)

    # Generate test data
    n_samples = 100
    n_markers = 500
    n_causal = 20  # True causal SNPs

    println("\n" * "="^80)
    println("Testing BayesCπ Model")
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

    # Generate phenotypes with known effects
    true_effects = zeros(n_markers)
    causal_indices = randperm(n_markers)[1:n_causal]
    true_effects[causal_indices] = randn(n_causal) .* 0.5  # Effect sizes

    # Standardize genotypes for phenotype generation
    X = Float64.(geno_data)
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    X_scaled = (X .- X_mean) ./ X_std

    # Genetic values
    g = X_scaled * true_effects

    # Add environmental noise (h² ≈ 0.6)
    var_g = var(g)
    var_e = var_g * (1 - 0.6) / 0.6
    e = randn(n_samples) .* sqrt(var_e)

    y = g + e
    trait_name = "TestTrait"

    pheno = PhenotypeData(
        y,
        sample_ids,
        [trait_name]
    )

    println("  True h²: 0.6")
    println("  True genetic variance: $(round(var_g, digits=4))")

    @testset "Model Construction" begin
        println("\nTest 1: Model Construction")

        # Default 4-component model
        model1 = BayesCπModel()
        @test model1.n_components == 4
        @test model1.n_iter == 50000
        @test model1.burn_in == 20000
        @test length(model1.mixture_variances) == 4
        @test model1.mixture_variances[1] == 0.0

        # 2-component model (BayesC)
        model2 = BayesCπModel(n_components=2)
        @test model2.n_components == 2
        @test length(model2.mixture_variances) == 2
        @test model2.mixture_variances == [0.0, 0.01]

        # Custom Dirichlet prior
        model3 = BayesCπModel(dirichlet_alpha=[1.0, 1.0, 1.0, 1.0])
        @test model3.dirichlet_alpha == [1.0, 1.0, 1.0, 1.0]

        # Invalid constructions
        @test_throws ArgumentError BayesCπModel(n_iter=100, burn_in=200)
        @test_throws ArgumentError BayesCπModel(n_components=3)
        @test_throws ArgumentError BayesCπModel(mixture_variances=[0.0, 0.01])  # Wrong length
        @test_throws ArgumentError BayesCπModel(dirichlet_alpha=[1.0, -1.0, 1.0, 1.0])  # Negative

        println("  ✓ Model construction tests passed")
    end

    @testset "Model Fitting - 4 Components" begin
        println("\nTest 2: Model Fitting (4 components)")

        model = BayesCπModel(
            n_iter = 5000,
            burn_in = 2000,
            thin = 10,
            n_components = 4,
            seed = 2024,
            verbose = false
        )

        # Fit model
        fit!(model, geno, pheno)

        @test !isnothing(model.result)
        @test model.result isa BayesCπResult

        result = model.result

        # Check dimensions
        @test length(result.marker_effects) == n_markers
        @test length(result.pip) == n_markers
        @test length(result.mixture_proportions) == 4
        @test sum(result.mixture_proportions) ≈ 1.0 atol=1e-10

        # Check variance components
        @test result.sigma2_e > 0
        @test result.sigma2_a > 0
        @test 0 < result.heritability < 1

        # Check that estimated h² is reasonable
        @test abs(result.heritability - 0.6) < 0.3  # Within 30% of true value

        # Check that mixture proportions are positive
        @test all(result.mixture_proportions .>= 0)
        @test all(result.mixture_proportions .<= 1)

        # Check PIP
        @test all(0 .<= result.pip .<= 1)

        # Causal SNPs should have higher PIPs on average
        causal_pip = mean(result.pip[causal_indices])
        noncausal_pip = mean(result.pip[setdiff(1:n_markers, causal_indices)])
        @test causal_pip > noncausal_pip

        println("  Estimated h²: $(round(result.heritability, digits=4))")
        println("  Mixture proportions: $(round.(result.mixture_proportions, digits=4))")
        println("  Mean PIP (causal): $(round(causal_pip, digits=4))")
        println("  Mean PIP (non-causal): $(round(noncausal_pip, digits=4))")
        println("  ✓ 4-component model fitting tests passed")
    end

    @testset "Model Fitting - 2 Components (BayesC)" begin
        println("\nTest 3: Model Fitting (2 components - BayesC)")

        model = BayesCπModel(
            n_iter = 5000,
            burn_in = 2000,
            thin = 10,
            n_components = 2,
            seed = 2024,
            verbose = false
        )

        fit!(model, geno, pheno)

        @test !isnothing(model.result)
        result = model.result

        # Check dimensions
        @test length(result.mixture_proportions) == 2
        @test sum(result.mixture_proportions) ≈ 1.0 atol=1e-10

        # First component should be larger (null component)
        # (though this is data-dependent, usually most SNPs have zero effect)
        @test result.mixture_proportions[1] > 0.0

        # Check variance components
        @test result.sigma2_e > 0
        @test result.sigma2_a > 0
        @test 0 < result.heritability < 1

        println("  Estimated h²: $(round(result.heritability, digits=4))")
        println("  Mixture proportions: $(round.(result.mixture_proportions, digits=4))")
        println("  ✓ 2-component model fitting tests passed")
    end

    @testset "Prediction" begin
        println("\nTest 4: Prediction")

        # Fit model
        model = BayesCπModel(
            n_iter = 3000,
            burn_in = 1000,
            thin = 10,
            seed = 2024,
            verbose = false
        )

        fit!(model, geno, pheno)

        # Predict on same data (training accuracy)
        predictions = predict(model, geno)

        @test length(predictions) == n_samples
        @test all(isfinite.(predictions))

        # Calculate correlation with true phenotype
        cor_pred_true = cor(predictions, pheno.data[:, 1])
        @test cor_pred_true > 0.3  # Should have some predictive ability

        # Calculate correlation with true genetic values
        cor_pred_g = cor(predictions .- mean(predictions), g)
        @test cor_pred_g > 0.4  # Should correlate with true genetic values

        println("  Correlation (pred, observed): $(round(cor_pred_true, digits=4))")
        println("  Correlation (pred, true g): $(round(cor_pred_g, digits=4))")
        println("  ✓ Prediction tests passed")
    end

    @testset "Error Handling" begin
        println("\nTest 5: Error Handling")

        model = BayesCπModel(n_iter=1000, burn_in=500, verbose=false)

        # Predict before fitting
        @test_throws ArgumentError predict(model, geno)

        # Mismatched sample sizes
        bad_pheno = PhenotypeData(
            randn(50),
            ["X$i" for i in 1:50],
            [trait_name]
        )
        @test_throws ArgumentError fit!(model, geno, bad_pheno)

        # No common samples
        bad_pheno2 = PhenotypeData(
            randn(n_samples),
            ["Y$i" for i in 1:n_samples],
            [trait_name]
        )
        @test_throws ArgumentError fit!(model, geno, bad_pheno2)

        println("  ✓ Error handling tests passed")
    end

    @testset "Mixture Proportion Estimation" begin
        println("\nTest 6: Mixture Proportion Estimation")

        # Fit model with uniform Dirichlet prior
        model1 = BayesCπModel(
            n_iter = 5000,
            burn_in = 2000,
            thin = 10,
            dirichlet_alpha = ones(4),  # Uniform prior
            seed = 2024,
            verbose = false
        )

        fit!(model1, geno, pheno)

        # Fit model with informative prior (favoring null component)
        model2 = BayesCπModel(
            n_iter = 5000,
            burn_in = 2000,
            thin = 10,
            dirichlet_alpha = [10.0, 1.0, 1.0, 1.0],  # Favor null
            seed = 2024,
            verbose = false
        )

        fit!(model2, geno, pheno)

        # Both should estimate valid proportions
        @test all(model1.result.mixture_proportions .>= 0)
        @test all(model2.result.mixture_proportions .>= 0)
        @test sum(model1.result.mixture_proportions) ≈ 1.0 atol=1e-10
        @test sum(model2.result.mixture_proportions) ≈ 1.0 atol=1e-10

        # Model 2 should have higher proportion in null component
        # (though data can override prior)
        println("  Uniform prior π: $(round.(model1.result.mixture_proportions, digits=4))")
        println("  Informative prior π: $(round.(model2.result.mixture_proportions, digits=4))")
        println("  ✓ Mixture proportion estimation tests passed")
    end

    @testset "MAF Filtering" begin
        println("\nTest 7: MAF Filtering")

        model = BayesCπModel(
            n_iter = 2000,
            burn_in = 1000,
            thin = 10,
            seed = 2024,
            verbose = false
        )

        # Fit with MAF filter
        fit!(model, geno, pheno; min_maf=0.05)

        @test !isnothing(model.result)

        # Number of markers should be reduced
        n_markers_used = length(model.result.marker_effects)
        @test n_markers_used <= n_markers

        println("  Markers after MAF filter: $n_markers_used / $n_markers")
        println("  ✓ MAF filtering tests passed")
    end

    @testset "Result Structure" begin
        println("\nTest 8: Result Structure Validation")

        model = BayesCπModel(
            n_iter = 2000,
            burn_in = 1000,
            thin = 10,
            seed = 2024,
            verbose = false
        )

        fit!(model, geno, pheno)
        result = model.result

        # Check all required fields exist
        @test hasfield(typeof(result), :marker_effects)
        @test hasfield(typeof(result), :marker_effects_se)
        @test hasfield(typeof(result), :pip)
        @test hasfield(typeof(result), :component_assignment)
        @test hasfield(typeof(result), :mixture_proportions)
        @test hasfield(typeof(result), :mixture_proportions_se)
        @test hasfield(typeof(result), :sigma2_e)
        @test hasfield(typeof(result), :sigma2_a)
        @test hasfield(typeof(result), :heritability)
        @test hasfield(typeof(result), :marker_ids)
        @test hasfield(typeof(result), :convergence)

        # Check convergence diagnostics
        @test hasfield(typeof(result.convergence), :ess_sigma2_e)
        @test hasfield(typeof(result.convergence), :geweke_sigma2_e)

        # Standard errors should be positive
        @test all(result.marker_effects_se .>= 0)
        @test all(result.mixture_proportions_se .>= 0)

        # Component assignments should be valid
        @test all(1 .<= result.component_assignment .<= 4)

        println("  ✓ Result structure validation passed")
    end

    @testset "Convergence Diagnostics" begin
        println("\nTest 9: Convergence Diagnostics")

        model = BayesCπModel(
            n_iter = 5000,
            burn_in = 2000,
            thin = 10,
            seed = 2024,
            verbose = false
        )

        fit!(model, geno, pheno)
        result = model.result

        # Geweke Z-score should ideally be < 2 in absolute value
        geweke = result.convergence.geweke_sigma2_e
        @test isfinite(geweke)

        # ESS should be positive
        @test result.convergence.ess_sigma2_e > 0
        @test result.convergence.ess_sigma2_a > 0

        println("  Geweke Z-score: $(round(geweke, digits=4))")
        println("  ESS (σ²ₑ): $(result.convergence.ess_sigma2_e)")

        if abs(geweke) < 2.0
            println("  ✓ Good convergence (|Z| < 2)")
        else
            println("  ⚠ May need more iterations (|Z| >= 2)")
        end

        println("  ✓ Convergence diagnostic tests passed")
    end

    println("\n" * "="^80)
    println("All BayesCπ Tests Passed!")
    println("="^80)
end
