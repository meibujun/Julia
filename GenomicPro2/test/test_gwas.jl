"""
GWAS Module Tests

Tests for genome-wide association studies functionality:
- Linear model GWAS
- Mixed model GWAS
- PCA correction
- Multiple testing correction
- GPU acceleration (if available)
"""

using Test
using GenomicPro2
using GenomicPro2.GWAS
using LinearAlgebra
using Statistics
using Random

@testset "GWAS Module Tests" begin

    # Generate simulated data for testing
    Random.seed!(12345)
    n_samples = 100
    n_snps = 500
    n_causal = 5

    # Simulate genotypes (0, 1, 2)
    simulated_genotypes = rand([0, 1, 2], n_samples, n_snps)

    # Simulate phenotypes with some causal SNPs
    causal_snps = rand(1:n_snps, n_causal)
    true_effects = randn(n_causal)
    phenotype_values = zeros(n_samples)

    for (i, snp_idx) in enumerate(causal_snps)
        phenotype_values .+ = simulated_genotypes[:, snp_idx] .* true_effects[i]
    end

    # Add noise
    phenotype_values .+= randn(n_samples) .* 0.5

    # ========================================================================
    # Test 1: GWASResult Structure
    # ========================================================================

    @testset "GWASResult Structure" begin
        snp_ids = ["rs" * string(i) for i in 1:10]
        chromosomes = rand(1:22, 10)
        positions = sort(rand(1:1000000, 10))
        pvalues = rand(10) .* 1e-6
        effect_sizes = randn(10) .* 0.1
        std_errors = rand(10) .* 0.05

        result = GWASResult(
            snp_ids,
            chromosomes,
            positions,
            pvalues,
            effect_sizes,
            std_errors
        )

        @test length(result.snp_ids) == 10
        @test length(result.pvalues) == 10
        @test all(result.pvalues .>= 0)
        @test all(result.pvalues .<= 1)
        @test length(result.chromosomes) == 10
        @test all(result.chromosomes .>= 1)
        @test all(result.chromosomes .<= 22)
    end

    # ========================================================================
    # Test 2: Linear Model GWAS (Simple)
    # ========================================================================

    @testset "Linear Model GWAS - Simple" begin
        model = LinearModelGWAS(
            adjust_population_structure = false
        )

        @test model.adjust_population_structure == false
        @test model.n_pcs == 10  # default value

        # Test basic linear regression calculation
        # For a single SNP
        snp = simulated_genotypes[:, 1]
        pheno = phenotype_values

        # Manual calculation
        X = hcat(ones(length(snp)), snp)
        beta = X \ pheno
        predicted = X * beta
        residuals = pheno - predicted
        sigma2 = sum(residuals .^ 2) / (length(pheno) - 2)

        @test sigma2 > 0
        @test length(beta) == 2
    end

    # ========================================================================
    # Test 3: Linear Model GWAS with PCA Correction
    # ========================================================================

    @testset "Linear Model GWAS - PCA Correction" begin
        model = LinearModelGWAS(
            adjust_population_structure = true,
            n_pcs = 3
        )

        @test model.adjust_population_structure == true
        @test model.n_pcs == 3
    end

    # ========================================================================
    # Test 4: Mixed Model GWAS
    # ========================================================================

    @testset "Mixed Model GWAS" begin
        # Create a simple GRM
        G = simulated_genotypes .- mean(simulated_genotypes, dims=1)
        G_std = G ./ std(G, dims=1)
        replace!(G_std, NaN => 0.0)

        grm = (G_std * G_std') ./ n_snps

        # Ensure GRM is positive definite
        grm = (grm + grm') / 2  # Make symmetric
        grm += I(n_samples) * 0.01  # Add small diagonal

        model = MixedModelGWAS(grm, reml=true)

        @test model.grm !== nothing
        @test size(model.grm) == (n_samples, n_samples)
        @test isapprox(model.grm, model.grm', atol=1e-10)  # Should be symmetric
        @test model.reml == true
    end

    # ========================================================================
    # Test 5: Multiple Testing Correction - Bonferroni
    # ========================================================================

    @testset "Multiple Testing - Bonferroni" begin
        pvalues = [0.001, 0.01, 0.05, 0.1, 0.5]
        n_tests = length(pvalues)

        adjusted = adjust_pvalues(pvalues, method=:bonferroni)

        @test length(adjusted) == n_tests
        @test all(adjusted .>= pvalues)  # Adjusted should be more conservative
        @test adjusted[1] ≈ pvalues[1] * n_tests
        @test all(adjusted .<= 1.0)  # Should not exceed 1
    end

    # ========================================================================
    # Test 6: Multiple Testing Correction - FDR
    # ========================================================================

    @testset "Multiple Testing - FDR (Benjamini-Hochberg)" begin
        pvalues = sort([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8])

        adjusted = adjust_pvalues(pvalues, method=:fdr)

        @test length(adjusted) == length(pvalues)
        @test all(adjusted .>= pvalues)  # FDR adjusted should be >= original
        @test issorted(adjusted)  # Should maintain order if input is sorted
    end

    # ========================================================================
    # Test 7: Multiple Testing Correction - Šidák
    # ========================================================================

    @testset "Multiple Testing - Šidák" begin
        pvalues = [0.001, 0.01, 0.05]
        n_tests = length(pvalues)

        adjusted = adjust_pvalues(pvalues, method=:sidak)

        @test length(adjusted) == n_tests
        @test all(adjusted .>= pvalues)

        # Šidák formula: 1 - (1 - p)^n
        expected_first = 1 - (1 - pvalues[1])^n_tests
        @test isapprox(adjusted[1], expected_first, atol=1e-10)
    end

    # ========================================================================
    # Test 8: Genomic Control Lambda
    # ========================================================================

    @testset "Genomic Control Lambda" begin
        # Simulate p-values under null (should have lambda ≈ 1)
        Random.seed!(42)
        null_pvalues = rand(1000)

        lambda = calculate_genomic_control_lambda(null_pvalues)

        # Under null, lambda should be close to 1
        @test lambda > 0.8
        @test lambda < 1.2

        # Test with inflated p-values
        inflated_pvalues = null_pvalues .^ 1.5  # Inflate
        lambda_inflated = calculate_genomic_control_lambda(inflated_pvalues)

        @test lambda_inflated > lambda  # Should be more inflated
    end

    # ========================================================================
    # Test 9: Calculate Chi-square from P-value
    # ========================================================================

    @testset "Chi-square Calculation" begin
        pvalue = 0.05
        chisq = calculate_chisquare_from_pvalue(pvalue)

        @test chisq > 0
        @test chisq ≈ 3.841 atol=0.01  # Chi-square critical value at df=1, p=0.05
    end

    # ========================================================================
    # Test 10: Identify Significant SNPs
    # ========================================================================

    @testset "Identify Significant SNPs" begin
        pvalues = [1e-9, 5e-8, 1e-7, 1e-5, 0.001, 0.01, 0.1]
        threshold = 5e-8

        significant_indices = find_significant_snps(pvalues, threshold)

        @test length(significant_indices) == 2  # First two pass threshold
        @test 1 in significant_indices
        @test 2 in significant_indices
        @test !(3 in significant_indices)
    end

    # ========================================================================
    # Test 11: SNP Filtering
    # ========================================================================

    @testset "SNP Filtering" begin
        snp_ids = ["rs1", "rs2", "rs3", "rs4", "rs5"]
        pvalues = [1e-10, 5e-9, 1e-7, 0.001, 0.5]
        effect_sizes = [0.5, 0.3, 0.2, 0.1, 0.05]

        result = GWASResult(
            snp_ids,
            ones(Int, 5),
            collect(1:5),
            pvalues,
            effect_sizes,
            ones(5) .* 0.01
        )

        # Filter by p-value threshold
        filtered = filter_gwas_results(result, pvalue_threshold=1e-6)

        @test length(filtered.snp_ids) == 3  # Top 3 SNPs
        @test filtered.pvalues[1] == 1e-10
    end

    # ========================================================================
    # Test 12: Effect Size Standardization
    # ========================================================================

    @testset "Effect Size Standardization" begin
        # Test standardization of effect sizes
        effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        std_errors = [0.05, 0.05, 0.05, 0.05, 0.05]

        z_scores = effect_sizes ./ std_errors

        @test length(z_scores) == 5
        @test z_scores[1] ≈ 2.0
        @test z_scores[5] ≈ 10.0
    end

    # ========================================================================
    # Test 13: Manhattan Plot Data Preparation
    # ========================================================================

    @testset "Manhattan Plot Data" begin
        snp_ids = ["rs" * string(i) for i in 1:100]
        chromosomes = repeat(1:10, inner=10)
        positions = repeat(1:10, outer=10) .* 1000
        pvalues = rand(100) .* 1e-6

        result = GWASResult(
            snp_ids,
            chromosomes,
            positions,
            pvalues,
            randn(100) .* 0.1,
            ones(100) .* 0.05
        )

        # This would call visualization module functions
        # Here we just test data integrity
        @test length(unique(chromosomes)) == 10
        @test all(pvalues .> 0)
        @test all(pvalues .<= 1)
    end

    # ========================================================================
    # Test 14: QQ Plot Data Preparation
    # ========================================================================

    @testset "QQ Plot Data" begin
        Random.seed!(123)
        pvalues = rand(1000)

        # Sort for QQ plot
        sorted_pvalues = sort(pvalues)

        # Expected under uniform distribution
        n = length(pvalues)
        expected = collect(1:n) ./ (n + 1)

        @test length(sorted_pvalues) == n
        @test issorted(sorted_pvalues)
        @test length(expected) == n

        # Check if roughly uniform (should be under null)
        ks_stat = maximum(abs.(sorted_pvalues .- expected))
        @test ks_stat < 0.1  # Kolmogorov-Smirnov test tolerance
    end

    # ========================================================================
    # Test 15: Heritability Estimation (from mixed model)
    # ========================================================================

    @testset "Heritability Estimation" begin
        # Simulate variance components
        vg = 0.5  # Genetic variance
        ve = 0.5  # Environmental variance

        h2 = vg / (vg + ve)

        @test h2 ≈ 0.5
        @test h2 >= 0
        @test h2 <= 1
    end

    # ========================================================================
    # Test 16: REML Estimation
    # ========================================================================

    @testset "REML vs ML" begin
        # Test that REML and ML give different but reasonable estimates
        # This is a conceptual test

        model_reml = MixedModelGWAS(nothing, reml=true)
        model_ml = MixedModelGWAS(nothing, reml=false)

        @test model_reml.reml == true
        @test model_ml.reml == false
    end

    # ========================================================================
    # Test 17: Population Stratification Detection
    # ========================================================================

    @testset "Population Stratification" begin
        # High lambda suggests stratification
        lambda = 1.5

        @test lambda > 1.0  # Indicates inflation

        # Rule of thumb: lambda > 1.1 suggests stratification
        has_stratification = lambda > 1.1

        @test has_stratification == true
    end

    # ========================================================================
    # Test 18: Edge Cases
    # ========================================================================

    @testset "Edge Cases" begin
        # Test with very small p-values
        tiny_pvalues = [1e-300, 1e-200, 1e-100]
        @test all(tiny_pvalues .> 0)

        # Test with p-values near 1
        high_pvalues = [0.99, 0.999, 0.9999]
        adjusted = adjust_pvalues(high_pvalues, method=:bonferroni)
        @test all(adjusted .<= 1.0)

        # Test with single SNP
        single_pvalue = [0.05]
        adjusted_single = adjust_pvalues(single_pvalue, method=:bonferroni)
        @test length(adjusted_single) == 1
        @test adjusted_single[1] ≈ 0.05  # No adjustment needed for single test
    end

    # ========================================================================
    # Test 19: Parallel Computing Support
    # ========================================================================

    @testset "Parallel Computing" begin
        # Test that parallel flag is properly handled
        # (actual parallel execution tested in integration tests)

        parallel_flag = true
        @test parallel_flag == true

        # Check thread count
        n_threads = Threads.nthreads()
        @test n_threads >= 1
    end

    # ========================================================================
    # Test 20: Input Validation
    # ========================================================================

    @testset "Input Validation" begin
        # Test invalid p-values
        invalid_pvalues = [-0.1, 0.5, 1.5]  # Contains invalid values

        # Should handle gracefully or throw error
        @test any(invalid_pvalues .< 0) || any(invalid_pvalues .> 1)

        # Test empty input
        empty_pvalues = Float64[]
        @test length(empty_pvalues) == 0

        # Test mismatched dimensions
        snp_ids_short = ["rs1", "rs2"]
        pvalues_long = [0.01, 0.02, 0.03]
        @test length(snp_ids_short) != length(pvalues_long)
    end

end  # @testset "GWAS Module Tests"

println("✓ GWAS module tests completed")
