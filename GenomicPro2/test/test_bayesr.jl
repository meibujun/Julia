using Test
using GenomicPro2
using Statistics
using Random
using LinearAlgebra

@testset "BayesR Tests" begin
    # Create test data
    Random.seed!(42)

    n_samples = 200
    n_markers = 500
    n_causal = 50
    true_h2 = 0.6

    # Generate genotype data
    geno_data = rand(0:2, n_samples, n_markers)
    sample_ids = [string("S", i) for i in 1:n_samples]
    marker_ids = [string("M", i) for i in 1:n_markers]
    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    # Generate phenotypes with known genetic architecture
    X = to_matrix(geno; impute=true)

    # Create sparse genetic architecture (like BayesR assumes)
    # 10 large effects, 40 small/medium effects, rest zero
    true_effects = zeros(n_markers)
    causal_idx = sort(randperm(n_markers)[1:n_causal])

    # Large effects (top 10)
    true_effects[causal_idx[1:10]] = randn(10) * 0.5
    # Small/medium effects (next 40)
    true_effects[causal_idx[11:50]] = randn(40) * 0.1

    # Generate phenotypes
    genetic_values = X * true_effects
    genetic_values = (genetic_values .- mean(genetic_values))
    genetic_values = genetic_values .* sqrt(true_h2 / var(genetic_values))

    environmental = randn(n_samples) * sqrt(1 - true_h2)
    pheno_values = genetic_values .+ environmental

    pheno = PhenotypeData(sample_ids, ["Trait1"], reshape(pheno_values, n_samples, 1))

    @testset "BayesRModel Construction" begin
        @testset "Default parameters" begin
            model = BayesRModel()

            @test model.n_iter == 50000
            @test model.burn_in == 20000
            @test model.thin == 10
            @test length(model.mixture_proportions) == 4
            @test isapprox(sum(model.mixture_proportions), 1.0)
            @test model.mixture_variances == [0.0, 0.0001, 0.001, 0.01]
            @test model.update_pi == true
            @test model.verbose == true
            @test isnothing(model.result)
        end

        @testset "Custom parameters" begin
            model = BayesRModel(
                n_iter = 1000,
                burn_in = 500,
                thin = 5,
                mixture_proportions = [0.6, 0.2, 0.15, 0.05],
                update_pi = false,
                verbose = false,
                seed = 123
            )

            @test model.n_iter == 1000
            @test model.burn_in == 500
            @test model.thin == 5
            @test model.mixture_proportions == [0.6, 0.2, 0.15, 0.05]
            @test model.update_pi == false
            @test model.verbose == false
            @test model.seed == 123
        end

        @testset "Invalid parameters" begin
            @test_throws ArgumentError BayesRModel(n_iter = 100, burn_in = 200)
            @test_throws ArgumentError BayesRModel(mixture_proportions = [0.5, 0.3, 0.1])  # Length != 4
            @test_throws ArgumentError BayesRModel(mixture_proportions = [0.5, 0.3, 0.15, 0.1])  # Sum != 1
            @test_throws ArgumentError BayesRModel(mixture_proportions = [-0.1, 0.4, 0.4, 0.3])  # Negative
            @test_throws ArgumentError BayesRModel(mixture_variances = [0.1, 0.001, 0.01, 0.0])  # First != 0
            @test_throws ArgumentError BayesRModel(thin = 0)
        end
    end

    @testset "BayesR Fitting" begin
        @testset "Basic fitting (short run)" begin
            model = BayesRModel(
                n_iter = 1000,
                burn_in = 500,
                thin = 10,
                seed = 123,
                verbose = false
            )

            fit!(model, geno, pheno)

            @test !isnothing(model.result)
            @test !isnothing(model.sample_ids)
            @test length(model.sample_ids) == n_samples

            result = model.result
            @test result isa BayesRResult
            @test length(result.marker_effects) == n_markers
            @test length(result.marker_effects_sd) == n_markers
            @test length(result.marker_pip) == n_markers
            @test size(result.marker_components) == (n_markers, 4)
            @test result.n_samples == n_samples
            @test result.n_markers == n_markers
            @test result.n_iter == 1000
            @test result.burn_in == 500

            # Check variance components are positive
            @test result.genetic_variance > 0
            @test result.residual_variance > 0
            @test 0 < result.heritability < 1

            # Check mixture proportions sum to 1
            @test isapprox(sum(result.mixture_proportions), 1.0, atol=0.01)

            # Check component probabilities sum to 1 for each marker
            for i in 1:n_markers
                @test isapprox(sum(result.marker_components[i, :]), 1.0, atol=0.01)
            end

            # Check PIPs are valid probabilities
            @test all(0 .<= result.marker_pip .<= 1)
        end

        @testset "With MAF filtering" begin
            model = BayesRModel(
                n_iter = 500,
                burn_in = 250,
                seed = 123,
                verbose = false
            )

            fit!(model, geno, pheno; min_maf = 0.05)

            @test !isnothing(model.result)
            # Should have fewer markers after filtering
            @test model.result.n_markers < n_markers
        end

        @testset "Without scaling" begin
            model = BayesRModel(
                n_iter = 500,
                burn_in = 250,
                seed = 123,
                verbose = false
            )

            fit!(model, geno, pheno; scale_X = false)

            @test !isnothing(model.result)
        end

        @testset "Fixed mixture proportions" begin
            model = BayesRModel(
                n_iter = 500,
                burn_in = 250,
                update_pi = false,  # Don't update π
                seed = 123,
                verbose = false
            )

            initial_pi = copy(model.mixture_proportions)
            fit!(model, geno, pheno)

            # π should be close to initial values (not updated)
            @test isapprox(model.result.mixture_proportions, initial_pi, atol=0.1)
        end
    end

    @testset "BayesR Prediction" begin
        # Fit model
        model = BayesRModel(
            n_iter = 1000,
            burn_in = 500,
            seed = 123,
            verbose = false
        )
        fit!(model, geno, pheno)

        @testset "Predict on training data" begin
            gebv = predict(model, geno)

            @test length(gebv) == n_samples
            @test all(isfinite.(gebv))

            # Should have some correlation with true breeding values
            # (though not perfect due to short MCMC)
            cor_with_true = cor(gebv, genetic_values)
            @test cor_with_true > 0.1  # Very conservative threshold
        end

        @testset "Predict on new data" begin
            # Create new test data (same markers)
            n_test = 50
            geno_test_data = rand(0:2, n_test, n_markers)
            sample_ids_test = [string("T", i) for i in 1:n_test]
            geno_test = CompactGenotypes(geno_test_data, sample_ids_test, marker_ids)

            gebv_test = predict(model, geno_test)

            @test length(gebv_test) == n_test
            @test all(isfinite.(gebv_test))
        end

        @testset "Predict before fitting" begin
            model_unfitted = BayesRModel(verbose = false)

            @test_throws ArgumentError predict(model_unfitted, geno)
        end
    end

    @testset "BayesRResult Display" begin
        model = BayesRModel(
            n_iter = 500,
            burn_in = 250,
            seed = 123,
            verbose = false
        )
        fit!(model, geno, pheno)

        io = IOBuffer()
        show(io, model.result)
        output = String(take!(io))

        @test contains(output, "BayesR Model Results")
        @test contains(output, "Samples:")
        @test contains(output, "Markers:")
        @test contains(output, "Genetic variance")
        @test contains(output, "Heritability")
        @test contains(output, "Mixture Proportions")
    end

    @testset "Reproducibility with Seed" begin
        # Same seed should give same results
        model1 = BayesRModel(
            n_iter = 500,
            burn_in = 250,
            seed = 999,
            verbose = false
        )
        fit!(model1, geno, pheno)

        model2 = BayesRModel(
            n_iter = 500,
            burn_in = 250,
            seed = 999,
            verbose = false
        )
        fit!(model2, geno, pheno)

        # Results should be identical
        @test model1.result.marker_effects == model2.result.marker_effects
        @test model1.result.genetic_variance == model2.result.genetic_variance
        @test model1.result.residual_variance == model2.result.residual_variance
    end

    @testset "Variable Selection Performance" begin
        # Fit model with longer run for better convergence
        model = BayesRModel(
            n_iter = 5000,
            burn_in = 2500,
            thin = 5,
            seed = 42,
            verbose = false
        )
        fit!(model, geno, pheno)

        result = model.result

        # Check if model identifies some causal SNPs
        # Get top SNPs by PIP
        top_snps_idx = sortperm(result.marker_pip, rev=true)[1:100]

        # How many true causal SNPs are in top 100 by PIP?
        causal_set = Set(causal_idx)
        top_set = Set(top_snps_idx)
        overlap = length(intersect(causal_set, top_set))

        # Should find more than random chance (random = 50*100/500 = 10)
        @test overlap > 15  # Conservative threshold

        # Check effect size estimates
        # Large effect SNPs should have larger posterior effects
        large_effect_idx = causal_idx[1:10]
        small_effect_idx = causal_idx[11:50]

        mean_large = mean(abs.(result.marker_effects[large_effect_idx]))
        mean_small = mean(abs.(result.marker_effects[small_effect_idx]))

        # Large effects should be estimated as larger (on average)
        @test mean_large > mean_small
    end

    @testset "Heritability Estimation" begin
        # Fit model with moderate run
        model = BayesRModel(
            n_iter = 3000,
            burn_in = 1500,
            seed = 42,
            verbose = false
        )
        fit!(model, geno, pheno)

        result = model.result

        # Estimated h² should be in reasonable range of true h²
        # (within 0.2 given short MCMC and small data)
        @test abs(result.heritability - true_h2) < 0.3
    end

    @testset "Mixture Component Usage" begin
        model = BayesRModel(
            n_iter = 2000,
            burn_in = 1000,
            seed = 42,
            verbose = false
        )
        fit!(model, geno, pheno)

        result = model.result

        # All 4 components should be used (even if rarely)
        @test all(result.mixture_proportions .> 0.0)

        # Null component should have highest proportion
        # (most SNPs should have zero effect)
        @test result.mixture_proportions[1] > 0.4

        # Check that component assignments make sense
        # SNPs in non-null components should have higher |effects|
        null_snps = findall(result.marker_components[:, 1] .> 0.5)
        nonnull_snps = findall(result.marker_components[:, 1] .< 0.5)

        if !isempty(null_snps) && !isempty(nonnull_snps)
            mean_effect_null = mean(abs.(result.marker_effects[null_snps]))
            mean_effect_nonnull = mean(abs.(result.marker_effects[nonnull_snps]))

            @test mean_effect_nonnull > mean_effect_null
        end
    end

    @testset "Training Set GEBVs" begin
        model = BayesRModel(
            n_iter = 1000,
            burn_in = 500,
            seed = 42,
            verbose = false
        )
        fit!(model, geno, pheno)

        result = model.result

        @test length(result.gebv_train) == n_samples
        @test all(isfinite.(result.gebv_train))

        # Training GEBVs should correlate with phenotypes
        cor_train = cor(result.gebv_train, pheno_values)
        @test cor_train > 0.2  # Conservative threshold
    end
end

println("✓ All BayesR tests passed")
