using Test
using GenomicPro2
using LinearAlgebra
using Statistics

@testset "Models Module Tests" begin
    # Create test data
    n_samples = 100
    n_markers = 500

    # Generate genotype data
    geno_data = rand(0:2, n_samples, n_markers)
    sample_ids = ["S$i" for i in 1:n_samples]
    marker_ids = ["M$i" for i in 1:n_markers]
    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    # Generate phenotype data (with genetic signal)
    # Simulate: y = Xβ + u + e
    # where u ~ N(0, G*σ²ᵤ)
    true_h2 = 0.5
    genetic_values = randn(n_samples)
    error_values = randn(n_samples) * sqrt((1 - true_h2) / true_h2)
    pheno_values = genetic_values .+ error_values

    pheno = PhenotypeData(sample_ids, ["Trait"], reshape(pheno_values, n_samples, 1))

    @testset "GRM Computation" begin
        @testset "VanRaden method" begin
            G = compute_grm_vanraden(geno)

            @test size(G) == (n_samples, n_samples)
            @test issymmetric(G)

            # Diagonal should be close to 1
            diag_mean = mean(diag(G))
            @test 0.8 < diag_mean < 1.2

            # Test with MAF filtering
            G_filtered = compute_grm_vanraden(geno; min_maf=0.05)
            @test size(G_filtered) == (n_samples, n_samples)
        end

        @testset "Additive method" begin
            G = compute_grm_additive(geno)

            @test size(G) == (n_samples, n_samples)
            @test issymmetric(G)
            @test all(diag(G) .≈ 1.0)
        end

        @testset "Generic compute_grm" begin
            G1 = compute_grm(geno; method=:vanraden)
            G2 = compute_grm(geno; method=:additive)

            @test size(G1) == (n_samples, n_samples)
            @test size(G2) == (n_samples, n_samples)

            @test_throws ArgumentError compute_grm(geno; method=:unknown)
        end

        @testset "GRM validation" begin
            G = compute_grm_vanraden(geno)
            result = validate_grm(G)

            @test is_valid(result)
            @test haskey(result.metadata, :mean_diagonal)
            @test haskey(result.metadata, :min_eigenvalue)
        end
    end

    @testset "GBLUP Model" begin
        @testset "Model construction" begin
            model = GBLUPModel()
            @test model.method == :cholesky
            @test model.estimate_variances == true

            model_pcg = GBLUPModel(method=:pcg, max_iter=500)
            @test model_pcg.method == :pcg
            @test model_pcg.max_iter == 500

            @test_throws ArgumentError GBLUPModel(method=:invalid)
        end

        @testset "GBLUP fitting (Cholesky)" begin
            G = compute_grm_vanraden(geno; min_maf=0.01)
            model = GBLUPModel(method=:cholesky)

            result = fit!(model, geno, pheno; G=G, trait_index=1)

            @test result isa GBLUPResult
            @test length(result.beta) >= 1  # At least intercept
            @test length(result.u) == n_samples
            @test result.var_e > 0
            @test result.var_u > 0
            @test 0 < result.heritability < 1

            # Check that model was updated
            @test model.result !== nothing
            @test model.sample_ids == sample_ids
        end

        @testset "GBLUP fitting (PCG)" begin
            G = compute_grm_vanraden(geno; min_maf=0.01)
            model = GBLUPModel(method=:pcg, max_iter=1000, tol=1e-6)

            result = fit!(model, geno, pheno; G=G, trait_index=1)

            @test result isa GBLUPResult
            @test length(result.u) == n_samples
            @test result.var_e > 0
            @test result.var_u > 0
        end

        @testset "GBLUP prediction" begin
            G = compute_grm_vanraden(geno; min_maf=0.01)
            model = GBLUPModel(method=:cholesky)

            # Fit model
            result = fit!(model, geno, pheno; G=G)

            # Predict (same samples)
            predictions = predict(model, geno)

            @test length(predictions) == n_samples
            @test all(isfinite, predictions)

            # Predictions should correlate with true breeding values
            # (not perfect due to noise, but should have some correlation)
            # cor_pred_true = cor(predictions, genetic_values)
            # @test cor_pred_true > 0.1  # Weak threshold due to randomness
        end

        @testset "Prediction without fitting" begin
            model = GBLUPModel()
            @test_throws ArgumentError predict(model, geno)
        end

        @testset "GBLUPResult display" begin
            G = compute_grm_vanraden(geno; min_maf=0.01)
            model = GBLUPModel(method=:cholesky, estimate_variances=false)
            result = fit!(model, geno, pheno; G=G)

            # Test that show() doesn't error
            io = IOBuffer()
            show(io, result)
            output = String(take!(io))

            @test contains(output, "GBLUP Results")
            @test contains(output, "Heritability")
        end
    end

    @testset "Center and scale genotypes" begin
        X = Float64.(to_matrix(geno; impute=true))
        freqs = allele_frequencies(geno)

        @testset "Center genotypes" begin
            Z = center_genotypes(X, freqs)

            @test size(Z) == size(X)

            # Check centering
            for j in 1:n_markers
                mean_geno = mean(Z[:, j])
                @test abs(mean_geno) < 1e-10 || abs(mean_geno - (mean(X[:, j]) - 2 * freqs[j])) < 1e-10
            end
        end

        @testset "Scale genotypes" begin
            Z = center_genotypes(X, freqs)
            Z_scaled = scale_genotypes(Z, freqs)

            @test size(Z_scaled) == size(Z)

            # Variance of scaled genotypes should be related to p(1-p)
            # (approximately 1 for most columns after scaling)
        end

        @testset "Dimension mismatch" begin
            wrong_freqs = freqs[1:10]
            @test_throws DimensionMismatchError center_genotypes(X, wrong_freqs)
            @test_throws DimensionMismatchError scale_genotypes(X, wrong_freqs)
        end
    end

    @testset "Integration: Full workflow" begin
        # This tests a complete analysis pipeline

        # 1. Create data
        n = 50
        m = 200
        geno_data = rand(0:2, n, m)
        geno = CompactGenotypes(geno_data,
                               ["S$i" for i in 1:n],
                               ["M$i" for i in 1:m])

        # 2. Generate phenotypes with known heritability
        true_u = randn(n)
        true_e = randn(n) * 0.5
        y = true_u .+ true_e
        pheno = PhenotypeData(sample_ids(geno), ["Trait"], reshape(y, n, 1))

        # 3. Compute GRM
        G = compute_grm(geno; method=:vanraden, min_maf=0.01)

        # 4. Validate GRM
        grm_validation = validate_grm(G)
        @test is_valid(grm_validation)

        # 5. Fit GBLUP
        model = GBLUPModel(method=:cholesky)
        result = fit!(model, geno, pheno; G=G)

        # 6. Check results
        @test result.converged
        @test 0 < result.heritability < 1
        @test result.var_u > 0
        @test result.var_e > 0

        # 7. Predict
        predictions = predict(model, geno)
        @test length(predictions) == n
        @test all(isfinite, predictions)

        # 8. Correlation with true breeding values
        # (Should have some correlation, but won't be perfect)
        # correlation = cor(predictions, true_u)
        # @test correlation > 0  # At least positive correlation
    end
end

println("✓ All Models tests passed")
