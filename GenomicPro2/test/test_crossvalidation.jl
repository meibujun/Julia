using Test
using GenomicPro2
using Statistics
using Random

@testset "Cross-Validation Tests" begin
    # Create test data
    Random.seed!(123)

    n_samples = 100
    n_markers = 200

    # Generate genotype data
    geno_data = rand(0:2, n_samples, n_markers)
    sample_ids = ["S$i" for i in 1:n_samples]
    marker_ids = ["M$i" for i in 1:n_markers]
    geno = CompactGenotypes(geno_data, sample_ids, marker_ids)

    # Generate phenotype data with genetic signal
    true_h2 = 0.5
    X = to_matrix(geno; impute=true)
    genetic_values = X[:, 1:50] * randn(50)  # Use first 50 markers
    genetic_values = (genetic_values .- mean(genetic_values)) ./ std(genetic_values)
    error_values = randn(n_samples) * sqrt((1 - true_h2) / true_h2)
    pheno_values = genetic_values .+ error_values

    pheno = PhenotypeData(sample_ids, ["Trait"], reshape(pheno_values, n_samples, 1))

    @testset "create_folds" begin
        @testset "Basic functionality" begin
            folds = create_folds(100, 5; shuffle=false)

            @test length(folds) == 5
            @test all(length(fold) == 20 for fold in folds)

            # Check no overlap
            all_idx = vcat(folds...)
            @test length(unique(all_idx)) == 100
            @test Set(all_idx) == Set(1:100)
        end

        @testset "Unequal folds" begin
            folds = create_folds(103, 5; shuffle=false)

            @test length(folds) == 5
            # First 3 folds get 21, last 2 get 20
            @test length(folds[1]) == 21
            @test length(folds[2]) == 21
            @test length(folds[3]) == 21
            @test length(folds[4]) == 20
            @test length(folds[5]) == 20

            # Total should be 103
            @test sum(length(fold) for fold in folds) == 103
        end

        @testset "Shuffle with seed" begin
            folds1 = create_folds(100, 5; shuffle=true, seed=42)
            folds2 = create_folds(100, 5; shuffle=true, seed=42)
            folds3 = create_folds(100, 5; shuffle=true, seed=43)

            # Same seed = same folds
            @test folds1 == folds2

            # Different seed = different folds (very likely)
            @test folds1 != folds3
        end

        @testset "Edge cases" begin
            # k = n (leave-one-out)
            folds = create_folds(10, 10; shuffle=false)
            @test length(folds) == 10
            @test all(length(fold) == 1 for fold in folds)

            # k = 2
            folds = create_folds(100, 2; shuffle=false)
            @test length(folds) == 2
            @test length(folds[1]) == 50
            @test length(folds[2]) == 50
        end

        @testset "Invalid parameters" begin
            @test_throws ArgumentError create_folds(100, 1)  # k < 2
            @test_throws ArgumentError create_folds(10, 20)  # k > n
        end
    end

    @testset "k-fold CV" begin
        @testset "5-fold CV" begin
            result = kfold_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                k = 5,
                seed = 123,
                verbose = false
            )

            @test result isa CVResult
            @test result.cv_method == :kfold
            @test length(result.predictions) == n_samples
            @test length(result.observed) == n_samples
            @test length(result.fold_results) == 5

            # All samples should be assigned to a fold
            @test all(1 .<= result.fold_assignments .<= 5)
            @test length(unique(result.fold_assignments)) == 5

            # Check metrics
            @test 0 <= result.metrics.correlation <= 1
            @test result.metrics.mse >= 0
            @test result.metrics.mae >= 0
            @test 0 <= result.metrics.r_squared <= 1

            # With genetic signal, should have some predictive ability
            @test result.metrics.correlation > 0.1
        end

        @testset "10-fold CV" begin
            result = kfold_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                k = 10,
                seed = 123,
                verbose = false
            )

            @test length(result.fold_results) == 10
            @test all(1 .<= result.fold_assignments .<= 10)
        end

        @testset "With GRM options" begin
            result = kfold_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                k = 5,
                grm_options = (min_maf = 0.05, method = :vanraden),
                seed = 123,
                verbose = false
            )

            @test result isa CVResult
        end

        @testset "Without recomputing GRM" begin
            result = kfold_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                k = 5,
                compute_grm = false,
                seed = 123,
                verbose = false
            )

            @test result isa CVResult
        end
    end

    @testset "Random CV" begin
        @testset "10 repetitions, 20% test" begin
            result = random_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                n_reps = 10,
                test_fraction = 0.2,
                seed = 123,
                verbose = false
            )

            @test result isa CVResult
            @test result.cv_method == :random
            @test length(result.fold_results) == 10

            # Total predictions = 10 reps × 20 test samples
            @test length(result.predictions) == 10 * round(Int, n_samples * 0.2)

            # Check metrics
            @test 0 <= result.metrics.correlation <= 1
            @test result.metrics.mse >= 0
        end

        @testset "Different test fractions" begin
            result_10 = random_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                n_reps = 5,
                test_fraction = 0.1,
                seed = 123,
                verbose = false
            )

            result_30 = random_cv(
                () -> GBLUPModel(method=:cholesky, estimate_variances=false),
                geno,
                pheno;
                n_reps = 5,
                test_fraction = 0.3,
                seed = 123,
                verbose = false
            )

            # More test data → more predictions
            @test length(result_30.predictions) > length(result_10.predictions)
        end

        @testset "Invalid parameters" begin
            @test_throws ArgumentError random_cv(
                () -> GBLUPModel(),
                geno,
                pheno;
                test_fraction = 0.0,  # Invalid
                verbose = false
            )

            @test_throws ArgumentError random_cv(
                () -> GBLUPModel(),
                geno,
                pheno;
                test_fraction = 1.0,  # Invalid
                verbose = false
            )
        end
    end

    @testset "CVResult display" begin
        result = kfold_cv(
            () -> GBLUPModel(method=:cholesky, estimate_variances=false),
            geno,
            pheno;
            k = 5,
            seed = 123,
            verbose = false
        )

        # Test that show() doesn't error
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test contains(output, "Cross-Validation Result")
        @test contains(output, "Correlation")
        @test contains(output, "MSE")
    end

    @testset "Reproducibility" begin
        # Same seed should give same results
        result1 = kfold_cv(
            () -> GBLUPModel(method=:cholesky, estimate_variances=false),
            geno,
            pheno;
            k = 5,
            seed = 999,
            verbose = false
        )

        result2 = kfold_cv(
            () -> GBLUPModel(method=:cholesky, estimate_variances=false),
            geno,
            pheno;
            k = 5,
            seed = 999,
            verbose = false
        )

        @test result1.predictions == result2.predictions
        @test result1.fold_assignments == result2.fold_assignments
        @test result1.metrics.correlation == result2.metrics.correlation
    end

    @testset "Fold results structure" begin
        result = kfold_cv(
            () -> GBLUPModel(method=:cholesky, estimate_variances=false),
            geno,
            pheno;
            k = 5,
            seed = 123,
            verbose = false
        )

        # Check each fold result has required fields
        for fold_res in result.fold_results
            @test haskey(fold_res, :fold)
            @test haskey(fold_res, :n_train)
            @test haskey(fold_res, :n_test)
            @test haskey(fold_res, :correlation)
            @test haskey(fold_res, :mse)
            @test haskey(fold_res, :mae)
            @test haskey(fold_res, :r_squared)

            # Train + test should equal total
            @test fold_res.n_train + fold_res.n_test == n_samples
        end
    end
end

println("✓ All Cross-Validation tests passed")
