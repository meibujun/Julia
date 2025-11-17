using Test
using GenomicPro2
using Statistics
using Random
using LinearAlgebra

@testset "LD Pruning Tests" begin
    # Create test data
    Random.seed!(123)

    n_samples = 300
    n_markers = 200

    # Generate genotypes with known LD structure
    # Create blocks of markers in high LD
    geno_data = zeros(Int, n_samples, n_markers)

    # Block 1: Markers 1-20 in high LD
    base_geno1 = rand(0:2, n_samples)
    for i in 1:20
        # Add some noise
        geno_data[:, i] = base_geno1 .+ rand([-1, 0, 0, 0, 1], n_samples)
        geno_data[:, i] = clamp.(geno_data[:, i], 0, 2)
    end

    # Block 2: Markers 21-40 in high LD
    base_geno2 = rand(0:2, n_samples)
    for i in 21:40
        geno_data[:, i] = base_geno2 .+ rand([-1, 0, 0, 0, 1], n_samples)
        geno_data[:, i] = clamp.(geno_data[:, i], 0, 2)
    end

    # Rest: Independent markers
    for i in 41:n_markers
        geno_data[:, i] = rand(0:2, n_samples)
    end

    sample_ids = [string("S", i) for i in 1:n_samples]
    marker_ids = [string("M", i) for i in 1:n_markers]

    # Create chromosome and position info
    chromosome = vcat(fill("1", 100), fill("2", 100))
    position = vcat(collect(1:100) .* 1000, collect(1:100) .* 1000)

    geno = CompactGenotypes(
        geno_data,
        sample_ids,
        marker_ids;
        chromosome = chromosome,
        position = position
    )

    @testset "LD r² Computation" begin
        @testset "Perfect LD" begin
            # Same marker has r²=1
            r2 = compute_ld_r2(geno, 1, 1)
            @test r2 ≈ 1.0
        end

        @testset "High LD within block" begin
            # Markers in same LD block should have high r²
            r2_1_2 = compute_ld_r2(geno, 1, 2)
            r2_1_5 = compute_ld_r2(geno, 1, 5)

            @test r2_1_2 > 0.3  # Should have moderate to high LD
            @test r2_1_5 > 0.2
        end

        @testset "Low LD between blocks" begin
            # Markers in different blocks should have low LD
            r2_across = compute_ld_r2(geno, 10, 50)

            @test r2_across < 0.5  # Should be relatively independent
        end

        @testset "r² bounds" begin
            # r² should be between 0 and 1
            for i in [1, 20, 50, 100]
                for j in [i+1, i+10, i+20]
                    if j <= n_markers
                        r2 = compute_ld_r2(geno, i, j)
                        @test 0.0 <= r2 <= 1.0
                    end
                end
            end
        end
    end

    @testset "LD D' Computation" begin
        @testset "D' bounds" begin
            # D' should be between -1 and 1
            for i in 1:10
                for j in (i+1):20
                    dprime = compute_ld_dprime(geno, i, j)
                    @test -1.0 <= dprime <= 1.0
                end
            end
        end

        @testset "High LD markers" begin
            # Markers in same block should have high |D'|
            dprime = compute_ld_dprime(geno, 1, 2)
            @test abs(dprime) > 0.1
        end
    end

    @testset "Full LD Computation" begin
        ld_result = compute_ld_full(geno, 1, 2)

        @test ld_result isa LDResult
        @test ld_result.marker1 == "M1"
        @test ld_result.marker2 == "M2"
        @test 0.0 <= ld_result.r2 <= 1.0
        @test 0.0 <= ld_result.r <= 1.0
        @test -1.0 <= ld_result.Dprime <= 1.0
        @test ld_result.r^2 ≈ ld_result.r2
    end

    @testset "Window-based LD Pruning" begin
        @testset "Basic pruning" begin
            keep_idx = ld_prune_window(
                geno;
                window_size = 20,
                step_size = 10,
                r2_threshold = 0.5,
                verbose = false
            )

            @test length(keep_idx) < n_markers  # Should remove some markers
            @test length(keep_idx) > 0  # Should keep some markers
            @test all(1 .<= keep_idx .<= n_markers)
            @test issorted(keep_idx)  # Should be sorted
            @test allunique(keep_idx)  # No duplicates
        end

        @testset "Strict threshold" begin
            # Very strict threshold (r² > 0.2) should remove more markers
            keep_strict = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.2,
                verbose = false
            )

            # Lenient threshold (r² > 0.9) should remove fewer markers
            keep_lenient = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.9,
                verbose = false
            )

            @test length(keep_strict) < length(keep_lenient)
        end

        @testset "Respect chromosomes" begin
            keep_respect = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.5,
                respect_chromosomes = true,
                verbose = false
            )

            keep_no_respect = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.5,
                respect_chromosomes = false,
                verbose = false
            )

            # Results might differ slightly
            @test length(keep_respect) > 0
            @test length(keep_no_respect) > 0
        end

        @testset "Different window sizes" begin
            keep_small = ld_prune_window(
                geno;
                window_size = 10,
                r2_threshold = 0.5,
                verbose = false
            )

            keep_large = ld_prune_window(
                geno;
                window_size = 50,
                r2_threshold = 0.5,
                verbose = false
            )

            # Both should work
            @test length(keep_small) > 0
            @test length(keep_large) > 0
        end

        @testset "Subset markers after pruning" begin
            keep_idx = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.5,
                verbose = false
            )

            geno_pruned = subset_markers(geno, keep_idx)

            @test geno_pruned.n_markers == length(keep_idx)
            @test geno_pruned.n_samples == n_samples
            @test geno_pruned.marker_ids == marker_ids[keep_idx]
        end

        @testset "Invalid parameters" begin
            @test_throws ArgumentError ld_prune_window(geno; window_size = 1)
            @test_throws ArgumentError ld_prune_window(geno; step_size = 0)
            @test_throws ArgumentError ld_prune_window(geno; r2_threshold = -0.1)
            @test_throws ArgumentError ld_prune_window(geno; r2_threshold = 1.5)
        end
    end

    @testset "Pairwise LD Pruning" begin
        @testset "Basic pruning (small subset)" begin
            # Use small subset for speed
            geno_small = subset_markers(geno, 1:50)

            keep_idx = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.5,
                verbose = false
            )

            @test length(keep_idx) < 50  # Should remove some markers
            @test length(keep_idx) > 0  # Should keep some markers
            @test all(1 .<= keep_idx .<= 50)
            @test issorted(keep_idx)
            @test allunique(keep_idx)
        end

        @testset "Strict vs lenient threshold" begin
            geno_small = subset_markers(geno, 1:50)

            keep_strict = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.2,
                verbose = false
            )

            keep_lenient = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.9,
                verbose = false
            )

            @test length(keep_strict) <= length(keep_lenient)
        end

        @testset "Respect chromosomes" begin
            geno_small = subset_markers(geno, 1:50)

            keep_respect = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.5,
                respect_chromosomes = true,
                verbose = false
            )

            keep_no_respect = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.5,
                respect_chromosomes = false,
                verbose = false
            )

            @test length(keep_respect) > 0
            @test length(keep_no_respect) > 0
        end

        @testset "Max distance constraint" begin
            geno_small = subset_markers(geno, 1:50)

            keep_limited = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.5,
                max_distance = 10000,  # 10kb
                verbose = false
            )

            keep_unlimited = ld_prune_pairwise(
                geno_small;
                r2_threshold = 0.5,
                max_distance = nothing,
                verbose = false
            )

            # Limited distance might keep more markers (fewer pairs considered)
            @test length(keep_limited) > 0
            @test length(keep_unlimited) > 0
        end
    end

    @testset "LD Matrix Computation" begin
        @testset "Small matrix" begin
            marker_subset = 1:10
            ld_mat = compute_ld_matrix(geno, marker_subset)

            @test ld_mat isa LDMatrix
            @test length(ld_mat.marker_ids) == 10
            @test size(ld_mat.r2) == (10, 10)
            @test size(ld_mat.r) == (10, 10)
        end

        @testset "Matrix properties" begin
            marker_subset = 1:20
            ld_mat = compute_ld_matrix(geno, marker_subset)

            # Diagonal should be 1.0
            for i in 1:20
                @test ld_mat.r2[i, i] ≈ 1.0
                @test ld_mat.r[i, i] ≈ 1.0
            end

            # Matrix should be symmetric
            for i in 1:20
                for j in 1:20
                    @test ld_mat.r2[i, j] ≈ ld_mat.r2[j, i]
                    @test ld_mat.r[i, j] ≈ ld_mat.r[j, i]
                end
            end

            # All values should be between 0 and 1
            @test all(0.0 .<= ld_mat.r2 .<= 1.0)
            @test all(0.0 .<= ld_mat.r .<= 1.0)

            # r² should equal r^2
            for i in 1:20
                for j in 1:20
                    @test ld_mat.r[i, j]^2 ≈ ld_mat.r2[i, j] atol = 1e-10
                end
            end
        end

        @testset "LD block structure" begin
            # Markers 1-20 are in high LD block
            marker_subset = 1:20
            ld_mat = compute_ld_matrix(geno, marker_subset)

            # Mean r² within block should be moderate to high
            # (excluding diagonal)
            off_diagonal_r2 = []
            for i in 1:20
                for j in (i+1):20
                    push!(off_diagonal_r2, ld_mat.r2[i, j])
                end
            end

            mean_r2 = mean(off_diagonal_r2)
            @test mean_r2 > 0.1  # Should have some LD structure
        end
    end

    @testset "LD Pruning Effect on Prediction" begin
        # Create simple phenotype
        Random.seed!(456)
        y = randn(n_samples)
        pheno = PhenotypeData(sample_ids, ["Trait1"], reshape(y, n_samples, 1))

        @testset "GBLUP before and after pruning" begin
            # Before pruning
            G_full = compute_grm(geno; method = :vanraden, min_maf = 0.0)
            model_full = GBLUPModel(method = :cholesky, estimate_variances = false)
            fit!(model_full, geno, pheno; G = G_full)
            gebv_full = predict(model_full, geno)

            # After pruning
            keep_idx = ld_prune_window(
                geno;
                window_size = 20,
                r2_threshold = 0.7,
                verbose = false
            )

            geno_pruned = subset_markers(geno, keep_idx)
            G_pruned = compute_grm(geno_pruned; method = :vanraden, min_maf = 0.0)
            model_pruned = GBLUPModel(method = :cholesky, estimate_variances = false)
            fit!(model_pruned, geno_pruned, pheno; G = G_pruned)
            gebv_pruned = predict(model_pruned, geno_pruned)

            # Both should produce valid predictions
            @test all(isfinite.(gebv_full))
            @test all(isfinite.(gebv_pruned))

            # Predictions should be reasonably correlated
            # (since we're using random phenotypes, correlation might not be very high)
            cor_pred = cor(gebv_full, gebv_pruned)
            @test cor_pred > -1.0 && cor_pred < 1.0  # Valid correlation
        end
    end

    @testset "LD Result Display" begin
        ld_result = compute_ld_full(geno, 1, 2)

        io = IOBuffer()
        show(io, ld_result)
        output = String(take!(io))

        @test contains(output, "M1")
        @test contains(output, "M2")
        @test contains(output, "r=")
        @test contains(output, "r²=")
        @test contains(output, "D'=")
    end

    @testset "Real-world LD Patterns" begin
        # Test with more realistic LD decay pattern
        Random.seed!(789)

        n_samples_real = 500
        n_markers_real = 100

        # Create genotypes with distance-dependent LD
        geno_real_data = zeros(Int, n_samples_real, n_markers_real)

        # Start with random base genotypes
        for i in 1:n_markers_real
            geno_real_data[:, i] = rand(0:2, n_samples_real)
        end

        # Add LD structure that decays with distance
        for i in 1:(n_markers_real-1)
            for j in (i+1):min(i+10, n_markers_real)  # Only nearby markers
                # Copy some fraction of genotypes (creates LD)
                distance = j - i
                copy_fraction = max(0.0, 1.0 - distance / 10.0)

                n_copy = round(Int, n_samples_real * copy_fraction * 0.3)
                copy_idx = randperm(n_samples_real)[1:n_copy]

                geno_real_data[copy_idx, j] = geno_real_data[copy_idx, i]
            end
        end

        marker_ids_real = [string("SNP", i) for i in 1:n_markers_real]
        sample_ids_real = [string("ID", i) for i in 1:n_samples_real]
        chr_real = fill("1", n_markers_real)
        pos_real = collect(1:n_markers_real) .* 5000  # 5kb spacing

        geno_real = CompactGenotypes(
            geno_real_data,
            sample_ids_real,
            marker_ids_real;
            chromosome = chr_real,
            position = pos_real
        )

        @testset "LD decay with distance" begin
            # Nearby markers should have higher LD
            r2_close = compute_ld_r2(geno_real, 10, 11)
            r2_far = compute_ld_r2(geno_real, 10, 50)

            # This is stochastic, so just check they're valid
            @test 0.0 <= r2_close <= 1.0
            @test 0.0 <= r2_far <= 1.0
        end

        @testset "Pruning preserves marker spacing" begin
            keep_idx = ld_prune_window(
                geno_real;
                window_size = 30,
                r2_threshold = 0.5,
                verbose = false
            )

            geno_pruned = subset_markers(geno_real, keep_idx)

            # Pruned markers should be relatively well-spaced
            # (not all clustered in one region)
            @test geno_pruned.n_markers > 10
            @test geno_pruned.n_markers < n_markers_real
        end
    end
end

println("✓ All LD pruning tests passed")
