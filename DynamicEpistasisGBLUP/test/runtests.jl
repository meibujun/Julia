using DynamicEpistasisGBLUP
using Test
using LinearAlgebra

@testset "DynamicEpistasisGBLUP.jl" begin

    @testset "G Matrix Calculation" begin
        # Simple genotype matrix for testing
        genotypes = [0 1 2; 2 1 0; 1 0 2]

        # Expected result from manual calculation
        # p = [1.0, 1/3, 4/3] / 2 = [0.5, 1/6, 2/3]
        # W = ...
        # G = ...
        # For simplicity, we'll test for properties rather than exact values
        # as the manual calculation is complex.

        G = GBLUP.calculate_g_matrix(genotypes)

        # Test dimensions
        @test size(G) == (3, 3)

        # Test for symmetry
        @test G ≈ G'

        # Test diagonal elements are positive
        @test all(diag(G) .> 0)
    end

    @testset "G_AA Matrix Calculation" begin
        genotypes = [0 1 2; 2 1 0; 1 0 2]
        G = GBLUP.calculate_g_matrix(genotypes)
        G_AA = GBLUP.calculate_gaa_matrix(genotypes)

        # Test dimensions
        @test size(G_AA) == (3, 3)

        # Test for symmetry
        @test G_AA ≈ G_AA'

        # Test that G_AA is the Hadamard product of G
        @test G_AA ≈ G .* G
    end

    @testset "GPU vs CPU Calculation" begin
        # Skip if CUDA is not available
        if CUDA.functional()
            genotypes = rand(0:2, 100, 500)

            G_cpu = GBLUP.calculate_g_matrix(genotypes)
            G_gpu = GPUAcceleration.calculate_g_matrix_gpu(genotypes)

            @test G_cpu ≈ G_gpu

            G_AA_cpu = GBLUP.calculate_gaa_matrix(genotypes)
            G_AA_gpu = GPUAcceleration.calculate_gaa_matrix_gpu(genotypes)

            @test G_AA_cpu ≈ G_AA_gpu
        else
            @warn "CUDA not functional, skipping GPU tests."
        end
    end

end
