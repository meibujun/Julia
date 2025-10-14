using Test
using LinearAlgebra
using SparseArrays
using AnimalBreeding

@testset "关系矩阵调度器" begin
    dm, _ = simulate_complete_dataset(n_generations=3,
                                      n_per_generation=20,
                                      n_markers=80,
                                      n_qtl=10,
                                      h2=0.4,
                                      add_omics=false)

    n_animals = nrow(dm.pedigree)

    A = compute_relationship_matrix(dm; type=:pedigree, compute_inverse=true)
    @test size(A) == (n_animals, n_animals)
    @test isapprox(A, transpose(A); atol=1e-8)
    @test all(diag(A) .>= 1.0)
    @test dm.A_inv_matrix isa SparseMatrixCSC
    @test size(dm.A_inv_matrix) == size(A)

    A_inv_direct = compute_relationship_matrix(dm; type=:pedigree_inv)
    @test A_inv_direct === dm.A_inv_matrix

    G = compute_relationship_matrix(dm; type=:genomic)
    @test size(G) == (n_animals, n_animals)
    @test isapprox(G, transpose(G); atol=1e-8)
    @test minimum(eigvals(Symmetric(G))) > -1e-6

    H_inv = compute_relationship_matrix(dm; type=:singlestep, blending_factor=0.03, ridge=1e-6)
    @test H_inv isa SparseMatrixCSC
    @test size(H_inv) == (n_animals, n_animals)
    @test isapprox(Matrix(H_inv), Matrix(transpose(H_inv)); atol=1e-8)
end
