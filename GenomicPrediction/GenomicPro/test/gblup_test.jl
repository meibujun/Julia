# test/gblup_test.jl

using Test
using GenomicPro
using LinearAlgebra

@testset "GBLUP" begin
    # Create dummy data
    n = 100
    m = 500
    geno = rand([0, 1, 2], n, m)
    pheno = randn(n)
    G = cor(geno')

    vc = estimate_variance_components(G, pheno)
    λ = vc.residual_variance / vc.genetic_variance

    results = solve_gblup(G, pheno, λ)

    @test length(results.breeding_values) == n
end
