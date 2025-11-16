"""
Main test suite for GenomicPro2.

Run all tests with: `julia --project test/runtests.jl`
Or in REPL: `using Pkg; Pkg.test("GenomicPro2")`
"""

using Test
using GenomicPro2
using LinearAlgebra
using Statistics

# Test suites
include("test_core.jl")
include("test_genotypes.jl")
include("test_io.jl")
include("test_models.jl")

@testset "GenomicPro2 Test Suite" begin
    @testset "Core" begin
        test_core()
    end

    @testset "Genotypes" begin
        test_genotypes()
    end

    # I/O tests are standalone (included above)

    # Models tests are standalone (included above)
end

println("\n" * "="^60)
println("All GenomicPro2 tests completed successfully!")
println("="^60)
