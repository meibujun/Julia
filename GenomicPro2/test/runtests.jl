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

@testset "GenomicPro2" begin
    @testset "Core" begin
        test_core()
    end

    @testset "Genotypes" begin
        test_genotypes()
    end
end
