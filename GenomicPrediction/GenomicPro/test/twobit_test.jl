# test/twobit_test.jl

using Test
using GenomicPro

@testset "TwoBitGenotypes" begin
    # Test constructor
    geno = [0 1 2 missing; 1 2 0 1]
    twobit_geno = TwoBitGenotypes(geno)
    @test size(twobit_geno) == (2, 4)

    # Test getindex
    @test twobit_geno[1, 1] == 0
    @test twobit_geno[1, 2] == 1
    @test twobit_geno[1, 3] == 2
    @test ismissing(twobit_geno[1, 4])
    @test twobit_geno[2, 1] == 1
    @test twobit_geno[2, 2] == 2
    @test twobit_geno[2, 3] == 0
    @test twobit_geno[2, 4] == 1
end
