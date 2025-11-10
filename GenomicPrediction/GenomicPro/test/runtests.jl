using Test
using GenomicPro

@testset "GenomicPro.jl" begin
    include("twobit_test.jl")
    include("gblup_test.jl")
end
