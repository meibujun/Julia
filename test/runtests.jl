using GenomicPro3
using Test
using CUDA

@testset "GenomicPro3 Test Suite" begin
    
    @info "Running Core Tests..."
    include("core_tests.jl")
    
    @info "Running Stats Tests..."
    include("stats_tests.jl")
    
    @info "Running AI Tests..."
    include("ai_tests.jl")
    
end
