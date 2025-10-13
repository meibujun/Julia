using Test
using AnimalBreeding
using LinearAlgebra
using DataFrames

@testset "Optimal contribution selection" begin
    ebvs = [0.1, 0.3, 0.2, 0.5]
    R = Matrix{Float64}(I, 4, 4)

    result = optimal_contribution_selection(ebvs, R, 2; max_relationship=0.05)
    @test length(result["selected_indices"]) == 2
    @test result["constraint_satisfied"]
    @test isapprox(result["avg_relationship"], 0.0; atol=1e-8)

    df = DataFrame(animal_id=string.(1:4), EBV=ebvs)
    result_df = optimal_contribution_selection(df, R, 3)
    @test length(result_df["selected_ids"]) == 3
    @test all(result_df["selected_ids"] .∈ string.(1:4))
end
