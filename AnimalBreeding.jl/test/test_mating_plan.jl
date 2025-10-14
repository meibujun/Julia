# AnimalBreeding.jl/test/test_mating_plan.jl

using Test
using AnimalBreeding
using DataFrames

@testset "Mating Plan" begin
    sires = [1, 2, 3]
    dams = [4, 5, 6]
    relationship_matrix = rand(6, 6)
    sire_ebvs = [10.0, 12.0, 8.0]
    dam_ebvs = [9.0, 11.0, 7.0]

    mating_plan = design_mating_plan(sires, dams, relationship_matrix, sire_ebvs, dam_ebvs)

    @test mating_plan isa DataFrame
    @test "sire" in names(mating_plan)
    @test "dam" in names(mating_plan)
    @test "inbreeding" in names(mating_plan)
    @test "expected_gain" in names(mating_plan)
end
