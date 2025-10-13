using Test
using AnimalBreeding

@testset "自动关系矩阵准备" begin
    dm, _ = simulate_complete_dataset(n_generations=2,
                                      n_per_generation=12,
                                      n_markers=60,
                                      n_qtl=8,
                                      h2=0.35,
                                      add_omics=false)

    dm.A_matrix = nothing
    dm.A_inv_matrix = nothing
    dm.H_inv_matrix = nothing
    dm.G_matrix = nothing

    model = define_model(traits=["trait"],
                         fixed=["herd"],
                         random=[("animal", :additive)])

    result = run_evaluation(model, dm; method=:BLUP, h2=0.3, estimate_variances=false)

    @test dm.A_inv_matrix !== nothing
    @test result isa AnimalBreeding.GeneticEvalResult
    @test nrow(result.breeding_values) == length(dm.animal_map)
end
