using Test
using AnimalBreeding
using CSV
using DataFrames

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

@testset "结果保存" begin
    dm, _ = simulate_complete_dataset(n_generations=2,
                                      n_per_generation=8,
                                      n_markers=40,
                                      n_qtl=6,
                                      h2=0.3,
                                      add_omics=false)

    model = define_model(traits=["trait"],
                         fixed=["herd"],
                         random=[("animal", :additive)])

    result = run_evaluation(model, dm; method=:BLUP, h2=0.3, estimate_variances=false)

    mktemp() do path, io
        close(io)
        save_results(result, dm, path; include_pedigree=true)
        saved = CSV.read(path, DataFrame)
        @test :EBV in names(saved)
        @test (:sire in names(saved)) || (:dam in names(saved))
        @test size(saved, 1) == nrow(result.breeding_values)
    end
end
