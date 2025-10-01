# 测试遗传评估模块 (GeneticEvaluation.jl)

using Test
using DataFrames
using CSV
using ..AnimalBreeding
using ..GeneticEvaluation

@testset "遗传评估模块: GeneticEvaluation.jl" begin

    # 使用与核心测试相同的数据
    ped_df = DataFrame(animal=[1, 2, 3, 4, 5], sire=[0, 0, 1, 1, 3], dam=[0, 0, 2, 2, 4])
    phen_df = DataFrame(animal=[3, 4, 5], milk=[102.0, 105.0, 110.0], herd=['A', 'B', 'A'])

    dm = DataManager("test")
    dm.pedigree = Pedigree(ped_df)
    dm.phenotypes = Phenotypes(phen_df, [:milk], [:herd])

    model = define_model(traits=[:milk], fixed=[:herd], random=[("animal", :additive)])

    @testset "BLUP 评估 (已知 h²)" begin
        # h²=0.5, λ = (1-0.5)/0.5 = 1.0
        result = run_evaluation(model, dm, method=:BLUP, h2=0.5)

        @test result isa BLUPResult
        @test haskey(result.breeding_values, "animal")
        @test length(result.breeding_values["animal"]) == 5

        # 检查育种值是否在合理范围内
        @test all(!isnan, result.breeding_values["animal"])
        @test std(result.breeding_values["animal"]) > 0

        # 检查可靠性计算
        @test haskey(result.reliabilities, "animal")
        @test length(result.reliabilities["animal"]) == 5
        @test all(0 .<= result.reliabilities["animal"] .<= 1.0)
        println("  BLUP 平均可靠性: ", round(mean(result.reliabilities["animal"]), digits=3))
    end

    @testset "REML 评估 (未知 h²)" begin
        # REML应该能估计出合理的方差组分
        # 由于数据量小，估计值可能不精确，但应在合理范围内
        result = run_evaluation(model, dm, method=:REML, maxiter=20, tol=1e-4)

        @test result isa BLUPResult
        @test haskey(result.variance_components, "animal")
        @test haskey(result.variance_components, "residual")

        vc_a = result.variance_components["animal"]
        vc_e = result.variance_components["residual"]
        h2_est = vc_a / (vc_a + vc_e)

        println("  REML 估计的遗传力: ", round(h2_est, digits=3))
        @test 0 < h2_est < 1.0
    end

end