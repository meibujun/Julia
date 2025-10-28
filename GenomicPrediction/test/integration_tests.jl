# test/integration_tests.jl

using Test
using DataFrames
using Random
using GenomicPrediction

@testset "Integration Tests" begin

    @testset "End-to-End GBLUP Workflow" begin
        println("\n--- [集成测试] 步骤 1: 加载数据 ---")
        geno_df = DataFrame(ID=1:5, m1=rand(0:2, 5), m2=rand(0:2, 5))
        pheno_df = DataFrame(ID=1:5, y=rand(5))
        data = GenomicData(geno_df, pheno_df)
        @test data isa GenomicData

        println("--- [集成测试] 步骤 2: 初始化 GBLUP 模型 ---")
        model = GBLUPModel(lambda=50.0)
        @test model.lambda == 50.0

        println("--- [集成测试] 步骤 3: 训练模型 ---")
        fit!(model, data)
        @test length(model.effects) == 2

        println("--- [集成测试] 步骤 4: 进行预测 ---")
        new_geno = DataFrame(ID=6:7, m1=rand(0:2, 2), m2=rand(0:2, 2))
        predictions = predict(model, new_geno)
        @test length(predictions) == 2

        println("--- [集成测试] 步骤 5: 评估结果 ---")
        # In a real scenario, you would have true values for the new data
        @test all(isfinite, predictions)

        println("--- [集成测试] GBLUP 工作流测试完成 ---")
    end

end
