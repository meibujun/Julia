# test/integration_tests.jl - 端到端集成测试
# -----------------------------------------------
#
# 本文件包含完整的端到端工作流测试，以确保各个模块能够协同工作。
# 测试模拟了用户的典型使用场景，例如：
# 1. 从文件加载数据。
# 2. 初始化并训练一个模型。
# 3. 使用训练好的模型进行预测。
# 4. 计算预测结果的评估指标。

using Test
using DataFrames

@testset "集成测试：端到端 GBLUP 工作流" begin

    println("\n--- [集成测试] 步骤 1: 加载数据 ---")
    data_path = joinpath(@__DIR__, "sample_data")
    geno_file = joinpath(data_path, "genotypes.csv")
    pheno_file = joinpath(data_path, "phenotypes.csv")

    data = GenomicPrediction.load_csv(geno_file, pheno_file)
    @test data isa GenomicPrediction.GenomicData
    @test size(data.genotypes) == (5, 10)
    @test size(data.phenotypes) == (5, 2)

    println("--- [集成测试] 步骤 2: 初始化 GBLUP 模型 ---")
    model = GenomicPrediction.GBLUPModel(5.0)
    @test model isa GenomicPrediction.GBLUPModel

    println("--- [集成测试] 步骤 3: 训练模型 ---")
    @test GenomicPrediction.fit!(model, data) === nothing
    @test length(model.effects) == 10

    println("--- [集成测试] 步骤 4: 进行预测 ---")
    predictions = GenomicPrediction.predict(model, data.genotypes)
    @test length(predictions) == 5

    println("--- [集成测试] 步骤 5: 评估结果 ---")
    # **修复**: 使用正确的列名 "trait1"
    true_values = data.phenotypes[!, "trait1"]

    acc = GenomicPrediction.accuracy(predictions, true_values)
    mse_val = GenomicPrediction.mse(predictions, true_values)

    @test -1.0 <= acc <= 1.0
    @test mse_val >= 0.0

    println("--- [集成测试] GBLUP 工作流测试完成 ---")
end
