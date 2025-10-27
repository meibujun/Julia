# test/eval_tests.jl - Evaluation 模块单元测试
# ----------------------------------------------
#
# 本文件包含对 `Evaluation` 模块中评估指标和工具的单元测试。
# 测试的目的是确保：
# - `accuracy` 和 `mse` 等指标的计算是数学上正确的。
# - `cross_validate` 函数能够正确地执行 K-折数据划分、模型训练和评估流程。

using Test
using Statistics
using DataFrames
using Random

@testset "Evaluation.jl - 评估与解释模块测试" begin

    @testset "accuracy (相关系数)" begin
        pred = [1.0, 2.0, 3.0, 4.0]
        truth = [1.1, 2.2, 3.3, 4.4]
        # 预期结果: 完美相关
        @test GenomicPrediction.accuracy(pred, truth) ≈ 1.0

        pred2 = [1.0, 2.0, 3.0, 4.0]
        truth2 = [4.0, 3.0, 2.0, 1.0]
        # 预期结果: 完美负相关
        @test GenomicPrediction.accuracy(pred2, truth2) ≈ -1.0

        pred3 = [1.0, 2.0, 1.0, 2.0]
        truth3 = [1.0, 1.0, 2.0, 2.0]
        # 预期结果: 不相关
        @test GenomicPrediction.accuracy(pred3, truth3) ≈ 0.0
    end

    @testset "mse (均方误差)" begin
        pred = [1.0, 2.0, 3.0]
        truth = [1.0, 2.0, 3.0]
        # 预期结果: 无误差
        @test GenomicPrediction.mse(pred, truth) == 0.0

        pred2 = [1.5, 2.5, 3.5]
        truth2 = [1.0, 2.0, 3.0]
        # 预期结果: (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.25
        @test GenomicPrediction.mse(pred2, truth2) ≈ 0.25
    end

    @testset "cross_validate (交叉验证)" begin
        # --- 1. 准备模拟数据 ---
        Random.seed!(42)
        G = rand(30, 10) # 30 个体, 10 标记
        y = rand(30)
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        # --- 2. 定义模型模板 ---
        # API 标准化：cross_validate 接收一个模型对象作为模板
        model_template = GenomicPrediction.GBLUPModel(5.0)

        k = 3 # 3-折交叉验证

        # --- 3. 运行交叉验证 ---
        cv_output = GenomicPrediction.cross_validate(model_template, mock_data, k)
        results = cv_output.metrics # 提取指标字典

        # --- 4. 验证结果 ---
        @test results isa Dict{String, Float64}
        @test haskey(results, "mean_accuracy")
        @test haskey(results, "mean_mse")
        @test -1.0 <= results["mean_accuracy"] <= 1.0
        @test results["mean_mse"] >= 0.0
        @test length(cv_output.raw_accuracies) == k # 检查原始结果数量
    end

end
