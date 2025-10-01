# 测试机器学习模块 (MachineLearning.jl)

using Test
using DataFrames
using Random
using Statistics
using ..AnimalBreeding
using ..MachineLearning

@testset "机器学习模块: MachineLearning.jl" begin

    # 生成一个包含非线性关系的简单数据集
    Random.seed!(456)
    n, p = 100, 10
    X = randn(n, p)

    # y = 5*sin(X1) + 3*X2^2 + noise
    y = 5 * sin.(X[:, 1]) .+ 3 * X[:, 2].^2 .+ randn(n)

    @testset "随机森林模型" begin
        # 训练模型
        result = train_ml_model(:RandomForest, X, y; n_trees=50, max_depth=5, verbose=false)

        @test result isa MLResult
        @test result.method == "Random Forest"
        @test length(result.predictions) == n

        # 检查训练集上的性能
        r2 = r2_score(y, result.predictions)
        println("  随机森林训练集R²: ", round(r2, digits=3))
        @test r2 > 0.6 # 模型应该能捕捉到大部分变异
    end

    @testset "排列重要性 (Permutation Importance)" begin
        result = train_ml_model(:RandomForest, X, y; n_trees=50, verbose=false)

        # 使用增强后的模型解释功能
        importance = model_explain(result.model, X, y; metric=r2_score, lower_is_better=false)

        @test length(importance) == p
        @test isapprox(sum(importance), 1.0, atol=1e-6)

        # 根据生成规则，特征1和2应该是最重要的
        top_two_features = sortperm(importance, rev=true)[1:2]
        println("  最重要的特征 (排列法): ", top_two_features)
        @test 1 in top_two_features
        @test 2 in top_two_features
    end

    @testset "交叉验证" begin
        scores = cross_validate(X, y, :RandomForest; n_folds=3, metric=r2_score, n_trees=30, verbose=false)

        @test length(scores) == 3
        @test all(s -> -1.0 < s < 1.0, scores) # R²应该在合理范围内

        mean_r2 = mean(scores)
        println("  3折交叉验证平均R²: ", round(mean_r2, digits=3))
        @test mean_r2 > 0.5 # 模型应具有良好的泛化能力
    end

end