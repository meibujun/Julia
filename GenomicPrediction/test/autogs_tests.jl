# test/autogs_tests.jl - AutoGS 模块单元测试
# -------------------------------------------
#
# 本文件包含对 `AutoGS` 模块中自动化功能的单元测试。
# 测试的目的是确保：
# - `grid_search` 等超参数搜索函数能够正确地遍历参数空间并找到最佳组合。
# - 自动化流程能够与核心模型和评估模块无缝协作。

using Test
using DataFrames
using Random
using Hyperopt

@testset "AutoGS.jl - 自动基因组选择模块测试" begin

    @testset "grid_search" begin
        # --- 1. 准备模拟数据 ---
        Random.seed!(789)
        G = rand(10, 5)
        y = rand(10)
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        # --- 2. 定义模型生成器和超参数网格 ---
        # model_generator 必须接收一个字典
        model_generator(params) = GenomicPrediction.GBLUPModel(params[:lambda])

        hyperparameters = Dict(
            :lambda => [1.0, 10.0, 100.0]
        )

        # --- 3. 运行网格搜索 ---
        # 使用 k=2 以加快测试速度
        search_result = GenomicPrediction.grid_search(model_generator, mock_data, hyperparameters; k=2)

        # --- 4. 验证结果 ---
        @test search_result.best_params isa Dict
        @test haskey(search_result.best_params, :lambda)
        @test search_result.best_params[:lambda] in [1.0, 10.0, 100.0]

        @test length(search_result.results) == 3
        @test haskey(search_result.results[1], :params)
        @test haskey(search_result.results[1], :metrics)
    end

    # @testset "bayesian_optimization" begin
    #     # --- 1. 準備模擬數據 ---
    #     Random.seed!(101)
    #     G = rand(10, 5)
    #     y = rand(10)
    #     geno_df = DataFrame(G, :auto)
    #     pheno_df = DataFrame(y = y)
    #     mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

    #     # --- 2. 定义模型生成器和搜索空间 ---
    #     model_generator(params) = GenomicPrediction.GBLUPModel(params[:lambda])

    #     # 使用 Hyperopt 语法定义搜索空间
    #     search_space = Dict(
    #         :lambda => Hyperopt.loguniform(log(1.0), log(1000.0))
    #     )

    #     # --- 3. 运行贝叶斯优化 ---
    #     # 使用 k=2 和 max_iters=1 以加快测试速度
    #     opt_result = GenomicPrediction.bayesian_optimization(model_generator, mock_data, search_space; k=2, max_iters=1)

    #     # --- 4. 验证结果 ---
    #     @test opt_result.best_params isa Dict
    #     @test haskey(opt_result.best_params, :lambda)
    #     # 检查 lambda 是否在定义的范围内
    #     @test 1.0 <= opt_result.best_params[:lambda] <= 1000.0
    # end

end
