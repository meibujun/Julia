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

@testset "AutoGS.jl - 自动基因组选择模块测试" begin

    @testset "grid_search" begin
        # --- 1. 准备模拟数据 ---
        Random.seed!(789)
        G = rand(30, 10)
        y = rand(30)
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

end
