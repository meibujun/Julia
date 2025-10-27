# test/fair_tests.jl - FAIRModeling 模块单元测试
# ------------------------------------------------
#
# 本文件包含对 `FAIRModeling` 模块中模型持久化功能的单元测试。
# 测试的目的是确保：
# - `save_model` 和 `load_model` 函数能够正确地序列化和反序列化模型对象。
# - 加载后的模型与原始模型在结构和参数上完全一致。
# - 加载后的模型能够产生与原始模型完全相同的预测结果，保证功能的一致性。

using Test
using DataFrames
using Random
using Dates

@testset "FAIRModeling.jl - FAIR 建模模块测试" begin

    @testset "模型保存与加载" begin
        # --- 1. 准备并训练一个模型 ---
        # 使用一个简单的 GBLUP 模型进行测试
        G = rand(10, 5)
        y = rand(10)
        geno_df = DataFrame(G, :auto)
        pheno_df = DataFrame(y = y)
        mock_data = GenomicPrediction.GenomicData(geno_df, pheno_df)

        original_model = GenomicPrediction.GBLUPModel(5.0)
        GenomicPrediction.fit!(original_model, mock_data)

        # --- 2. 定义临时文件路径并保存模型 ---
        # mktemp() 会创建一个临时文件并返回其路径，确保测试不会污染工作目录
        temp_path = mktemp()[1]
        GenomicPrediction.save_model(original_model, temp_path)

        @test isfile(temp_path) # 确认文件已创建

        # --- 3. 从文件加载模型 ---
        loaded_model = GenomicPrediction.load_model(temp_path)

        # --- 4. 验证加载的模型与原始模型是否一致 ---
        @test loaded_model isa typeof(original_model)
        @test loaded_model.lambda == original_model.lambda
        @test loaded_model.intercept ≈ original_model.intercept atol=1e-9
        @test loaded_model.effects ≈ original_model.effects atol=1e-9
        @test loaded_model.allele_freqs ≈ original_model.allele_freqs atol=1e-9

        # --- 5. 验证加载的模型的预测结果是否一致 ---
        new_G = rand(5, 5)
        new_geno_df = DataFrame(new_G, :auto)

        original_predictions = GenomicPrediction.predict(original_model, new_geno_df)
        loaded_predictions = GenomicPrediction.predict(loaded_model, new_geno_df)

        @test original_predictions ≈ loaded_predictions atol=1e-9

        # --- 6. 清理临时文件 ---
        rm(temp_path)
    end

    # 测试元数据保存和查看功能
    @testset "元数据管理" begin
        # --- 1. 准备并训练模型 ---
        model = GenomicPrediction.GBLUPModel(25.0)
        G = rand(5, 2); y = rand(5)
        data = GenomicPrediction.GenomicData(DataFrame(G, :auto), DataFrame(y=y))
        GenomicPrediction.fit!(model, data)

        # --- 2. 保存模型并检查元数据 ---
        temp_path = mktemp()[1]
        GenomicPrediction.save_model(model, temp_path)

        metadata = GenomicPrediction.view_model_metadata(temp_path)

        # --- 3. 验证元数据内容 ---
        @test metadata isa Dict
        @test haskey(metadata, :model_type)
        @test metadata[:model_type] == string(typeof(model))

        @test haskey(metadata, :model_parameters)
        @test metadata[:model_parameters][:lambda] == 25.0

        @test haskey(metadata, :save_timestamp)
        @test metadata[:save_timestamp] isa DateTime

        @test haskey(metadata, :julia_version)
        @test metadata[:julia_version] == string(VERSION)

        @test haskey(metadata, :package_version)
        @test isnothing(metadata[:package_version]) || metadata[:package_version] isa String

        # --- 4. 清理 ---
        rm(temp_path)
    end

end
