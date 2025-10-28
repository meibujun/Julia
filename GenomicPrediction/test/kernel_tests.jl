# test/kernel_tests.jl
# ==========================================================
# Unit tests for KernelModels.jl module.
#
# This file has been updated to correctly test the refactored ssGBLUPModel,
# including the creation of mock pedigree data.
# ==========================================================

using Test
using .GenomicPrediction
using DataFrames
using CSV

@testset "KernelModels.jl" begin

    @testset "ssGBLUPModel" begin
        # 1. 创建模拟数据
        # 基因型数据
        geno_df = DataFrame(ID = 1:5, m1 = [0,1,2,0,1], m2 = [2,1,0,2,1], m3 = [0,0,1,1,2])
        # 表型数据 (仅针对部分个体)
        pheno_df = DataFrame(ID = [3, 4, 5], y = [3.2, 1.2, 2.3])
        # 系谱数据 (包含有基因型和无基因型的个体)
        ped_df = DataFrame(
            ID = [1, 2, 3, 4, 5, 6],
            Sire = [0, 0, 1, 1, 2, 4],
            Dam = [0, 0, 2, 2, 3, 5]
        )

        # 将模拟数据写入临时文件
        geno_path = "temp_geno.csv"
        pheno_path = "temp_pheno.csv"
        ped_path = "temp_ped.csv"
        CSV.write(geno_path, geno_df)
        CSV.write(pheno_path, pheno_df)
        CSV.write(ped_path, ped_df)

        # 2. 加载数据
        # ssGBLUP 使用所有系谱信息，但只有部分个体有基因型和表型
        data = GenomicPrediction.load_csv(geno_path, pheno_path; ped_path=ped_path)

        # 3. 初始化和训练模型
        model = ssGBLUPModel(10.0)

        # fit! 函数现在从 data.pedigree 获取系谱
        @test_nowarn GenomicPrediction.fit!(model, data)

        @test model.intercept != 0.0
        @test !isempty(model.breeding_values)
        @test length(model.breeding_values) == nrow(ped_df) # 应为所有个体计算育种值
        @test !isempty(model.snp_effects)

        # 4. 测试预测
        # 预测已知个体的育种值
        known_ids = [3, 5, 6]
        predictions = GenomicPrediction.predict(model, known_ids)
        @test length(predictions) == length(known_ids)
        @test all(!isnan, predictions)

        # 预测全新个体 (基于 SNP 效应)
        new_geno_df = DataFrame(ID = [101, 102], m1=[1,0], m2=[2,1], m3=[0,2])
        snp_predictions = GenomicPrediction.predict(model, new_geno_df)
        @test length(snp_predictions) == 2
        @test all(!isnan, snp_predictions)

        # 5. 清理临时文件
        rm(geno_path); rm(pheno_path); rm(ped_path)
    end

end
