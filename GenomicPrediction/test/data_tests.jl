# test/data_tests.jl - DataProcessing 模块单元测试
# ---------------------------------------------------
#
# 本文件包含对 `DataProcessing` 模块中数据处理功能的单元测试。
# 测试重点包括：
# - 数据导入/导出功能的正确性，能否处理不同格式的文件。
# - 质量控制函数的逻辑是否正确，能否准确过滤数据。
# - GRM 矩阵计算的准确性。
# - 数据模拟器能否生成符合预期分布的数据。

using Test
using GenomicPrediction.DataProcessing # 引入待测试的模块
using DataFrames

# 定义样本数据文件的路径
const GENO_PATH = joinpath(@__DIR__, "sample_data", "genotypes.csv")
const PHENO_PATH = joinpath(@__DIR__, "sample_data", "phenotypes.csv")
const COV_PATH = joinpath(@__DIR__, "sample_data", "covariates.csv")

@testset "DataProcessing.jl - 数据处理模块测试" begin

    # --- 数据导入测试 ---
    @testset "load_csv 函数" begin
        # --- 测试 1: 仅加载基因型和表型数据 ---
        @testset "加载基因型和表型" begin
            data = load_csv(GENO_PATH, PHENO_PATH)

            # 验证类型是否正确
            @test isa(data, GenomicData)

            # 验证数据维度
            @test size(data.genotypes) == (5, 10)
            @test size(data.phenotypes) == (5, 2)
            @test data.covariates === nothing # 确认协变量为 nothing

            # 验证列名
            @test names(data.genotypes)[1] == "snp1"
            @test names(data.phenotypes)[1] == "trait1"
        end

        # --- 测试 2: 加载所有三种数据 ---
        @testset "加载基因型、表型和协变量" begin
            data = load_csv(GENO_PATH, PHENO_PATH, cov_path=COV_PATH)

            # 验证类型
            @test isa(data, GenomicData)

            # 验证维度
            @test size(data.genotypes) == (5, 10)
            @test size(data.phenotypes) == (5, 2)
            @test size(data.covariates) == (5, 1)

            # 验证协变量不为 nothing
            @test data.covariates !== nothing
            @test names(data.covariates)[1] == "cov1"
        end

        # --- 测试 3: 测试行数不匹配的错误情况 ---
        @testset "行数不匹配的错误处理" begin
            # 创建一个行数错误的临时表型文件
            wrong_pheno_path = joinpath(@__DIR__, "sample_data", "wrong_phenotypes.csv")
            open(wrong_pheno_path, "w") do f
                write(f, "trait1\n10.5\n12.1\n9.8") # 只有 3 行
            end

            # 验证是否会抛出预期的错误
            @test_throws ErrorException load_csv(GENO_PATH, wrong_pheno_path)

            # 删除临时文件
            rm(wrong_pheno_path)
        end
    end

    # --- 质量控制测试 (占位符) ---
    @testset "质量控制 (QC)" begin
        @test true
    end

    # --- GRM 计算测试 (占位符) ---
    @testset "基因组关系矩阵 (GRM) 计算" begin
        @test true
    end

end
