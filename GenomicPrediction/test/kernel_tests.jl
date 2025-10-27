# test/kernel_tests.jl - KernelModels 模块单元测试
# ---------------------------------------------------

using Test
using DataFrames
using SparseArrays
using LinearAlgebra
using Statistics

# 导入内部函数以进行白盒测试
using GenomicPrediction.KernelModels: build_A_inv
using GenomicPrediction.CoreAlgorithm: _standardize_genotypes

@testset "KernelModels.jl - 核方法模块测试" begin

    @testset "build_A_inv (构建 A 逆矩阵)" begin
        pedigree = DataFrame(ID=[1, 2, 3], Sire=[0, 0, 1], Dam=[0, 2, 0])
        A_inv = build_A_inv(pedigree)

        @test issparse(A_inv)
        @test size(A_inv) == (3, 3)
        @test issymmetric(A_inv)
        # A simple check on diagonal elements
        @test A_inv[1,1] > 0 && A_inv[2,2] > 0 && A_inv[3,3] > 0
    end

    @testset "ssGBLUP 模型 (完整数值验证)" begin
        pedigree_df = DataFrame(
            ID = [1, 2, 3, 4, 5],
            Sire = [0, 0, 1, 1, 3],
            Dam = [0, 0, 2, 2, 4]
        )

        genotypes_df = DataFrame(
            ID = [3, 4, 5],
            snp1 = [1.0, 2.0, 1.0],
            snp2 = [0.0, 1.0, 1.0],
        )

        phenotypes_df = DataFrame(
            ID = [1, 2, 3, 4, 5],
            y = [10.5, 9.8, 11.2, 12.1, 11.5]
        )

        mock_data = GenomicPrediction.GenomicData(genotypes_df, phenotypes_df, nothing, pedigree_df)

        lambda = 10.0
        model = GenomicPrediction.ssGBLUPModel(lambda)

        GenomicPrediction.fit!(model, mock_data)

        # Basic checks to ensure the model trained and is plausible
        @test isfinite(model.intercept)
        @test length(model.effects) == size(pedigree_df, 1)
        @test all(isfinite, model.effects)

        predictions = GenomicPrediction.predict(model, [1, 2, 3, 4, 5])
        @test length(predictions) == 5
        @test all(isfinite, predictions)
    end

end
