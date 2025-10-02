# ============================================================================
# 测试: 核心数据层 (data & relationships)
# ============================================================================

using Test
using AnimalBreeding
using DataFrames
using LinearAlgebra

@testset "核心数据与关系矩阵测试" begin

    # --- 1. 准备测试数据 ---
    ped_df = DataFrame(
        animal = [1, 2, 3, 4, 5],
        sire   = [0, 0, 1, 1, 3],
        dam    = [0, 0, 2, 2, 4]
    )

    # 动物3, 4, 5有基因型
    gen_df = DataFrame(
        animal_id = [3, 4, 5],
        snp1 = [0, 1, 2],
        snp2 = [1, 1, 0],
        snp3 = [2, 0, 1]
    )

    dm = DataManager()
    dm.pedigree = ped_df
    dm.genotypes = gen_df

    @testset "A 矩阵计算" begin
        dm.A_matrix, sorted_ped, id_map = compute_A_matrix(dm.pedigree)
        dm.pedigree = sorted_ped
        dm.animal_map = id_map
        dm.A_inv_matrix, sorted_ped_inv, id_map_inv = compute_A_inv_matrix(dm.pedigree)
        @test sorted_ped_inv == sorted_ped
        @test id_map_inv == id_map

        @test size(dm.A_matrix) == (5, 5)
        @test issymmetric(dm.A_matrix)
        @test dm.A_matrix[5, 5] > 1.0 # 检查近交

        @test size(dm.A_inv_matrix) == (5, 5)
        # 验证 A * A⁻¹ ≈ I
        @test Matrix(dm.A_matrix * Matrix(dm.A_inv_matrix)) ≈ I(5) atol=1e-8
    end

    @testset "G 矩阵计算" begin
        dm.G_matrix = compute_G_matrix(dm.genotypes)

        @test size(dm.G_matrix) == (3, 3)
        @test issymmetric(dm.G_matrix)
        @test isapprox(mean(diag(dm.G_matrix)), 1.0, atol=0.2)
    end

    @testset "H⁻¹ 矩阵计算" begin
        # 确保G矩阵的动物顺序与A矩阵中的顺序一致
        # `compute_H_matrix_inv` 内部会处理对齐
        dm.H_inv_matrix = compute_H_matrix_inv(dm)

        @test size(dm.H_inv_matrix) == (5, 5)
        @test issymmetric(dm.H_inv_matrix)

        # H⁻¹ 应该与 A⁻¹ 不同，因为加入了基因组信息
        @test !isapprox(Matrix(dm.H_inv_matrix), Matrix(dm.A_inv_matrix))

        # 检查基因分型个体的对角元是否已更新
        genotyped_idx = findall(x -> x in [3,4,5], dm.pedigree.animal)
        @test dm.H_inv_matrix[genotyped_idx[1], genotyped_idx[1]] != dm.A_inv_matrix[genotyped_idx[1], genotyped_idx[1]]
    end

    println("\n✓ 核心数据层测试通过。")
end