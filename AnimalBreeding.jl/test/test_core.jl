# 测试核心数据结构和功能 (AnimalBreeding.jl)

using Test
using DataFrames
using CSV
using ..AnimalBreeding  # 假设从主runtests.jl中调用

@testset "核心模块: AnimalBreeding.jl" begin

    # 使用小规模、可预测的数据进行精确验证
    ped_df = DataFrame(animal=[1, 2, 3, 4, 5], sire=[0, 0, 1, 1, 3], dam=[0, 0, 2, 2, 4])
    gen_df = DataFrame(animal=[3, 4, 5], snp1=[0,1,2], snp2=[1,1,0])
    phen_df = DataFrame(animal=[3, 4, 5], milk=[102.0, 105.0, 110.0], herd=['A', 'B', 'A'])

    @testset "Pedigree 结构与排序" begin
        ped = Pedigree(ped_df)
        @test ped.n_animals == 5
        # 验证排序是否正确（父母在子代前）
        @test ped.data.animal == [1, 2, 3, 4, 5]
        @test ped.generation == [1, 1, 2, 2, 3]
    end

    @testset "数据加载器" begin
        # 模拟从文件加载
        ped = load_pedigree(CSV.File(IOBuffer(sprint(CSV.write, ped_df))))
        @test ped isa Pedigree

        gen = load_genotypes(CSV.File(IOBuffer(sprint(CSV.write, gen_df))))
        @test gen isa Genotypes
        @test gen.n_animals == 3

        phen = load_phenotypes(CSV.File(IOBuffer(sprint(CSV.write, phen_df))), trait_cols=["milk"], fixed_cols=["herd"])
        @test phen isa Phenotypes
        @test phen.n_traits == 1
    end

    @testset "A矩阵计算 (已修复)" begin
        ped = Pedigree(ped_df)
        compute_A_matrix!(ped)
        A = Matrix(ped.A_matrix)

        # 理论值
        # F_5 = 0.5 * A[3,4] = 0.5 * (0.5*A[1,2] + 0.5*A[1,4] + 0.5*A[2,2] + 0.5*A[2,4])
        # A[1,2]=0, A[1,4]=0.5, A[2,2]=1, A[2,4]=0.5
        # F_5 = 0.5 * (0.5*0 + 0.5*0.5 + 0.5*1 + 0.5*0.5) = 0.5 * (0.25 + 0.5 + 0.25) = 0.5
        # A[3,4] = 0.5, F_5=0.25 - Let's re-verify logic.
        # A[3,4] = 0.5(A[1,4]+A[2,4]) = 0.5(0.5+0.5)=0.5
        # F_5 = 0.5 * A[3,4] = 0.25. A[5,5] = 1.25.

        @test A[5, 5] ≈ 1.25
        @test A[3, 4] ≈ 0.5
        @test A[1, 5] ≈ 0.5 * (A[1,3] + A[1,4]) # 0.5 * (0.5 + 0.5) = 0.5
        @test A[1, 5] ≈ 0.5
    end

end