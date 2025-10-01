# --- AnimalBreeding.jl 测试套件 ---
# 这是项目的主测试文件，用于验证所有模块功能的正确性。
# 运行此文件将执行一系列单元测试和集成测试。

# --- 导入所需模块 ---
using AnimalBreeding # 导入我们自己的模块
using Test           # 导入Julia内置的测试框架
using DataFrames     # 用于数据操作
using SparseArrays   # 用于稀疏矩阵操作
using LinearAlgebra  # 用于线性代数计算

# 定义一个常量，指向包含示例数据的文件夹
const DATADIR = joinpath(@__DIR__, "..", "data")

# --- 主测试集 ---
# `@testset` 将相关的测试组织在一起。
@testset "AnimalBreeding.jl 全方位测试" begin

    # --- 1. 数据管理模块测试 ---
    @testset "数据管理 (Data Management)" begin
        # 定义测试数据文件路径
        ped_path = joinpath(DATADIR, "pedigree.csv")
        pheno_path = joinpath(DATADIR, "phenotypes.csv")
        geno_path = joinpath(DATADIR, "genotypes.csv")

        # 测试文件是否存在
        @test isfile(ped_path)
        @test isfile(pheno_path)
        @test isfile(geno_path)

        # 测试 `load_pedigree` 函数
        ped = load_pedigree(ped_path)
        @test ped isa DataFrame
        @test names(ped) == ["animal", "sire", "dam"]
        @test nrow(ped) == 5

        # 测试 `load_phenotypes` 函数
        pheno = load_phenotypes(pheno_path, trait_cols=["milk"], fixed_cols=["herd"])
        @test pheno isa DataFrame
        @test "milk" in metadat(pheno, "trait_cols") # 检查元数据是否正确设置
        @test nrow(pheno) == 3

        # 测试 `load_genotypes` 函数
        geno = load_genotypes(geno_path, animal_id_col="animal")
        @test geno isa DataFrame
        @test names(geno)[1] == "animal"
        @test ncol(geno) == 5 # 1 animal_id + 4 snps
        @test nrow(geno) == 2

        # --- 测试数据验证功能 ---
        dm = DataManager()
        dm.pedigree = ped
        dm.phenotypes = pheno
        dm.genotypes = geno

        # 对于有效数据，`validate_data` 应返回 true
        @test validate_data(dm) == true

        # 测试验证失败的情况：表型数据中的个体在谱系中不存在
        bad_pheno = deepcopy(pheno)
        push!(bad_pheno, (animal=99, herd="C", milk=100)) # 添加一个无效个体
        dm_bad = DataManager()
        dm_bad.pedigree = ped
        dm_bad.phenotypes = bad_pheno
        # `@test_throws` 检查函数是否如预期那样抛出错误
        @test_throws ErrorException validate_data(dm_bad)
    end

    # --- 2. 关系矩阵计算测试 ---
    @testset "关系矩阵 (Relationship Matrices)" begin
        dm = DataManager()
        dm.pedigree = load_pedigree(joinpath(DATADIR, "pedigree.csv"))
        dm.genotypes = load_genotypes(joinpath(DATADIR, "genotypes.csv"), animal_id_col="animal")

        # --- 测试 A 矩阵 ---
        compute_relationship_matrix(dm, type=:A) # 使用别名 :A
        A = dm.A
        @test A isa SparseMatrixCSC
        @test size(A) == (5, 5)
        # 理论值 (通过手工计算或与其他软件对比得出)
        # A = [1.0  0.0   0.5   0.5   0.5;
        #      0.0  1.0   0.0   0.5   0.25;
        #      0.5  0.0   1.0   0.25  0.625;
        #      0.5  0.5   0.25  1.0   0.625;
        #      0.5  0.25  0.625 0.625 1.125]
        # 检查一些关键的亲缘关系和近交系数
        @test round(A[3, 1], digits=3) == 0.5   # 个体3与父本1的关系
        @test round(A[4, 2], digits=3) == 0.5   # 个体4与母本2的关系
        @test round(A[5, 1], digits=3) == 0.5   # 个体5与祖父1的关系
        @test round(A[5, 5], digits=3) == 1.125 # 个体5的对角线元素，反映了近交

        # --- 测试 G 矩阵 ---
        compute_relationship_matrix(dm, type=:G) # 使用别名 :G
        G = dm.G
        @test G isa Matrix
        @test size(G) == (2, 2)
        # 理论值 (根据VanRaden方法手工计算)
        # M = [1 2 0 1; 2 1 1 0]
        # p = [0.75, 0.75, 0.25, 0.25]
        # Z = [-0.5 0.5 -0.5 0.5; 0.5 -0.5 0.5 -0.5]
        # Denom = 1.5
        # G = ZZ' / 1.5 = [0.667 -0.667; -0.667 0.667]
        @test isapprox(G[1, 1], 2/3, atol=1e-3)
        @test isapprox(G[1, 2], -2/3, atol=1e-3)
    end

    # --- 3. BLUP 评估流程集成测试 ---
    @testset "BLUP 评估 (Integration Test)" begin
        dm = DataManager()
        dm.pedigree = load_pedigree(joinpath(DATADIR, "pedigree.csv"))
        dm.phenotypes = load_phenotypes(joinpath(DATADIR, "phenotypes.csv"), trait_cols=["milk"], fixed_cols=["herd"])

        # 这是一个集成测试，它将前面几个模块的功能串联起来
        compute_relationship_matrix(dm, type=:pedigree)

        model = define_model(
            traits = ["milk"],
            fixed = ["herd"],
            random = [("animal", :additive)]
        )

        # 运行评估
        result = run_evaluation(model, dm, h2=0.5)

        # 验证结果的类型和基本结构
        @test result isa EvaluationResult
        @test length(result.fixed_effects) == 2 # 截距 + herdB
        @test length(result.random_effects) == 5 # 谱系中所有5个动物的育种值

        # 验证计算结果是否有效（不是NaN或Inf）
        @test !any(isnan, result.random_effects)
        @test !any(isinf, result.random_effects)

        # 测试结果保存功能
        save_path = joinpath(@__DIR__, "test_results.csv")
        save_results(result, save_path)
        @test isfile(save_path)

        # 清理测试生成的文件
        rm(save_path)
    end

end