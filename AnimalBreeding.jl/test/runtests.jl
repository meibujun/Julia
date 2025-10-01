# AnimalBreeding.jl - 主测试套件入口
# 作者：AnimalBreeding.jl 开发团队
# 版本：1.0.1

using Test

@testset "AnimalBreeding.jl - 全面测试" begin
    println("\n" * "="^70)
    println("  启动 AnimalBreeding.jl 完整测试套件")
    println("="^70)

    # --- 1. 加载核心模块 ---
    # 这种结构确保了模块在测试环境中被正确加载
    # `..` 表示从当前文件目录（test/）向上到项目根目录
    include("../src/AnimalBreeding.jl")
    using .AnimalBreeding
    println("\n✓ 核心模块 (AnimalBreeding.jl) 加载成功")

    # --- 2. 包含并运行各个模块的测试 ---

    @testset "核心数据与工具测试" begin
        include("test_core.jl")
    end

    # --- 3. 加载并测试遗传评估模块 ---
    include("../src/GeneticEvaluation.jl")
    using .GeneticEvaluation
    println("\n✓ 遗传评估模块 (GeneticEvaluation.jl) 加载成功")

    @testset "遗传评估测试" begin
        include("test_evaluation.jl")
    end

    # --- 4. 加载并测试贝叶斯分析模块 ---
    include("../src/BayesianAnalysis.jl")
    using .BayesianAnalysis
    println("\n✓ 贝叶斯分析模块 (BayesianAnalysis.jl) 加载成功")

    @testset "贝叶斯分析测试" begin
        include("test_bayesian.jl")
    end

    # --- 5. 加载并测试机器学习模块 ---
    include("../src/MachineLearning.jl")
    using .MachineLearning
    println("\n✓ 机器学习模块 (MachineLearning.jl) 加载成功")

    @testset "机器学习测试" begin
        include("test_ml.jl")
    end

    println("\n" * "="^70)
    println("  所有测试模块已执行完毕")
    println("="^70)
end