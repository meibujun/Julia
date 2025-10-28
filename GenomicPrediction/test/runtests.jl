# test/runtests.jl - 软件包主测试入口
# =======================================
#
# 这个文件是 `GenomicPrediction.jl` 整个测试套件的入口点。

using Test
using GenomicPrediction
using Hyperopt # Ensure Hyperopt is available for all tests

println("="^60)
println(" 正在运行 GenomicPrediction.jl 软件包测试套件...")
println(" Julia 版本: ", VERSION)
println("="^60)

@testset "GenomicPrediction.jl" begin
    println("\n[1/8] 正在测试: DataProcessing (数据处理)...")
    include("data_tests.jl")

    println("\n[2/8] 正在测试: CoreAlgorithm (核心算法)...")
    include("core_tests.jl")

    println("\n[3/8] 正在测试: KernelModels (核方法)...")
    include("kernel_tests.jl")

    println("\n[4/8] 正在测试: DeepLearning (深度学习)...")
    include("dl_tests.jl")

    println("\n[5/8] 正在测试: Evaluation (评估与解释)...")
    include("eval_tests.jl")

    println("\n[6/8] 正在测试: FAIRModeling (FAIR 建模)...")
    include("fair_tests.jl")

    println("\n[7/8] 正在测试: AutoGS (自动基因组选择)...")
    include("autogs_tests.jl")

    println("\n[8/8] 正在运行: 端到端集成测试...")
    include("integration_tests.jl")
end

println("\n" * "="^60)
println(" 所有测试已完成!")
println("="^60)
