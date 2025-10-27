# test/runtests.jl - 软件包主测试入口
# =======================================
#
# 这个文件是 `GenomicPrediction.jl` 整个测试套件的入口点。
# 当执行 `Pkg.test("GenomicPrediction")` 时，Julia 的包管理器会自动运行此文件。
#
# 主要职责:
# 1. 引入 `Test` 标准库和待测试的 `GenomicPrediction` 主模块。
# 2. 使用 `include` 语句，将各个子模块的测试文件包含进来，确保所有测试都被执行。
# 3. 可以定义一些全局的测试辅助函数或数据，供所有测试文件使用。

using Test
using GenomicPrediction # 引入我们的主模块
using Hyperopt         # 确保 Hyperopt 在所有测试中都可用

# --- 测试欢迎语 ---
println("="^60)
println(" 正在运行 GenomicPrediction.jl 软件包测试套件...")
println(" Julia 版本: ", VERSION)
println("="^60)

# --- 依次执行各个模块的测试 ---
# 每个文件内部都使用 @testset 进行了组织。
# 在这里直接 include 文件即可。

println("\n[1/6] 正在测试: CoreAlgorithm (核心算法)...")
include("core_tests.jl")

println("[2/6] 正在测试: DataProcessing (数据处理)...")
include("data_tests.jl")

println("[3/6] 正在测试: DeepLearning (深度学习)...")
include("dl_tests.jl")

println("[4/6] 正在测试: Evaluation (评估与解释)...")
include("eval_tests.jl")

println("[5/6] 正在测试: FAIRModeling (FAIR 建模)...")
include("fair_tests.jl")

println("[6/6] 正在测试: AutoGS (自动基因组选择)...")
include("autogs_tests.jl")

println("\n[7/8] 正在测试: KernelModels (核方法)...")
include("kernel_tests.jl")

println("\n[8/8] 正在运行: 端到端集成测试...")
include("integration_tests.jl")


println("\n" * "="^60)
println(" 所有测试已完成!")
println("="^60)
