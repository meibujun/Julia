# =============================================================================
# AnimalBreeding.jl - 示例脚本
# 展示数据模拟、关系矩阵与 BLUP 评估的协同流程
# =============================================================================

try
    using AnimalBreeding
catch
    include("../src/AnimalBreeding.jl")
    using .AnimalBreeding
end

using Statistics

"""
    run_all_examples() -> Dict

运行三个核心示例：关系矩阵计算、BLUP 评估以及结果保存。返回每个示例
的输出，便于在 REPL 中快速检查。
"""
function run_all_examples()
    results = Dict{String,Any}()
    results["relationship_matrices"] = example_relationship_matrices()
    results["gblup"] = example_gblup_pipeline()
    results["save_results"] = example_save_results()
    return results
end

"""
    example_relationship_matrices()

加载模拟数据后，计算 A、A⁻¹ 与 G 矩阵，并返回矩阵尺寸摘要。
"""
function example_relationship_matrices()
    dm, _ = simulate_complete_dataset(n_generations=3, n_per_generation=80, n_markers=250)
    matrices = compute_relationship_matrix(dm, type=:all)
    summary = Dict{Symbol,Tuple{Int,Int}}()
    for (name, mat) in matrices
        summary[name] = size(mat)
    end
    return summary
end

"""
    example_gblup_pipeline()

演示 GBLUP 流程：模拟数据、计算 G 矩阵、构建模型并运行评估。
返回育种值的均值与标准差。
"""
function example_gblup_pipeline()
    dm, _ = simulate_complete_dataset(n_generations=4, n_per_generation=120, n_markers=1000, h2=0.35)
    compute_relationship_matrix(dm, type=:genomic)
    model = define_model(traits=["trait"], fixed=["herd"], random=[("animal", :additive)])
    result = run_evaluation(model, dm, method=:GBLUP, h2=0.35)
    mean_ebv = mean(result.breeding_values.EBV)
    std_ebv = std(result.breeding_values.EBV)
    return (; mean_ebv, std_ebv, result)
end

"""
    example_save_results()

调用 `quickstart_gblup` 并将结果保存到临时文件，返回写入的文件路径。
"""
function example_save_results()
    result, dm, _ = quickstart_gblup(n_generations=2, n_per_generation=40, n_markers=200, h2=0.25)
    filepath = joinpath(mktempdir(), "quickstart_results.csv")
    save_results(result, filepath)
    return filepath
end

"""
    quick_test() -> Bool

运行轻量级回归测试以确认模拟、矩阵计算与 BLUP 求解能顺利协同。
"""
function quick_test()
    try
        dm, _ = simulate_complete_dataset(n_generations=2, n_per_generation=30, n_markers=100, h2=0.3)
        compute_relationship_matrix(dm, type=:pedigree)
        model = define_model(traits=["trait"], fixed=["herd"], random=[("animal", :additive)])
        run_evaluation(model, dm, method=:BLUP, estimate_variances=false)
        return true
    catch
        return false
    end
end
