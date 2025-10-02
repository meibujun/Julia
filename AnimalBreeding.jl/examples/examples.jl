# ============================================================================
# AnimalBreeding.jl - 完整使用示例和测试
# 展示软件系统的各种功能和典型应用场景
# ============================================================================

# 加载主模块
# This assumes the file is run from a context where AnimalBreeding is available.
# In a package context, this would be handled by the test runner.
try
    using AnimalBreeding
catch
    # If running as a standalone script, include the main module file.
    include("../src/AnimalBreeding.jl")
    using .AnimalBreeding
end

using Printf
using Statistics
using Random

"""
    run_all_examples()

运行此文件中定义的所有示例，全面展示软件功能。
"""
function run_all_examples()
    println("\n\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^20 * "开始 AnimalBreeding.jl 完整功能演示" * " "^20 * "║")
    println("╚" * "═"^78 * "╝")

    examples = [
        ("基础BLUP分析", example_basic_blup),
        ("基因组选择", example_genomic_selection),
        ("贝叶斯分析", example_bayesian_analysis),
        ("机器学习方法", example_machine_learning),
        ("测定日模型", example_test_day_model),
        ("上位性分析", example_epistasis_analysis),
        ("G×E互作", example_gxe_interaction),
        ("育种规划", example_breeding_plan)
    ]

    results = Dict()

    for (i, (name, func)) in enumerate(examples)
        println("\n\n▶ 运行示例 $i/$(length(examples)): $name")
        try
            results[name] = func()
            println("\n✓ 示例 '$name' 完成")
        catch e
            println("\n✗ 示例 '$name' 出错: $e")
            println(stacktrace(catch_backtrace()))
        end
        GC.gc() # 回收内存
    end

    println("\n\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^28 * "所有示例运行完成" * " "^28 * "║")
    println("╚" * "═"^78 * "╝")

    return results
end

# ============================================================================
# 示例实现
# ============================================================================
function example_basic_blup()
    dm, true_params = simulate_complete_dataset(n_generations=4, n_per_generation=150, n_markers=100, h2=0.35, add_omics=false)
    compute_relationship_matrix(dm, type=:pedigree)
    model = define_model(traits=["trait"], fixed=["herd"], random=[("animal", :additive)])
    result = run_evaluation(model, dm, method=:BLUP, estimate_variances=true)
    accuracy = cor(result.breeding_values.EBV, true_params["phenotype"]["true_breeding_values"])
    @info @sprintf("  预测准确性 (相关系数): %.4f", accuracy)
    return result
end

function example_genomic_selection()
    dm, true_params = simulate_complete_dataset(n_generations=5, n_per_generation=200, n_markers=2000, h2=0.3)
    dm.A_matrix, dm.pedigree, dm.animal_map = compute_A_matrix(dm.pedigree)
    dm.G_matrix = compute_G_matrix(dm.genotypes)
    model = define_model(traits=["trait"], fixed=["herd"], random=[("animal", :additive)])
    result_gblup = run_evaluation(model, dm, method=:GBLUP, h2=0.3)
    accuracy = cor(result_gblup.breeding_values.EBV, true_params["phenotype"]["true_breeding_values"])
    @info @sprintf("  GBLUP 预测准确性: %.4f", accuracy)
    return result_gblup
end

function example_bayesian_analysis()
    dm, true_params = simulate_complete_dataset(n_generations=3, n_per_generation=100, n_markers=500, n_qtl=10, h2=0.4)
    y = Vector{Float64}(dm.phenotypes.trait)
    X = ones(Float64, length(y), 1)
    Z = Matrix{Float64}(dm.genotypes[:, 2:end])
    result = run_bayesian_evaluation(y, X, Z, method=:BayesB, n_iter=2000, burn_in=500, thin=5, π=0.95, save_samples=true)
    accuracy = cor(result.breeding_values.GEBV, true_params["phenotype"]["true_breeding_values"])
    @info @sprintf("  BayesB 预测准确性: %.4f", accuracy)
    return result
end

function example_machine_learning()
    dm, _ = simulate_complete_dataset(n_generations=4, n_per_generation=100, n_markers=1000, h2=0.5)
    X = Matrix{Float64}(dm.genotypes[:, 2:end])
    y = Vector{Float64}(dm.phenotypes.trait)
    cv_results = cross_validate(X, y, :RandomForest, n_folds=3, hyperparams=Dict("n_trees"=>50))
    @info "  随机森林交叉验证平均相关性: $(mean(cv_results["fold_scores"]))"
    return cv_results
end

function example_test_day_model()
    n_cows = 30; test_days = 5:30:305
    milk_data = DataFrame(animal=Int[], dim=Int[], milk=Float64[])
    for cow in 1:n_cows
        a=rand(30:45); b=rand(0.05:0.01:0.08); c=rand(0.002:0.0005:0.004)
        for day in test_days
            milk = a * (day^b) * exp(-c * day) + randn() * 2
            push!(milk_data, (cow, day, max(0, milk)))
        end
    end
    td_model = fit_random_regression_model(milk_data, animal_col=:animal, time_col=:dim, trait_col=:milk, polynomial_degree=3)
    @info "  测定日模型拟合完成。随机效应方差(tr(G)): $(tr(td_model.variance_components["genetic_covariance"]))"
    return td_model
end

function example_epistasis_analysis()
    dm, _ = simulate_complete_dataset(n_markers=500, h2=0.3)
    phenotypes = Vector{Float64}(dm.phenotypes.trait)
    # Add epistasis effect for demonstration
    phenotypes += 0.5 .* dm.genotypes[:, 10] .* dm.genotypes[:, 20]
    epi_results = fit_epistasis_model(dm.genotypes, phenotypes, relationship_matrix=compute_G_matrix(dm.genotypes))
    @info "  上位性模型拟合完成。加性方差: $(epi_results["variance_components"]["additive"]) 上位性方差: $(epi_results["variance_components"]["epistasis"])"
    return epi_results
end

function example_gxe_interaction()
    dm, _ = simulate_complete_dataset(n_animals=100, n_markers=500, h2=0.4)
    multi_env_data = DataFrame(animal=Int[], environment=Int[], trait=Float64[])
    true_slopes = randn(100) * 0.5
    for env in 1:3
        for i in 1:100
            base = dm.phenotypes.trait[i]; gxe = true_slopes[i] * (env-2)
            push!(multi_env_data, (i, env, base + gxe + randn()*0.2))
        end
    end
    rnm = fit_reaction_norm_model(multi_env_data, compute_G_matrix(dm.genotypes), animal_col=:animal, trait_col=:trait, env_col=:environment)
    @info "  GxE模型拟合完成。斜率方差: $(rnm.variance_components["slope_var"])"
    return rnm
end

function example_breeding_plan()
    dm, _ = simulate_complete_dataset(n_animals=100, n_markers=1000, h2=0.5)
    ebvs = dm.phenotypes.TBV # Use true BV for this example
    G = compute_G_matrix(dm.genotypes)
    ocs_results = optimal_contribution_selection(ebvs, G, 10)
    @info "  OCS完成, 选留 $(length(ocs_results["selected_indices"])) 个个体。"
    return ocs_results
end

"""
    quick_test()

运行一个快速测试来验证安装和基本功能。
"""
function quick_test()
    @info "运行快速安装验证测试..."
    try
        dm, _ = simulate_complete_dataset(n_generations=2, n_per_generation=20, n_markers=50, h2=0.3, add_omics=false)
        model = define_model(traits=["trait"], fixed=["herd"], random=[("animal", :additive)])
        dm.A_matrix, dm.pedigree, dm.animal_map = compute_A_matrix(dm.pedigree)
        run_evaluation(model, dm, method=:BLUP, h2=0.3)
        @info "✓ 快速测试通过！"
        return true
    catch e
        @error "✗ 快速测试失败: $e"
        return false
    end
end