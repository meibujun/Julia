# 测试贝叶斯分析模块 (BayesianAnalysis.jl)

using Test
using DataFrames
using Random
using Statistics
using ..AnimalBreeding
using ..BayesianAnalysis

@testset "贝叶斯分析模块: BayesianAnalysis.jl" begin

    # 生成一个小规模的、可预测的数据集
    Random.seed!(123)
    n, p = 50, 100 # 50个个体, 100个标记
    X = ones(n, 1)
    Z = randn(n, p)

    # 模拟稀疏效应
    true_effects = zeros(p)
    qtl_indices = sample(1:p, 10, replace=false)
    true_effects[qtl_indices] = randn(10) * 2.0

    y = X * [10.0] + Z * true_effects + randn(n)

    # 使用较短的MCMC链以加速测试
    mcmc_params = (n_iter=200, burn_in=50, thin=2, verbose=false)

    @testset "BayesA" begin
        result = run_bayesian_evaluation(y, X, Z; method=:BayesA, mcmc_params...)
        @test result isa BayesianResult
        @test length(result.marker_effects) == p
        @test all(result.marker_pip .== 1.0) # BayesA includes all markers
        @test !any(isnan, result.breeding_values)
    end

    @testset "BayesB" begin
        result = run_bayesian_evaluation(y, X, Z; method=:BayesB, π=0.8, mcmc_params...)
        @test result isa BayesianResult
        @test length(result.marker_pip) == p
        @test 0.0 <= mean(result.marker_pip) <= 1.0
        # 应该能识别出一些有效应的标记
        @test sum(result.marker_pip) > 0
        println("  BayesB 平均PIP: ", round(mean(result.marker_pip), digits=3))
    end

    @testset "BayesC" begin
        result = run_bayesian_evaluation(y, X, Z; method=:BayesC, π=0.8, mcmc_params...)
        @test result isa BayesianResult
        @test haskey(result.variance_components, "marker") # 检查共同方差
        @test result.variance_components["marker"] > 0
    end

    @testset "Bayesian LASSO" begin
        result = run_bayesian_evaluation(y, X, Z; method=:BayesianLASSO, mcmc_params...)
        @test result isa BayesianResult
        @test haskey(result.hyperparameters, "lambda")
        # 检查收缩效应
        @test sum(abs.(result.marker_effects) .< 0.01) > 0
    end

end