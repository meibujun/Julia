#!/usr/bin/env julia

using LinearAlgebra
using Random
using Distributions
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearModels
const Ch1 = LinearModels.LinearModelsChapter1
const Ch2 = LinearModels.LinearModelsChapter2
const Ch3 = LinearModels.LinearModelsChapter3
const Ch4 = LinearModels.LinearModelsChapter4
const Ch5 = LinearModels.LinearModelsChapter5
const Ch6 = LinearModels.LinearModelsChapter6
const Ch7 = LinearModels.LinearModelsChapter7
const Ch8 = LinearModels.LinearModelsChapter8
const Ch9 = LinearModels.LinearModelsChapter9
const Ch10 = LinearModels.LinearModelsChapter10

# ---------------- 第1章：线性代数基础与数据生成 ----------------
Random.seed!(2025)
X = [1.0 0.5; 1.0 1.5; 1.0 2.5; 1.0 3.5]
β_true = [2.0, -0.8]
σ2 = 0.25
design = Ch1.LinearModelDesign(X, β_true, σ2)
y = Ch1.simulate_response(design)
Q, R = Ch1.gram_schmidt_orthonormal_basis(X)
P = Ch1.projection_matrix(X)
Xscaled, μ, σ = Ch1.center_and_scale(X)

println("第1章示例：")
println("  模拟响应 y = ", y)
println("  正交矩阵 Q = \n", Q)
println("  投影矩阵对称性: ", isapprox(P, P'))
println("  归一化后的设计矩阵第一列均值 = ", mean(Xscaled[:, 1]))
println()

# ---------------- 第2章：最小二乘估计与诊断 ----------------
β̂, κ = Ch2.least_squares_estimator(design, y)
ŷ, residual, rss, mse, H = Ch2.fitted_and_residual(design, β̂, y)
diagnostics = Ch2.leave_one_out_diagnostics(design, y)

println("第2章示例：")
println("  最小二乘估计 β̂ = ", β̂, ", 条件数 = ", κ)
println("  残差平方和 RSS = ", rss, ", 均方误差 MSE = ", mse)
println("  留一交叉验证MSE = ", diagnostics.cv_mse)
println()

# ---------------- 第3章：假设检验与区间估计 ----------------
C = [0.0 1.0]
d = [0.0]
f_test = Ch3.hypothesis_f_test(design, y, C, d)
t_test = Ch3.parameter_t_test(design, y, 2)
ci = Ch3.confidence_interval(design, y, 2)

println("第3章示例：")
println("  F检验统计量 = ", f_test.fstat, ", p值 = ", f_test.pvalue)
println("  t检验统计量 = ", t_test.tstat, ", p值 = ", t_test.pvalue)
println("  95%置信区间 = ", ci)
println()

# ---------------- 第4章：混合模型与REML ----------------
Z = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
G = 0.5 .* Matrix(I, 2, 2)
R = 0.25 .* Matrix(I, 4, 4)
mm_struct = Ch4.MixedModelStructure(X, Z, Matrix(G), Matrix(R))
y_mixed, u_true, e_true = Ch4.simulate_mixed_model(mm_struct, β_true)
β̂_mixed, û, V = Ch4.solve_mixed_model_equations(mm_struct, y_mixed)
û_pred, Cuu = Ch4.predict_random_effects(mm_struct, y_mixed)
loglik = Ch4.reml_loglikelihood(mm_struct, y_mixed)

println("第4章示例：")
println("  固定效应估计 = ", β̂_mixed)
println("  随机效应预测 = ", û_pred)
println("  REML 对数似然 = ", loglik)
println()

# ---------------- 第5章：残差诊断与影响分析 ----------------
res_diag = Ch5.residual_diagnostics(y, X, β̂)
influence = Ch5.influence_measures(y, X, β̂)
DW = Ch5.durbin_watson_statistic(res_diag.residuals)
partial = Ch5.partial_residuals(y, X, β̂)

println("第5章示例：")
println("  Durbin-Watson统计量 = ", DW)
println("  最大帽子值 = ", maximum(res_diag.hatvalues))
println("  最大库克距离 = ", maximum(influence.cooks_distance))
println("  第一列部分残差均值 = ", mean(partial[:, 1]))
println()

# ---------------- 第6章：模型选择与信息准则 ----------------
cols = collect(1:size(X, 2))
stepwise = Ch6.forward_stepwise_selection(X, y, cols; criterion = :AIC)
criteria = [Ch6.model_selection_criteria(y, X[:, 1:i], X[:, 1:i] \ y).AIC for i in 1:2]
weights = Ch6.information_weights(criteria)

println("第6章示例：")
println("  逐步选择路径 = ", stepwise.path)
println("  Akaike权重 = ", weights)
println()

# ---------------- 第7章：预测与重采样评估 ----------------
XtX_inv = inv(X' * X)
pred_int = Ch7.prediction_interval([1.0, 2.0], β̂, res_diag.σ̂2, XtX_inv)
cv_rmse = Ch7.cross_validated_rmse(X, y; k = 2)
press = Ch7.predictive_residuals(X, y)
boot = Ch7.bootstrap_prediction_intervals(X, y, [1.0, 2.0]; B = 200)

println("第7章示例：")
println("  预测区间 = ", pred_int)
println("  2折交叉验证RMSE = ", cv_rmse)
println("  PRESS残差范数 = ", norm(press))
println("  Bootstrap预测区间 = ", (boot.lower, boot.upper))
println()

# ---------------- 第8章：方差分量与REML迭代 ----------------
reml = Ch8.em_reml_variance_components(X, Z, y_mixed; σe2_init = 0.2, σu2_init = 0.4, tol = 1e-4, maxiter = 50)
logreml = Ch8.log_reml_likelihood(X, Z, y_mixed, reml.σu2, reml.σe2)

println("第8章示例：")
println("  EM-REML估计 = ", (reml.σu2, reml.σe2), "，是否收敛 = ", reml.converged)
println("  REML对数似然 = ", logreml)
println()

# ---------------- 第9章：贝叶斯线性模型 ----------------
posterior = Ch9.conjugate_posterior(X, y, zeros(2), Matrix(I, 2, 2), 2.0, 1.0)
draws = Ch9.gibbs_sampler_linear_model(X, y; n_samples = 300, burnin = 50, α0 = 2.0, β0 = 1.0)
pred = Ch9.posterior_predictive([1.0 1.0; 1.0 2.0], draws)

println("第9章示例：")
println("  后验均值 = ", posterior.μn)
println("  σ²样本均值 = ", mean(draws[:σ2_samples]))
println("  后验预测区间(第一行) = ", (pred.lower[1], pred.upper[1]))
println()

# ---------------- 第10章：计算策略与矩阵更新 ----------------
A = X' * X + Matrix(I, 2, 2)
b = X' * y
cg = Ch10.conjugate_gradient_solver(A, b)
S = [4.0 2.0; 2.0 5.0]
Ch10.sweep_operator!(S, 1)
L = cholesky(Symmetric(X' * X))
updated = Ch10.block_cholesky_update!(L, X)
woodbury = Ch10.sparse_woodbury_inverse(Matrix(I, 2, 2), X', Matrix(I, size(X, 1), size(X, 1)), X)

println("第10章示例：")
println("  共轭梯度是否收敛 = ", cg.converged)
println("  扫掠算子后的矩阵 = \n", S)
println("  更新后Cholesky对角线 = ", diag(updated.U))
println("  Woodbury逆矩阵迹 = ", tr(woodbury))

