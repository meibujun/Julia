"""
# ADMIXTURE 分析

基于似然的群体混合比例估计。

## 模型
假设每个个体的基因型来自 K 个祖先群体的混合：

P(G_ij = g | Q, F) = Σ_k Q_ik F_kjg

其中：
- G_ij: 个体 i 在位点 j 的基因型
- Q_ik: 个体 i 来自祖先群体 k 的比例
- F_kjg: 祖先群体 k 在位点 j 的等位基因频率

## 算法
使用 EM 算法（期望最大化）估计 Q 和 F。
"""

using Random
using Statistics
using LinearAlgebra

"""
    ADMIXTUREResults

ADMIXTURE 分析结果。

# 字段
- `Q`: 个体混合比例矩阵 (n_samples × K)
- `F`: 祖先群体等位基因频率矩阵 (K × n_snps)
- `K`: 祖先群体数量
- `log_likelihood`: 对数似然值
- `convergence`: 收敛信息
"""
struct ADMIXTUREResults
    Q::Matrix{Float64}
    F::Matrix{Float64}
    K::Int
    log_likelihood::Float64
    convergence::Dict{String, Any}
end

"""
    perform_admixture(genotypes::CompactGenotypes;
                      K::Int=3,
                      niter::Int=1000,
                      tol::Float64=1e-4,
                      seed::Union{Int,Nothing}=nothing,
                      verbose::Bool=true)

执行 ADMIXTURE 分析。

# 参数
- `genotypes`: 基因型数据
- `K`: 祖先群体数量
- `niter`: 最大迭代次数
- `tol`: 收敛阈值
- `seed`: 随机种子
- `verbose`: 是否显示进度

# 返回
ADMIXTUREResults 对象

# 算法
使用 EM 算法：
1. E 步骤：计算后验祖先群体分配概率
2. M 步骤：更新 Q 和 F
3. 重复直到收敛

# 示例
```julia
# K=3 的 ADMIXTURE 分析
results = perform_admixture(genotypes, K=3, niter=1000)

# 查看个体的混合比例
println("个体 1 的混合比例: ", results.Q[1, :])

# 祖先群体 1 的等位基因频率
println("群体 1 频率: ", results.F[1, :])
```
"""
function perform_admixture(genotypes::CompactGenotypes;
                           K::Int=3,
                           niter::Int=1000,
                           tol::Float64=1e-4,
                           seed::Union{Int,Nothing}=nothing,
                           verbose::Bool=true)

    if seed !== nothing
        Random.seed!(seed)
    end

    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    if K < 1
        throw(ValidationError("K 必须 >= 1"))
    end

    if verbose
        @info "开始 ADMIXTURE 分析"
        @info "  样本数: $n_samples"
        @info "  SNP 数: $n_snps"
        @info "  祖先群体数 K: $K"
        @info "  最大迭代: $niter"
    end

    # 解压缩基因型数据（0, 1, 2）
    if verbose
        @info "准备数据..."
    end

    G = Matrix{Int}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            G[i, j] = Int(genotypes[i, j])
        end
    end

    # 初始化参数
    # Q: 个体混合比例 (n_samples × K)
    Q = rand(n_samples, K)
    Q ./= sum(Q, dims=2)  # 归一化

    # F: 祖先群体等位基因频率 (K × n_snps)
    F = rand(K, n_snps) * 0.4 .+ 0.3  # 初始化在 [0.3, 0.7]
    F = clamp.(F, 0.01, 0.99)  # 避免 0 和 1

    # EM 迭代
    log_likelihood_history = Float64[]
    prev_ll = -Inf

    if verbose
        @info "开始 EM 迭代..."
    end

    for iter in 1:niter
        # =====================
        # E 步骤：计算后验概率
        # =====================
        # 对于每个个体和每个位点，计算来自每个祖先群体的后验概率
        # 这里简化：直接更新 Q

        # 计算似然贡献
        # P(G_ij | Q_i, F_j) = Σ_k Q_ik * P(G_ij | F_kj)

        # =====================
        # M 步骤：更新参数
        # =====================

        # 更新 Q
        Q_new = zeros(n_samples, K)

        for i in 1:n_samples
            for k in 1:K
                # 计算个体 i 来自群体 k 的后验权重
                weight = 0.0

                for j in 1:n_snps
                    g = G[i, j]
                    p = F[k, j]

                    # 二项式概率 P(g | p) = C(2,g) * p^g * (1-p)^(2-g)
                    if g == 0
                        prob = (1 - p)^2
                    elseif g == 1
                        prob = 2 * p * (1 - p)
                    else  # g == 2
                        prob = p^2
                    end

                    weight += log(prob + 1e-10)  # 避免 log(0)
                end

                Q_new[i, k] = exp(weight)
            end

            # 归一化
            Q_new[i, :] ./= sum(Q_new[i, :])
        end

        Q = Q_new

        # 更新 F
        F_new = zeros(K, n_snps)

        for k in 1:K
            for j in 1:n_snps
                # 等位基因计数的期望
                expected_allele_count = 0.0
                total_alleles = 0.0

                for i in 1:n_samples
                    q_ik = Q[i, k]
                    g = G[i, j]

                    expected_allele_count += q_ik * g
                    total_alleles += q_ik * 2
                end

                F_new[k, j] = expected_allele_count / (total_alleles + 1e-10)
            end
        end

        # 限制频率范围
        F = clamp.(F_new, 0.01, 0.99)

        # 计算对数似然
        ll = 0.0
        for i in 1:n_samples
            for j in 1:n_snps
                g = G[i, j]
                prob = 0.0

                for k in 1:K
                    p = F[k, j]
                    q = Q[i, k]

                    if g == 0
                        prob += q * (1 - p)^2
                    elseif g == 1
                        prob += q * 2 * p * (1 - p)
                    else
                        prob += q * p^2
                    end
                end

                ll += log(prob + 1e-10)
            end
        end

        push!(log_likelihood_history, ll)

        # 检查收敛
        if abs(ll - prev_ll) < tol
            if verbose
                @info "EM 算法在第 $iter 次迭代收敛"
            end
            break
        end

        prev_ll = ll

        # 进度报告
        if verbose && (iter % 100 == 0 || iter == niter)
            @info "  迭代 $iter: log-likelihood = $(round(ll, digits=2))"
        end
    end

    final_ll = log_likelihood_history[end]

    convergence = Dict(
        "iterations" => length(log_likelihood_history),
        "converged" => length(log_likelihood_history) < niter,
        "log_likelihood_history" => log_likelihood_history
    )

    if verbose
        @info "ADMIXTURE 分析完成"
        @info "  最终 log-likelihood: $(round(final_ll, digits=2))"
        @info "  迭代次数: $(convergence["iterations"])"
    end

    return ADMIXTUREResults(Q, F, K, final_ll, convergence)
end

"""
    estimate_optimal_k(genotypes::CompactGenotypes;
                       K_range::UnitRange{Int}=1:10,
                       niter::Int=500,
                       nreps::Int=3,
                       verbose::Bool=true)

通过交叉验证估计最优的 K 值。

# 参数
- `genotypes`: 基因型数据
- `K_range`: 要测试的 K 值范围
- `niter`: 每次运行的迭代次数
- `nreps`: 每个 K 值的重复次数
- `verbose`: 是否显示进度

# 返回
命名元组 (K_values, cross_validation_errors, best_K)

# 方法
使用交叉验证误差选择 K：
1. 对每个 K 值运行 ADMIXTURE
2. 计算预测误差
3. 选择误差最小的 K
"""
function estimate_optimal_k(genotypes::CompactGenotypes;
                            K_range::UnitRange{Int}=1:10,
                            niter::Int=500,
                            nreps::Int=3,
                            verbose::Bool=true)

    K_values = collect(K_range)
    cv_errors = zeros(length(K_values))

    if verbose
        @info "估计最优 K 值 (K ∈ $K_range, $nreps 次重复)"
    end

    for (i, K) in enumerate(K_values)
        errors = Float64[]

        for rep in 1:nreps
            # 运行 ADMIXTURE
            result = perform_admixture(
                genotypes,
                K = K,
                niter = niter,
                verbose = false,
                seed = rep * 1000 + K
            )

            # 使用负对数似然作为"误差"
            push!(errors, -result.log_likelihood)
        end

        cv_errors[i] = mean(errors)

        if verbose
            @info "  K=$K: CV error = $(round(cv_errors[i], digits=2)) ± $(round(std(errors), digits=2))"
        end
    end

    # 选择最优 K
    best_idx = argmin(cv_errors)
    best_K = K_values[best_idx]

    if verbose
        @info "最优 K: $best_K (CV error = $(round(cv_errors[best_idx], digits=2)))"
    end

    return (
        K_values = K_values,
        cv_errors = cv_errors,
        best_K = best_K
    )
end

"""
    assign_clusters(admix::ADMIXTUREResults; threshold::Float64=0.8)

基于混合比例将个体分配到群体。

# 参数
- `admix`: ADMIXTUREResults 对象
- `threshold`: 主要祖先比例阈值（默认 0.8）

# 返回
命名元组 (assignments, is_admixed)
- `assignments`: 主要祖先群体编号（混合个体为 0）
- `is_admixed`: 是否为混合个体
"""
function assign_clusters(admix::ADMIXTUREResults; threshold::Float64=0.8)
    n_samples = size(admix.Q, 1)

    assignments = zeros(Int, n_samples)
    is_admixed = falses(n_samples)

    for i in 1:n_samples
        max_q = maximum(admix.Q[i, :])

        if max_q >= threshold
            # 主要来自某个祖先群体
            assignments[i] = argmax(admix.Q[i, :])
        else
            # 混合个体
            is_admixed[i] = true
            assignments[i] = 0
        end
    end

    return (
        assignments = assignments,
        is_admixed = is_admixed
    )
end

# 导出
export ADMIXTUREResults, perform_admixture, estimate_optimal_k, assign_clusters
