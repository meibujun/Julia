"""
# 主成分分析（PCA）

用于基因型数据的降维和群体结构可视化。

## 方法
1. 数据标准化
2. 协方差矩阵计算（或使用 SVD）
3. 特征值分解
4. 提取主成分

## 解释
- PC1, PC2: 捕捉最大的遗传变异
- 用于检测群体分层、离群个体
- 可作为 GWAS 的协变量
"""

using LinearAlgebra
using Statistics

"""
    PCAResults

PCA 分析结果。

# 字段
- `scores`: 主成分得分矩阵 (n_samples × n_components)
- `loadings`: SNP 载荷矩阵 (n_snps × n_components)
- `eigenvalues`: 特征值
- `explained_variance`: 解释的方差比例
- `cumulative_variance`: 累积解释方差比例
- `n_components`: 主成分数量
"""
struct PCAResults
    scores::Matrix{Float64}
    loadings::Matrix{Float64}
    eigenvalues::Vector{Float64}
    explained_variance::Vector{Float64}
    cumulative_variance::Vector{Float64}
    n_components::Int
end

"""
    perform_pca(genotypes::CompactGenotypes;
                n_components::Int=10,
                method::Symbol=:svd,
                center::Bool=true,
                scale::Bool=true,
                verbose::Bool=true)

对基因型数据进行主成分分析。

# 参数
- `genotypes`: CompactGenotypes 基因型数据
- `n_components`: 提取的主成分数量（默认 10）
- `method`: 计算方法 (:svd 或 :eigen)
- `center`: 是否中心化（默认 true）
- `scale`: 是否标准化（默认 true）
- `verbose`: 是否显示进度信息

# 返回
PCAResults 对象

# 方法说明
- `:svd`: 使用奇异值分解（推荐，更稳定）
- `:eigen`: 使用协方差矩阵的特征值分解

# 示例
```julia
# 提取前 20 个主成分
pca = perform_pca(genotypes, n_components=20)

# 查看解释方差
println("PC1-PC3 解释方差: ", pca.explained_variance[1:3])

# 获取个体在 PC1-PC2 上的坐标
pc1 = pca.scores[:, 1]
pc2 = pca.scores[:, 2]
```
"""
function perform_pca(genotypes::CompactGenotypes;
                     n_components::Int=10,
                     method::Symbol=:svd,
                     center::Bool=true,
                     scale::Bool=true,
                     verbose::Bool=true)

    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    if n_components > min(n_samples, n_snps)
        throw(ValidationError("n_components 不能超过 min(n_samples, n_snps)"))
    end

    if verbose
        @info "开始 PCA 分析"
        @info "  样本数: $n_samples"
        @info "  SNP 数: $n_snps"
        @info "  提取主成分数: $n_components"
        @info "  方法: $method"
    end

    # 解压缩基因型数据
    if verbose
        @info "解压缩基因型数据..."
    end

    X = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            X[i, j] = Float64(genotypes[i, j])
        end
    end

    # 中心化和标准化
    if verbose
        @info "数据预处理（中心化和标准化）..."
    end

    if center
        X_mean = mean(X, dims=1)
        X .-= X_mean
    end

    if scale
        X_std = std(X, dims=1)
        # 避免除零
        X_std[X_std .== 0] .= 1.0
        X ./= X_std
    end

    # 执行 PCA
    if verbose
        @info "计算主成分..."
    end

    if method == :svd
        # 使用 SVD 方法
        U, S, V = svd(X)

        # 主成分得分
        scores = U[:, 1:n_components] * Diagonal(S[1:n_components])

        # SNP 载荷
        loadings = V[:, 1:n_components]

        # 特征值（奇异值的平方除以 n-1）
        eigenvalues = (S[1:n_components] .^ 2) / (n_samples - 1)

    elseif method == :eigen
        # 使用协方差矩阵的特征值分解
        # 计算协方差矩阵 X'X / (n-1)
        if verbose
            @info "计算协方差矩阵..."
        end

        C = (X' * X) / (n_samples - 1)

        # 特征值分解
        if verbose
            @info "特征值分解..."
        end

        eigen_result = eigen(Symmetric(C))

        # 按特征值降序排列
        idx = sortperm(eigen_result.values, rev=true)
        eigenvalues = eigen_result.values[idx[1:n_components]]
        eigenvectors = eigen_result.vectors[:, idx[1:n_components]]

        # 主成分得分 = X * 特征向量
        scores = X * eigenvectors

        # SNP 载荷
        loadings = eigenvectors

    else
        throw(ValidationError("未知的 PCA 方法: $method（支持 :svd 或 :eigen）"))
    end

    # 计算解释方差比例
    total_variance = sum(var(X, dims=1))
    explained_variance = eigenvalues / total_variance
    cumulative_variance = cumsum(explained_variance)

    if verbose
        @info "PCA 分析完成"
        @info "  PC1 解释方差: $(round(explained_variance[1] * 100, digits=2))%"
        @info "  前 $n_components 个PC累积解释方差: $(round(cumulative_variance[end] * 100, digits=2))%"
    end

    return PCAResults(
        scores,
        loadings,
        eigenvalues,
        explained_variance,
        cumulative_variance,
        n_components
    )
end

"""
    scree_plot(pca::PCAResults)

生成碎石图数据（用于确定保留主成分数量）。

# 返回
命名元组 (component_numbers, eigenvalues, explained_variance)
"""
function scree_plot(pca::PCAResults)
    return (
        component_numbers = 1:pca.n_components,
        eigenvalues = pca.eigenvalues,
        explained_variance = pca.explained_variance
    )
end

"""
    biplot_pca(pca::PCAResults, pc_x::Int=1, pc_y::Int=2, top_snps::Int=10)

生成 PCA 双标图数据（样本得分 + SNP 载荷）。

# 参数
- `pca`: PCAResults 对象
- `pc_x`: X 轴主成分编号
- `pc_y`: Y 轴主成分编号
- `top_snps`: 显示载荷最大的 SNP 数量

# 返回
命名元组，包含样本坐标和 SNP 载荷
"""
function biplot_pca(pca::PCAResults, pc_x::Int=1, pc_y::Int=2, top_snps::Int=10)
    if pc_x > pca.n_components || pc_y > pca.n_components
        throw(ValidationError("主成分编号超出范围"))
    end

    # 样本得分
    sample_coords = (
        x = pca.scores[:, pc_x],
        y = pca.scores[:, pc_y]
    )

    # 选择载荷最大的 SNPs
    loading_magnitude = sqrt.(pca.loadings[:, pc_x] .^ 2 + pca.loadings[:, pc_y] .^ 2)
    top_idx = sortperm(loading_magnitude, rev=true)[1:min(top_snps, length(loading_magnitude))]

    snp_loadings = (
        x = pca.loadings[top_idx, pc_x],
        y = pca.loadings[top_idx, pc_y],
        indices = top_idx
    )

    return (
        samples = sample_coords,
        snps = snp_loadings,
        pc_x = pc_x,
        pc_y = pc_y
    )
end

"""
    project_new_samples(pca::PCAResults, genotypes_new::CompactGenotypes)

将新样本投影到现有的 PCA 空间。

# 参数
- `pca`: 已拟合的 PCAResults
- `genotypes_new`: 新样本的基因型数据

# 返回
新样本的主成分得分矩阵
"""
function project_new_samples(pca::PCAResults, genotypes_new::CompactGenotypes)
    n_samples_new = size(genotypes_new.data, 1)
    n_snps = size(genotypes_new.data, 2)

    if size(pca.loadings, 1) != n_snps
        throw(ValidationError("SNP 数量不匹配"))
    end

    # 解压缩新数据
    X_new = Matrix{Float64}(undef, n_samples_new, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples_new
            X_new[i, j] = Float64(genotypes_new[i, j])
        end
    end

    # 应用相同的标准化
    # 注意：这里简化处理，实际应保存训练集的均值和标准差
    X_new_mean = mean(X_new, dims=1)
    X_new .-= X_new_mean
    X_new_std = std(X_new, dims=1)
    X_new_std[X_new_std .== 0] .= 1.0
    X_new ./= X_new_std

    # 投影
    scores_new = X_new * pca.loadings

    return scores_new
end

"""
    compute_fst(genotypes::CompactGenotypes, populations::Vector{Int})

计算群体间的 FST（固定指数）。

# 参数
- `genotypes`: 基因型数据
- `populations`: 每个样本的群体标签（整数向量）

# 返回
FST 矩阵（群体间成对 FST）

# 公式
FST = (Ht - Hs) / Ht

其中：
- Ht: 总群体杂合度
- Hs: 亚群内平均杂合度
"""
function compute_fst(genotypes::CompactGenotypes, populations::Vector{Int})
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    if length(populations) != n_samples
        throw(ValidationError("群体标签数量与样本数不匹配"))
    end

    unique_pops = unique(populations)
    n_pops = length(unique_pops)

    @info "计算 $n_pops 个群体的成对 FST"

    # 初始化 FST 矩阵
    FST = zeros(n_pops, n_pops)

    for i in 1:n_pops
        for j in (i+1):n_pops
            pop1_idx = findall(populations .== unique_pops[i])
            pop2_idx = findall(populations .== unique_pops[j])

            fst_values = Float64[]

            # 计算每个 SNP 的 FST
            for snp in 1:n_snps
                # 提取基因型
                geno1 = [Float64(genotypes[k, snp]) for k in pop1_idx]
                geno2 = [Float64(genotypes[k, snp]) for k in pop2_idx]

                # 计算等位基因频率
                p1 = mean(geno1) / 2
                p2 = mean(geno2) / 2

                # 总群体等位基因频率
                pt = (length(geno1) * p1 + length(geno2) * p2) / (length(geno1) + length(geno2))

                # 杂合度
                Ht = 2 * pt * (1 - pt)
                Hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2

                # FST
                if Ht > 0
                    fst = (Ht - Hs) / Ht
                    push!(fst_values, max(0, fst))  # FST 不应为负
                end
            end

            # 平均 FST
            FST[i, j] = mean(fst_values)
            FST[j, i] = FST[i, j]
        end
    end

    return FST
end

"""
    compute_kinship(genotypes::CompactGenotypes)

计算亲缘关系矩阵（等价于标准化的 GRM）。

# 返回
亲缘关系矩阵 (n_samples × n_samples)
"""
function compute_kinship(genotypes::CompactGenotypes)
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    @info "计算亲缘关系矩阵 ($n_samples × $n_samples)"

    # 解压缩基因型
    X = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            X[i, j] = Float64(genotypes[i, j])
        end
    end

    # 中心化和标准化
    X_mean = mean(X, dims=1)
    X .-= X_mean
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    X ./= X_std

    # 亲缘关系矩阵 = X X' / m
    K = (X * X') / n_snps

    return K
end

# 导出
export PCAResults, perform_pca, scree_plot, biplot_pca
export project_new_samples, compute_fst, compute_kinship
