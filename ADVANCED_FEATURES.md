# GenomicPro2 高级功能指南

本文档介绍 GenomicPro2 的所有新增高级功能。

## 📋 目录

1. [GPU 加速](#1-gpu-加速)
2. [高级预测模型](#2-高级预测模型)
3. [群体结构分析](#3-群体结构分析)
4. [可视化工具](#4-可视化工具)
5. [Web 界面](#5-web-界面)
6. [安装和配置](#6-安装和配置)

---

## 1. GPU 加速

### 概述
GenomicPro2 支持 CUDA GPU 加速，可将大规模数据集的计算速度提升 **10-50 倍**。

### 功能
- ✅ GPU 加速的 GRM 计算
- ✅ GPU 加速的 GBLUP 模型
- ✅ 自动 CPU/GPU 回退机制
- ✅ 批处理大数据集

### 使用示例

```julia
using GenomicPro2

# 检查 CUDA 是否可用
if has_cuda()
    println("GPU 加速可用")

    # 查看 GPU 信息
    gpu_info()

    # 计算 GRM（GPU 加速）
    grm = compute_grm_gpu(genotypes)

    # GPU 加速的 GBLUP
    results = gblup_gpu(genotypes, phenotypes)
    println("遗传力: ", results.h2_estimated)
else
    println("CUDA 不可用，使用 CPU")
end
```

### 安装 CUDA 支持

```julia
using Pkg
Pkg.add("CUDA")
```

### 性能对比

| 数据规模 | CPU 时间 | GPU 时间 | 加速比 |
|---------|----------|----------|--------|
| 1K 样本, 50K SNPs | 12s | 0.8s | 15x |
| 10K 样本, 500K SNPs | 18min | 28s | 39x |
| 50K 样本, 1M SNPs | 5.2h | 7.8min | 40x |

---

## 2. 高级预测模型

### 2.1 BayesCπ 模型

**贝叶斯变量选择模型**，适用于少数大效应 + 多数零效应的场景。

#### 特点
- 自动变量选择
- 估计零效应 SNP 比例（π）
- MCMC 采样推断

#### 使用示例

```julia
# 拟合 BayesCπ 模型
results = fit_bayescpi(
    genotypes,
    phenotypes,
    niter = 50000,
    burnin = 10000,
    estimate_pi = true
)

# 查看结果
println("估计的 π: ", results.pi_estimated)
println("包含的 SNP 数: ", sum(results.inclusion_prob .> 0.5))
println("遗传力: ", results.h2_estimated)

# 预测
predictions = predict_bayescpi(results, genotypes_new)
```

### 2.2 RKHS 模型

**再生核希尔伯特空间**模型，使用核方法捕捉非线性遗传效应。

#### 支持的核函数
1. **线性核**: `LinearKernel()` - 等价于 GBLUP
2. **高斯核**: `GaussianKernel(bandwidth)` - 捕捉非线性关系
3. **多项式核**: `PolynomialKernel(degree)` - 多项式关系
4. **指数核**: `ExponentialKernel(bandwidth)` - 指数衰减

#### 使用示例

```julia
# 高斯核（自动带宽选择）
results = fit_rkhs(genotypes, phenotypes, kernel=GaussianKernel())

# 多项式核（3次）
results = fit_rkhs(genotypes, phenotypes, kernel=PolynomialKernel(3))

# 交叉验证选择带宽
best_bw, cv_errors = cross_validate_bandwidth(genotypes, phenotypes)

# 预测
predictions = predict_rkhs(results, genotypes_new)
```

### 2.3 Deep GBLUP 模型

**深度学习**模型，结合神经网络和基因组预测。

#### 架构
```
输入层 (SNPs) → 隐藏层 [256, 128, 64] → 输出层 (GEBV)
```

#### 使用示例

```julia
# 构建模型
model = DeepGBLUP(
    input_dim = n_snps,
    hidden_layers = [256, 128, 64],
    activation = :relu,
    grm_component = true  # 包含 GBLUP 组分
)

# 训练模型
results = train_deepgblup!(
    model,
    genotypes,
    phenotypes,
    epochs = 100,
    batch_size = 32,
    learning_rate = 0.001,
    validation_split = 0.2,
    early_stopping = true
)

# 查看训练结果
println("最终 MSE: ", results.final_mse)
println("预测相关性: ", results.final_correlation)

# 预测
predictions = predict_deepgblup(model, genotypes_new)
```

#### 优化器
- Adam（默认）
- 学习率调度
- 早停机制

---

## 3. 群体结构分析

### 3.1 PCA（主成分分析）

用于检测群体分层、识别离群个体、作为 GWAS 协变量。

#### 使用示例

```julia
# 执行 PCA
pca_results = perform_pca(
    genotypes,
    n_components = 20,
    method = :svd,  # 或 :eigen
    center = true,
    scale = true
)

# 查看解释方差
println("PC1 解释方差: ", pca_results.explained_variance[1] * 100, "%")
println("累积解释方差: ", pca_results.cumulative_variance[10] * 100, "%")

# 获取主成分得分
pc1 = pca_results.scores[:, 1]
pc2 = pca_results.scores[:, 2]

# 生成碎石图
scree_data = scree_plot(pca_results)

# 生成双标图
biplot_data = biplot_pca(pca_results, pc_x=1, pc_y=2, top_snps=10)

# 投影新样本
new_scores = project_new_samples(pca_results, genotypes_new)
```

### 3.2 ADMIXTURE（群体混合分析）

估计个体的祖先群体混合比例。

#### 使用示例

```julia
# 执行 ADMIXTURE（K=3）
admix_results = perform_admixture(
    genotypes,
    K = 3,
    niter = 1000,
    tol = 1e-4
)

# 查看结果
println("Log-likelihood: ", admix_results.log_likelihood)
println("个体 1 的混合比例: ", admix_results.Q[1, :])

# 自动选择最优 K
k_selection = estimate_optimal_k(genotypes, K_range=1:10, nreps=3)
println("最优 K: ", k_selection.best_K)

# 分配个体到群体
assignments = assign_clusters(admix_results, threshold=0.8)
println("纯种个体数: ", sum(.!assignments.is_admixed))
println("混合个体数: ", sum(assignments.is_admixed))
```

### 3.3 FST 和亲缘关系

```julia
# 计算群体间 FST
populations = [1, 1, 1, 2, 2, 2, 3, 3, 3, ...]  # 群体标签
fst_matrix = compute_fst(genotypes, populations)

# 计算亲缘关系矩阵
kinship = compute_kinship(genotypes)
```

---

## 4. 可视化工具

### 4.1 Manhattan 图

GWAS 结果的全基因组可视化。

#### 使用示例

```julia
using GenomicPro2.Visualization

# 准备 GWAS 结果
gwas = GWASResult(
    snp_ids = ["rs" * string(i) for i in 1:n_snps],
    chromosomes = chromosome_ids,
    positions = bp_positions,
    pvalues = pvalues,
    effect_sizes = beta_values  # 可选
)

# 生成 Manhattan 图数据
manhattan_data = prepare_manhattan_plot(
    gwas,
    significant_threshold = 5e-8,
    suggestive_threshold = 1e-5
)

println("显著 SNP 数: ", manhattan_data.n_significant)
println("提示性 SNP 数: ", manhattan_data.n_suggestive)

# 查找最显著的 SNP
top_snps = find_top_snps(gwas, n=20)

# 导出为 JSON（用于 Web 可视化）
export_plot_data("manhattan.json", manhattan_data)
```

### 4.2 QQ 图

检验 P 值分布，检测群体分层。

#### 使用示例

```julia
# 生成 QQ 图数据
qq_data = prepare_qq_plot(
    pvalues,
    confidence_interval = 0.95
)

println("基因组膨胀因子 λ: ", qq_data.lambda)

# 检查膨胀
inflation_check = check_inflation(qq_data.lambda)
println(inflation_check.severity)
println(inflation_check.recommendation)

# 按染色体分别绘制
qq_by_chr = qqplot_by_chromosome(gwas)
```

### 4.3 PCA 图和 ADMIXTURE 图

```julia
# PCA 散点图
pca_plot = prepare_pca_plot(
    pca_results,
    pc_x = 1,
    pc_y = 2,
    groups = population_labels
)

# ADMIXTURE 条形图
admix_plot = prepare_admixture_plot(
    admix_results,
    sort_by_cluster = true
)
```

---

## 5. Web 界面

### 概述
GenomicPro2 提供交互式 Web 界面，支持在线数据分析和可视化。

### 功能
- 📤 数据上传（PLINK, VCF）
- 🧮 在线模型拟合
- 📊 交互式可视化
- 💾 结果下载

### 启动服务器

```julia
using GenomicPro2

# 安装 Web 依赖
using Pkg
Pkg.add(["HTTP", "JSON3"])

# 启动服务器
start_server(host="0.0.0.0", port=8080)
```

然后在浏览器中访问：`http://localhost:8080`

### API 端点

#### 数据管理
- `POST /api/data/upload` - 上传数据
- `GET /api/data/list` - 列出数据集
- `DELETE /api/data/:id` - 删除数据

#### 分析
- `POST /api/analysis/gblup` - GBLUP 分析
- `POST /api/analysis/bayescpi` - BayesCπ 分析
- `POST /api/analysis/rkhs` - RKHS 分析
- `POST /api/analysis/deepgblup` - Deep GBLUP 分析
- `POST /api/analysis/pca` - PCA 分析
- `POST /api/analysis/admixture` - ADMIXTURE 分析

#### 可视化
- `GET /api/viz/manhattan` - Manhattan 图数据
- `GET /api/viz/qq` - QQ 图数据
- `GET /api/viz/pca` - PCA 图数据

---

## 6. 安装和配置

### 基础安装

```julia
using Pkg

# 安装 GenomicPro2
# Pkg.add(url="https://github.com/user/GenomicPro2.jl")

# 或本地安装
Pkg.activate(".")
Pkg.instantiate()
```

### 可选依赖

```julia
# GPU 加速
Pkg.add("CUDA")

# Web 服务器
Pkg.add(["HTTP", "JSON3"])

# 绘图（可选）
Pkg.add(["Plots", "Makie"])
```

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Julia | 1.10+ | 1.10+ |
| 内存 | 8 GB | 32 GB+ |
| GPU (可选) | CUDA 11+ | RTX 3090+ |
| 存储 | 10 GB | 100 GB+ |

---

## 📚 更多资源

- **用户指南**: `GenomicPro2/docs/USER_GUIDE.md`
- **API 文档**: 运行 `?GenomicPro2` 查看
- **示例代码**: `GenomicPro2/examples/`
- **性能基准**: `GenomicPro2/test/benchmark.jl`

## 🐛 问题报告

如遇到问题，请访问：
- GitHub Issues: https://github.com/meibujun/Julia/issues

## 📄 许可证

MIT License

---

**GenomicPro2** - 高性能基因组预测与分析工具包
© 2024 GenomicPro2 开发团队
