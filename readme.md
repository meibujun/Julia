# AnimalBreeding.jl

[![Julia Version](https://img.shields.io/badge/Julia-1.11.6-blue.svg)](https://julialang.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`AnimalBreeding.jl` 是一个使用 Julia 语言开发的开源多物种动物育种分析平台，面向牛、猪、羊、家禽等物种的遗传评估与基因组选择工作流程。项目集成了数据管理、混合线性模型遗传评估、贝叶斯基因组选择、机器学习预测、命令行接口以及性能优化配置，强调模块化、可扩展和科学严谨性。

## ✨ 功能亮点

### 数据管理与整合
- 统一加载表型、谱系、基因型、环境及多组学数据。
- 自动验证数据完整性，检测缺失 ID、重复个体、谱系回路等问题。
- 支持谱系关系矩阵 (A)、基因组关系矩阵 (G，VanRaden 方法) 以及单步关系矩阵 (H) 计算。
- 关系矩阵自动缓存，可通过 `clear_cache!` 手动清理避免内存占用。

### 模型定义与遗传评估
- 通过 `define_model` 配置多性状、固定效应与随机效应结构。
- `run_evaluation` 支持 BLUP、GBLUP、SS-GBLUP，自动构建 Henderson 混合模型方程求解育种值。
- 输出固定效应估计、个体育种值与可靠度，并提供方差组分估计。

### 贝叶斯基因组选择
- 实现 BayesA、BayesB、BayesC、Bayesian LASSO 采样器。
- 支持单链 Gibbs 采样、后验均值及方差组件汇总，提供简易诊断工具。
- 可直接基于模型与数据仓库调用，或传入矩阵运行自定义贝叶斯分析。

### 机器学习与深度学习
- 封装随机森林、梯度提升树与多层感知机等方法。
- 自动标准化特征，支持交叉验证并返回预测相关系数等指标。
- 可作为传统遗传评估的补充或基准对比。

### 命令行接口
- `validate`：快速完成数据加载与质量检查。
- `evaluate`：执行 BLUP/GBLUP/SS-GBLUP 遗传评估。
- `bayes`：运行贝叶斯基因组选择采样。
- `ml`：对给定特征和标签执行机器学习交叉验证。

### 性能与可扩展性
- 配置多线程、分布式和 GPU 使用策略的统一入口 `configure_performance`。
- 关键矩阵运算使用线性代数优化，并对大数据集提供稀疏/正则化处理。

## 🚀 快速开始

```julia
using AnimalBreeding

# 1. 构建数据仓库
repo = DataRepository()
repo.phenotypes = load_phenotypes("phenotypes.csv")
repo.pedigrees  = load_pedigree("pedigree.csv")
repo.genotypes  = load_genotypes("genotypes.csv")
integrate_data!(repo)
validate_data(repo) |> println
# 如需释放缓存
clear_cache!(repo)

# 2. 定义模型
model = define_model(
    traits = [:milk],
    fixed  = [:herd, :year],
    random = [("animal", :additive)]
)

# 3. 运行混合线性模型评估
result = run_evaluation(model, repo; method = :GBLUP, h2 = 0.35)
println(result)

# 4. 贝叶斯基因组选择
bayes_res = run_bayesian_evaluation(model, repo; method = :BayesC, n_iter = 2000, burn_in = 500)
println(bayes_res.posterior_means[:marker_variance])

# 5. 机器学习预测（随机森林）
X = Matrix(repo.genotypes[:, Not(:animal)])
y = Float64.(repo.phenotypes[:, :milk])
rf = train_ml_model(:RandomForest, X, y; n_trees = 200)
predict(rf, X)[1:5]

# 6. CLI 示例
# julia --project=. -e 'using AnimalBreeding; run_cli(["validate", "--phenotype", "phenotypes.csv"])'
```

## 🧪 测试

项目提供 `test/runtests.jl` 覆盖核心功能：
- 数据加载与关系矩阵计算
- 混合线性模型 BLUP/GBLUP 评估
- 贝叶斯采样正确性（小规模示例）
- 机器学习训练与交叉验证
- 命令行配置解析

运行测试：

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

> 注：如需 GPU 相关功能，请提前安装 `CUDA.jl` 并确认硬件支持。

## 📚 参考文献

1. Henderson, C. R. (1984). *Applications of Linear Models in Animal Breeding*.
2. VanRaden, P. M. (2008). Efficient methods to compute genomic predictions. *Journal of Dairy Science*.
3. Meuwissen, T. et al. (2001). Prediction of total genetic value using genome-wide dense marker maps. *Genetics*.
4. Legarra, A. et al. (2009). A relationship matrix including full pedigree and genomic information. *Journal of Dairy Science*.

## 🤝 贡献

欢迎提交 Issue、Pull Request 或分享使用案例。提交代码时请：
- 遵循 Julia 代码风格并补充文档字符串；
- 为新增功能编写单元测试；
- 更新 README 或相关文档。

## 📄 许可证

项目遵循 MIT 许可证，详见 [LICENSE](LICENSE)。
