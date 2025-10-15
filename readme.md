# RareEpistasisMeta.jl

高性能的家畜全基因组稀有变异上位性 Meta 分析平台，面向 90K 级别 SNP 标记、
多组学整合与实际育种应用场景。软件覆盖稀有变异检测、贝叶斯推断、
RKHS/EG-BLUP 预测、Meta 分析、育种方案模拟与大语言模型自动报告等模块，
以满足国际领先科研与生产需求。

## 核心特性

- **多格式数据支持**：兼容 VCF、PLINK、CSV 等常见格式，提供缺失填补与质控。
- **12 种稀有变异检验**：包含 Burden、CMC、VT、SKAT、ACAT-V 等最新方法。
- **多样化贝叶斯模型**：Gibbs 多元回归、Bayesian LASSO、BayesB 全面覆盖。
- **上位性解析**：GLM 交互、方差组分、互信息、梯度提升等 4 大框架。
- **Meta 分析框架**：固定、随机、贝叶斯层级、ACAT 与稳健加权 5 种模型。
- **多组学与育种应用**：提供模拟、整合核构建、EG-BLUP 预测与方案优化。
- **LLM 智能助理**：自动生成中文报告、候选基因排序与方案建议。
- **高质量可视化**：曼哈顿图、QQ 图、森林图、互作网络快速呈现结果。

## 快速开始

```julia
using RareEpistasisMeta

multiomics = simulate_multiomics(n_samples = 300)
integrate_multiomics!(multiomics)
G = multiomics[:genomics]
y = multiomics[:phenomics].DailyGain

rvat = collapsing_test(G, y; method = :acatv)
epi = epistasis_scan(G[:, 1:100], y)
meta = meta_analyze(DataFrame(effect = randn(5), variance = rand(5) .+ 0.01, study = 1:5); model = :bayesian)

println(rvat)
println(first(epi, 5))
println(meta)
```

更多示例请查看 `examples/demo.jl` 与 `test/runtests.jl`。

## 许可证

MIT License
