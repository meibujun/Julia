# GenomicPro

GenomicPro 是一个使用 Julia 语言实现的多组学（multi-omics）分析流水线框架，提供从数据加载、质量控制、归一化、组学数据对齐、集成分析到报告生成的一站式能力。该实现基于 Julia v1.12.1，侧重于易用性与可扩展性，支持 CLI 运行以及通过配置文件管理复杂工作流。

## 关键特性

- **多组学数据加载**：支持从 CSV/TSV 表格读取不同组学层的数据，灵活配置分隔符、特征列、缺失值标记等。
- **缺失值填补与质量控制**：内置均值/中位数/零值填补策略，支持对低方差、缺失率过高的特征进行过滤。
- **标准化与变换**：提供 z-score、min-max、robust scaling 以及可选对数变换，适配不同测序平台的特性。
- **数据对齐与集成**：自动对齐样本，支持特征拼接（concatenate）与加权求和（weighted sum）两种集成策略。
- **统计/机器学习分析**：内置 PCA 降维与 KMeans 聚类，可根据配置决定是否执行并调整参数。
- **报告生成**：输出 JSON 与文本摘要，涵盖数据概览、质量指标、分析结果，方便集成到下游系统。

## 项目结构

```
Project.toml
src/
  GenomicPro.jl        # 模块入口
  types.jl             # 核心类型定义
  io.jl                # 数据读取与配置解析
  preprocessing.jl     # 缺失值填补、归一化、对齐等预处理
  integration.jl       # 多组学集成算法
  analysis.jl          # PCA、KMeans 等分析
  reporting.jl         # 报告与摘要生成
  validation.jl        # 数据校验与质量过滤
  utils.jl             # 通用辅助函数
  pipeline.jl          # 流水线编排
scripts/
  genomicpro_cli.jl    # 命令行入口
config/
  example_pipeline.json# 配置示例
```

## 安装依赖

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## 运行示例流水线

准备示例数据（可参考 `test/data` 目录中的 CSV），并在根目录创建输出目录：

```bash
mkdir -p data reports
cp test/data/*.csv data/
```

然后运行命令行脚本：

```bash
julia scripts/genomicpro_cli.jl config/example_pipeline.json
```

执行完成后，JSON 与文本报告分别保存在 `reports/pipeline_report.json` 与 `reports/pipeline_report.txt`。

## 在代码中使用

```julia
using GenomicPro

config = read_pipeline_config("config/example_pipeline.json")
result = run_pipeline(config)
println(render_summary(result))
```

## 测试

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## 扩展建议

- 接入更多集成策略（如基于网络的融合、协同矩阵分解等）。
- 支持批量效应校正与更丰富的归一化方法。
- 引入更高级的聚类与分类算法，或与 MLJ 生态衔接。
- 构建交互式可视化报告，支持在浏览器中探索多组学结果。
