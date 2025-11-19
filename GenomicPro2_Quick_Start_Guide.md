# GenomicPro 2.0 快速入门指南

**文档版本**: 1.0
**更新日期**: 2025-11-15
**适用版本**: GenomicPro2 v2.0.0

---

## 📦 安装与设置

### 前置要求

- **Julia**: ≥ 1.10
- **内存**: 建议 8GB+ （处理大规模数据时需要更多）
- **操作系统**: Linux, macOS, Windows

### 安装步骤

#### 方法 1: 从本地路径安装

```julia
using Pkg

# 从本地路径安装
Pkg.develop(path="/home/user/Julia/GenomicPro2")

# 或者添加到项目
Pkg.add(path="/home/user/Julia/GenomicPro2")
```

#### 方法 2: 从 GitHub 安装（推送到远程后）

```julia
using Pkg
Pkg.add(url="https://github.com/meibujun/Julia", subdir="GenomicPro2")
```

### 验证安装

```julia
using GenomicPro2

# 查看版本
println("GenomicPro2 version: ", GenomicPro2.VERSION)

# 检查核心类型是否可用
@assert isdefined(GenomicPro2, :CompactGenotypes)
@assert isdefined(GenomicPro2, :ValidationResult)

println("✓ GenomicPro2 安装成功！")
```

---

## 🚀 基础使用

### 1. 创建基因型数据

#### 从矩阵创建

```julia
using GenomicPro2

# 创建示例数据 (样本 × 标记)
# 基因型值: 0, 1, 2 或 missing
n_samples = 1000
n_markers = 5000

genotype_matrix = rand(0:2, n_samples, n_markers)

# 添加一些缺失值
for i in 1:100
    genotype_matrix[rand(1:n_samples), rand(1:n_markers)] = missing
end

# 创建样本和标记 ID
sample_ids = ["S$i" for i in 1:n_samples]
marker_ids = ["M$i" for i in 1:n_markers]

# 创建 CompactGenotypes 对象
geno = CompactGenotypes(genotype_matrix, sample_ids, marker_ids)

println("✓ 创建了 $(n_samples) 个样本 × $(n_markers) 个标记的基因型数据")
```

#### 添加可选的基因组注释信息

```julia
# 创建带完整注释的基因型数据
geno = CompactGenotypes(
    genotype_matrix,
    sample_ids,
    marker_ids;
    chromosome = ["chr$((i-1) ÷ 1000 + 1)" for i in 1:n_markers],
    position = [1000 * i for i in 1:n_markers],
    ref_allele = fill("A", n_markers),
    alt_allele = fill("G", n_markers)
)
```

### 2. 数据访问和查询

```julia
# 获取基本信息
println("样本数: ", n_samples(geno))
println("标记数: ", n_markers(geno))
println("缺失率: ", missing_rate(geno))

# 获取 ID 列表
samples = sample_ids(geno)
markers = marker_ids(geno)

println("前 5 个样本: ", samples[1:5])
println("前 5 个标记: ", markers[1:5])

# 获取等位基因频率
freqs = allele_frequencies(geno)
println("前 5 个标记的等位基因频率: ", freqs[1:5])

# 访问单个基因型
genotype_value = geno[10, 20]  # 第 10 个样本，第 20 个标记
println("样本 10，标记 20 的基因型: ", genotype_value)

# 检查是否为缺失值
is_missing = ismissing(geno, 10, 20)
println("该位点是否缺失: ", is_missing)
```

### 3. 数据验证

```julia
# 验证数据完整性
result = validate(geno)

# 检查验证结果
if is_valid(result)
    println("✓ 数据验证通过")
else
    println("✗ 数据验证失败")

    # 显示错误
    for error in result.errors
        println("  错误: ", error)
    end

    # 显示警告
    for warning in result.warnings
        println("  警告: ", warning)
    end
end

# 查看验证元数据
println("\n验证元数据:")
for (key, value) in result.metadata
    println("  $key: $value")
end
```

### 4. 内存使用分析

```julia
# 计算内存使用情况
mem = memory_usage(geno)

println("内存使用统计:")
println("  原始数据大小 (Float64): $(mem.original / 1e6) MB")
println("  压缩后大小 (2-bit): $(mem.total / 1e6) MB")
println("  节省: $(mem.savings * 100)%")
println("  压缩比: $(mem.compression_ratio)x")

# 示例输出:
# 内存使用统计:
#   原始数据大小 (Float64): 40.0 MB
#   压缩后大小 (2-bit): 1.25 MB
#   节省: 96.875%
#   压缩比: 32.0x
```

---

## 🔬 高级功能

### 数据子集提取

```julia
# 提取特定样本
sample_subset = [1, 5, 10, 15, 20]
geno_subset = subset_samples(geno, sample_subset)

# 提取特定标记
marker_subset = 1:1000  # 前 1000 个标记
geno_subset = subset_markers(geno, marker_subset)

# 同时提取样本和标记
geno_subset = subset(geno, sample_subset, marker_subset)
```

### 质量控制

```julia
# 应用质量控制过滤
# （注意：这些函数将在后续版本中实现）

# 示例: 手动过滤高缺失率的标记
marker_missing_rates = [
    sum(ismissing(geno, i, j) for i in 1:n_samples(geno)) / n_samples(geno)
    for j in 1:n_markers(geno)
]

# 保留缺失率 < 10% 的标记
good_markers = findall(marker_missing_rates .< 0.1)
geno_qc = subset_markers(geno, good_markers)

println("QC 后保留 $(length(good_markers)) / $(n_markers(geno)) 个标记")
```

### 等位基因频率过滤

```julia
# 获取等位基因频率
freqs = allele_frequencies(geno)

# 过滤低频等位基因 (MAF < 0.01)
maf = min.(freqs, 1 .- freqs)  # 次等位基因频率
good_markers = findall(maf .>= 0.01)

geno_filtered = subset_markers(geno, good_markers)

println("MAF 过滤后保留 $(length(good_markers)) / $(n_markers(geno)) 个标记")
```

---

## 💡 实用技巧

### 1. 处理大规模数据

```julia
# 对于非常大的数据集，分批处理
batch_size = 10000

for start in 1:batch_size:n_markers(geno)
    stop = min(start + batch_size - 1, n_markers(geno))

    # 处理这一批标记
    batch = subset_markers(geno, start:stop)

    # 进行计算...
    freqs_batch = allele_frequencies(batch)

    println("处理标记 $start 到 $stop")
end
```

### 2. 数据导出

```julia
# 导出等位基因频率
using DataFrames, CSV

df = DataFrame(
    marker_id = marker_ids(geno),
    chromosome = geno.chromosome,
    position = geno.position,
    ref_allele = geno.ref_allele,
    alt_allele = geno.alt_allele,
    allele_freq = allele_frequencies(geno)
)

CSV.write("allele_frequencies.csv", df)
println("✓ 等位基因频率已导出")
```

### 3. 性能优化

```julia
# 预计算等位基因频率（如果频繁使用）
freqs = allele_frequencies(geno)  # 首次计算会缓存

# 使用类型稳定的代码
function compute_statistics(geno::CompactGenotypes{T}) where T
    n = n_samples(geno)
    m = n_markers(geno)
    freqs = allele_frequencies(geno)

    # ... 进行计算

    return results
end

# 避免在循环中创建临时数组
# 不推荐:
# for i in 1:1000
#     temp = allele_frequencies(geno)  # 重复计算
# end

# 推荐:
# freqs = allele_frequencies(geno)  # 计算一次
# for i in 1:1000
#     # 使用 freqs
# end
```

---

## 🧪 测试代码

### 运行完整测试套件

```bash
# 在 GenomicPro2 目录下
cd /home/user/Julia/GenomicPro2

# 运行所有测试
julia --project -e 'using Pkg; Pkg.test()'

# 运行特定测试文件
julia --project test/test_genotypes.jl
julia --project test/test_core.jl
```

### 编写自定义测试

```julia
using Test
using GenomicPro2

@testset "自定义测试" begin
    # 创建测试数据
    data = [0 1 2; 1 2 0; 2 0 1]
    sample_ids = ["S1", "S2", "S3"]
    marker_ids = ["M1", "M2", "M3"]

    geno = CompactGenotypes(data, sample_ids, marker_ids)

    # 测试基本属性
    @test n_samples(geno) == 3
    @test n_markers(geno) == 3

    # 测试数据访问
    @test geno[1, 1] == 0
    @test geno[2, 2] == 2

    # 测试等位基因频率
    freqs = allele_frequencies(geno)
    @test length(freqs) == 3
    @test all(0 .<= freqs .<= 1)

    println("✓ 所有自定义测试通过")
end
```

---

## 📊 性能基准测试

### 内存效率测试

```julia
using BenchmarkTools

# 比较不同存储方式的内存使用
function compare_memory_usage()
    sizes = [
        (1000, 5000),
        (5000, 10000),
        (10000, 50000)
    ]

    println("数据规模 (样本 × 标记) | Float64 | CompactGenotypes | 节省")
    println("-" ^ 70)

    for (n, m) in sizes
        data = rand(0:2, n, m)

        # Float64 存储
        mem_float64 = sizeof(Float64) * n * m

        # CompactGenotypes 存储
        geno = CompactGenotypes(data,
                               ["S$i" for i in 1:n],
                               ["M$i" for i in 1:m])
        mem_usage = memory_usage(geno)

        savings = (1 - mem_usage.total / mem_float64) * 100

        @printf("%6d × %6d | %8.2f MB | %8.2f MB | %5.1f%%\n",
                n, m,
                mem_float64 / 1e6,
                mem_usage.total / 1e6,
                savings)
    end
end

compare_memory_usage()
```

### 计算性能测试

```julia
# 测试等位基因频率计算速度
function benchmark_allele_frequencies()
    data = rand(0:2, 10000, 50000)
    geno = CompactGenotypes(data,
                           ["S$i" for i in 1:10000],
                           ["M$i" for i in 1:50000])

    println("等位基因频率计算性能:")
    @btime allele_frequencies($geno)
end

benchmark_allele_frequencies()
```

---

## 🐛 故障排查

### 常见问题

#### 1. 维度不匹配错误

```julia
# 错误示例
data = rand(0:2, 100, 1000)
sample_ids = ["S$i" for i in 1:100]
marker_ids = ["M$i" for i in 1:999]  # ❌ 数量不匹配

# 错误信息: DimensionMismatchError

# 解决方法: 确保 ID 数量匹配
marker_ids = ["M$i" for i in 1:1000]  # ✓ 正确
```

#### 2. 无效的基因型值

```julia
# 错误示例
data = rand(0:3, 100, 1000)  # ❌ 包含值 3

# 错误信息: DataValidationError: Invalid genotype value

# 解决方法: 确保所有值都是 0, 1, 2 或 missing
data = rand(0:2, 100, 1000)  # ✓ 正确
```

#### 3. 重复的 ID

```julia
# 错误示例
sample_ids = ["S1", "S2", "S1"]  # ❌ S1 重复

# 验证会失败
geno = CompactGenotypes(data, sample_ids, marker_ids)
result = validate(geno)
# result.valid == false
# result.errors 包含 "Sample IDs are not unique"

# 解决方法: 使用唯一的 ID
sample_ids = ["S1", "S2", "S3"]  # ✓ 正确
```

---

## 📚 下一步学习

### Phase 1 完成后的功能

当 Phase 1 完全实现后，你将能够：

1. **文件 I/O**: 读取 VCF, PLINK, HDF5 格式
2. **GRM 计算**: 计算基因组关系矩阵
3. **GBLUP**: 基因组最佳线性无偏预测
4. **Pipeline**: 完整的预测工作流

### 示例: 完整的 GBLUP 工作流（即将支持）

```julia
using GenomicPro2

# 1. 加载数据
geno = read_genotypes("data.vcf.gz")
pheno = read_phenotypes("phenotypes.csv")

# 2. 质量控制
geno_qc, pheno_qc = quality_control(geno, pheno;
                                    max_missing = 0.1,
                                    min_maf = 0.01)

# 3. 计算 GRM
G = compute_grm(geno_qc; method = :vanraden)

# 4. 训练模型
model = GBLUPModel()
fit!(model, geno_qc, pheno_qc; G = G)

# 5. 预测育种值
gebv = predict(model, geno_qc)

# 6. 交叉验证
cv_results = cross_validate(model, geno_qc, pheno_qc; folds = 5)
println("预测准确性: ", cv_results.accuracy)

# 7. 保存结果
save_model(model, "gblup_model.jld2")
save_predictions(gebv, "breeding_values.csv")
```

---

## 📞 获取帮助

### 文档资源

- **架构设计**: `/home/user/Julia/GenomicPro_2.0_Architecture_Design.md`
- **代码示例**: `/home/user/Julia/GenomicPro_2.0_Code_Examples.md`
- **性能工程**: `/home/user/Julia/GenomicPro_2.0_Performance_Engineering.md`
- **高级架构**: `/home/user/Julia/GenomicPro_2.0_Advanced_Architecture.md`

### 在线资源

- **Julia 官方文档**: https://docs.julialang.org/
- **LinearAlgebra 文档**: https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
- **性能优化指南**: https://docs.julialang.org/en/v1/manual/performance-tips/

---

## ✅ 检查清单

开始使用 GenomicPro2 之前，确保：

- [ ] Julia 1.10 或更高版本已安装
- [ ] GenomicPro2 包已添加到项目
- [ ] 能够成功 `using GenomicPro2`
- [ ] 了解基因型数据的格式 (0/1/2 编码)
- [ ] 准备好样本和标记的 ID 列表
- [ ] 阅读了基础使用示例

---

**文档更新**: 2025-11-15
**GenomicPro2 版本**: 2.0.0-dev
**Julia 要求**: ≥ 1.10

祝你使用愉快！如有问题，请参考设计文档或提交 Issue。
