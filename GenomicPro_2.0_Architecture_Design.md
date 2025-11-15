# GenomicPro 2.0 架构设计文档

**版本**: 2.0.0
**设计日期**: 2025-11-15
**状态**: 架构设计阶段

---

## 目录

1. [设计原则](#1-设计原则)
2. [系统架构概览](#2-系统架构概览)
3. [分层架构设计](#3-分层架构设计)
4. [核心模块设计](#4-核心模块设计)
5. [数据流和交互](#5-数据流和交互)
6. [性能优化策略](#6-性能优化策略)
7. [错误处理和日志](#7-错误处理和日志)
8. [测试策略](#8-测试策略)
9. [部署和扩展](#9-部署和扩展)
10. [迁移路线图](#10-迁移路线图)

---

## 1. 设计原则

### 1.1 核心原则

**SOLID 原则**
- **S** - Single Responsibility: 每个模块只负责一个功能
- **O** - Open/Closed: 对扩展开放，对修改封闭
- **L** - Liskov Substitution: 抽象类型可互换
- **I** - Interface Segregation: 小而专注的接口
- **D** - Dependency Inversion: 依赖抽象而非具体实现

**Julia 最佳实践**
- 类型稳定性优先
- 利用多重派发
- 避免类型联合（Union{T, Nothing}）
- 使用参数化类型
- 零成本抽象

### 1.2 设计目标

| 目标 | 指标 | 实现策略 |
|------|------|----------|
| **正确性** | 100% 核心算法测试覆盖 | 单元测试 + 数值验证 |
| **性能** | 比 v1.0 快 10x | GPU 加速 + 算法优化 |
| **可扩展性** | 支持 100 万+ SNP | 分块计算 + 稀疏矩阵 |
| **可维护性** | 模块耦合度 < 20% | 接口抽象 + 依赖注入 |
| **易用性** | 5 行代码完成基本分析 | 高层 API + 合理默认值 |

---

## 2. 系统架构概览

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        GenomicPro 2.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               用户接口层 (User Interface)                │   │
│  ├─────────────┬──────────────┬──────────────┬─────────────┤   │
│  │  CLI/REPL   │  Jupyter API │   Web API    │  Python Bridge│  │
│  └─────────────┴──────────────┴──────────────┴─────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            工作流引擎 (Workflow Engine)                  │   │
│  ├─────────────┬──────────────┬──────────────┬─────────────┤   │
│  │  Pipeline   │  配置管理     │   任务调度    │  结果管理   │   │
│  └─────────────┴──────────────┴──────────────┴─────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              算法层 (Algorithm Layer)                    │   │
│  ├────────┬─────────┬─────────┬──────────┬─────────────────┤   │
│  │ GBLUP  │ BayesR  │  SSGBLUP│  DL Models│  Multi-trait   │   │
│  └────────┴─────────┴─────────┴──────────┴─────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            核心计算层 (Core Compute)                      │   │
│  ├─────────────┬──────────────┬──────────────┬─────────────┤   │
│  │  线性代数   │   统计计算    │   优化器      │  MCMC引擎  │   │
│  └─────────────┴──────────────┴──────────────┴─────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              数据层 (Data Layer)                         │   │
│  ├──────────┬───────────┬───────────┬──────────┬───────────┤   │
│  │ Genotype │ Phenotype │ Pedigree  │ Annotations│ Omics   │   │
│  └──────────┴───────────┴───────────┴──────────┴───────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            存储层 (Storage Layer)                        │   │
│  ├──────────┬───────────┬───────────┬──────────┬───────────┤   │
│  │  VCF/BCF │  PLINK    │   HDF5    │   Arrow  │   Cloud   │   │
│  └──────────┴───────────┴───────────┴──────────┴───────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          计算后端层 (Compute Backend)                     │   │
│  ├──────────┬───────────┬───────────┬──────────┬───────────┤   │
│  │  CPU     │  GPU      │ Multi-GPU │ Distributed│ Cloud   │   │
│  └──────────┴───────────┴───────────┴──────────┴───────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计模式

1. **分层架构 (Layered Architecture)**
   - 每层只依赖下层
   - 上层对下层透明

2. **策略模式 (Strategy Pattern)**
   - 算法可插拔
   - 运行时切换

3. **工厂模式 (Factory Pattern)**
   - 统一对象创建
   - 隐藏实现细节

4. **观察者模式 (Observer Pattern)**
   - 进度监控
   - 事件通知

5. **适配器模式 (Adapter Pattern)**
   - 统一数据格式接口
   - 兼容多种文件格式

---

## 3. 分层架构设计

### 3.1 数据层 (Data Layer)

#### 设计目标
- 统一的数据访问接口
- 延迟加载（Lazy Loading）
- 内存映射大文件
- 类型安全

#### 核心接口

```julia
# 抽象接口
abstract type AbstractGenomicData{T} end
abstract type AbstractGenotypeData{T} <: AbstractGenomicData{T} end
abstract type AbstractPhenotypeData{T} <: AbstractGenomicData{T} end
abstract type AbstractPedigreeData{T} <: AbstractGenomicData{T} end

# 必须实现的接口
Base.size(::AbstractGenomicData)
Base.getindex(::AbstractGenomicData, i...)
validate(::AbstractGenomicData) -> ValidationResult
sample_ids(::AbstractGenomicData) -> Vector{String}
```

#### 具体实现

**1. 基因型数据**

```julia
# 内存高效的基因型存储
struct CompactGenotypes{T<:Integer} <: AbstractGenotypeData{T}
    data::Vector{UInt8}        # 2-bit 编码
    n_samples::Int
    n_markers::Int
    sample_ids::Vector{String}
    marker_ids::Vector{String}
    allele_freqs::Vector{Float64}
    missing_mask::BitMatrix    # 缺失值标记

    # 元数据
    chromosome::Vector{String}
    position::Vector{Int}
    ref_allele::Vector{String}
    alt_allele::Vector{String}
end

# 稀疏基因型（用于插补数据）
struct SparseGenotypes{T<:AbstractFloat} <: AbstractGenotypeData{T}
    data::SparseMatrixCSC{T, Int}
    sample_ids::Vector{String}
    marker_ids::Vector{String}
    quality_scores::Union{SparseMatrixCSC{T, Int}, Nothing}
end

# 大规模数据的内存映射版本
struct MappedGenotypes{T} <: AbstractGenotypeData{T}
    filepath::String
    mmap::Vector{UInt8}
    metadata::GenotypeMetadata
    chunk_size::Int
end
```

**2. 表型数据**

```julia
struct PhenotypeData{T<:Real} <: AbstractPhenotypeData{T}
    data::DataFrame
    sample_ids::Vector{String}
    trait_names::Vector{Symbol}
    fixed_effects::Vector{Symbol}
    random_effects::Vector{Symbol}
    weights::Union{Vector{T}, Nothing}

    # 验证器
    validators::Vector{<:AbstractValidator}
end
```

**3. 系谱数据**

```julia
struct PedigreeData <: AbstractPedigreeData{String}
    id::Vector{String}
    sire::Vector{Union{String, Missing}}
    dam::Vector{Union{String, Missing}}

    # 预计算的矩阵（缓存）
    A_inv::Union{SparseMatrixCSC{Float64, Int}, Nothing}
    inbreeding::Union{Vector{Float64}, Nothing}

    # 索引加速
    id_map::Dict{String, Int}
end
```

#### 数据验证框架

```julia
abstract type AbstractValidator end

struct ValidationResult
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
    metadata::Dict{Symbol, Any}
end

# 具体验证器
struct IDConsistencyValidator <: AbstractValidator
    strict::Bool
end

struct MissingDataValidator <: AbstractValidator
    max_missing_rate::Float64
end

struct AlleleFrequencyValidator <: AbstractValidator
    min_maf::Float64
    max_maf::Float64
end

# 验证函数
function validate(data::AbstractGenomicData, validators::Vector{<:AbstractValidator})
    results = ValidationResult[]
    for validator in validators
        push!(results, validate(data, validator))
    end
    return merge(results)
end
```

---

### 3.2 核心计算层 (Core Compute Layer)

#### 设计目标
- 数值稳定性
- 性能优化
- 设备抽象（CPU/GPU）
- 可测试性

#### 计算后端抽象

```julia
abstract type ComputeBackend end

struct CPUBackend <: ComputeBackend
    num_threads::Int
end

struct GPUBackend{T} <: ComputeBackend
    device::T  # CuDevice
    stream::T  # CuStream
end

struct DistributedBackend <: ComputeBackend
    workers::Vector{Int}
end

# 设备管理
struct ComputeContext{B<:ComputeBackend}
    backend::B
    memory_pool::MemoryPool
    profiler::Union{Profiler, Nothing}
end

# 统一接口
function allocate(ctx::ComputeContext, dims...; dtype=Float64)
    # 根据后端分配内存
end

function compute!(result, op::AbstractOperation, ctx::ComputeContext)
    # 根据后端执行计算
end
```

#### 线性代数核心

```julia
# 关系矩阵计算
abstract type AbstractRelationshipMatrix end

struct GRM{T<:AbstractFloat} <: AbstractRelationshipMatrix
    matrix::AbstractMatrix{T}
    method::Symbol  # :vanraden, :astle_balding, :robust
    scaled::Bool
end

# 高性能 GRM 计算
function compute_grm(
    genotypes::AbstractGenotypeData,
    backend::ComputeBackend;
    method::Symbol = :vanraden,
    min_maf::Float64 = 0.01,
    center::Bool = true,
    scale::Bool = true,
    chunk_size::Int = 1000,
    progress::Bool = true
) -> GRM
    # 参数验证
    @argcheck method in [:vanraden, :astle_balding, :robust]
    @argcheck 0 < min_maf < 0.5
    @argcheck chunk_size > 0

    # 根据后端选择实现
    if backend isa GPUBackend
        return compute_grm_gpu(genotypes, method, chunk_size)
    elseif backend isa CPUBackend
        return compute_grm_cpu(genotypes, method, chunk_size, backend.num_threads)
    else
        return compute_grm_distributed(genotypes, method, backend.workers)
    end
end

# CPU 实现（优化版本）
function compute_grm_cpu(
    geno::AbstractGenotypeData,
    method::Symbol,
    chunk_size::Int,
    nthreads::Int
)
    n = n_samples(geno)
    m = n_markers(geno)

    # 预分配结果矩阵
    G = zeros(Float64, n, n)

    # 计算等位基因频率（Kahan求和）
    freqs = compute_allele_frequencies(geno, :kahan)

    # 过滤低频标记
    valid_markers = freqs .>= min_maf .&& freqs .<= (1 - min_maf)

    # 分块计算（减少内存占用）
    @threads for chunk_start in 1:chunk_size:m
        chunk_end = min(chunk_start + chunk_size - 1, m)

        # 提取并标准化
        Z = extract_and_standardize(geno, chunk_start:chunk_end, freqs)

        # 累加到 G
        BLAS.syrk!('U', 'N', 1.0, Z, 1.0, G)
    end

    # 对称化
    copytri!(G, 'U')

    # 缩放
    if method == :vanraden
        G ./= sum(2 .* freqs .* (1 .- freqs))
    elseif method == :astle_balding
        # ... 其他方法
    end

    return GRM(G, method, true)
end

# GPU 实现
function compute_grm_gpu(geno::AbstractGenotypeData, method::Symbol, chunk_size::Int)
    # CUDA 核函数实现
    # ...
end
```

#### 求解器框架

```julia
abstract type AbstractSolver end
abstract type LinearSolver <: AbstractSolver end
abstract type IterativeSolver <: LinearSolver end

# 直接求解器
struct CholeskySolver <: LinearSolver
    pivoting::Bool
    tolerance::Float64
end

struct LUSolver <: LinearSolver
    pivoting::Bool
end

# 迭代求解器
struct PCGSolver <: IterativeSolver
    preconditioner::Symbol  # :jacobi, :ichol, :ilu
    max_iterations::Int
    tolerance::Float64
    restart::Int
end

struct GMRESSolver <: IterativeSolver
    preconditioner::Symbol
    max_iterations::Int
    tolerance::Float64
    restart::Int
end

# 统一求解接口
function solve(
    A::AbstractMatrix,
    b::AbstractVector,
    solver::AbstractSolver,
    backend::ComputeBackend
) -> SolverResult
    # 参数验证
    @argcheck size(A, 1) == size(A, 2)
    @argcheck size(A, 1) == length(b)

    # 分发到具体实现
    return solve_impl(A, b, solver, backend)
end

struct SolverResult{T}
    solution::Vector{T}
    converged::Bool
    iterations::Int
    residual_norm::T
    solve_time::Float64
    memory_used::Int
end
```

---

### 3.3 算法层 (Algorithm Layer)

#### 设计目标
- 模块化算法实现
- 统一的训练/预测接口
- 超参数管理
- 模型序列化

#### 算法基类

```julia
abstract type AbstractGenomicModel end

# 所有模型必须实现的接口
function fit!(
    model::AbstractGenomicModel,
    genotypes::AbstractGenotypeData,
    phenotypes::AbstractPhenotypeData;
    kwargs...
)
    # 训练模型
end

function predict(
    model::AbstractGenomicModel,
    genotypes::AbstractGenotypeData
)
    # 预测
end

function cross_validate(
    model::AbstractGenomicModel,
    data::GenomicDataset;
    folds::Int = 5,
    metric::Symbol = :correlation
)
    # 交叉验证
end

function save_model(model::AbstractGenomicModel, filepath::String)
    # 序列化模型
end

function load_model(filepath::String) -> AbstractGenomicModel
    # 反序列化
end
```

#### GBLUP 实现

```julia
struct GBLUPModel{T<:AbstractFloat} <: AbstractGenomicModel
    # 模型参数
    fixed_effects::Vector{T}
    breeding_values::Vector{T}

    # 方差组分
    σ²_g::T  # 遗传方差
    σ²_e::T  # 残差方差

    # 关系矩阵
    G::Union{GRM, Nothing}

    # 配置
    config::GBLUPConfig

    # 元数据
    sample_ids::Vector{String}
    convergence_info::Union{ConvergenceInfo, Nothing}
end

struct GBLUPConfig
    solver::AbstractSolver
    variance_estimator::Symbol  # :reml, :ml, :minque
    compute_se::Bool
    backend::ComputeBackend
end

function fit!(
    model::GBLUPModel,
    geno::AbstractGenotypeData,
    pheno::AbstractPhenotypeData;
    G::Union{GRM, Nothing} = nothing,
    progress::Bool = true
)
    # 1. 数据验证
    validate_data_consistency(geno, pheno)

    # 2. 构建 G 矩阵（如果未提供）
    if isnothing(G)
        G = compute_grm(geno, model.config.backend)
    end

    # 3. 估计方差组分
    σ²_g, σ²_e = estimate_variance_components(
        pheno,
        G,
        model.config.variance_estimator
    )

    # 4. 求解混合模型方程
    λ = σ²_e / σ²_g
    result = solve_mme(pheno, G, λ, model.config.solver)

    # 5. 更新模型
    model.breeding_values = result.solution
    model.σ²_g = σ²_g
    model.σ²_e = σ²_e
    model.G = G
    model.convergence_info = result.convergence

    return model
end
```

#### BayesR 实现

```julia
struct BayesRModel{T<:AbstractFloat} <: AbstractGenomicModel
    # 标记效应
    marker_effects::Vector{T}
    marker_variances::Vector{T}

    # 混合比例
    π::Vector{T}  # [π₀, π₁, π₂, π₃]

    # 方差组分
    σ²::Vector{T}  # [0, σ₁², σ₂², σ₃²]
    σ²_e::T

    # MCMC 样本
    samples::Union{MCMCSamples, Nothing}

    # 配置
    config::BayesRConfig
end

struct BayesRConfig
    num_iterations::Int
    burn_in::Int
    thin::Int
    variance_components::Vector{Float64}  # [0.0, 0.0001, 0.001, 0.01]
    update_variance::Bool
    backend::ComputeBackend
    seed::Union{Int, Nothing}
end

struct MCMCSamples{T}
    marker_effects::Matrix{T}  # iterations × markers
    π::Matrix{T}
    σ²::Matrix{T}
    σ²_e::Vector{T}

    # 诊断信息
    acceptance_rate::Vector{T}
    ess::Vector{T}  # Effective sample size
    rhat::Vector{T}  # Gelman-Rubin statistic
end

function fit!(
    model::BayesRModel,
    geno::AbstractGenotypeData,
    pheno::AbstractPhenotypeData;
    progress::Bool = true
)
    # MCMC 采样
    cfg = model.config
    n_iter = cfg.num_iterations

    # 初始化
    X = to_matrix(geno)
    y = get_phenotypes(pheno)
    n, m = size(X)

    # 存储样本
    β_samples = zeros(n_iter, m)
    π_samples = zeros(n_iter, 4)
    σ²_samples = zeros(n_iter, 4)
    σ²_e_samples = zeros(n_iter)

    # Gibbs 采样
    prog = Progress(n_iter; enabled=progress)

    for iter in 1:n_iter
        # 1. 采样标记效应
        for j in 1:m
            β_samples[iter, j] = sample_marker_effect(j, X, y, ...)
        end

        # 2. 采样混合比例
        π_samples[iter, :] = sample_mixture_proportions(...)

        # 3. 采样方差组分
        if cfg.update_variance
            σ²_samples[iter, :] = sample_variances(...)
        end

        # 4. 采样残差方差
        σ²_e_samples[iter] = sample_residual_variance(...)

        next!(prog)
    end

    # 后处理
    model.samples = MCMCSamples(
        β_samples[(cfg.burn_in+1):cfg.thin:end, :],
        π_samples[(cfg.burn_in+1):cfg.thin:end, :],
        σ²_samples[(cfg.burn_in+1):cfg.thin:end, :],
        σ²_e_samples[(cfg.burn_in+1):cfg.thin:end]
    )

    # 计算后验均值
    model.marker_effects = mean(model.samples.marker_effects, dims=1)[:]
    model.π = mean(model.samples.π, dims=1)[:]

    # MCMC 诊断
    compute_diagnostics!(model.samples)

    return model
end
```

#### 深度学习模型

```julia
struct DeepGBLUPModel <: AbstractGenomicModel
    # 神经网络
    network::Lux.AbstractExplicitLayer
    parameters::ComponentArray
    states::NamedTuple

    # GBLUP 组件
    breeding_values::Vector{Float64}
    G::Union{GRM, Nothing}

    # 配置
    config::DeepGBLUPConfig

    # 训练历史
    training_history::Union{TrainingHistory, Nothing}
end

struct DeepGBLUPConfig
    # 网络架构
    hidden_layers::Vector{Int}
    activation::Function
    dropout_rate::Float64

    # 训练参数
    optimizer::Optimisers.AbstractRule
    batch_size::Int
    epochs::Int
    early_stopping::Bool
    patience::Int

    # 计算后端
    backend::ComputeBackend
end

function build_network(input_dim::Int, config::DeepGBLUPConfig)
    layers = []

    # 输入层
    push!(layers, Dense(input_dim, config.hidden_layers[1], config.activation))
    push!(layers, Dropout(config.dropout_rate))

    # 隐藏层
    for i in 1:(length(config.hidden_layers)-1)
        push!(layers, Dense(
            config.hidden_layers[i],
            config.hidden_layers[i+1],
            config.activation
        ))
        push!(layers, Dropout(config.dropout_rate))
    end

    # 输出层
    push!(layers, Dense(config.hidden_layers[end], 1))

    return Chain(layers...)
end
```

---

### 3.4 工作流引擎 (Workflow Engine)

#### 设计目标
- 声明式配置
- 自动化流程
- 断点续跑
- 结果追踪

#### Pipeline 框架

```julia
abstract type AbstractPipeline end

struct GenomicPredictionPipeline <: AbstractPipeline
    name::String
    stages::Vector{PipelineStage}
    config::PipelineConfig
    state::PipelineState
end

struct PipelineStage
    name::String
    operation::Function
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    dependencies::Vector{Symbol}
    cacheable::Bool
end

struct PipelineConfig
    # 数据路径
    genotype_file::String
    phenotype_file::String
    output_dir::String

    # QC 参数
    qc_params::QCParams

    # 模型参数
    model_type::Symbol
    model_params::Dict{Symbol, Any}

    # 计算资源
    backend::ComputeBackend
    num_workers::Int

    # 其他
    seed::Int
    verbose::Bool
end

# 定义完整的预测流程
function create_prediction_pipeline(config::PipelineConfig)
    stages = [
        PipelineStage(
            "load_data",
            load_genomic_data,
            [:genotype_file, :phenotype_file],
            [:genotypes, :phenotypes],
            Symbol[],
            true
        ),

        PipelineStage(
            "quality_control",
            apply_quality_control,
            [:genotypes, :phenotypes, :qc_params],
            [:genotypes_qc, :phenotypes_qc, :qc_report],
            [:load_data],
            true
        ),

        PipelineStage(
            "compute_grm",
            compute_relationship_matrix,
            [:genotypes_qc],
            [:G],
            [:quality_control],
            true
        ),

        PipelineStage(
            "train_model",
            train_genomic_model,
            [:genotypes_qc, :phenotypes_qc, :G, :model_params],
            [:model, :training_report],
            [:compute_grm],
            false
        ),

        PipelineStage(
            "cross_validate",
            perform_cross_validation,
            [:genotypes_qc, :phenotypes_qc, :model_params],
            [:cv_results],
            [:quality_control],
            false
        ),

        PipelineStage(
            "generate_report",
            create_analysis_report,
            [:model, :training_report, :cv_results, :qc_report],
            [:final_report],
            [:train_model, :cross_validate],
            false
        )
    ]

    return GenomicPredictionPipeline(
        "genomic_prediction",
        stages,
        config,
        PipelineState()
    )
end

# 执行 pipeline
function run!(pipeline::GenomicPredictionPipeline)
    # 检查依赖
    validate_dependencies(pipeline)

    # 创建输出目录
    mkpath(pipeline.config.output_dir)

    # 执行每个阶段
    for stage in pipeline.stages
        if should_run_stage(stage, pipeline.state)
            @info "Running stage: $(stage.name)"

            # 准备输入
            inputs = prepare_inputs(stage, pipeline.state)

            # 执行
            outputs = try
                stage.operation(inputs...)
            catch e
                @error "Stage $(stage.name) failed" exception=e
                rethrow(e)
            end

            # 保存输出
            save_outputs(stage, outputs, pipeline.config.output_dir)

            # 更新状态
            update_state!(pipeline.state, stage, outputs)
        else
            @info "Skipping stage: $(stage.name) (already completed)"
        end
    end

    return pipeline
end
```

---

## 4. 核心模块设计

### 4.1 模块组织结构

```
GenomicPro2.jl/
├── src/
│   ├── GenomicPro2.jl                    # 主模块
│   │
│   ├── Core/                              # 核心模块
│   │   ├── Core.jl
│   │   ├── types.jl                       # 类型定义
│   │   ├── interfaces.jl                  # 接口定义
│   │   ├── validation.jl                  # 数据验证
│   │   └── errors.jl                      # 自定义异常
│   │
│   ├── Data/                              # 数据层
│   │   ├── Data.jl
│   │   ├── genotypes.jl                   # 基因型数据
│   │   ├── phenotypes.jl                  # 表型数据
│   │   ├── pedigrees.jl                   # 系谱数据
│   │   ├── annotations.jl                 # 注释数据
│   │   └── datasets.jl                    # 数据集容器
│   │
│   ├── IO/                                # 输入输出
│   │   ├── IO.jl
│   │   ├── vcf.jl                         # VCF/BCF 读取
│   │   ├── plink.jl                       # PLINK 格式
│   │   ├── hdf5.jl                        # HDF5 格式
│   │   └── serialization.jl               # 序列化
│   │
│   ├── QualityControl/                    # 质量控制
│   │   ├── QualityControl.jl
│   │   ├── filters.jl                     # 过滤器
│   │   ├── validators.jl                  # 验证器
│   │   ├── imputation.jl                  # 插补
│   │   └── phasing.jl                     # 定相
│   │
│   ├── LinearAlgebra/                     # 线性代数
│   │   ├── LinearAlgebra.jl
│   │   ├── grm.jl                         # 关系矩阵
│   │   ├── solvers.jl                     # 求解器
│   │   ├── decompositions.jl              # 矩阵分解
│   │   └── sparse.jl                      # 稀疏矩阵操作
│   │
│   ├── Statistics/                        # 统计计算
│   │   ├── Statistics.jl
│   │   ├── distributions.jl               # 分布函数
│   │   ├── variance_components.jl         # 方差组分
│   │   └── hypothesis_tests.jl            # 假设检验
│   │
│   ├── Models/                            # 模型层
│   │   ├── Models.jl
│   │   ├── base.jl                        # 基类和接口
│   │   ├── gblup.jl                       # GBLUP
│   │   ├── ssgblup.jl                     # ssGBLUP
│   │   ├── bayesian/                      # 贝叶斯方法
│   │   │   ├── bayesr.jl
│   │   │   ├── bayesrc.jl
│   │   │   ├── bayesa.jl
│   │   │   └── mcmc.jl
│   │   ├── deep_learning/                 # 深度学习
│   │   │   ├── deep_gblup.jl
│   │   │   ├── cnn.jl
│   │   │   ├── transformer.jl
│   │   │   └── layers.jl
│   │   └── ensemble/                      # 集成方法
│   │       ├── stacking.jl
│   │       └── boosting.jl
│   │
│   ├── MultiTrait/                        # 多性状分析
│   │   ├── MultiTrait.jl
│   │   ├── mtgblup.jl
│   │   └── selection_index.jl
│   │
│   ├── MultiOmics/                        # 多组学
│   │   ├── MultiOmics.jl
│   │   ├── integration.jl
│   │   └── fusion.jl
│   │
│   ├── Workflows/                         # 工作流
│   │   ├── Workflows.jl
│   │   ├── pipeline.jl
│   │   ├── tasks.jl
│   │   └── dag.jl
│   │
│   ├── Backends/                          # 计算后端
│   │   ├── Backends.jl
│   │   ├── cpu.jl
│   │   ├── gpu.jl
│   │   ├── distributed.jl
│   │   └── cloud.jl
│   │
│   ├── Validation/                        # 模型验证
│   │   ├── Validation.jl
│   │   ├── cross_validation.jl
│   │   └── metrics.jl
│   │
│   ├── Utils/                             # 工具
│   │   ├── Utils.jl
│   │   ├── logging.jl
│   │   ├── config.jl
│   │   └── profiling.jl
│   │
│   └── API/                               # 用户接口
│       ├── API.jl
│       ├── high_level.jl                  # 高层 API
│       └── cli.jl                         # 命令行接口
│
├── test/                                   # 测试
│   ├── runtests.jl
│   ├── data/                              # 测试数据
│   ├── unit/                              # 单元测试
│   ├── integration/                       # 集成测试
│   └── performance/                       # 性能测试
│
├── examples/                              # 示例
│   ├── 01_basic_gblup.jl
│   ├── 02_bayesian_methods.jl
│   ├── 03_deep_learning.jl
│   ├── 04_multi_trait.jl
│   └── 05_production_pipeline.jl
│
├── docs/                                  # 文档
│   ├── make.jl
│   └── src/
│       ├── index.md
│       ├── tutorials/
│       ├── manual/
│       └── api/
│
├── Project.toml
└── README.md
```

### 4.2 依赖管理

```toml
# Project.toml
[deps]
# 核心依赖
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

# 数据处理
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"

# 文件 I/O
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
CodecZlib = "944b1d66-785c-5afd-91f1-9de20f533193"

# 统计和优化
Distributions = "31c24e10-a181-5473-b8eb-7853c613294"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"

# 深度学习
Lux = "b2108857-7c20-4412-a14c-986f62d294a5"
Optimisers = "3a2145e1-a3c6-433b-a48e-927c1f82e88a"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

# GPU 支持（可选）
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

# 并行计算
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
ThreadsX = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"

# 工具
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
ArgParse = "c7e460c6-2fb1-5374-8c6c-847e6a715a3a"

# 验证和测试
ArgCheck = "dce04be8-c92d-5529-be00-80e4d2c0e197"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
julia = "1.10"
```

---

## 5. 数据流和交互

### 5.1 典型数据流

```
用户输入
   ↓
┌──────────────────────────────────────┐
│  1. 数据加载                          │
│  - 读取基因型 (VCF/PLINK/HDF5)       │
│  - 读取表型 (CSV/DataFrame)          │
│  - 读取系谱 (可选)                   │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  2. 数据验证                          │
│  - ID 一致性检查                     │
│  - 数据类型验证                      │
│  - 完整性检查                        │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  3. 质量控制                          │
│  - 缺失率过滤                        │
│  - MAF 过滤                          │
│  - HWE 检验                          │
│  - 样本 QC                           │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  4. 特征工程                          │
│  - 计算关系矩阵                      │
│  - 标准化/中心化                     │
│  - 降维 (PCA)                        │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  5. 模型训练                          │
│  - 方差组分估计                      │
│  - 参数估计                          │
│  - 超参数调优                        │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  6. 模型验证                          │
│  - 交叉验证                          │
│  - 预测准确性评估                    │
│  - 模型诊断                          │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  7. 结果输出                          │
│  - 育种值预测                        │
│  - 模型参数保存                      │
│  - 报告生成                          │
└──────────────────────────────────────┘
   ↓
用户输出
```

### 5.2 API 示例

#### 基础 API

```julia
using GenomicPro2

# 1. 加载数据
geno = read_genotypes("data.vcf.gz")
pheno = read_phenotypes("phenotypes.csv")

# 2. 质量控制
geno_qc, pheno_qc = quality_control(
    geno, pheno;
    max_missing_geno = 0.1,
    max_missing_sample = 0.1,
    min_maf = 0.01
)

# 3. 计算 G 矩阵
G = compute_grm(geno_qc; method=:vanraden)

# 4. 训练模型
model = GBLUP()
fit!(model, geno_qc, pheno_qc; G=G)

# 5. 预测
gebv = predict(model, geno_qc)

# 6. 评估
accuracy = cross_validate(model, geno_qc, pheno_qc; folds=5)
println("预测准确性: $(accuracy)")
```

#### 高级 API（Pipeline）

```julia
using GenomicPro2

# 定义配置
config = PipelineConfig(
    genotype_file = "data.vcf.gz",
    phenotype_file = "phenotypes.csv",
    output_dir = "results/",

    qc_params = QCParams(
        max_missing_geno = 0.1,
        min_maf = 0.01
    ),

    model_type = :bayesr,
    model_params = Dict(
        :num_iterations => 50000,
        :burn_in => 10000,
        :thin => 10
    ),

    backend = GPUBackend(),
    verbose = true
)

# 创建并运行 pipeline
pipeline = create_prediction_pipeline(config)
run!(pipeline)

# 查看结果
report = load_report(joinpath(config.output_dir, "report.html"))
```

---

## 6. 性能优化策略

### 6.1 算法优化

1. **避免矩阵求逆**
   - 使用 Cholesky 分解
   - 直接求解线性方程组
   - 利用稀疏性

2. **分块计算**
   - GRM 分块计算
   - 减少内存占用
   - 提高缓存命中率

3. **数值稳定性**
   - Kahan 求和
   - 对数空间计算
   - 避免下溢/上溢

### 6.2 并行化

1. **多线程**
```julia
using ThreadsX

# 多线程 GRM 计算
function compute_grm_parallel(geno, nthreads=Threads.nthreads())
    chunks = partition(1:n_markers(geno), nthreads)

    results = ThreadsX.map(chunks) do chunk
        compute_grm_chunk(geno, chunk)
    end

    return reduce(+, results)
end
```

2. **GPU 加速**
```julia
using CUDA

function compute_grm_gpu(geno::AbstractGenotypeData)
    # 转移到 GPU
    X_gpu = CuArray(to_matrix(geno))

    # 标准化
    μ = mean(X_gpu, dims=1)
    X_centered = X_gpu .- μ

    # 计算 G = X'X
    G = X_centered' * X_centered

    # 转回 CPU
    return Array(G)
end
```

3. **分布式计算**
```julia
using Distributed

@everywhere function compute_partial_grm(geno_chunk)
    # 计算部分 GRM
end

# 分布式执行
results = pmap(chunks) do chunk
    compute_partial_grm(chunk)
end
```

### 6.3 内存优化

1. **内存映射**
```julia
using Mmap

struct MappedGenotypes
    data::Vector{UInt8}

    function MappedGenotypes(filepath::String)
        io = open(filepath, "r")
        data = Mmap.mmap(io, Vector{UInt8})
        new(data)
    end
end
```

2. **延迟加载**
```julia
struct LazyGenotypes
    loader::Function
    cache::Union{Matrix, Nothing}

    function Base.getindex(g::LazyGenotypes, i, j)
        if isnothing(g.cache)
            g.cache = g.loader()
        end
        return g.cache[i, j]
    end
end
```

### 6.4 性能监控

```julia
using Profile, BenchmarkTools

# 性能分析
@profview fit!(model, geno, pheno)

# 基准测试
@benchmark compute_grm($geno) samples=10 evals=3

# 内存分配跟踪
@allocated compute_grm(geno)
```

---

## 7. 错误处理和日志

### 7.1 自定义异常

```julia
# 核心异常类型
abstract type GenomicProException <: Exception end

struct DataValidationError <: GenomicProException
    msg::String
    field::Symbol
    value::Any
end

struct DimensionMismatchError <: GenomicProException
    expected::Tuple
    actual::Tuple
end

struct ConvergenceError <: GenomicProException
    msg::String
    iterations::Int
    residual::Float64
end

struct FileFormatError <: GenomicProException
    filepath::String
    expected_format::String
    msg::String
end

# 异常处理
Base.showerror(io::IO, e::DataValidationError) =
    print(io, "DataValidationError: $(e.msg)\n  Field: $(e.field)\n  Value: $(e.value)")
```

### 7.2 日志系统

```julia
using Logging

# 自定义 Logger
struct GenomicProLogger <: AbstractLogger
    level::LogLevel
    stream::IO
    show_progress::Bool
end

# 日志宏
macro loginfo(msg)
    quote
        @info $(esc(msg)) _module=@__MODULE__ _file=@__FILE__ _line=@__LINE__
    end
end

# 使用示例
function fit!(model::GBLUPModel, args...; kwargs...)
    @loginfo "开始训练 GBLUP 模型"
    @loginfo "样本数: $(n_samples(geno)), 标记数: $(n_markers(geno))"

    try
        # 训练逻辑
    catch e
        @logerror "模型训练失败" exception=(e, catch_backtrace())
        rethrow(e)
    end

    @loginfo "模型训练完成"
end
```

### 7.3 参数验证

```julia
using ArgCheck

function compute_grm(geno; min_maf=0.01, max_missing=0.1)
    # 参数验证
    @argcheck 0 < min_maf < 0.5 "min_maf 必须在 (0, 0.5) 范围内"
    @argcheck 0 <= max_missing <= 1 "max_missing 必须在 [0, 1] 范围内"
    @argcheck n_markers(geno) > 0 "基因型数据为空"

    # 实现...
end
```

---

## 8. 测试策略

### 8.1 测试层次

```julia
# 1. 单元测试
@testset "GRM Computation" begin
    geno = generate_test_genotypes(100, 1000)
    G = compute_grm(geno)

    @test size(G) == (100, 100)
    @test issymmetric(G)
    @test all(diag(G) .>= 0)
end

# 2. 数值精度测试
@testset "Numerical Accuracy" begin
    # 与 R/GCTA 结果对比
    geno = load_test_data("test_genotypes.vcf")
    G_julia = compute_grm(geno)
    G_reference = load("reference_grm.h5")

    @test isapprox(G_julia, G_reference; rtol=1e-10)
end

# 3. 性能测试
@testset "Performance" begin
    for n in [100, 1000, 10000]
        geno = generate_test_genotypes(n, 5000)

        t = @elapsed compute_grm(geno)
        @test t < expected_time(n)  # 性能回归检测
    end
end

# 4. 集成测试
@testset "End-to-End Pipeline" begin
    config = PipelineConfig(...)
    pipeline = create_prediction_pipeline(config)

    @test_nowarn run!(pipeline)
    @test isfile(joinpath(config.output_dir, "model.jls"))
end

# 5. 边界条件测试
@testset "Edge Cases" begin
    # 空数据
    @test_throws ArgumentError compute_grm(empty_genotypes())

    # 单样本
    geno = generate_test_genotypes(1, 100)
    @test size(compute_grm(geno)) == (1, 1)

    # 全缺失
    geno = missing_genotypes(100, 100)
    @test_throws DataValidationError compute_grm(geno)
end
```

### 8.2 测试数据生成

```julia
module TestData

using Random

function generate_test_genotypes(n_samples, n_markers; maf=0.3, seed=123)
    Random.seed!(seed)

    data = zeros(UInt8, n_samples, n_markers)
    for j in 1:n_markers
        p = rand() * (0.5 - maf) + maf  # MAF ∈ [maf, 0.5]
        for i in 1:n_samples
            data[i, j] = rand(Binomial(2, p))
        end
    end

    return CompactGenotypes(data)
end

function generate_test_phenotypes(n_samples, h²=0.5; seed=123)
    Random.seed!(seed)

    # 模拟遗传值
    g = randn(n_samples) * sqrt(h²)

    # 模拟环境效应
    e = randn(n_samples) * sqrt(1 - h²)

    # 表型 = 遗传值 + 环境
    y = g + e

    return PhenotypeData(
        DataFrame(id=string.(1:n_samples), phenotype=y),
        string.(1:n_samples),
        [:phenotype]
    )
end

end  # module
```

### 8.3 持续集成

```yaml
# .github/workflows/CI.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        julia-version: ['1.10', '1.11', 'nightly']

    steps:
      - uses: actions/checkout@v2

      - name: Setup Julia
        uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}

      - name: Install dependencies
        run: julia --project -e 'using Pkg; Pkg.instantiate()'

      - name: Run tests
        run: julia --project -e 'using Pkg; Pkg.test(coverage=true)'

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 9. 部署和扩展

### 9.1 云部署

```julia
# AWS 部署配置
struct AWSConfig
    region::String
    instance_type::String
    storage_bucket::String
    credentials::AWSCredentials
end

function deploy_to_cloud(pipeline::GenomicPredictionPipeline, config::AWSConfig)
    # 1. 上传数据到 S3
    upload_data_to_s3(pipeline.config.genotype_file, config.storage_bucket)

    # 2. 启动 EC2 实例
    instance = launch_ec2_instance(config.instance_type, config.region)

    # 3. 部署代码
    deploy_code(instance, get_repo_url())

    # 4. 提交任务
    submit_job(instance, pipeline.config)

    # 5. 监控进度
    monitor_job(instance)

    # 6. 下载结果
    download_results(instance, config.storage_bucket)
end
```

### 9.2 容器化

```dockerfile
# Dockerfile
FROM julia:1.10

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY Project.toml Manifest.toml ./
RUN julia --project -e 'using Pkg; Pkg.instantiate()'

COPY src/ src/
COPY examples/ examples/

# 设置入口点
ENTRYPOINT ["julia", "--project", "src/API/cli.jl"]
```

```bash
# 构建镜像
docker build -t genomicpro2:latest .

# 运行容器
docker run -v $(pwd)/data:/data \
           -v $(pwd)/results:/results \
           genomicpro2:latest \
           --genotype /data/genotypes.vcf.gz \
           --phenotype /data/phenotypes.csv \
           --output /results \
           --model bayesr
```

### 9.3 可扩展性

1. **插件系统**
```julia
# 注册自定义模型
abstract type AbstractCustomModel <: AbstractGenomicModel end

function register_model(model_type::Type{<:AbstractGenomicModel})
    MODEL_REGISTRY[nameof(model_type)] = model_type
end

# 用户自定义模型
struct MyCustomModel <: AbstractCustomModel
    # ...
end

function fit!(model::MyCustomModel, args...; kwargs...)
    # 自定义实现
end

# 注册
register_model(MyCustomModel)
```

2. **扩展接口**
```julia
# 允许用户扩展数据格式
function read_genotypes(filepath::String, format::Symbol)
    if format == :custom
        return read_custom_format(filepath)
    else
        # 默认实现
    end
end

# 用户实现
function read_custom_format(filepath::String)
    # 自定义读取逻辑
    return CompactGenotypes(...)
end
```

---

## 10. 迁移路线图

### 10.1 第一阶段（1-2个月）：核心重构

**目标**: 修复关键bug，建立稳定基础

- [ ] 实现新的数据层架构
  - [ ] CompactGenotypes with 2-bit encoding
  - [ ] PhenotypeData with validation
  - [ ] PedigreeData with efficient A-inverse
- [ ] 重写 GRM 计算
  - [ ] CPU 优化版本
  - [ ] 数值稳定性改进
  - [ ] 单元测试 (覆盖率 > 90%)
- [ ] 重写 GBLUP 求解器
  - [ ] 修复矩阵运算错误
  - [ ] PCG 优化实现
  - [ ] 方差组分估计
- [ ] 建立错误处理框架
- [ ] 建立日志系统
- [ ] CI/CD 配置

**交付物**:
- 可用的 GBLUP 实现
- 完整的单元测试
- 技术文档

### 10.2 第二阶段（2-3个月）：高级算法

**目标**: 实现贝叶斯和深度学习方法

- [ ] BayesR/BayesRC 实现
  - [ ] 高效 Gibbs 采样
  - [ ] MCMC 诊断
  - [ ] GPU 加速
- [ ] Deep GBLUP
  - [ ] CNN 架构
  - [ ] Transformer 架构
  - [ ] 训练优化
- [ ] ssGBLUP
  - [ ] H 矩阵构建
  - [ ] 高效求解
- [ ] 多性状分析
  - [ ] MT-GBLUP
  - [ ] 选择指数

**交付物**:
- 完整的算法库
- 示例代码
- 性能基准

### 10.3 第三阶段（2-3个月）：生产特性

**目标**: 工业级特性和优化

- [ ] GPU 全面支持
  - [ ] CUDA 内核优化
  - [ ] 多 GPU 并行
- [ ] 分布式计算
  - [ ] 数据并行
  - [ ] 模型并行
- [ ] 文件格式支持
  - [ ] VCF/BCF 读写
  - [ ] PLINK 格式
  - [ ] HDF5 优化
- [ ] Pipeline 系统
  - [ ] 配置管理
  - [ ] 任务调度
  - [ ] 结果追踪
- [ ] Web API
  - [ ] RESTful 接口
  - [ ] 在线预测服务

**交付物**:
- 生产级系统
- API 文档
- 部署指南

### 10.4 第四阶段（持续）：扩展和维护

**目标**: 社区建设和持续改进

- [ ] 文档完善
  - [ ] 教程
  - [ ] 最佳实践
  - [ ] 案例研究
- [ ] 社区支持
  - [ ] Issue 处理
  - [ ] PR 审查
  - [ ] 版本发布
- [ ] 性能优化
  - [ ] Profile 驱动优化
  - [ ] 算法改进
- [ ] 新功能
  - [ ] 多组学整合
  - [ ] 联邦学习
  - [ ] AutoML

---

## 11. 总结

### 11.1 关键改进

| 方面 | v1.0 问题 | v2.0 解决方案 |
|------|-----------|--------------|
| **架构** | 紧耦合，难扩展 | 分层架构，插件系统 |
| **正确性** | 多处 bug | 100% 测试覆盖，CI/CD |
| **性能** | 低效算法 | GPU 加速，并行化 |
| **易用性** | 接口混乱 | 统一 API，Pipeline |
| **文档** | 不完整 | 完整教程和示例 |

### 11.2 预期效果

- **性能**: 10-100x 提升（取决于问题规模和硬件）
- **可扩展性**: 支持百万级 SNP 和样本
- **可靠性**: 工业级质量
- **易用性**: 5 行代码完成基本分析
- **社区**: 活跃的开源社区

### 11.3 成功指标

- [ ] 测试覆盖率 > 90%
- [ ] 文档覆盖率 > 95%
- [ ] 性能达到 GCTA/BLUPF90 水平
- [ ] GitHub Stars > 500
- [ ] 至少 3 篇使用 GenomicPro2 的论文发表

---

**文档版本**: 1.0
**最后更新**: 2025-11-15
**维护者**: GenomicPro Development Team

