# GenomicPro 2.0 高级架构设计

**版本**: 2.0
**日期**: 2025-11-15
**类型**: 企业级/研究级架构设计

---

## 目录

1. [高级架构模式](#1-高级架构模式)
2. [依赖注入和控制反转](#2-依赖注入和控制反转)
3. [事件驱动架构](#3-事件驱动架构)
4. [插件生态系统](#4-插件生态系统)
5. [领域驱动设计](#5-领域驱动设计)
6. [CQRS和事件溯源](#6-cqrs和事件溯源)
7. [微服务架构支持](#7-微服务架构支持)
8. [响应式编程](#8-响应式编程)

---

## 1. 高级架构模式

### 1.1 六边形架构（Hexagonal Architecture）

将业务逻辑与外部依赖完全隔离。

```
┌─────────────────────────────────────────────────────────────┐
│                     外部适配器层                              │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │   CLI    │   Web    │  Python  │  R Bridge│  Cloud   │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                       端口层                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            应用服务接口（Ports）                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                    核心业务逻辑层                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  领域模型 │ 业务规则 │ 算法实现 │ 数据验证              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                       端口层                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         基础设施接口（Ports）                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                    基础设施适配器层                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ Database │  FileIO  │   GPU    │ Message  │  Cache   │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 实现示例

```julia
# ============================================================================
# 端口定义（接口）
# ============================================================================

"""
基因型数据存储端口
"""
abstract type GenotypeRepository end

# 必须实现的方法
function save(repo::GenotypeRepository, data::AbstractGenotypeData, id::String) end
function load(repo::GenotypeRepository, id::String)::AbstractGenotypeData end
function exists(repo::GenotypeRepository, id::String)::Bool end
function delete(repo::GenotypeRepository, id::String) end
function list_all(repo::GenotypeRepository)::Vector{String} end

"""
模型持久化端口
"""
abstract type ModelRepository end

function save_model(repo::ModelRepository, model::AbstractGenomicModel, metadata::Dict) end
function load_model(repo::ModelRepository, id::String)::AbstractGenomicModel end

"""
计算引擎端口
"""
abstract type ComputeEngine end

function execute(engine::ComputeEngine, task::ComputeTask)::ComputeResult end
function get_available_resources(engine::ComputeEngine)::ResourceInfo end

# ============================================================================
# 适配器实现
# ============================================================================

"""
HDF5 基因型存储适配器
"""
struct HDF5GenotypeRepository <: GenotypeRepository
    base_path::String
    compression::Symbol

    function HDF5GenotypeRepository(path::String; compression::Symbol=:gzip)
        mkpath(path)
        new(path, compression)
    end
end

function save(repo::HDF5GenotypeRepository, data::CompactGenotypes, id::String)
    filepath = joinpath(repo.base_path, "$id.h5")

    h5open(filepath, "w") do file
        # 保存数据
        write(file, "data", data.data)
        write(file, "n_samples", data.n_samples)
        write(file, "n_markers", data.n_markers)
        write(file, "sample_ids", data.sample_ids)
        write(file, "marker_ids", data.marker_ids)

        # 元数据
        attrs(file)["created"] = string(now())
        attrs(file)["version"] = "2.0"
    end

    @info "已保存基因型数据: $id"
end

function load(repo::HDF5GenotypeRepository, id::String)::CompactGenotypes
    filepath = joinpath(repo.base_path, "$id.h5")

    if !isfile(filepath)
        throw(DataNotFoundError("基因型数据不存在: $id"))
    end

    h5open(filepath, "r") do file
        # 读取并重建对象
        data = read(file, "data")
        n_samples = read(file, "n_samples")
        n_markers = read(file, "n_markers")
        sample_ids = read(file, "sample_ids")
        marker_ids = read(file, "marker_ids")

        # 重建 CompactGenotypes（需要添加构造函数）
        return reconstruct_genotypes(data, n_samples, n_markers, sample_ids, marker_ids)
    end
end

"""
Arrow 基因型存储适配器（更快的读取速度）
"""
struct ArrowGenotypeRepository <: GenotypeRepository
    base_path::String
end

function save(repo::ArrowGenotypeRepository, data::CompactGenotypes, id::String)
    filepath = joinpath(repo.base_path, "$id.arrow")

    # 转换为 Arrow 表
    table = (
        sample_ids = data.sample_ids,
        marker_ids = data.marker_ids,
        # 基因型数据需要特殊处理
        genotypes = [collect(data[i, :]) for i in 1:data.n_samples]
    )

    Arrow.write(filepath, table)
end

"""
GPU 计算引擎适配器
"""
struct CUDAComputeEngine <: ComputeEngine
    device_id::Int
    stream::CuStream
    memory_pool::CuMemoryPool

    function CUDAComputeEngine(device_id::Int=0)
        device!(device_id)
        stream = CuStream()
        pool = CuMemoryPool()
        new(device_id, stream, pool)
    end
end

function execute(engine::CUDAComputeEngine, task::ComputeGRMTask)::ComputeResult
    # 将数据传输到 GPU
    X_gpu = CuArray(task.genotypes)

    # 在 GPU 上计算
    G_gpu = X_gpu' * X_gpu

    # 传回 CPU
    G = Array(G_gpu)

    return ComputeResult(G, metadata=Dict(:device => "cuda:$(engine.device_id)"))
end

# ============================================================================
# 核心业务逻辑（与基础设施无关）
# ============================================================================

"""
基因组预测应用服务

核心业务逻辑，不依赖任何具体的基础设施实现
"""
struct GenomicPredictionService
    genotype_repo::GenotypeRepository
    model_repo::ModelRepository
    compute_engine::ComputeEngine
    event_bus::EventBus
    logger::Logger
end

"""
执行基因组预测工作流
"""
function predict_genomic_values(
    service::GenomicPredictionService,
    genotype_id::String,
    phenotype_data::PhenotypeData,
    model_config::Dict{Symbol, Any}
)
    # 1. 发布事件：工作流开始
    publish(service.event_bus, WorkflowStartedEvent(genotype_id, now()))

    try
        # 2. 加载基因型数据
        log_info(service.logger, "加载基因型数据: $genotype_id")
        genotypes = load(service.genotype_repo, genotype_id)

        publish(service.event_bus, DataLoadedEvent(genotype_id, n_samples(genotypes)))

        # 3. 数据验证
        validation_result = validate(genotypes)
        if !validation_result.valid
            throw(DataValidationError("基因型数据验证失败", validation_result.errors))
        end

        # 4. 创建计算任务
        grm_task = ComputeGRMTask(genotypes, model_config[:grm_method])

        # 5. 执行计算（委托给计算引擎）
        log_info(service.logger, "计算 GRM")
        grm_result = execute(service.compute_engine, grm_task)

        publish(service.event_bus, GRMComputedEvent(grm_result.metadata))

        # 6. 训练模型
        model = create_model(model_config[:model_type], model_config[:params])
        fit!(model, genotypes, phenotype_data; G=grm_result.value)

        # 7. 保存模型
        model_id = generate_model_id()
        save_model(service.model_repo, model, Dict(
            :genotype_id => genotype_id,
            :created_at => now(),
            :config => model_config
        ))

        # 8. 预测
        predictions = predict(model, genotypes)

        # 9. 发布成功事件
        publish(service.event_bus, WorkflowCompletedEvent(model_id, predictions))

        log_info(service.logger, "预测完成: $model_id")

        return predictions

    catch e
        # 发布失败事件
        publish(service.event_bus, WorkflowFailedEvent(genotype_id, e))
        log_error(service.logger, "预测失败", exception=e)
        rethrow(e)
    end
end

# ============================================================================
# 依赖注入容器
# ============================================================================

"""
应用配置和依赖注入
"""
struct ApplicationContext
    config::Configuration
    repositories::Dict{Symbol, Any}
    services::Dict{Symbol, Any}

    function ApplicationContext(config::Configuration)
        ctx = new(config, Dict(), Dict())

        # 根据配置创建适配器
        setup_repositories!(ctx)
        setup_services!(ctx)

        return ctx
    end
end

function setup_repositories!(ctx::ApplicationContext)
    # 根据配置选择存储后端
    storage_backend = get(ctx.config, "storage.backend", "hdf5")

    if storage_backend == "hdf5"
        ctx.repositories[:genotype] = HDF5GenotypeRepository(
            get(ctx.config, "storage.path", "./data")
        )
    elseif storage_backend == "arrow"
        ctx.repositories[:genotype] = ArrowGenotypeRepository(
            get(ctx.config, "storage.path", "./data")
        )
    else
        error("未知的存储后端: $storage_backend")
    end

    # 模型存储
    ctx.repositories[:model] = JLDModelRepository(
        get(ctx.config, "models.path", "./models")
    )
end

function setup_services!(ctx::ApplicationContext)
    # 根据配置选择计算引擎
    compute_backend = get(ctx.config, "compute.backend", "cpu")

    if compute_backend == "cuda" && CUDA.functional()
        engine = CUDAComputeEngine(get(ctx.config, "compute.device_id", 0))
    else
        engine = CPUComputeEngine(get(ctx.config, "compute.num_threads", Threads.nthreads()))
    end

    # 创建应用服务
    ctx.services[:prediction] = GenomicPredictionService(
        ctx.repositories[:genotype],
        ctx.repositories[:model],
        engine,
        EventBus(),
        create_logger(ctx.config)
    )
end

# ============================================================================
# 使用示例
# ============================================================================

# 创建配置
config = Configuration(Dict(
    "storage.backend" => "hdf5",
    "storage.path" => "./data",
    "compute.backend" => "cuda",
    "compute.device_id" => 0
))

# 创建应用上下文（依赖注入）
app = ApplicationContext(config)

# 获取服务
prediction_service = app.services[:prediction]

# 执行预测（业务逻辑与基础设施完全解耦）
results = predict_genomic_values(
    prediction_service,
    "my_genotypes",
    phenotype_data,
    Dict(
        :grm_method => :vanraden,
        :model_type => :gblup,
        :params => Dict(:solver => :pcg)
    )
)
```

### 1.2 洋葱架构（Onion Architecture）

依赖关系由外向内，核心领域模型完全独立。

```
┌─────────────────────────────────────────────┐
│         Infrastructure Layer                 │
│  (Database, FileIO, External APIs)          │
│  ┌───────────────────────────────────────┐  │
│  │      Application Services Layer       │  │
│  │   (Use Cases, Workflows, DTOs)        │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │    Domain Services Layer        │  │  │
│  │  │  (Business Logic, Algorithms)   │  │  │
│  │  │  ┌───────────────────────────┐  │  │  │
│  │  │  │   Domain Model (Core)     │  │  │  │
│  │  │  │  (Entities, Value Objects)│  │  │  │
│  │  │  └───────────────────────────┘  │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

#### 实现

```julia
# ============================================================================
# 第 1 层：领域模型（最内层，无依赖）
# ============================================================================

module DomainModel

"""
基因型值对象（Value Object）
不可变，包含业务规则
"""
struct GenotypeValue
    value::UInt8

    function GenotypeValue(val::Integer)
        if !(val in [0, 1, 2])
            throw(DomainException("基因型值必须是 0, 1, 或 2"))
        end
        new(UInt8(val))
    end
end

Base.:(==)(a::GenotypeValue, b::GenotypeValue) = a.value == b.value

"""
等位基因频率值对象
包含业务规则验证
"""
struct AlleleFrequency
    value::Float64

    function AlleleFrequency(freq::Real)
        if !(0 <= freq <= 1)
            throw(DomainException("等位基因频率必须在 [0, 1] 范围内"))
        end
        new(Float64(freq))
    end
end

function is_rare(af::AlleleFrequency, threshold::Float64=0.01)::Bool
    return af.value < threshold || af.value > (1 - threshold)
end

"""
标记实体（Entity）
有唯一标识
"""
mutable struct Marker
    id::String
    chromosome::String
    position::Int
    ref_allele::String
    alt_allele::String
    allele_frequency::AlleleFrequency

    # 业务规则：验证染色体编号
    function Marker(id, chr, pos, ref, alt, freq)
        if !is_valid_chromosome(chr)
            throw(DomainException("无效的染色体编号: $chr"))
        end
        if pos <= 0
            throw(DomainException("位置必须 > 0"))
        end
        new(id, chr, pos, ref, alt, freq)
    end
end

function is_valid_chromosome(chr::String)::Bool
    return chr in ["1", "2", ..., "X", "Y", "MT"] || occursin(r"^chr\d+$", chr)
end

"""
个体实体
"""
mutable struct Individual
    id::String
    genotypes::Vector{Union{GenotypeValue, Missing}}
    phenotype::Union{Float64, Missing}
    metadata::Dict{Symbol, Any}

    # 业务规则
    function Individual(id, geno, pheno, meta=Dict())
        if isempty(id)
            throw(DomainException("个体 ID 不能为空"))
        end
        new(id, geno, pheno, meta)
    end
end

"""
基因组关系矩阵聚合根（Aggregate Root）
"""
struct GenomicRelationshipMatrix
    individuals::Vector{Individual}
    matrix::Matrix{Float64}
    method::Symbol

    # 不变式：矩阵必须是对称的
    function GenomicRelationshipMatrix(indiv, mat, meth)
        if !issymmetric(mat)
            throw(DomainException("GRM 必须是对称矩阵"))
        end
        if size(mat, 1) != length(indiv)
            throw(DomainException("矩阵维度与个体数量不匹配"))
        end
        new(indiv, mat, meth)
    end
end

# 领域服务
function compute_relationship(i1::Individual, i2::Individual, markers::Vector{Marker})::Float64
    # 业务逻辑：计算两个个体的遗传关系
    # ...
end

end  # module DomainModel

# ============================================================================
# 第 2 层：领域服务（依赖领域模型）
# ============================================================================

module DomainServices

using ..DomainModel

"""
GRM 计算领域服务
纯粹的业务逻辑，无基础设施依赖
"""
struct GRMCalculator
    method::Symbol
    min_maf::Float64
end

function calculate(
    calc::GRMCalculator,
    individuals::Vector{Individual},
    markers::Vector{Marker}
)::GenomicRelationshipMatrix
    # 1. 过滤低频标记（业务规则）
    valid_markers = filter(m -> !is_rare(m.allele_frequency, calc.min_maf), markers)

    # 2. 提取基因型矩阵
    n = length(individuals)
    m = length(valid_markers)
    X = Matrix{Float64}(undef, n, m)

    for (i, ind) in enumerate(individuals)
        for (j, marker) in enumerate(valid_markers)
            geno = ind.genotypes[j]
            X[i, j] = ismissing(geno) ? 2 * marker.allele_frequency.value : Float64(geno.value)
        end
    end

    # 3. 标准化
    for j in 1:m
        p = valid_markers[j].allele_frequency.value
        X[:, j] .-= 2*p
        X[:, j] ./= sqrt(2*p*(1-p))
    end

    # 4. 计算关系矩阵
    G = X * X'

    # 5. 缩放
    if calc.method == :vanraden
        scaling = sum(2 .* [m.allele_frequency.value for m in valid_markers] .*
                      (1 .- [m.allele_frequency.value for m in valid_markers]))
        G ./= scaling
    end

    # 6. 创建聚合根
    return GenomicRelationshipMatrix(individuals, G, calc.method)
end

"""
质量控制领域服务
"""
struct QualityController
    max_missing_rate::Float64
    min_maf::Float64
end

function apply_qc(
    qc::QualityController,
    individuals::Vector{Individual},
    markers::Vector{Marker}
)::Tuple{Vector{Individual}, Vector{Marker}}
    # 业务规则：过滤
    valid_individuals = filter(ind -> calculate_missing_rate(ind) <= qc.max_missing_rate, individuals)
    valid_markers = filter(m -> !is_rare(m.allele_frequency, qc.min_maf), markers)

    return valid_individuals, valid_markers
end

end  # module DomainServices

# ============================================================================
# 第 3 层：应用服务（用例编排）
# ============================================================================

module ApplicationServices

using ..DomainModel
using ..DomainServices

"""
基因组预测用例
编排领域服务，但不包含业务逻辑
"""
struct GenomicPredictionUseCase
    grm_calculator::GRMCalculator
    qc_controller::QualityController
    individual_repo::IndividualRepository  # 端口（接口）
    marker_repo::MarkerRepository          # 端口（接口）
    model_trainer::ModelTrainer            # 端口（接口）
end

function execute(
    usecase::GenomicPredictionUseCase,
    request::PredictionRequest
)::PredictionResponse
    # 1. 加载数据（通过端口）
    individuals = load_individuals(usecase.individual_repo, request.dataset_id)
    markers = load_markers(usecase.marker_repo, request.dataset_id)

    # 2. 质量控制（领域服务）
    qc_individuals, qc_markers = apply_qc(
        usecase.qc_controller,
        individuals,
        markers
    )

    # 3. 计算 GRM（领域服务）
    grm = calculate(usecase.grm_calculator, qc_individuals, qc_markers)

    # 4. 训练模型（通过端口）
    model = train(usecase.model_trainer, grm, request.model_config)

    # 5. 预测
    predictions = predict(model, qc_individuals)

    # 6. 返回响应
    return PredictionResponse(
        predictions = predictions,
        n_individuals = length(qc_individuals),
        n_markers = length(qc_markers),
        accuracy = calculate_accuracy(predictions, qc_individuals)
    )
end

end  # module ApplicationServices
```

---

## 2. 依赖注入和控制反转

### 2.1 依赖注入容器

```julia
module DependencyInjection

"""
服务生命周期
"""
@enum ServiceLifetime begin
    Singleton    # 单例：整个应用生命周期
    Scoped       # 作用域：每个请求/会话
    Transient    # 瞬时：每次请求创建新实例
end

"""
服务描述符
"""
struct ServiceDescriptor
    service_type::Type
    implementation_type::Union{Type, Nothing}
    factory::Union{Function, Nothing}
    instance::Union{Any, Nothing}
    lifetime::ServiceLifetime
end

"""
依赖注入容器
"""
mutable struct ServiceContainer
    services::Dict{Type, ServiceDescriptor}
    singletons::Dict{Type, Any}

    function ServiceContainer()
        new(Dict{Type, ServiceDescriptor}(), Dict{Type, Any}())
    end
end

# ============================================================================
# 注册服务
# ============================================================================

"""
注册单例服务
"""
function add_singleton!(
    container::ServiceContainer,
    service_type::Type,
    implementation_type::Type
)
    container.services[service_type] = ServiceDescriptor(
        service_type,
        implementation_type,
        nothing,
        nothing,
        Singleton
    )
end

function add_singleton!(
    container::ServiceContainer,
    service_type::Type,
    factory::Function
)
    container.services[service_type] = ServiceDescriptor(
        service_type,
        nothing,
        factory,
        nothing,
        Singleton
    )
end

function add_singleton!(
    container::ServiceContainer,
    service_type::Type,
    instance::Any
)
    container.services[service_type] = ServiceDescriptor(
        service_type,
        nothing,
        nothing,
        instance,
        Singleton
    )
    container.singletons[service_type] = instance
end

"""
注册作用域服务
"""
function add_scoped!(
    container::ServiceContainer,
    service_type::Type,
    implementation_type::Type
)
    container.services[service_type] = ServiceDescriptor(
        service_type,
        implementation_type,
        nothing,
        nothing,
        Scoped
    )
end

"""
注册瞬时服务
"""
function add_transient!(
    container::ServiceContainer,
    service_type::Type,
    implementation_type::Type
)
    container.services[service_type] = ServiceDescriptor(
        service_type,
        implementation_type,
        nothing,
        nothing,
        Transient
    )
end

# ============================================================================
# 解析服务
# ============================================================================

"""
解析服务实例
"""
function resolve(container::ServiceContainer, service_type::Type)
    if !haskey(container.services, service_type)
        error("服务未注册: $service_type")
    end

    descriptor = container.services[service_type]

    # 单例：检查是否已创建
    if descriptor.lifetime == Singleton
        if haskey(container.singletons, service_type)
            return container.singletons[service_type]
        end
    end

    # 创建实例
    instance = if !isnothing(descriptor.instance)
        descriptor.instance
    elseif !isnothing(descriptor.factory)
        descriptor.factory(container)
    elseif !isnothing(descriptor.implementation_type)
        create_instance(container, descriptor.implementation_type)
    else
        error("无法创建服务实例: $service_type")
    end

    # 缓存单例
    if descriptor.lifetime == Singleton
        container.singletons[service_type] = instance
    end

    return instance
end

"""
通过构造函数注入创建实例
"""
function create_instance(container::ServiceContainer, type::Type)
    # 获取构造函数
    ctors = methods(type)

    if length(ctors) == 0
        error("类型没有构造函数: $type")
    end

    # 使用第一个构造函数（简化版本，实际应选择最合适的）
    ctor = first(ctors)

    # 解析构造函数参数
    param_types = ctor.sig.parameters[2:end]  # 跳过类型参数
    params = [resolve(container, pt) for pt in param_types]

    # 创建实例
    return type(params...)
end

# ============================================================================
# 使用示例
# ============================================================================

# 定义接口
abstract type ILogger end
abstract type IGenotypeRepository end
abstract type IComputeEngine end

# 定义实现
struct ConsoleLogger <: ILogger
    level::Symbol
end

struct HDF5Repository <: IGenotypeRepository
    path::String
end

struct CUDAEngine <: IComputeEngine
    device_id::Int
end

# 定义服务（依赖注入）
struct GenomicPredictionService
    logger::ILogger
    genotype_repo::IGenotypeRepository
    compute_engine::IComputeEngine

    # 构造函数自动注入
    function GenomicPredictionService(
        logger::ILogger,
        repo::IGenotypeRepository,
        engine::IComputeEngine
    )
        new(logger, repo, engine)
    end
end

# 创建容器并注册服务
container = ServiceContainer()

# 注册基础设施服务
add_singleton!(container, ILogger, () -> ConsoleLogger(:info))
add_singleton!(container, IGenotypeRepository, () -> HDF5Repository("./data"))
add_singleton!(container, IComputeEngine, () -> CUDAEngine(0))

# 注册应用服务
add_scoped!(container, GenomicPredictionService, GenomicPredictionService)

# 解析服务（自动注入所有依赖）
service = resolve(container, GenomicPredictionService)

# 使用服务
# service 的所有依赖已自动注入
```

### 2.2 配置驱动的依赖注入

```julia
"""
从配置文件自动配置服务
"""
function configure_services(config::Configuration)::ServiceContainer
    container = ServiceContainer()

    # 日志服务
    log_config = get(config, "logging", Dict())
    if get(log_config, "type", "console") == "console"
        add_singleton!(container, ILogger, () -> ConsoleLogger(
            Symbol(get(log_config, "level", "info"))
        ))
    elseif log_config["type"] == "file"
        add_singleton!(container, ILogger, () -> FileLogger(
            log_config["path"],
            Symbol(get(log_config, "level", "info"))
        ))
    end

    # 存储服务
    storage_config = get(config, "storage", Dict())
    backend = get(storage_config, "backend", "hdf5")

    if backend == "hdf5"
        add_singleton!(container, IGenotypeRepository, () -> HDF5Repository(
            get(storage_config, "path", "./data")
        ))
    elseif backend == "arrow"
        add_singleton!(container, IGenotypeRepository, () -> ArrowRepository(
            get(storage_config, "path", "./data")
        ))
    end

    # 计算引擎
    compute_config = get(config, "compute", Dict())
    backend = get(compute_config, "backend", "cpu")

    if backend == "cuda" && CUDA.functional()
        add_singleton!(container, IComputeEngine, () -> CUDAEngine(
            get(compute_config, "device_id", 0)
        ))
    else
        add_singleton!(container, IComputeEngine, () -> CPUEngine(
            get(compute_config, "num_threads", Threads.nthreads())
        ))
    end

    # 应用服务
    add_scoped!(container, GenomicPredictionService, GenomicPredictionService)

    return container
end

# 使用
config = load_configuration("config.toml")
container = configure_services(config)
service = resolve(container, GenomicPredictionService)
```

---

## 3. 事件驱动架构

### 3.1 事件系统设计

```julia
# ============================================================================
# 事件定义
# ============================================================================

"""
领域事件基类
"""
abstract type DomainEvent end

"""
事件元数据
"""
struct EventMetadata
    event_id::String
    timestamp::DateTime
    correlation_id::String
    causation_id::String
    user_id::Union{String, Nothing}
end

"""
数据加载完成事件
"""
struct DataLoadedEvent <: DomainEvent
    metadata::EventMetadata
    dataset_id::String
    n_samples::Int
    n_markers::Int
end

"""
GRM 计算完成事件
"""
struct GRMComputedEvent <: DomainEvent
    metadata::EventMetadata
    method::Symbol
    computation_time::Float64
    matrix_size::Tuple{Int, Int}
end

"""
模型训练完成事件
"""
struct ModelTrainedEvent <: DomainEvent
    metadata::EventMetadata
    model_id::String
    model_type::Symbol
    accuracy::Float64
end

"""
预测完成事件
"""
struct PredictionCompletedEvent <: DomainEvent
    metadata::EventMetadata
    model_id::String
    n_predictions::Int
    output_path::String
end

# ============================================================================
# 事件总线
# ============================================================================

"""
事件处理器接口
"""
abstract type EventHandler{T<:DomainEvent} end

function handle(handler::EventHandler{T}, event::T) where T
    error("未实现 handle 方法")
end

"""
事件总线
"""
mutable struct EventBus
    handlers::Dict{Type{<:DomainEvent}, Vector{EventHandler}}
    middleware::Vector{Function}

    function EventBus()
        new(Dict(), Function[])
    end
end

"""
注册事件处理器
"""
function subscribe!(bus::EventBus, event_type::Type{T}, handler::EventHandler{T}) where T<:DomainEvent
    if !haskey(bus.handlers, event_type)
        bus.handlers[event_type] = EventHandler[]
    end
    push!(bus.handlers[event_type], handler)
end

"""
发布事件
"""
function publish(bus::EventBus, event::DomainEvent)
    event_type = typeof(event)

    # 应用中间件
    for mw in bus.middleware
        event = mw(event)
    end

    # 分发到处理器
    if haskey(bus.handlers, event_type)
        for handler in bus.handlers[event_type]
            try
                handle(handler, event)
            catch e
                @error "事件处理失败" event=event_type handler=typeof(handler) exception=e
            end
        end
    end
end

"""
添加中间件
"""
function add_middleware!(bus::EventBus, middleware::Function)
    push!(bus.middleware, middleware)
end

# ============================================================================
# 事件处理器实现
# ============================================================================

"""
日志事件处理器
"""
struct LoggingEventHandler{T} <: EventHandler{T}
    logger::ILogger
end

function handle(handler::LoggingEventHandler{DataLoadedEvent}, event::DataLoadedEvent)
    log_info(handler.logger,
        "数据加载完成: $(event.dataset_id), 样本: $(event.n_samples), 标记: $(event.n_markers)")
end

function handle(handler::LoggingEventHandler{GRMComputedEvent}, event::GRMComputedEvent)
    log_info(handler.logger,
        "GRM 计算完成: $(event.method), 耗时: $(event.computation_time)s")
end

"""
指标收集事件处理器
"""
struct MetricsEventHandler{T} <: EventHandler{T}
    metrics_collector::MetricsCollector
end

function handle(handler::MetricsEventHandler{GRMComputedEvent}, event::GRMComputedEvent)
    record_metric(handler.metrics_collector,
        "grm.computation_time",
        event.computation_time,
        tags=Dict("method" => string(event.method)))
end

"""
通知事件处理器
"""
struct NotificationEventHandler{T} <: EventHandler{T}
    notifier::INotifier
end

function handle(handler::NotificationEventHandler{PredictionCompletedEvent}, event::PredictionCompletedEvent)
    send_notification(handler.notifier,
        "预测完成",
        "模型 $(event.model_id) 已完成 $(event.n_predictions) 个样本的预测"
    )
end

"""
数据持久化事件处理器
"""
struct EventStoreHandler{T} <: EventHandler{T}
    event_store::IEventStore
end

function handle(handler::EventStoreHandler{T}, event::T) where T<:DomainEvent
    save_event(handler.event_store, event)
end

# ============================================================================
# 事件中间件
# ============================================================================

"""
事件增强中间件：添加 correlation_id
"""
function correlation_middleware(event::DomainEvent)
    if isnothing(event.metadata.correlation_id)
        # 添加 correlation_id
        new_metadata = EventMetadata(
            event.metadata.event_id,
            event.metadata.timestamp,
            generate_correlation_id(),
            event.metadata.causation_id,
            event.metadata.user_id
        )
        return reconstruct_event(event, new_metadata)
    end
    return event
end

"""
性能监控中间件
"""
function performance_middleware(event::DomainEvent)
    t_start = time()

    # 事件处理完成后记录
    # (需要异步机制)

    return event
end

# ============================================================================
# 使用示例
# ============================================================================

# 创建事件总线
bus = EventBus()

# 添加中间件
add_middleware!(bus, correlation_middleware)

# 注册处理器
subscribe!(bus, DataLoadedEvent, LoggingEventHandler{DataLoadedEvent}(logger))
subscribe!(bus, GRMComputedEvent, LoggingEventHandler{GRMComputedEvent}(logger))
subscribe!(bus, GRMComputedEvent, MetricsEventHandler{GRMComputedEvent}(metrics))
subscribe!(bus, PredictionCompletedEvent, NotificationEventHandler{PredictionCompletedEvent}(notifier))

# 所有事件都持久化
for event_type in [DataLoadedEvent, GRMComputedEvent, ModelTrainedEvent, PredictionCompletedEvent]
    subscribe!(bus, event_type, EventStoreHandler{event_type}(event_store))
end

# 发布事件
publish(bus, DataLoadedEvent(
    EventMetadata(generate_event_id(), now(), "", "", "user123"),
    "dataset_001",
    5000,
    50000
))
```

### 3.2 异步事件处理

```julia
using Channels

"""
异步事件总线
"""
mutable struct AsyncEventBus
    event_channel::Channel{DomainEvent}
    handlers::Dict{Type{<:DomainEvent}, Vector{EventHandler}}
    worker_tasks::Vector{Task}
    running::Bool

    function AsyncEventBus(buffer_size::Int=1000, num_workers::Int=4)
        bus = new(
            Channel{DomainEvent}(buffer_size),
            Dict(),
            Task[],
            false
        )

        # 启动工作线程
        for i in 1:num_workers
            task = @async process_events(bus)
            push!(bus.worker_tasks, task)
        end

        bus.running = true
        return bus
    end
end

"""
事件处理工作线程
"""
function process_events(bus::AsyncEventBus)
    while bus.running
        try
            event = take!(bus.event_channel)
            event_type = typeof(event)

            if haskey(bus.handlers, event_type)
                # 并行处理所有处理器
                @sync for handler in bus.handlers[event_type]
                    @async begin
                        try
                            handle(handler, event)
                        catch e
                            @error "异步事件处理失败" exception=e
                        end
                    end
                end
            end
        catch e
            if isa(e, InvalidStateException)
                break  # Channel 已关闭
            else
                @error "事件处理线程错误" exception=e
            end
        end
    end
end

"""
异步发布事件
"""
function publish_async(bus::AsyncEventBus, event::DomainEvent)
    put!(bus.event_channel, event)
end

"""
关闭事件总线
"""
function close!(bus::AsyncEventBus)
    bus.running = false
    close(bus.event_channel)

    # 等待所有工作线程完成
    for task in bus.worker_tasks
        wait(task)
    end
end
```

---

## 4. 插件生态系统

### 4.1 插件架构

```julia
# ============================================================================
# 插件接口定义
# ============================================================================

"""
插件元数据
"""
struct PluginMetadata
    name::String
    version::String
    author::String
    description::String
    dependencies::Vector{String}
    license::String
end

"""
插件接口
"""
abstract type Plugin end

# 插件生命周期
function initialize(plugin::Plugin, context::PluginContext) end
function activate(plugin::Plugin) end
function deactivate(plugin::Plugin) end
function unload(plugin::Plugin) end

# 插件信息
function get_metadata(plugin::Plugin)::PluginMetadata end

"""
插件上下文
"""
struct PluginContext
    config::Configuration
    services::ServiceContainer
    event_bus::EventBus
    logger::ILogger
end

# ============================================================================
# 插件管理器
# ============================================================================

"""
插件管理器
"""
mutable struct PluginManager
    plugins::Dict{String, Plugin}
    plugin_paths::Vector{String}
    context::PluginContext

    function PluginManager(context::PluginContext)
        new(Dict(), String[], context)
    end
end

"""
添加插件搜索路径
"""
function add_plugin_path!(manager::PluginManager, path::String)
    push!(manager.plugin_paths, path)
end

"""
发现插件
"""
function discover_plugins(manager::PluginManager)::Vector{String}
    discovered = String[]

    for path in manager.plugin_paths
        if isdir(path)
            for file in readdir(path)
                if endswith(file, "_plugin.jl")
                    push!(discovered, joinpath(path, file))
                end
            end
        end
    end

    return discovered
end

"""
加载插件
"""
function load_plugin!(manager::PluginManager, plugin_file::String)
    # 动态加载 Julia 文件
    include(plugin_file)

    # 假设插件定义了一个 create_plugin() 函数
    plugin = Base.invokelatest(create_plugin)

    # 初始化
    initialize(plugin, manager.context)

    # 注册
    metadata = get_metadata(plugin)
    manager.plugins[metadata.name] = plugin

    @info "已加载插件: $(metadata.name) v$(metadata.version)"

    return plugin
end

"""
激活插件
"""
function activate_plugin!(manager::PluginManager, plugin_name::String)
    if !haskey(manager.plugins, plugin_name)
        error("插件未加载: $plugin_name")
    end

    plugin = manager.plugins[plugin_name]
    activate(plugin)

    @info "已激活插件: $plugin_name"
end

"""
停用插件
"""
function deactivate_plugin!(manager::PluginManager, plugin_name::String)
    if !haskey(manager.plugins, plugin_name)
        error("插件未加载: $plugin_name")
    end

    plugin = manager.plugins[plugin_name]
    deactivate(plugin)

    @info "已停用插件: $plugin_name"
end

# ============================================================================
# 扩展点定义
# ============================================================================

"""
数据加载器扩展点
"""
abstract type DataLoaderExtension <: Plugin end

function load_data(ext::DataLoaderExtension, filepath::String)::AbstractGenomicData end
function supported_formats(ext::DataLoaderExtension)::Vector{String} end

"""
模型扩展点
"""
abstract type ModelExtension <: Plugin end

function create_model(ext::ModelExtension, config::Dict)::AbstractGenomicModel end
function model_type(ext::ModelExtension)::Symbol end

"""
可视化扩展点
"""
abstract type VisualizationExtension <: Plugin end

function create_plot(ext::VisualizationExtension, data, plot_type::Symbol) end
function supported_plot_types(ext::VisualizationExtension)::Vector{Symbol} end

# ============================================================================
# 插件示例
# ============================================================================

"""
VCF 数据加载器插件
"""
struct VCFLoaderPlugin <: DataLoaderExtension
    context::Union{PluginContext, Nothing}
    active::Bool

    VCFLoaderPlugin() = new(nothing, false)
end

function get_metadata(plugin::VCFLoaderPlugin)::PluginMetadata
    return PluginMetadata(
        "VCFLoader",
        "1.0.0",
        "GenomicPro Team",
        "加载 VCF/BCF 格式的基因型数据",
        String[],
        "MIT"
    )
end

function initialize(plugin::VCFLoaderPlugin, context::PluginContext)
    @set! plugin.context = context
    log_info(context.logger, "VCF 加载器插件已初始化")
end

function activate(plugin::VCFLoaderPlugin)
    @set! plugin.active = true

    # 注册数据加载器
    register_data_loader(plugin.context.services, "vcf", plugin)
    register_data_loader(plugin.context.services, "bcf", plugin)
end

function supported_formats(plugin::VCFLoaderPlugin)::Vector{String}
    return ["vcf", "vcf.gz", "bcf"]
end

function load_data(plugin::VCFLoaderPlugin, filepath::String)::CompactGenotypes
    # 实现 VCF 加载逻辑
    # ...
end

# 导出插件创建函数
create_plugin() = VCFLoaderPlugin()

"""
高级可视化插件
"""
struct AdvancedVisualizationPlugin <: VisualizationExtension
    context::Union{PluginContext, Nothing}
    active::Bool

    AdvancedVisualizationPlugin() = new(nothing, false)
end

function get_metadata(plugin::AdvancedVisualizationPlugin)::PluginMetadata
    return PluginMetadata(
        "AdvancedVisualization",
        "1.0.0",
        "Community",
        "高级可视化功能：曼哈顿图、QQ图、PCA图等",
        ["Plots", "StatsPlots"],
        "MIT"
    )
end

function activate(plugin::AdvancedVisualizationPlugin)
    @set! plugin.active = true

    # 注册可视化函数
    register_visualization(plugin.context.services, :manhattan_plot, plugin)
    register_visualization(plugin.context.services, :qq_plot, plugin)
    register_visualization(plugin.context.services, :pca_plot, plugin)
end

function supported_plot_types(plugin::AdvancedVisualizationPlugin)::Vector{Symbol}
    return [:manhattan_plot, :qq_plot, :pca_plot, :kinship_heatmap]
end

function create_plot(plugin::AdvancedVisualizationPlugin, data, plot_type::Symbol)
    if plot_type == :manhattan_plot
        return create_manhattan_plot(data)
    elseif plot_type == :qq_plot
        return create_qq_plot(data)
    # ...
    end
end

# ============================================================================
# 使用示例
# ============================================================================

# 创建插件管理器
context = PluginContext(config, services, event_bus, logger)
plugin_manager = PluginManager(context)

# 添加插件路径
add_plugin_path!(plugin_manager, "./plugins")
add_plugin_path!(plugin_manager, "~/.genomicpro/plugins")

# 发现并加载所有插件
discovered = discover_plugins(plugin_manager)
for plugin_file in discovered
    load_plugin!(plugin_manager, plugin_file)
end

# 激活插件
activate_plugin!(plugin_manager, "VCFLoader")
activate_plugin!(plugin_manager, "AdvancedVisualization")

# 使用插件功能
genotypes = load_data_with_plugin("data.vcf.gz", format="vcf")
plot = create_plot_with_plugin(gwas_results, :manhattan_plot)
```

### 4.2 插件配置和热加载

```julia
"""
插件配置
"""
struct PluginConfiguration
    enabled_plugins::Vector{String}
    plugin_settings::Dict{String, Dict{Symbol, Any}}
    auto_discover::Bool
    plugin_paths::Vector{String}
end

function load_plugin_config(config_file::String)::PluginConfiguration
    config = TOML.parsefile(config_file)

    return PluginConfiguration(
        get(config, "enabled_plugins", String[]),
        get(config, "plugin_settings", Dict()),
        get(config, "auto_discover", true),
        get(config, "plugin_paths", String["./plugins"])
    )
end

"""
热加载支持
"""
function watch_plugins(manager::PluginManager)
    # 监视插件目录变化
    @async begin
        while true
            sleep(5)  # 每 5 秒检查一次

            for path in manager.plugin_paths
                check_plugin_updates(manager, path)
            end
        end
    end
end

function check_plugin_updates(manager::PluginManager, path::String)
    # 检查文件修改时间
    # 如果有更新，重新加载插件
    # ...
end
```

---

## 5. 领域驱动设计（DDD）

### 5.1 限界上下文（Bounded Contexts）

```
GenomicPro 2.0 领域模型

┌────────────────────────────────────────────────────────────┐
│         数据管理上下文 (Data Management Context)            │
│                                                              │
│  实体: Genotype, Phenotype, Pedigree                        │
│  值对象: GenotypeValue, AlleleFrequency                     │
│  聚合: Dataset                                              │
│  服务: QualityController, DataValidator                     │
└────────────────────────────────────────────────────────────┘
                           ↓↑
┌────────────────────────────────────────────────────────────┐
│         分析上下文 (Analysis Context)                        │
│                                                              │
│  实体: GRM, Model                                           │
│  值对象: Accuracy, PredictedValue                           │
│  聚合: AnalysisWorkflow                                     │
│  服务: GRMCalculator, ModelTrainer                          │
└────────────────────────────────────────────────────────────┘
                           ↓↑
┌────────────────────────────────────────────────────────────┐
│         育种决策上下文 (Breeding Decision Context)           │
│                                                              │
│  实体: SelectionCandidate, BreedingPlan                     │
│  值对象: SelectionIndex, GeneticGain                        │
│  聚合: BreedingProgram                                      │
│  服务: SelectionOptimizer, MatingPlanner                    │
└────────────────────────────────────────────────────────────┘
```

### 5.2 聚合设计

```julia
# ============================================================================
# 数据管理上下文
# ============================================================================

module DataManagementContext

"""
数据集聚合根
"""
mutable struct Dataset
    id::String
    genotypes::Vector{Individual}
    markers::Vector{Marker}
    metadata::DatasetMetadata

    # 聚合根不变式
    function Dataset(id, geno, markers, meta)
        # 业务规则：个体数量必须一致
        if !all(ind -> length(ind.genotypes) == length(markers), geno)
            throw(DomainException("个体基因型数量与标记数量不一致"))
        end

        new(id, geno, markers, meta)
    end
end

# 只能通过聚合根修改内部实体
function add_individual!(dataset::Dataset, individual::Individual)
    # 验证
    if length(individual.genotypes) != length(dataset.markers)
        throw(DomainException("个体基因型数量与数据集标记数量不匹配"))
    end

    push!(dataset.genotypes, individual)
end

function remove_low_quality_markers!(dataset::Dataset, min_quality::Float64)
    # 业务逻辑：移除低质量标记
    valid_markers_idx = findall(m -> m.quality >= min_quality, dataset.markers)

    # 同时更新标记和所有个体的基因型
    dataset.markers = dataset.markers[valid_markers_idx]
    for ind in dataset.genotypes
        ind.genotypes = ind.genotypes[valid_markers_idx]
    end
end

"""
应用服务
"""
function create_dataset_from_files(
    genotype_file::String,
    phenotype_file::String
)::Dataset
    # 领域服务
    loader = DataLoader()
    validator = DataValidator()

    # 加载
    raw_genotypes = load_genotypes(loader, genotype_file)
    raw_phenotypes = load_phenotypes(loader, phenotype_file)

    # 验证
    validation_result = validate(validator, raw_genotypes, raw_phenotypes)
    if !validation_result.valid
        throw(DataValidationError("数据验证失败", validation_result.errors))
    end

    # 创建领域对象
    individuals = map_to_individuals(raw_genotypes, raw_phenotypes)
    markers = map_to_markers(raw_genotypes)

    # 创建聚合
    dataset = Dataset(
        generate_dataset_id(),
        individuals,
        markers,
        DatasetMetadata(now(), "uploaded", Dict())
    )

    # 发布领域事件
    publish(event_bus, DatasetCreatedEvent(dataset.id, length(individuals), length(markers)))

    return dataset
end

end  # module DataManagementContext

# ============================================================================
# 分析上下文
# ============================================================================

module AnalysisContext

using ..DataManagementContext

"""
分析工作流聚合根
"""
mutable struct AnalysisWorkflow
    id::String
    dataset_id::String
    steps::Vector{AnalysisStep}
    status::WorkflowStatus
    results::Dict{Symbol, Any}

    function AnalysisWorkflow(dataset_id::String)
        new(
            generate_workflow_id(),
            dataset_id,
            AnalysisStep[],
            WorkflowStatus(:created),
            Dict{Symbol, Any}()
        )
    end
end

"""
添加分析步骤
"""
function add_step!(workflow::AnalysisWorkflow, step::AnalysisStep)
    # 业务规则：不能向已完成的工作流添加步骤
    if workflow.status == WorkflowStatus(:completed)
        throw(DomainException("无法向已完成的工作流添加步骤"))
    end

    push!(workflow.steps, step)
end

"""
执行工作流
"""
function execute!(workflow::AnalysisWorkflow, dataset::Dataset)
    # 状态转换
    workflow.status = WorkflowStatus(:running)

    for step in workflow.steps
        try
            result = execute_step(step, dataset, workflow.results)
            workflow.results[step.name] = result

            # 发布步骤完成事件
            publish(event_bus, StepCompletedEvent(workflow.id, step.name))
        catch e
            workflow.status = WorkflowStatus(:failed)
            publish(event_bus, WorkflowFailedEvent(workflow.id, step.name, e))
            rethrow(e)
        end
    end

    workflow.status = WorkflowStatus(:completed)
    publish(event_bus, WorkflowCompletedEvent(workflow.id))
end

end  # module AnalysisContext
```

---

*（续下一部分...）*

**当前进度**: 已完成高级架构模式、依赖注入、事件驱动架构、插件系统、DDD 设计

**下一部分将包含**:
- CQRS 和事件溯源
- 微服务架构
- 响应式编程
- 性能工程
- 可观测性设计
- API 设计规范
- 安全性设计

文档字数已达到约 **26,000** 字。是否继续完成剩余部分？
