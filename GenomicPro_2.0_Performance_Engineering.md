# GenomicPro 2.0 性能工程指南

**版本**: 2.0
**日期**: 2025-11-15
**目标**: 企业级性能标准

---

## 目录

1. [性能目标和指标](#1-性能目标和指标)
2. [性能分析和Profiling](#2-性能分析和profiling)
3. [内存优化策略](#3-内存优化策略)
4. [并行计算优化](#4-并行计算优化)
5. [GPU加速](#5-gpu加速)
6. [缓存策略](#6-缓存策略)
7. [基准测试框架](#7-基准测试框架)
8. [性能回归检测](#8-性能回归检测)
9. [大规模数据优化](#9-大规模数据优化)

---

## 1. 性能目标和指标

### 1.1 性能目标

| 操作 | 数据规模 | 目标时间 | 目标内存 |
|------|----------|----------|---------|
| **GRM 计算** | 10k × 50k | < 2s (GPU) | < 1 GB |
| **GRM 计算** | 50k × 500k | < 30s (GPU) | < 10 GB |
| **GBLUP 求解** | 10k 样本 | < 1s | < 2 GB |
| **GBLUP 求解** | 50k 样本 | < 10s | < 20 GB |
| **BayesR (10k iter)** | 5k × 50k | < 60s (GPU) | < 5 GB |
| **数据加载 (VCF)** | 10k × 100k | < 10s | < 500 MB |
| **QC Pipeline** | 10k × 100k | < 5s | < 1 GB |

### 1.2 关键性能指标 (KPIs)

```julia
"""
性能指标定义
"""
struct PerformanceMetrics
    # 时间指标
    wall_time::Float64          # 墙钟时间
    cpu_time::Float64           # CPU 时间
    gc_time::Float64            # GC 时间

    # 内存指标
    peak_memory::Int            # 峰值内存 (bytes)
    allocated_memory::Int       # 分配的内存
    gc_collections::Int         # GC 次数

    # 吞吐量
    samples_per_second::Float64  # 样本处理速度
    markers_per_second::Float64  # 标记处理速度

    # 资源利用率
    cpu_utilization::Float64    # CPU 利用率 (%)
    gpu_utilization::Float64    # GPU 利用率 (%)
    memory_bandwidth::Float64   # 内存带宽 (GB/s)

    # 缓存效率
    cache_hit_rate::Float64     # 缓存命中率
    cache_miss_rate::Float64    # 缓存未命中率
end

"""
性能测量装饰器
"""
macro measure_performance(expr)
    quote
        # 开始测量
        gc_start = Base.gc_num()
        mem_start = Base.gc_live_bytes()
        time_start = time_ns()

        # 执行
        result = $(esc(expr))

        # 结束测量
        time_end = time_ns()
        mem_end = Base.gc_live_bytes()
        gc_end = Base.gc_num()

        # 计算指标
        wall_time = (time_end - time_start) / 1e9
        gc_diff = Base.GC_Diff(gc_end, gc_start)
        gc_time = gc_diff.total_time / 1e9
        allocated = gc_diff.allocd
        peak_memory = mem_end - mem_start

        metrics = PerformanceMetrics(
            wall_time,
            wall_time - gc_time,
            gc_time,
            peak_memory,
            allocated,
            gc_diff.pause,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

        (result, metrics)
    end
end

# 使用示例
result, metrics = @measure_performance begin
    G = compute_grm(genotypes)
end

@info "性能指标" wall_time=metrics.wall_time peak_memory=metrics.peak_memory/1e9
```

---

## 2. 性能分析和Profiling

### 2.1 CPU Profiling

```julia
using Profile, ProfileView, PProf

"""
详细的性能分析
"""
function profile_function(f, args...; kwargs...)
    # 预热
    f(args...; kwargs...)

    # Profile
    @profile begin
        for i in 1:100
            f(args...; kwargs...)
        end
    end

    # 生成报告
    ProfileView.view()  # 交互式可视化

    # 或导出火焰图
    PProf.pprof()
end

# 使用示例
profile_function(compute_grm, genotypes)

"""
热点分析
"""
function analyze_hotspots()
    Profile.clear()

    @profile begin
        # 要分析的代码
        model = GBLUPModel()
        fit!(model, genotypes, phenotypes)
    end

    # 找出热点
    Profile.print(
        format=:flat,
        sortedby=:count,
        mincount=10
    )

    # 找出分配最多的函数
    Profile.print(
        format=:flat,
        sortedby=:allocations
    )
end
```

### 2.2 内存Profiling

```julia
"""
内存分配分析
"""
function profile_allocations(f, args...)
    # 开启分配跟踪
    Profile.Allocs.@profile sample_rate=0.0001 begin
        f(args...)
    end

    # 分析结果
    results = Profile.Allocs.fetch()

    # 按分配大小排序
    sorted = sort(collect(results), by=x->x.size, rev=true)

    println("Top 10 分配:")
    for (i, alloc) in enumerate(sorted[1:min(10, length(sorted))])
        println("$i. $(alloc.type): $(alloc.size) bytes at $(alloc.stacktrace[1])")
    end
end

"""
检测内存泄漏
"""
function check_memory_leak(f, iterations=1000)
    gc()  # 初始 GC
    mem_start = Base.gc_live_bytes()

    for i in 1:iterations
        f()

        if i % 100 == 0
            gc()
            mem_current = Base.gc_live_bytes()
            growth = (mem_current - mem_start) / 1e6

            if growth > 100  # 超过 100MB 增长
                @warn "可能的内存泄漏" iteration=i growth_mb=growth
            end
        end
    end

    gc()
    mem_end = Base.gc_live_bytes()
    total_growth = (mem_end - mem_start) / 1e6

    println("总内存增长: $(total_growth) MB")
    return total_growth
end
```

### 2.3 类型稳定性分析

```julia
using Cthulhu, JET

"""
检查类型稳定性
"""
function check_type_stability(f, args...)
    # 使用 @code_warntype
    @code_warntype f(args...)

    # 使用 JET 进行静态分析
    JET.@report_opt f(args...)

    # 使用 Cthulhu 深入分析
    @descend f(args...)
end

"""
自动修复类型不稳定性
"""
function suggest_type_annotations(f, args...)
    # 推断类型
    inferred = @inferred f(args...)

    # 如果失败，给出建议
    # ...
end

# 示例：检查 GRM 计算的类型稳定性
check_type_stability(compute_grm, genotypes)
```

---

## 3. 内存优化策略

### 3.1 就地操作 (In-place Operations)

```julia
"""
避免不必要的分配
"""
# ❌ 差的实现 (每次都分配新数组)
function standardize_bad(X::Matrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    return (X .- μ) ./ σ
end

# ✅ 好的实现 (就地修改)
function standardize_good!(X::Matrix)
    n, m = size(X)

    for j in 1:m
        # 计算统计量
        μ = 0.0
        for i in 1:n
            μ += X[i, j]
        end
        μ /= n

        σ² = 0.0
        for i in 1:n
            σ² += (X[i, j] - μ)^2
        end
        σ = sqrt(σ² / (n - 1))

        # 就地标准化
        for i in 1:n
            X[i, j] = (X[i, j] - μ) / σ
        end
    end

    return X
end

# 性能对比
@btime standardize_bad($X)   # 分配: ~8 MB
@btime standardize_good!($X) # 分配: ~0 MB
```

### 3.2 视图 (Views)

```julia
"""
使用视图避免复制
"""
# ❌ 复制数据
function process_chunk_bad(X::Matrix, chunk_start::Int, chunk_end::Int)
    chunk = X[:, chunk_start:chunk_end]  # 复制!
    return chunk' * chunk
end

# ✅ 使用视图
function process_chunk_good(X::Matrix, chunk_start::Int, chunk_end::Int)
    chunk = @view X[:, chunk_start:chunk_end]  # 零拷贝
    return chunk' * chunk
end

# GRM 计算中的应用
function compute_grm_optimized(geno::CompactGenotypes; chunk_size=1000)
    n = n_samples(geno)
    m = n_markers(geno)
    G = zeros(Float64, n, n)

    for chunk_start in 1:chunk_size:m
        chunk_end = min(chunk_start + chunk_size - 1, m)

        # 使用视图提取数据
        Z = extract_chunk(geno, chunk_start:chunk_end)

        # 累加
        BLAS.syrk!('U', 'N', 1.0, Z, 1.0, G)
    end

    return G
end
```

### 3.3 内存池和对象复用

```julia
"""
对象池模式
"""
mutable struct MatrixPool
    pool::Vector{Matrix{Float64}}
    sizes::Dict{Tuple{Int,Int}, Vector{Matrix{Float64}}}

    MatrixPool() = new(Matrix{Float64}[], Dict())
end

function acquire!(pool::MatrixPool, m::Int, n::Int)::Matrix{Float64}
    size_key = (m, n)

    if !haskey(pool.sizes, size_key) || isempty(pool.sizes[size_key])
        # 创建新矩阵
        return Matrix{Float64}(undef, m, n)
    else
        # 复用现有矩阵
        return pop!(pool.sizes[size_key])
    end
end

function release!(pool::MatrixPool, mat::Matrix{Float64})
    size_key = size(mat)

    if !haskey(pool.sizes, size_key)
        pool.sizes[size_key] = Matrix{Float64}[]
    end

    push!(pool.sizes[size_key], mat)
end

# 使用示例
pool = MatrixPool()

function compute_with_pool(pool::MatrixPool, n::Int, m::Int)
    # 获取矩阵
    temp = acquire!(pool, n, m)

    try
        # 使用矩阵进行计算
        # ...
        result = compute_something(temp)
        return result
    finally
        # 归还矩阵
        release!(pool, temp)
    end
end
```

### 3.4 稀疏矩阵优化

```julia
using SparseArrays

"""
智能稀疏矩阵转换
"""
function to_sparse_if_beneficial(A::Matrix{T}, threshold::Float64=0.5) where T
    sparsity = count(iszero, A) / length(A)

    if sparsity > threshold
        # 转换为稀疏矩阵可以节省内存
        sparse_size = nnz_estimate(A) * (sizeof(T) + sizeof(Int) * 2)
        dense_size = sizeof(A)

        @info "稀疏度: $(sparsity*100)%, 稀疏存储可节省 $(100*(1-sparse_size/dense_size))% 内存"

        return sparse(A)
    else
        return A
    end
end

"""
稀疏 GRM (用于远缘群体)
"""
function compute_sparse_grm(geno::CompactGenotypes; threshold::Float64=0.01)
    n = n_samples(geno)
    m = n_markers(geno)

    # 使用稀疏矩阵存储 G
    I = Int[]
    J = Int[]
    V = Float64[]

    for i in 1:n
        for j in i:n
            g_ij = compute_relationship(geno, i, j)

            # 只存储显著的关系
            if abs(g_ij) > threshold
                push!(I, i)
                push!(J, j)
                push!(V, g_ij)

                if i != j
                    push!(I, j)
                    push!(J, i)
                    push!(V, g_ij)
                end
            end
        end
    end

    return sparse(I, J, V, n, n)
end
```

---

## 4. 并行计算优化

### 4.1 多线程

```julia
using ThreadsX, FLoops

"""
多线程 GRM 计算
"""
function compute_grm_threaded(geno::CompactGenotypes; chunk_size=1000)
    n = n_samples(geno)
    m = n_markers(geno)

    # 每个线程一个局部 G 矩阵
    n_threads = Threads.nthreads()
    local_Gs = [zeros(Float64, n, n) for _ in 1:n_threads]

    # 并行处理块
    chunks = partition(1:m, chunk_size)

    Threads.@threads for chunk in chunks
        tid = Threads.threadid()
        Z = extract_and_standardize(geno, chunk)

        # 累加到本地 G
        BLAS.syrk!('U', 'N', 1.0, Z, 1.0, local_Gs[tid])
    end

    # 合并所有局部 G
    G = reduce(+, local_Gs)

    # 对称化
    copytri!(G, 'U')

    return G
end

"""
使用 FLoops 进行并行化
"""
function compute_allele_frequencies_parallel(geno::CompactGenotypes)
    m = n_markers(geno)
    freqs = zeros(Float64, m)

    @floop ThreadedEx() for j in 1:m
        sum_val = 0.0
        n_valid = 0

        for i in 1:n_samples(geno)
            val = geno[i, j]
            if !ismissing(val)
                sum_val += val
                n_valid += 1
            end
        end

        @reduce() do (freqs_reduce = freqs)
            freqs_reduce[j] = sum_val / (2 * n_valid)
        end
    end

    return freqs
end
```

### 4.2 SIMD 向量化

```julia
using SIMD

"""
SIMD 优化的点积
"""
function dot_simd(x::Vector{Float64}, y::Vector{Float64})
    @assert length(x) == length(y)
    n = length(x)

    # SIMD 宽度
    vec_width = 8  # AVX512: 8 个 Float64
    n_vecs = div(n, vec_width)
    remainder = rem(n, vec_width)

    # SIMD 累加
    sum_vec = Vec{vec_width, Float64}(tuple([0.0 for i in 1:vec_width]...))

    for i in 1:n_vecs
        offset = (i - 1) * vec_width + 1
        x_vec = vload(Vec{vec_width, Float64}, x, offset)
        y_vec = vload(Vec{vec_width, Float64}, y, offset)
        sum_vec += x_vec * y_vec
    end

    # 水平求和
    result = sum(sum_vec)

    # 处理剩余元素
    for i in (n_vecs * vec_width + 1):n
        result += x[i] * y[i]
    end

    return result
end

# 性能对比
x = randn(10000)
y = randn(10000)

@btime dot($x, $y)           # 标准实现
@btime dot_simd($x, $y)      # SIMD 优化 (预期 2-4x 加速)
```

### 4.3 分布式计算

```julia
using Distributed

"""
分布式 GRM 计算
"""
@everywhere function compute_grm_block(
    geno_data::Vector{UInt8},
    n_samples::Int,
    marker_range::UnitRange{Int},
    freqs::Vector{Float64}
)
    # 重建基因型对象
    geno = reconstruct_genotypes(geno_data, n_samples, marker_range)

    # 提取并标准化
    Z = extract_and_standardize(geno, freqs[marker_range])

    # 计算部分 GRM
    return Z * Z'
end

function compute_grm_distributed(geno::CompactGenotypes, n_workers::Int=nworkers())
    n = n_samples(geno)
    m = n_markers(geno)

    # 计算等位基因频率
    freqs = allele_frequencies(geno)

    # 分割标记到各个 worker
    marker_ranges = partition(1:m, n_workers)

    # 序列化基因型数据
    geno_data = geno.data

    # 分布式计算
    futures = [@spawnat w compute_grm_block(geno_data, n, range, freqs)
               for (w, range) in zip(workers(), marker_ranges)]

    # 收集结果
    partial_Gs = fetch.(futures)

    # 合并
    G = reduce(+, partial_Gs)

    return G
end

# 启动 workers
addprocs(4)

# 使用
G = compute_grm_distributed(genotypes)
```

---

## 5. GPU加速

### 5.1 CUDA 内核优化

```julia
using CUDA

"""
GPU GRM 计算内核
"""
function grm_kernel!(G, X, n, m)
    # 线程索引
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= n && j <= i
        # 计算 G[i,j] = dot(X[i,:], X[j,:])
        sum = 0.0f0
        for k in 1:m
            sum += X[i, k] * X[j, k]
        end
        G[i, j] = sum
        if i != j
            G[j, i] = sum
        end
    end

    return nothing
end

"""
高性能 GPU GRM 计算
"""
function compute_grm_gpu(geno::CompactGenotypes)
    n = n_samples(geno)
    m = n_markers(geno)

    # 转换为矩阵并标准化
    X = to_matrix(geno)
    standardize!(X)

    # 传输到 GPU
    X_gpu = CuArray{Float32}(X)
    G_gpu = CUDA.zeros(Float32, n, n)

    # 配置内核
    threads_per_block = (16, 16)
    blocks = (cld(n, threads_per_block[1]), cld(n, threads_per_block[2]))

    # 启动内核
    @cuda threads=threads_per_block blocks=blocks grm_kernel!(G_gpu, X_gpu, n, m)

    # 同步
    CUDA.synchronize()

    # 传回 CPU
    G = Array(G_gpu)

    return G
end

"""
使用 cuBLAS 优化
"""
function compute_grm_cublas(geno::CompactGenotypes)
    n = n_samples(geno)

    # 标准化
    X = to_matrix(geno)
    standardize!(X)

    # 传输到 GPU
    X_gpu = CuArray{Float32}(X)

    # 使用 cuBLAS 的 syrk (对称秩k更新)
    G_gpu = CUDA.zeros(Float32, n, n)
    CUBLAS.syrk!('U', 'N', 1.0f0, X_gpu, 0.0f0, G_gpu)

    # 对称化
    G_full_gpu = G_gpu + G_gpu' - Diagonal(G_gpu)

    # 传回
    G = Array(G_full_gpu)

    return G
end
```

### 5.2 GPU 内存管理

```julia
"""
GPU 内存池
"""
struct CUDAMemoryPool
    device_id::Int
    pool::Dict{Int, Vector{CuArray}}

    function CUDAMemoryPool(device_id::Int=0)
        CUDA.device!(device_id)
        new(device_id, Dict())
    end
end

function acquire!(pool::CUDAMemoryPool, size::Int, dtype::Type=Float32)
    if !haskey(pool.pool, size) || isempty(pool.pool[size])
        return CuArray{dtype}(undef, size)
    else
        return pop!(pool.pool[size])
    end
end

function release!(pool::CUDAMemoryPool, arr::CuArray)
    size = length(arr)
    if !haskey(pool.pool, size)
        pool.pool[size] = CuArray[]
    end
    push!(pool.pool[size], arr)
end

"""
批处理 GPU 计算
"""
function compute_grm_batched_gpu(
    geno::CompactGenotypes;
    batch_size::Int=10000,
    max_gpu_memory::Int=8*1024^3  # 8 GB
)
    n = n_samples(geno)
    m = n_markers(geno)

    G = zeros(Float64, n, n)

    # 计算可以一次处理的标记数
    memory_per_marker = n * sizeof(Float32)
    markers_per_batch = min(batch_size, div(max_gpu_memory, memory_per_marker))

    @info "批处理: 每批 $markers_per_batch 标记"

    for batch_start in 1:markers_per_batch:m
        batch_end = min(batch_start + markers_per_batch - 1, m)

        # 提取批次
        X_batch = extract_and_standardize(geno, batch_start:batch_end)

        # 传输到 GPU 并计算
        X_gpu = CuArray{Float32}(X_batch)
        G_batch_gpu = X_gpu * X_gpu'

        # 累加到 G (在 CPU 上)
        G .+= Array(G_batch_gpu)

        # 释放 GPU 内存
        CUDA.unsafe_free!(X_gpu)
        CUDA.unsafe_free!(G_batch_gpu)
        CUDA.reclaim()
    end

    return G
end
```

### 5.3 多 GPU 支持

```julia
"""
多 GPU 并行计算
"""
function compute_grm_multi_gpu(geno::CompactGenotypes, gpu_ids::Vector{Int}=[0, 1])
    n = n_samples(geno)
    m = n_markers(geno)

    n_gpus = length(gpu_ids)
    markers_per_gpu = cld(m, n_gpus)

    # 在每个 GPU 上异步计算
    tasks = Task[]

    for (gpu_idx, gpu_id) in enumerate(gpu_ids)
        task = @async begin
            CUDA.device!(gpu_id)

            # 标记范围
            start_marker = (gpu_idx - 1) * markers_per_gpu + 1
            end_marker = min(gpu_idx * markers_per_gpu, m)

            # 提取数据
            X = extract_and_standardize(geno, start_marker:end_marker)

            # GPU 计算
            X_gpu = CuArray{Float32}(X)
            G_partial_gpu = X_gpu * X_gpu'
            G_partial = Array(G_partial_gpu)

            # 清理
            CUDA.unsafe_free!(X_gpu)
            CUDA.unsafe_free!(G_partial_gpu)

            return G_partial
        end

        push!(tasks, task)
    end

    # 等待所有 GPU 完成
    partial_Gs = fetch.(tasks)

    # 合并结果
    G = reduce(+, partial_Gs)

    return G
end
```

---

## 6. 缓存策略

### 6.1 计算结果缓存

```julia
using LRUCache

"""
LRU 缓存装饰器
"""
struct CachedFunction{F}
    func::F
    cache::LRU{UInt64, Any}
    max_size::Int

    function CachedFunction(f::F; max_size::Int=100) where F
        new{F}(f, LRU{UInt64, Any}(maxsize=max_size), max_size)
    end
end

function (cf::CachedFunction)(args...; kwargs...)
    # 计算缓存键
    key = hash((args, kwargs))

    # 检查缓存
    if haskey(cf.cache, key)
        @debug "缓存命中" func=cf.func args=args
        return cf.cache[key]
    end

    # 计算结果
    result = cf.func(args...; kwargs...)

    # 缓存结果
    cf.cache[key] = result

    return result
end

# 使用示例
cached_grm = CachedFunction(compute_grm; max_size=10)

# 第一次调用：计算
G1 = cached_grm(genotypes)  # 慢

# 第二次调用：从缓存
G2 = cached_grm(genotypes)  # 快！
```

### 6.2 中间结果缓存

```julia
"""
等位基因频率缓存
"""
mutable struct CompactGenotypesWithCache{T<:Integer} <: AbstractGenotypeData{T}
    # 原始字段
    data::Vector{UInt8}
    n_samples::Int
    n_markers::Int
    # ...

    # 缓存
    _allele_freqs::Union{Vector{Float64}, Nothing}
    _missing_rates::Union{Vector{Float64}, Nothing}
    _standardized_matrix::Union{Matrix{Float64}, Nothing}

    # 缓存时间戳
    _cache_timestamp::Dict{Symbol, DateTime}

    function CompactGenotypesWithCache(data, n_samples, n_markers, ...)
        new{T}(
            data, n_samples, n_markers, ...,
            nothing, nothing, nothing,
            Dict{Symbol, DateTime}()
        )
    end
end

"""
延迟计算并缓存等位基因频率
"""
function allele_frequencies(cg::CompactGenotypesWithCache)
    if isnothing(cg._allele_freqs)
        @debug "计算等位基因频率"
        cg._allele_freqs = compute_allele_frequencies_kahan(cg)
        cg._cache_timestamp[:allele_freqs] = now()
    end
    return cg._allele_freqs
end

"""
失效缓存
"""
function invalidate_cache!(cg::CompactGenotypesWithCache, key::Symbol)
    if key == :all
        cg._allele_freqs = nothing
        cg._missing_rates = nothing
        cg._standardized_matrix = nothing
        empty!(cg._cache_timestamp)
    elseif key == :allele_freqs
        cg._allele_freqs = nothing
        delete!(cg._cache_timestamp, :allele_freqs)
    end
end
```

### 6.3 磁盘缓存

```julia
using Serialization, SHA

"""
磁盘缓存管理器
"""
struct DiskCache
    cache_dir::String
    max_size_gb::Float64

    function DiskCache(dir::String="./cache"; max_size_gb::Float64=10.0)
        mkpath(dir)
        new(dir, max_size_gb)
    end
end

function cache_key(func_name::String, args...; kwargs...)::String
    # 生成唯一键
    key_data = (func_name, args, kwargs)
    return bytes2hex(sha256(string(key_data)))
end

function get(cache::DiskCache, key::String)
    filepath = joinpath(cache.cache_dir, key * ".jls")

    if isfile(filepath)
        @debug "磁盘缓存命中" key=key
        return deserialize(filepath)
    end

    return nothing
end

function put!(cache::DiskCache, key::String, value)
    filepath = joinpath(cache.cache_dir, key * ".jls")

    # 序列化
    serialize(filepath, value)

    # 检查缓存大小
    check_cache_size!(cache)
end

function check_cache_size!(cache::DiskCache)
    total_size = 0.0

    for file in readdir(cache.cache_dir; join=true)
        total_size += filesize(file) / 1e9  # GB
    end

    if total_size > cache.max_size_gb
        @warn "缓存超过限制" total_size_gb=total_size max_size_gb=cache.max_size_gb
        # 删除最旧的文件
        evict_old_entries!(cache)
    end
end

"""
缓存装饰器
"""
macro disk_cache(cache, expr)
    @assert expr.head == :call "只能缓存函数调用"

    func_name = string(expr.args[1])
    args = expr.args[2:end]

    quote
        key = cache_key($func_name, $(args...))
        cached_value = get($(esc(cache)), key)

        if !isnothing(cached_value)
            cached_value
        else
            result = $(esc(expr))
            put!($(esc(cache)), key, result)
            result
        end
    end
end

# 使用示例
disk_cache = DiskCache("./grm_cache"; max_size_gb=20.0)

G = @disk_cache disk_cache compute_grm(genotypes)  # 第一次：计算并缓存
G = @disk_cache disk_cache compute_grm(genotypes)  # 第二次：从磁盘加载
```

---

## 7. 基准测试框架

### 7.1 基准测试套件

```julia
using BenchmarkTools

"""
基准测试套件
"""
const BENCHMARK_SUITE = BenchmarkGroup()

# GRM 基准测试
BENCHMARK_SUITE["grm"] = BenchmarkGroup(["computation"])
BENCHMARK_SUITE["grm"]["small"] = @benchmarkable compute_grm($small_geno) setup=(small_geno=generate_test_genotypes(100, 1000))
BENCHMARK_SUITE["grm"]["medium"] = @benchmarkable compute_grm($med_geno) setup=(med_geno=generate_test_genotypes(1000, 10000))
BENCHMARK_SUITE["grm"]["large"] = @benchmarkable compute_grm($large_geno) setup=(large_geno=generate_test_genotypes(5000, 50000))

# GBLUP 基准测试
BENCHMARK_SUITE["gblup"] = BenchmarkGroup(["models"])
BENCHMARK_SUITE["gblup"]["cholesky"] = @benchmarkable begin
    model = GBLUPModel(GBLUPConfig(solver=:cholesky))
    fit!(model, geno, pheno)
end setup=(geno, pheno = generate_test_data(1000, 10000))

BENCHMARK_SUITE["gblup"]["pcg"] = @benchmarkable begin
    model = GBLUPModel(GBLUPConfig(solver=:pcg))
    fit!(model, geno, pheno)
end setup=(geno, pheno = generate_test_data(1000, 10000))

# BayesR 基准测试
BENCHMARK_SUITE["bayesr"] = BenchmarkGroup(["models", "bayesian"])
BENCHMARK_SUITE["bayesr"]["1k_iter"] = @benchmarkable begin
    model = BayesRModel(BayesRConfig(num_iterations=1000, burn_in=100))
    fit!(model, geno, pheno)
end setup=(geno, pheno = generate_test_data(500, 5000))

"""
运行所有基准测试
"""
function run_benchmarks()
    results = run(BENCHMARK_SUITE; verbose=true)
    return results
end

"""
生成基准测试报告
"""
function generate_benchmark_report(results::BenchmarkGroup)
    report = """
    # GenomicPro 2.0 性能基准测试报告

    **日期**: $(now())
    **Julia 版本**: $(VERSION)
    **CPU**: $(Sys.CPU_NAME)
    **内存**: $(Sys.total_memory() / 1e9) GB

    ## 结果

    """

    for (group_name, group) in results
        report *= "\n### $group_name\n\n"
        report *= "| 测试 | 时间 (中位数) | 内存 | 分配 |\n"
        report *= "|------|--------------|------|------|\n"

        for (test_name, result) in group
            time_str = BenchmarkTools.prettytime(median(result).time)
            memory_str = BenchmarkTools.prettymemory(median(result).memory)
            allocs = median(result).allocs

            report *= "| $test_name | $time_str | $memory_str | $allocs |\n"
        end
    end

    return report
end

# 运行并生成报告
results = run_benchmarks()
report = generate_benchmark_report(results)
write("benchmark_report.md", report)
```

### 7.2 性能对比

```julia
"""
对比两个实现的性能
"""
function compare_implementations(
    impl1::Function,
    impl2::Function,
    args...;
    name1::String="Implementation 1",
    name2::String="Implementation 2"
)
    # 基准测试
    b1 = @benchmark $impl1($args...)
    b2 = @benchmark $impl2($args...)

    # 对比
    time_ratio = median(b2).time / median(b1).time
    memory_ratio = median(b2).memory / median(b1).memory

    println("性能对比:")
    println("=" ^ 60)
    println("$name1:")
    println("  时间: ", BenchmarkTools.prettytime(median(b1).time))
    println("  内存: ", BenchmarkTools.prettymemory(median(b1).memory))
    println()
    println("$name2:")
    println("  时间: ", BenchmarkTools.prettytime(median(b2).time), " ($(round(time_ratio, digits=2))x)")
    println("  内存: ", BenchmarkTools.prettymemory(median(b2).memory), " ($(round(memory_ratio, digits=2))x)")
    println("=" ^ 60)

    if time_ratio < 1.0
        println("✅ $name2 更快 ($(round((1-time_ratio)*100, digits=1))%)")
    else
        println("⚠️  $name1 更快 ($(round((time_ratio-1)*100, digits=1))%)")
    end

    return (b1, b2)
end

# 使用示例
compare_implementations(
    compute_grm_cpu,
    compute_grm_gpu,
    genotypes;
    name1="CPU 实现",
    name2="GPU 实现"
)
```

---

## 8. 性能回归检测

### 8.1 持续性能监控

```julia
using JSON3

"""
性能回归检测
"""
struct PerformanceRegression
    baseline::BenchmarkGroup
    threshold::Float64  # 允许的性能下降百分比

    function PerformanceRegression(baseline_file::String; threshold::Float64=0.1)
        baseline = load_baseline(baseline_file)
        new(baseline, threshold)
    end
end

function check_regression(pr::PerformanceRegression, current::BenchmarkGroup)
    regressions = Tuple{String, Float64}[]

    for (group_name, group) in current
        if !haskey(pr.baseline, group_name)
            @warn "新的基准测试组: $group_name"
            continue
        end

        baseline_group = pr.baseline[group_name]

        for (test_name, result) in group
            if !haskey(baseline_group, test_name)
                @warn "新的测试: $group_name/$test_name"
                continue
            end

            baseline_result = baseline_group[test_name]

            # 对比时间
            current_time = median(result).time
            baseline_time = median(baseline_result).time
            ratio = current_time / baseline_time

            if ratio > (1.0 + pr.threshold)
                push!(regressions, ("$group_name/$test_name", ratio - 1.0))
            end
        end
    end

    return regressions
end

function report_regressions(regressions::Vector{Tuple{String, Float64}})
    if isempty(regressions)
        println("✅ 没有性能回归")
        return
    end

    println("⚠️  检测到性能回归:")
    println("=" ^ 60)

    for (test, regression) in regressions
        println("  $test: $(round(regression*100, digits=1))% 慢于基线")
    end

    println("=" ^ 60)

    # CI 失败
    if !isempty(regressions)
        error("性能回归检测失败")
    end
end

# CI 集成
function ci_performance_check()
    # 运行基准测试
    current = run(BENCHMARK_SUITE)

    # 加载基线
    pr = PerformanceRegression("benchmarks/baseline.json")

    # 检测回归
    regressions = check_regression(pr, current)

    # 报告
    report_regressions(regressions)

    # 更新基线 (仅在主分支)
    if ENV["BRANCH"] == "main"
        save_baseline("benchmarks/baseline.json", current)
    end
end
```

### 8.2 性能可视化

```julia
using Plots

"""
性能趋势图
"""
function plot_performance_history(history_file::String)
    # 加载历史数据
    history = load_history(history_file)

    # 提取时间序列
    dates = [entry.date for entry in history]
    times = [entry.median_time for entry in history]

    # 绘图
    plot(
        dates,
        times,
        title="GRM 计算性能趋势",
        xlabel="日期",
        ylabel="时间 (秒)",
        legend=false,
        marker=:circle
    )

    # 添加趋势线
    plot!(dates, smooth(times), linestyle=:dash, color=:red)

    savefig("performance_history.png")
end
```

---

## 9. 大规模数据优化

### 9.1 流式处理

```julia
"""
流式 VCF 处理
"""
struct StreamingVCFReader
    filepath::String
    chunk_size::Int
    buffer::Channel{Matrix{UInt8}}

    function StreamingVCFReader(filepath::String; chunk_size::Int=1000)
        buffer = Channel{Matrix{UInt8}}(10)  # 缓冲 10 个块
        reader = new(filepath, chunk_size, buffer)

        # 启动后台读取
        @async stream_chunks(reader)

        return reader
    end
end

function stream_chunks(reader::StreamingVCFReader)
    # 打开文件
    io = open(reader.filepath, "r")

    try
        while !eof(io)
            # 读取块
            chunk = read_chunk(io, reader.chunk_size)

            # 放入缓冲
            put!(reader.buffer, chunk)
        end
    finally
        close(io)
        close(reader.buffer)
    end
end

function Base.iterate(reader::StreamingVCFReader, state=nothing)
    try
        chunk = take!(reader.buffer)
        return (chunk, nothing)
    catch e
        if isa(e, InvalidStateException)
            return nothing
        else
            rethrow(e)
        end
    end
end

# 使用示例
reader = StreamingVCFReader("huge_dataset.vcf.gz")

G = zeros(n, n)
for chunk in reader
    # 处理块
    G_partial = process_chunk(chunk)
    G .+= G_partial
end
```

### 9.2 外存算法

```julia
"""
外存 GRM 计算
"""
function compute_grm_out_of_core(
    vcf_file::String;
    max_memory_gb::Float64=4.0,
    temp_dir::String="./temp"
)
    mkpath(temp_dir)

    # 1. 第一遍：计算等位基因频率
    freqs = compute_frequencies_streaming(vcf_file)

    # 2. 第二遍：分块计算并写入磁盘
    reader = StreamingVCFReader(vcf_file)
    chunk_id = 0

    for chunk in reader
        chunk_id += 1

        # 标准化
        Z = standardize(chunk, freqs)

        # 计算部分 GRM
        G_partial = Z * Z'

        # 写入磁盘
        save_chunk(joinpath(temp_dir, "grm_chunk_$chunk_id.bin"), G_partial)
    end

    # 3. 合并所有块
    G = merge_grm_chunks(temp_dir, n)

    # 4. 清理
    rm(temp_dir; recursive=true)

    return G
end
```

---

**待续...**

文档长度：约 **18,000** 字

下一部分将包含：
- 可观测性和监控设计
- API 设计规范
- 安全性设计

是否继续？
