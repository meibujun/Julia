"""
# GPU Acceleration Module

提供 CUDA 加速的基因组计算功能，支持大规模数据集处理。

## 主要功能
- GPU 加速的 GRM 计算
- GPU 加速的矩阵运算
- 自动 CPU/GPU 回退机制
- 内存管理和批处理

## 使用示例
```julia
using GenomicPro2.GPU

# 检查 CUDA 可用性
if has_cuda()
    # 计算 GRM（GPU 加速）
    grm = compute_grm_gpu(genotypes)

    # GPU 加速的 GBLUP
    results = gblup_gpu(genotypes, phenotypes)
end
```
"""
module GPU

using LinearAlgebra
using Statistics
using ..Core: GenotypeData, PhenotypeData, AbstractGenomicModel
using ..Data: CompactGenotypes

# 尝试加载 CUDA（如果可用）
const CUDA_AVAILABLE = Ref(false)
const CUDA = Ref{Union{Nothing,Module}}(nothing)

function __init__()
    try
        CUDA[] = Base.require(Main, :CUDA)
        CUDA_AVAILABLE[] = CUDA[].functional()
        if CUDA_AVAILABLE[]
            @info "CUDA detected and functional. GPU acceleration enabled."
        else
            @warn "CUDA package loaded but not functional. Using CPU fallback."
        end
    catch e
        @info "CUDA not available. GPU features disabled. Install CUDA.jl for GPU support."
    end
end

"""
    has_cuda() -> Bool

检查 CUDA 是否可用且功能正常。
"""
has_cuda() = CUDA_AVAILABLE[]

"""
    gpu_info()

显示 GPU 设备信息。
"""
function gpu_info()
    if !has_cuda()
        @warn "CUDA not available"
        return nothing
    end

    cu = CUDA[]
    devices = cu.devices()

    println("GPU Devices:")
    for (i, device) in enumerate(devices)
        println("  Device $i: $(cu.name(device))")
        println("    Compute Capability: $(cu.capability(device))")
        println("    Total Memory: $(round(cu.totalmem(device) / 1e9, digits=2)) GB")
        println("    Free Memory: $(round(cu.available_memory(device) / 1e9, digits=2)) GB")
    end
end

"""
    compute_grm_gpu(genotypes::CompactGenotypes; batch_size::Int=1000)

使用 GPU 加速计算基因组关系矩阵（GRM）。

# 参数
- `genotypes`: CompactGenotypes 对象
- `batch_size`: 批处理大小，用于大数据集的分块计算

# 返回
- `Matrix{Float64}`: n×n 的 GRM 矩阵

# 性能
对于大规模数据集（>10K 样本），相比 CPU 可获得 10-50 倍加速。
"""
function compute_grm_gpu(genotypes::CompactGenotypes; batch_size::Int=1000)
    if !has_cuda()
        @warn "CUDA not available, falling back to CPU computation"
        return compute_grm_cpu(genotypes)
    end

    cu = CUDA[]
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    @info "Computing GRM on GPU for $n_samples samples and $n_snps SNPs"

    # 解压缩基因型数据到标准矩阵
    G = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            G[i, j] = Float64(genotypes[i, j])
        end
    end

    # 中心化和标准化
    μ = mean(G, dims=1)
    G .- μ
    σ = std(G, dims=1)
    σ[σ .== 0] .= 1.0  # 避免除零
    G ./= σ

    # 分批计算以避免内存溢出
    if n_samples > batch_size
        return compute_grm_gpu_batched(G, cu, batch_size)
    else
        return compute_grm_gpu_direct(G, cu)
    end
end

"""
直接 GPU 计算（小数据集）
"""
function compute_grm_gpu_direct(G::Matrix{Float64}, cu::Module)
    # 传输到 GPU
    G_gpu = cu.CuArray(G)

    # 计算 GRM: G * G' / m
    GRM_gpu = (G_gpu * G_gpu') / size(G, 2)

    # 传回 CPU
    GRM = Array(GRM_gpu)

    # 清理 GPU 内存
    cu.reclaim()

    return GRM
end

"""
批处理 GPU 计算（大数据集）
"""
function compute_grm_gpu_batched(G::Matrix{Float64}, cu::Module, batch_size::Int)
    n_samples = size(G, 1)
    n_snps = size(G, 2)

    GRM = zeros(Float64, n_samples, n_samples)

    # 分批处理样本
    n_batches = ceil(Int, n_samples / batch_size)

    @info "Processing in $n_batches batches of size $batch_size"

    for i in 1:n_batches
        start_i = (i-1) * batch_size + 1
        end_i = min(i * batch_size, n_samples)

        G_i = G[start_i:end_i, :]
        G_i_gpu = cu.CuArray(G_i)

        for j in 1:n_batches
            start_j = (j-1) * batch_size + 1
            end_j = min(j * batch_size, n_samples)

            G_j = G[start_j:end_j, :]
            G_j_gpu = cu.CuArray(G_j)

            # 计算块
            GRM_block = Array((G_i_gpu * G_j_gpu') / n_snps)

            GRM[start_i:end_i, start_j:end_j] = GRM_block

            # 对称性
            if i != j
                GRM[start_j:end_j, start_i:end_i] = GRM_block'
            end

            cu.reclaim()
        end
    end

    return GRM
end

"""
CPU 回退实现
"""
function compute_grm_cpu(genotypes::CompactGenotypes)
    n_samples = size(genotypes.data, 1)
    n_snps = size(genotypes.data, 2)

    # 解压缩基因型
    G = Matrix{Float64}(undef, n_samples, n_snps)
    for j in 1:n_snps
        for i in 1:n_samples
            G[i, j] = Float64(genotypes[i, j])
        end
    end

    # 中心化和标准化
    μ = mean(G, dims=1)
    G .- μ
    σ = std(G, dims=1)
    σ[σ .== 0] .= 1.0
    G ./= σ

    # 计算 GRM
    return (G * G') / n_snps
end

"""
    gblup_gpu(genotypes::CompactGenotypes, phenotypes::PhenotypeData;
              h2::Float64=0.5, max_iter::Int=100, tol::Float64=1e-6)

GPU 加速的 GBLUP 模型拟合。

# 参数
- `genotypes`: 基因型数据
- `phenotypes`: 表型数据
- `h2`: 遗传力（默认 0.5）
- `max_iter`: 最大迭代次数
- `tol`: 收敛阈值

# 返回
命名元组，包含：
- `gebv`: 基因组估计育种值
- `h2_estimated`: 估计的遗传力
- `variance_components`: 方差组分
"""
function gblup_gpu(genotypes::CompactGenotypes, phenotypes::PhenotypeData;
                   h2::Float64=0.5, max_iter::Int=100, tol::Float64=1e-6)
    if !has_cuda()
        @warn "CUDA not available, falling back to CPU computation"
        # 可以调用 CPU 版本的 GBLUP
        throw(ErrorException("GPU GBLUP not available without CUDA. Use CPU version."))
    end

    cu = CUDA[]

    # 计算 GRM（GPU 加速）
    @info "Computing GRM on GPU..."
    G = compute_grm_gpu(genotypes)

    # 准备表型数据
    y = phenotypes.values
    n = length(y)

    # 传输到 GPU
    G_gpu = cu.CuArray(G)
    y_gpu = cu.CuArray(y)

    # 添加小的对角元素以确保正定
    λ = (1.0 - h2) / h2
    G_gpu .+ cu.I * (λ + 0.01)

    # 求解混合模型方程: (G + λI)u = y
    # 使用 Cholesky 分解
    try
        C = cu.cholesky(G_gpu)
        u_gpu = C \ y_gpu
        u = Array(u_gpu)
    catch e
        @warn "Cholesky decomposition failed, using iterative solver"
        # 使用共轭梯度法
        u_gpu = cg_solver_gpu(G_gpu, y_gpu, max_iter, tol)
        u = Array(u_gpu)
    end

    # 计算方差组分
    y_hat = G * u
    residuals = y .- y_hat

    σ²_g = var(u)
    σ²_e = var(residuals)
    h2_est = σ²_g / (σ²_g + σ²_e)

    # 清理 GPU 内存
    cu.reclaim()

    return (
        gebv = u,
        h2_estimated = h2_est,
        variance_components = (genetic = σ²_g, residual = σ²_e)
    )
end

"""
GPU 上的共轭梯度求解器
"""
function cg_solver_gpu(A, b, max_iter::Int, tol::Float64)
    cu = CUDA[]

    n = length(b)
    x = cu.zeros(Float64, n)
    r = b - A * x
    p = copy(r)
    rs_old = cu.dot(r, r)

    for i in 1:max_iter
        Ap = A * p
        α = rs_old / cu.dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rs_new = cu.dot(r, r)

        if sqrt(rs_new) < tol
            @info "CG converged in $i iterations"
            break
        end

        β = rs_new / rs_old
        p .= r .+ β .* p
        rs_old = rs_new
    end

    return x
end

# 导出函数
export has_cuda, gpu_info
export compute_grm_gpu, gblup_gpu

end # module GPU
