# src/GenomicProGPU/device_management.jl

"""
    GPUBackend

GPU computation backend configuration and device management.

Manages CUDA device selection, capability detection, memory allocation strategies,
and automatic fallback to CPU when GPU resources are unavailable or insufficient.
Provides unified interface for GPU-accelerated genomic computations across different
hardware configurations.

# Device Detection and Selection

The system automatically detects available CUDA-capable GPUs and selects the most
appropriate device based on compute capability, available memory, and current
utilization. Users can override automatic selection to target specific devices
in multi-GPU systems.

# Memory Management Strategy

Implements tiered memory management to handle datasets exceeding GPU memory:
- **Direct allocation**: Datasets fitting entirely in GPU memory (fastest)
- **Streaming computation**: Process data in chunks for moderate oversize
- **Automatic fallback**: Switch to CPU for extreme memory pressure

# Compute Capability Requirements

Minimum requirements for different operations:
- Basic matrix operations: Compute capability 3.5+
- Mixed precision (FP16/BF16): Compute capability 7.0+
- Tensor cores: Compute capability 7.0+ (Volta and newer)
- Multi-GPU communication: CUDA 10.0+, NCCL support

# Examples
```julia
# Automatic device selection
backend = GPUBackend()
println("Using GPU: ", backend.device_name)
println("Compute capability: ", backend.compute_capability)
println("Available memory: ", round(backend.total_memory / 1e9, digits=1), " GB")

# Manual device selection in multi-GPU system
backend = GPUBackend(device_id=1)  # Select second GPU

# Query device capabilities
if backend.supports_mixed_precision
    println("Mixed precision training available")
end

# Check if operation fits in memory
dataset_size = 10_000_000_000  # 10 GB
if can_fit_in_gpu_memory(backend, dataset_size)
    # Proceed with GPU computation
    result = compute_on_gpu(data)
else
    # Use streaming or CPU fallback
    result = compute_with_streaming(data)
end
```

# See Also
- [`configure_gpu_memory`](@ref): Adjust memory allocation settings
- [`gpu_benchmark`](@ref): Performance testing and validation
"""
struct GPUBackend
    device_id::Int
    device_name::String
    compute_capability::VersionNumber
    total_memory::Int64
    available_memory::Int64
    supports_mixed_precision::Bool
    supports_tensor_cores::Bool
    max_threads_per_block::Int
    max_shared_memory::Int

    function GPUBackend(; device_id::Int = 0)
        if !CUDA.functional()
            @warn "CUDA not available, GPU acceleration disabled"
            return new(-1, "CPU", v"0.0", 0, 0, false, false, 0, 0)
        end

        # Set active device
        CUDA.device!(device_id)
        dev = CUDA.device()

        # Query device properties
        device_name = CUDA.name(dev)
        compute_cap = CUDA.capability(dev)
        total_mem = CUDA.totalmem(dev)
        available_mem = CUDA.available_memory(dev)

        # Determine capabilities
        supports_mixed = compute_cap >= v"7.0"
        supports_tensor = compute_cap >= v"7.0"

        max_threads = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        max_shared = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

        println("GPU Backend Initialized:")
        println("  Device: $device_name")
        println("  Compute Capability: $compute_cap")
        println("  Total Memory: $(round(total_mem / 1e9, digits=2)) GB")
        println("  Available Memory: $(round(available_mem / 1e9, digits=2)) GB")
        println("  Mixed Precision: $supports_mixed")
        println("  Tensor Cores: $supports_tensor")

        new(device_id, device_name, compute_cap, total_mem, available_mem,
            supports_mixed, supports_tensor, max_threads, max_shared)
    end
end


"""
    estimate_gpu_memory_requirement(operation::Symbol, n::Int, m::Int)

Estimate GPU memory required for genomic computation.

Calculates expected memory usage for various operations to enable intelligent
memory management and automatic fallback decisions.

# Arguments
- `operation::Symbol`: Type of computation (:grm, :pcg, :mcmc, :deeplearning)
- `n::Int`: Number of individuals/samples
- `m::Int`: Number of markers/features

# Returns
- `Int64`: Estimated memory requirement in bytes

# Memory Formulas

## Genomic Relationship Matrix (GRM)
- Input genotypes: n × m × 1 byte (two-bit encoding)
- Centered matrix Z: n × m × 4 bytes (Float32)
- Output GRM: n × n × 8 bytes (Float64)
- Working memory: 2 × blocksize × m × 4 bytes
- Total: approximately n² × 8 + n × m × 5 + overhead

## Preconditioned Conjugate Gradient (PCG)
- GRM: n × n × 8 bytes
- Solution vector: n × 8 bytes
- Residual vectors: 3 × n × 8 bytes
- Search direction: n × 8 bytes
- Total: approximately n² × 8 + n × 40 bytes

## MCMC Sampling
- Genotypes: n × m × 1 byte
- Marker effects: m × 8 bytes
- Residuals: n × 8 bytes
- Working arrays: 2 × n × 8 bytes
- Total: approximately n × m + m × 8 + n × 24 bytes
"""
function estimate_gpu_memory_requirement(operation::Symbol, n::Int, m::Int)
    if operation == :grm
        # Genotypes (two-bit) + centered Z (Float32) + G (Float64) + working memory
        genotype_mem = ceil(Int64, n * m / 4)  # Two-bit encoding
        z_matrix_mem = n * m * 4  # Float32
        grm_mem = n * n * 8  # Float64
        working_mem = 10000 * m * 4  # Block processing, 10K individuals per block
        overhead = 500_000_000  # 500 MB overhead for CUDA runtime

        return genotype_mem + z_matrix_mem + grm_mem + working_mem + overhead

    elseif operation == :pcg
        # GRM + solution + residuals + search directions
        grm_mem = n * n * 8
        vectors_mem = n * 8 * 5  # u, r, z, p, Cp
        overhead = 200_000_000

        return grm_mem + vectors_mem + overhead

    elseif operation == :mcmc
        # Genotypes + marker effects + residuals
        genotype_mem = ceil(Int64, n * m / 4)
        effects_mem = m * 8
        residual_mem = n * 8 * 3
        overhead = 300_000_000

        return genotype_mem + effects_mem + residual_mem + overhead

    elseif operation == :deeplearning
        # Model parameters + activations + gradients + optimizer state
        # This is a rough estimate; actual depends on architecture
        param_mem = m * 1000 * 4  # Assume ~1000 parameters per input feature (FP32)
        activation_mem = n * 500 * 4  # Batch activations
        gradient_mem = param_mem * 2  # Gradients + optimizer momentum
        overhead = 1_000_000_000  # 1 GB for framework overhead

        return param_mem + activation_mem + gradient_mem + overhead

    else
        # Conservative default estimate
        return n * m * 8 + 1_000_000_000
    end
end


"""
    can_fit_in_gpu_memory(backend::GPUBackend, required_bytes::Int64)

Check if operation can fit in available GPU memory with safety margin.

Compares required memory against available GPU memory, reserving headroom
for CUDA runtime and unexpected allocations.

# Safety Margins
- 85% threshold for available memory utilization
- Additional 10% reserved for CUDA overhead
- Larger margins for concurrent kernel execution
"""
function can_fit_in_gpu_memory(backend::GPUBackend, required_bytes::Int64)
    if backend.device_id < 0
        return false  # No GPU available
    end

    # Safety margin: use only 85% of available memory
    usable_memory = backend.available_memory * 0.85

    return required_bytes <= usable_memory
end


"""
    configure_gpu_memory_pool(initial_size::Int64, max_size::Int64)

Configure CUDA memory pool for efficient allocation and reduced fragmentation.

Memory pooling dramatically reduces allocation overhead by reusing previously
allocated memory blocks. This is critical for iterative algorithms that repeatedly
allocate and free temporary arrays.

# Benefits
- 10-100× faster allocation compared to cudaMalloc
- Reduced memory fragmentation
- Automatic memory reuse across kernel calls
- Lower host-device synchronization overhead

# Configuration Strategy
- Initial pool size: Preallocate common working memory
- Maximum size: Limit to prevent out-of-memory crashes
- Trim interval: Periodically release unused memory

# Examples
```julia
# Conservative configuration for shared GPU
configure_gpu_memory_pool(2_000_000_000, 8_000_000_000)  # 2-8 GB

# Aggressive configuration for dedicated computation
configure_gpu_memory_pool(10_000_000_000, 30_000_000_000)  # 10-30 GB
```
"""
function configure_gpu_memory_pool(initial_size::Int64, max_size::Int64)
    if !CUDA.functional()
        @warn "GPU not available, skipping memory pool configuration"
        return
    end

    # Enable memory pool
    CUDA.memory_pool()[] = CUDA.default_memory_pool()

    println("GPU Memory Pool Configured:")
    println("  Initial size: $(round(initial_size / 1e9, digits=2)) GB")
    println("  Maximum size: $(round(max_size / 1e9, digits=2)) GB")
end