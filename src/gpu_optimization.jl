# ===== src/gpu_optimization.jl =====
"""
Advanced GPU optimization techniques for DynamicEpistasisGBLUP.
Includes auto-tuning, fused kernels, mixed-precision computation,
streaming, optimized memory access, dynamic parallelism, persistent kernels,
warp-level primitives, and cooperative groups.
Many of these are highly advanced and depend on specific GPU hardware capabilities
and mature CUDA.jl features.
"""

module GPUOptimization

using CUDA
using KernelAbstractions
# using CUDAKernels # This was in original, but KA is usually the primary abstraction layer.
                  # CUDAKernels might refer to specific lower-level kernel constructs if used.
using Adapt       # For adapting structures to GPU
using StaticArrays # For small fixed-size arrays in kernels if beneficial

# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export GPUConfig, auto_tune_gpu_config, # Renamed auto_tune_gpu
       compute_fused_grm!, # Renamed from fused_grm_kernel! to be a higher-level call
       compute_mixed_precision_epistasis_grm, # Renamed from mixed_precision_epistasis
       perform_streaming_grm_computation # Renamed from streaming_grm_computation
       # Other optimization concepts are more about kernel design patterns than exported functions.

"""
    GPUConfig

Stores configuration details for the active GPU device, aiding in optimizing kernel launches.
"""
struct GPUConfig
    device::CuDevice
    max_threads_per_block::Int
    max_blocks_per_grid_dim::NamedTuple{(:x, :y, :z), Tuple{Int,Int,Int}} # Max grid dimensions
    shared_memory_per_block::Int
    warp_size::Int
    compute_capability::VersionNumber
end

"""
    auto_tune_gpu_config(; verbose::Bool=true) -> GPUConfig

Queries the active CUDA device for its capabilities and returns a `GPUConfig` object.
This information can be used to tailor kernel launch parameters.
"""
function auto_tune_gpu_config(; verbose::Bool = true)
    if !CUDA.functional()
        error("CUDA is not functional. Cannot auto-tune GPU configuration.")
    end
    dev = CUDA.device() # Get current/default device

    config = GPUConfig(
        dev,
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        (x=CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
         y=CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
         z=CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)),
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE),
        CUDA.capability(dev)
    )

    if verbose
        println("GPU Auto-Configuration:")
        println("  Device Name: $(CUDA.name(dev))")
        println("  Compute Capability: $(config.compute_capability)")
        println("  Max Threads/Block: $(config.max_threads_per_block)")
        println("  Max Grid Dims (X,Y,Z): $(config.max_blocks_per_grid_dim.x), $(config.max_blocks_per_grid_dim.y), $(config.max_blocks_per_grid_dim.z)")
        println("  Shared Memory/Block: $(config.shared_memory_per_block) bytes")
        println("  Warp Size: $(config.warp_size)")
    end

    return config
end

"""
    compute_fused_grm!(G_output::CuArray{T,2}, genotypes_input::CuArray{T,2}, allele_freq_input::CuVector{T}) where T

Computes GRM using a fused kernel that combines genotype centering and matrix multiplication
in a single pass to improve memory locality and reduce kernel launch overhead.
`G_output` is modified in-place.
Requires `fused_grm_compute_kernel!` to be defined in `gpu_kernels.jl`.
"""
function compute_fused_grm!(
    G_output::CuArray{T,2},
    genotypes_input::CuArray{T,2}, # Raw genotypes (0,1,2)
    allele_freq_input::CuVector{T}; # Allele frequencies for centering
    gpu_config::Union{Nothing, GPUConfig} = nothing
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_input)
    if size(G_output) != (n_individuals, n_individuals) || length(allele_freq_input) != n_snps
        error("Dimension mismatch for fused GRM computation.")
    end

    current_gpu_config = gpu_config === nothing ? auto_tune_gpu_config(verbose=false) : gpu_config

    # Determine optimal block and grid sizes for the fused kernel
    # Example: Tiled approach, BLOCK_DIM x BLOCK_DIM threads per block
    # The kernel `fused_grm_compute!` (from original, now in gpu_kernels.jl) used Val(BLOCK_SIZE)
    # and dynamic shared memory.
    block_dim_heuristic = 32 # Typical for matrix multiplication tiles
    # Ensure block_dim does not exceed device max_threads_per_block / block_dim (if 2D block)
    # Max threads = block_dim_heuristic * block_dim_heuristic.
    if block_dim_heuristic * block_dim_heuristic > current_gpu_config.max_threads_per_block
        block_dim_heuristic = floor(Int, sqrt(current_gpu_config.max_threads_per_block))
    end

    threads_per_block = (block_dim_heuristic, block_dim_heuristic)
    grid_blocks = (cld(n_individuals, block_dim_heuristic), cld(n_individuals, block_dim_heuristic))

    # Shared memory: for two tiles of genotypes (BLOCK_DIM x BLOCK_DIM)
    # The kernel `fused_grm_compute!` used `@cuDynamicSharedMem`.
    # The size of dynamic shared memory is passed as `shmem` argument to @cuda.
    # shmem_size_bytes = sizeof(T) * 2 * block_dim_heuristic * block_dim_heuristic
    # This needs to be less than `current_gpu_config.shared_memory_per_block`.
    # For now, assume kernel handles shared memory internally based on Val(BLOCK_SIZE).

    backend = KernelAbstractions.get_backend(G_output)
    # Kernel `fused_grm_compute_kernel!` should be in `gpu_kernels.jl`
    # Its signature was: (G, genotypes, allele_freq, n_ind, n_snps, Val(BLOCK_SIZE))
    # The KA version might not use Val(BLOCK_SIZE) in the same way if block size is fixed by launch params.
    # KA kernels usually get block/grid info from launch.
    # For now, assume a KA-compatible kernel `fused_grm_ka_kernel!` exists.
    # This is a placeholder for the actual KA kernel call.
    # The original code had a direct @cuda call, not KA for this fused kernel.
    # If using direct @cuda:
    # @cuda threads=threads_per_block blocks=grid_blocks shmem=shmem_size_bytes fused_grm_compute_cuda_kernel!(...)
    # For KA, the kernel structure is different.
    # This is a stub, actual kernel call would be more nuanced.
    # For now, this function indicates intent; actual kernel is in gpu_kernels.jl.

    # Placeholder: If fused_grm_compute_kernel! is a direct CUDA kernel:
    # This part will be implemented when the kernel itself is finalized.
    # For now, let's assume it's callable via KA if it was refactored.
    # If `fused_grm_compute!` was the KA kernel name in `gpu_kernels.jl`:
    # kernel! = fused_grm_compute!(backend, threads_per_block) # Workgroup size
    # kernel!(G_output, genotypes_input, allele_freq_input, n_individuals, n_snps,
    #         ndrange= (n_individuals, n_individuals) .* threads_per_block ) # Global size
    # KernelAbstractions.synchronize(backend)
    # This call structure is not quite right for tiled matmul.
    # The launch in original was: @cuda threads=(block_size,block_size) blocks=grid_size ...
    # This implies a direct CUDA C kernel, not a KA one in that specific form.
    # This function will be refined once the kernel in gpu_kernels.jl is set.
    error("compute_fused_grm! requires its CUDA kernel (fused_grm_compute_kernel!) to be properly defined and called.")

    return G_output # Modified in-place
end


"""
    compute_mixed_precision_epistasis_grm(genotypes_fp32::CuArray{Float32,2}; use_tensor_cores_if_available::Bool=true) -> CuArray{Float32,2}

Computes epistatic GRM using mixed precision (e.g., FP16 for computation, FP32 for accumulation)
to leverage hardware like Tensor Cores for performance if available and applicable.
Input `genotypes_fp32` are standard Float32. Result is Float32.
"""
function compute_mixed_precision_epistasis_grm(
    genotypes_fp32::CuArray{Float32,2};
    use_tensor_cores_if_available::Bool = true
) where T <: AbstractFloat # T is Float32 from input

    n_individuals, n_snps = size(genotypes_fp32)
    current_gpu_config = auto_tune_gpu_config(verbose=false)

    can_use_tensor_cores = use_tensor_cores_if_available &&
                           current_gpu_config.compute_capability >= v"7.0" # Volta and newer for FP16 tensor cores

    G_aa_final_fp32 = CUDA.zeros(Float32, n_individuals, n_individuals)

    if can_use_tensor_cores
        # print("  Using Tensor Cores for mixed-precision epistatic GRM.")
        # Convert input genotypes to FP16 for Tensor Core operations
        genotypes_fp16 = CuArray{Float16}(genotypes_fp32) # Downcast

        # Pad matrices for WMMA tile dimensions (e.g., 16x16 or 32x32)
        # WMMA_DIM = 16 (typical for many ops)
        # n_ind_padded = cld(n_individuals, WMMA_DIM) * WMMA_DIM
        # n_snps_padded = cld(n_snps, WMMA_DIM) * WMMA_DIM
        # W_padded_fp16 = CUDA.zeros(Float16, n_ind_padded, n_snps_padded)
        # W_padded_fp16[1:n_individuals, 1:n_snps] = genotypes_fp16
        # G_aa_accum_fp32 = CUDA.zeros(Float32, n_ind_padded, n_ind_padded) # Accumulate in FP32

        # Call the specialized tensor core kernel (e.g., `tensor_epistasis_kernel!`)
        # This kernel needs to be defined in `gpu_kernels.jl` using WMMA intrinsics.
        # Example conceptual call:
        # tensor_epistasis_kernel!(G_aa_accum_fp32, W_padded_fp16, ...)
        # G_aa_final_fp32 = G_aa_accum_fp32[1:n_individuals, 1:n_individuals]
        error("Tensor core epistatic GRM kernel (tensor_epistasis_kernel!) not fully implemented yet.")

    else # Standard mixed-precision (or just FP32 if no specific mixed-precision path without tensor cores)
        # print("  Using standard GPU computation for epistatic GRM (Tensor Cores not used/available).")
        # This might fall back to a regular epistatic GRM computation, possibly still benefiting
        # from FP16 if intermediate products are handled carefully, but without specific Tensor Core calls.
        # Or, it just uses the standard FP32 epistatic GRM computation.
        # For now, assume it calls a standard epistatic GRM (e.g. Hadamard based) if no tensor cores.
        # This function should ideally have a non-tensor-core mixed-precision path if that's intended.
        # If it's only about tensor cores, then this branch might just use full FP32.

        # Placeholder: Call standard epistatic GRM, which should be in grm_computation.jl
        # This means `genotypes_fp32` should be standardized first.
        # This function's scope is optimization; actual GRM logic is elsewhere.
        # For now, this is a stub.
        # standardized_genotypes_fp32 = compute_standardized_genotypes(GenotypeMatrix(genotypes_fp32,...)) # Needs full GenotypeMatrix
        # G_aa_final_fp32 = compute_epistatic_grm!(standardized_genotypes_fp32, method=:hadamard)
        error("Standard mixed-precision path for epistatic GRM (non-Tensor Core) not fully stubbed.")
    end

    return G_aa_final_fp32
end


"""
    perform_streaming_grm_computation(genotypes_host::Matrix{T}; n_streams=4, snp_chunk_size=1000) -> CuArray{T,2}

Computes GRM using multiple CUDA streams to overlap computation with data transfers
between host and device. Genotype data is processed in chunks of SNPs.
`genotypes_host` is the full matrix on CPU. Result is GRM on GPU.
"""
function perform_streaming_grm_computation(
    genotypes_host::Matrix{T}; # Full genotypes on CPU (Individuals x SNPs)
    allele_freq_host::Vector{T}, # Allele frequencies on CPU
    n_streams::Int = 4,
    snp_chunk_size::Int = 1000 # Number of SNPs per chunk
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_host)
    if length(allele_freq_host) != n_snps
        error("Allele frequency vector length does not match number of SNPs.")
    end

    # Create CUDA streams
    streams = [CuStream() for _ in 1:n_streams]

    # Allocate pinned host memory for faster H2D transfers (optional but good practice)
    # For now, directly use `genotypes_host`.

    # Device memory for genotype chunks and partial GRM results per stream
    # Each stream processes one chunk of SNPs at a time.
    # W_chunk_gpu = [CuArray{T}(undef, n_individuals, snp_chunk_size) for _ in 1:n_streams]
    # G_partial_gpu = [CUDA.zeros(T, n_individuals, n_individuals) for _ in 1:n_streams]
    # For GRM = WW', we sum contributions from SNP chunks.
    # G = sum_chunks (W_chunk * W_chunk')
    # This means each stream computes a full GRM based on its SNP chunk, and these are summed.
    # This interpretation is incorrect for standard GRM from SNP chunks.
    # GRM_ij = sum_k (W_ik * W_jk). Each chunk of SNPs contributes to this sum_k.

    # Correct approach for streaming GRM:
    # G_final_gpu = CUDA.zeros(T, n_individuals, n_individuals)
    # W_centered_chunk_gpu = [CuArray{T}(undef, n_individuals, snp_chunk_size) for _ in 1:n_streams]

    # For each chunk of SNPs:
    # 1. Copy SNP data & corresponding allele freqs to device (stream specific buffer).
    # 2. Center this SNP chunk on GPU (stream specific kernel). W_chunk_centered.
    # 3. Compute W_chunk_centered * W_chunk_centered' and add to global G_final_gpu (stream specific kernel, needs atomic add or careful sync).
    # This is complex. The original `streaming_grm_computation` logic was simpler:
    # `compute_grm_chunk!` was called, implying it computed a partial sum for G.
    # Let's assume `compute_grm_chunk!` adds its part to a global G.

    # This function is highly complex to implement correctly with true overlap.
    # For now, this is a high-level stub.
    # The original code's `compute_grm_chunk!` needs to be defined.
    error("perform_streaming_grm_computation is a complex feature and not fully stubbed.")

    # Placeholder return
    # return CUDA.zeros(T, n_individuals, n_individuals)
end


# Other concepts mentioned in original (more like design patterns for kernels):
# - `optimize_memory_access!`: This would be about how data is laid out or accessed in kernels.
# - `adaptive_epistasis_kernel!`: Using dynamic parallelism. (Parent and child kernels)
# - `persistent_grm_kernel!`: For continuous work queues.
# - `warp_reduce_sum`: Utility for warp-level operations.
# - `cooperative_epistasis!`: Using cooperative groups.

# Global counter for dynamic parallelism example (child_kernel output indexing)
# This needs to be a CuArray of length 1 to be modifiable by atomic operations on GPU.
const DYNAMIC_INTERACTION_COUNTER = Ref(CUDA.zeros(Int32, 1))

"""
    reset_dynamic_interaction_counter!()

Resets the global interaction counter used by `child_kernel!` in dynamic parallelism examples.
Should be called before launching `adaptive_epistasis_kernel!` if it uses this shared counter.
"""
function reset_dynamic_interaction_counter!()
    DYNAMIC_INTERACTION_COUNTER[] = CUDA.zeros(Int32, 1)
    CUDA.synchronize() # Ensure reset is complete on GPU before next use
end

# Export necessary items including the new counter and its reset function if they are part of public API.
# For now, keeping them internal to this module or for use by kernels in gpu_kernels.jl.
# The adaptive_epistasis_kernel! function itself would be the public API.

"""
    adaptive_epistasis_kernel_launcher!(...)

Launcher function for the adaptive epistasis detection using dynamic parallelism.
Manages the parent kernel launch and potentially output data structures.
`interactions_output_buffer` and `scores_output_buffer` should be pre-allocated
CuArrays to store results from child kernels. Max size determines buffer capacity.
"""
function adaptive_epistasis_kernel_launcher!(
    interactions_output_buffer::CuArray{Tuple{Int32, Int32}, 1}, # Buffer for (snp_i, snp_j)
    scores_output_buffer::CuArray{T, 1},                          # Buffer for scores
    genotypes_gpu::CuArray{T, 2},
    marginal_score_threshold::T;
    # parent_kernel_config, # Launch parameters for parent kernel
    # child_kernel_config   # Launch parameters for child kernel (if configurable from host)
    verbose::Bool = false
) where T <: AbstractFloat

    n_snps = size(genotypes_gpu, 2)
    if n_snps == 0 return 0 end

    reset_dynamic_interaction_counter!() # Reset counter before new run

    # Max number of interactions that can be stored is length of output buffers
    max_storable_interactions = length(interactions_output_buffer)
    if length(scores_output_buffer) != max_storable_interactions
        error("Output buffers for interactions and scores must have the same length.")
    end

    # Launch parent kernel
    # Parent kernel iterates over SNPs. Each thread (or block) for a SNP.
    # `parent_kernel!` is defined in `gpu_kernels.jl`.
    # It needs access to `DYNAMIC_INTERACTION_COUNTER` and output buffers.
    backend = KernelAbstractions.get_backend(genotypes_gpu)
    parent_kernel_ka! = parent_kernel_dynamic!(backend) # Assuming name in gpu_kernels.jl

    # Example launch configuration (adjust as needed)
    threads_parent = 256
    blocks_parent = cld(n_snps, threads_parent)

    if verbose println("Launching parent kernel for adaptive epistasis with $blocks_parent blocks, $threads_parent threads each...") end

    parent_kernel_ka!(
        interactions_output_buffer, scores_output_buffer,
        DYNAMIC_INTERACTION_COUNTER[], # Pass the CuArray{Int32,1} counter
        genotypes_gpu, marginal_score_threshold,
        max_storable_interactions, # Pass max capacity to kernels
        ndrange=n_snps # Launch one "task" per SNP for parent kernel
    )
    KernelAbstractions.synchronize(backend)

    # Number of interactions actually found is in DYNAMIC_INTERACTION_COUNTER[][1]
    num_found_interactions = Array(DYNAMIC_INTERACTION_COUNTER[])[1]

    if verbose println("Adaptive epistasis search complete. Found $num_found_interactions interactions (up to buffer capacity).") end

    return num_found_interactions
end


# Export functions if this file were a module
# export GPUConfig, auto_tune_gpu_config, compute_fused_grm!,
#        compute_mixed_precision_epistasis_grm, perform_streaming_grm_computation,
#        adaptive_epistasis_kernel_launcher!, reset_dynamic_interaction_counter!


end # module GPUOptimization
