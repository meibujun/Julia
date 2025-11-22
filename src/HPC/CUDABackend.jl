module CUDABackend

using CUDA
using LinearAlgebra
using ...GenomicCore

export has_gpu, select_gpu!, unpack_genotypes_gpu

"""
    has_gpu()

Check if a CUDA-capable GPU is available.
"""
function has_gpu()
    return CUDA.functional()
end

"""
    select_gpu!(id::Int)

Select a specific GPU device.
"""
function select_gpu!(id::Int)
    if has_gpu()
        device!(id)
    else
        @warn "No GPU available."
    end
end

# CUDA Kernel for unpacking
function unpack_kernel!(dest, src, n_samples, n_snps)
    # Grid stride loop or simple 2D indexing
    # src: (n_packed_rows, n_snps)
    # dest: (n_samples, n_snps)
    
    # We parallelize over SNPs (x) and Samples (y)
    snp_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sample_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if snp_idx <= n_snps && sample_idx <= n_samples
        # Calculate byte index
        # Row in src corresponds to packed samples
        # sample_idx is 1-based
        
        # 4 samples per byte
        # row_idx = div(sample_idx - 1, 4) + 1
        # bit_offset = 2 * ((sample_idx - 1) % 4)
        
        # Optimization: compute indices
        row_idx = ((sample_idx - 1) >> 2) + 1
        bit_offset = ((sample_idx - 1) & 3) << 1
        
        @inbounds byte = src[row_idx, snp_idx]
        val = (byte >> bit_offset) & 0x03
        
        if val == 0x00      # 00 -> Homo Ref (0)
            dest[sample_idx, snp_idx] = 0.0f0
        elseif val == 0x02  # 10 -> Heteroz (1)
            dest[sample_idx, snp_idx] = 1.0f0
        elseif val == 0x03  # 11 -> Homo Alt (2)
            dest[sample_idx, snp_idx] = 2.0f0
        else                # 01 -> Missing
            dest[sample_idx, snp_idx] = NaN32
        end
    end
    return nothing
end

"""
    unpack_genotypes_gpu(g::CompactGenotypes)

Transfer compressed genotypes to GPU and unpack to Float32.
Returns `CuArray{Float32, 2}`.
"""
function unpack_genotypes_gpu(g::CompactGenotypes)
    if !has_gpu()
        error("No GPU available for unpacking.")
    end
    
    n_samples = g.n_samples
    n_snps = g.n_snps
    
    # 1. Transfer compressed data to GPU
    # This is small: N*M/4 bytes
    d_src = CuArray(g.data)
    
    # 2. Allocate output
    # This is large: N*M*4 bytes
    d_dest = CuArray{Float32}(undef, n_samples, n_snps)
    
    # 3. Launch Kernel
    threads = (16, 16)
    blocks = (ceil(Int, n_snps/16), ceil(Int, n_samples/16))
    
    @cuda threads=threads blocks=blocks unpack_kernel!(d_dest, d_src, n_samples, n_snps)
    
    return d_dest
end

end # module CUDABackend
