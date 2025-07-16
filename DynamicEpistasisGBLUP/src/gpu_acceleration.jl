module GPUAcceleration

using ..DynamicEpistasisGBLUP: CUDA, LinearAlgebra

export calculate_g_matrix_gpu, calculate_gaa_matrix_gpu

"""
    calculate_g_matrix_gpu(genotypes::Matrix) -> Matrix

Calculates the additive genomic relationship matrix (G) on the GPU using `CUDA.jl`.

This function performs the same calculation as `calculate_g_matrix` but moves the
data to the GPU to leverage parallel processing for faster computation, especially
on large datasets.

# Arguments
- `genotypes::Matrix`: The genotype matrix on the CPU.

# Returns
- `Matrix`: The G matrix, copied back to the CPU.
"""
function calculate_g_matrix_gpu(genotypes::Matrix)
    n_individuals, n_markers = size(genotypes)

    # Move genotype data from CPU to GPU
    genotypes_gpu = CuMatrix(genotypes)

    # Calculate allele frequencies on the GPU
    p_gpu = vec(mean(genotypes_gpu, dims=1) ./ 2)

    # Center the genotype matrix on the GPU
    W_gpu = genotypes_gpu .- 2 .* p_gpu'

    # Calculate the denominator for scaling on the GPU
    denominator_gpu = sum(2 .* p_gpu .* (1 .- p_gpu))

    # Calculate G on the GPU
    G_gpu = (W_gpu * W_gpu') / denominator_gpu

    # Copy the result from GPU back to CPU memory
    G_cpu = Matrix(G_gpu)

    return G_cpu
end


"""
    calculate_gaa_matrix_gpu(genotypes::Matrix) -> Matrix

Calculates the epistatic genomic relationship matrix (G_AA) on the GPU.

It first computes the G matrix on the GPU, then performs the element-wise
Hadamard product on the GPU for maximum efficiency.

# Arguments
- `genotypes::Matrix`: The genotype matrix on the CPU.

# Returns
- `Matrix`: The G_AA matrix, copied back to the CPU.
"""
function calculate_gaa_matrix_gpu(genotypes::Matrix)
    # First, calculate G on the GPU
    G_gpu = CuMatrix(calculate_g_matrix_gpu(genotypes))

    # Perform the Hadamard product on the GPU
    G_AA_gpu = G_gpu .* G_gpu

    # Copy the result back to the CPU
    G_AA_cpu = Matrix(G_AA_gpu)

    return G_AA_cpu
end

end # module GPUAcceleration
