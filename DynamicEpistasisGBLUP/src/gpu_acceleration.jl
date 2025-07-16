module GPUAcceleration

using ..DynamicEpistasisGBLUP: CUDA, LinearAlgebra

export calculate_g_matrix_gpu, calculate_gaa_matrix_gpu

"""
    calculate_g_matrix_gpu(genotypes_gpu::CuMatrix)

Calculates the additive genomic relationship matrix (G) on the GPU.
"""
function calculate_g_matrix_gpu(genotypes::Matrix)
    n_individuals, n_markers = size(genotypes)

    # Move data to GPU
    genotypes_gpu = CuMatrix(genotypes)

    # Calculate allele frequencies on the GPU
    p_gpu = vec(mean(genotypes_gpu, dims=1) ./ 2)

    # Center genotypes
    W_gpu = genotypes_gpu .- 2 .* p_gpu'

    # Standardize W
    W_std_gpu = W_gpu ./ sqrt.(2 .* p_gpu' .* (1 .- p_gpu'))

    # Calculate G matrix
    G_gpu = (W_std_gpu * W_std_gpu') ./ n_markers

    # Copy result back to CPU
    G_cpu = Matrix(G_gpu)

    return G_cpu
end


"""
    calculate_gaa_matrix_gpu(genotypes::Matrix)

Calculates the epistatic genomic relationship matrix (G_AA) on the GPU.
"""
function calculate_gaa_matrix_gpu(genotypes::Matrix)
    # First, calculate G on the GPU
    G = calculate_g_matrix_gpu(genotypes)

    # Move G to GPU to perform Hadamard product
    G_gpu = CuMatrix(G)

    # Perform Hadamard product on the GPU
    G_AA_gpu = G_gpu .* G_gpu

    # Copy result back to CPU
    G_AA_cpu = Matrix(G_AA_gpu)

    return G_AA_cpu
end

end # module GPUAcceleration
