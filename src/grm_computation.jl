# ===== src/grm_computation.jl =====
"""
    DynamicEpistasisGBLUP.GRMComputation

This module provides functions for computing Genomic Relationship Matrices (GRMs),
both additive (G) and epistatic (G_aa). It leverages GPU acceleration for efficiency.
Key functions include `compute_grm!` for additive GRMs, `compute_epistatic_grm!`
for epistatic GRMs, and helper functions for genotype centering and standardization.
Cross-population GRM functions (`compute_grm_cross!`, `compute_epistatic_grm_cross!`)
are also included for prediction scenarios.
"""

using CUDA
using LinearAlgebra # For I in G + eps*I
using KernelAbstractions # For GPU kernels
# Assuming types.jl (GenotypeMatrix) and gpu_kernels.jl are included before this file in the main module.
# And utils.jl for update_allele_frequencies!

# Helper to get Float type, assuming it's defined in the scope that includes this file.
_Float() = DynamicEpistasisGBLUP.Float


"""
    compute_grm!(genotypes::GenotypeMatrix{T}; method::Symbol = :vanraden, use_gpu::Bool = true) where T -> Union{CuArray{T,2}, Matrix{T}}

Computes the additive Genomic Relationship Matrix (GRM) using primarily VanRaden's Method 1.
The function updates allele frequencies stored in `genotypes.allele_freq` based on the
current `genotypes.data` before GRM computation.
It supports both GPU (default) and CPU computation paths.

The GRM is calculated as `G = W W' / scale`, where `W` is the centered genotype matrix
(X_ij - 2p_j) and `scale` is `sum(2*p_j*(1-p_j))` over all SNPs.

# Arguments
- `genotypes::GenotypeMatrix{T}`: A `GenotypeMatrix` object containing genotype data, allele frequencies (will be updated), and metadata.
- `method::Symbol = :vanraden`: The method for GRM computation. Currently, only `:vanraden` is fully supported by this function's structure.
- `use_gpu::Bool = true`: If `true` and CUDA is functional, computation occurs on the GPU and returns a `CuArray`. Otherwise, uses CPU and returns a `Matrix`.

# Returns
- `Union{CuArray{T,2}, Matrix{T}}`: The computed additive GRM. Type depends on `use_gpu` and CUDA availability.

# Details
- Calls `update_allele_frequencies!(genotypes)` internally.
- Uses `compute_centered_genotypes` to get the `W` matrix.
- Employs `grm_kernel!` (from `gpu_kernels.jl`) for GPU computation.
- Has a CPU fallback if `use_gpu` is `false` or CUDA is not available.
"""
function compute_grm!(
    genotypes::GenotypeMatrix{T};
    method::Symbol = :vanraden,
    use_gpu::Bool = true
) where T <: AbstractFloat

    n_individuals = genotypes.n_individuals
    n_snps = genotypes.n_snps

    # Ensure allele frequencies are up-to-date or computed if not already
    # The `update_allele_frequencies!` function is assumed to be in utils.jl
    update_allele_frequencies!(genotypes) # Modifies genotypes.allele_freq in-place

    # Compute centered genotype matrix W: W_ij = X_ij - 2p_j
    # X_ij is genotype of individual i at SNP j (0, 1, or 2)
    # p_j is allele frequency of SNP j
    W_centered = compute_centered_genotypes(genotypes) # Returns a new CuArray

    # Initialize GRM matrix
    # If use_gpu is false, operations should ideally happen on CPU then result moved if needed.
    # For now, assuming if use_gpu is true, result is CuArray, else Array.
    # The kernels are GPU kernels, so CPU path needs separate logic.

    local G_matrix # Declare G_matrix to be assigned in if/else

    if method == :vanraden
        # VanRaden Method 1: G = WW' / sum(2 * p_j * (1-p_j) for all j)
        scaling_factor_sum_2pq = compute_sum_2pq_scaling_factor(genotypes)

        if scaling_factor_sum_2pq <= eps(T)
            error("Sum of 2*p_j*(1-p_j) is zero or negative, cannot compute GRM. Check allele frequencies.")
        end

        actual_scale_for_kernel = one(T) / scaling_factor_sum_2pq

        if use_gpu && CUDA.functional()
            G_gpu = CuArray{T}(undef, n_individuals, n_individuals)
            backend = KernelAbstractions.get_backend(G_gpu) # Should be CUDABackend

            # Kernel launch for G = W_centered * W_centered' * actual_scale_for_kernel
            # The grm_kernel! computes sum(W[i,k]*W[j,k]) * scale
            kernel! = grm_kernel!(backend) # From gpu_kernels.jl
            kernel!(G_gpu, W_centered, actual_scale_for_kernel, ndrange=(n_individuals, n_individuals))
            KernelAbstractions.synchronize(backend)
            G_matrix = G_gpu
        else # CPU fallback
            W_centered_cpu = Array(W_centered) # Move W to CPU
            G_cpu = zeros(T, n_individuals, n_individuals)
            # G_cpu = (W_centered_cpu * W_centered_cpu') .* actual_scale_for_kernel # Direct computation
            # Or, loop version similar to GPU kernel for consistency:
            for i in 1:n_individuals
                for j in i:n_individuals # Upper triangle
                    sum_ij = zero(T)
                    for k in 1:n_snps
                        sum_ij += W_centered_cpu[i,k] * W_centered_cpu[j,k]
                    end
                    G_cpu[i,j] = sum_ij * actual_scale_for_kernel
                    if i != j
                        G_cpu[j,i] = G_cpu[i,j]
                    end
                end
            end
            G_matrix = G_cpu
        end
    else
        error("GRM computation method '$method' not implemented.")
    end

    return G_matrix
end


"""
    compute_epistatic_grm!(genotypes::GenotypeMatrix{T}; method::Symbol = :hadamard, use_gpu::Bool = true) where T -> Union{CuArray{T,2}, Matrix{T}}

Computes the additive-by-additive epistatic Genomic Relationship Matrix (G_aa).
The primary method implemented is `:hadamard`, which follows the logic:
  `G_aa(i,j) = (1/NumPairs) * sum_{k<l} (W_std_ik * W_std_il) * (W_std_jk * W_std_jl)`
where `W_std` is the standardized genotype matrix `(X_ij - 2p_j) / sqrt(2p_j(1-p_j))`,
and `NumPairs` is the total number of unique SNP pairs.

Allele frequencies in `genotypes.allele_freq` are updated before standardizing genotypes.
Supports GPU (default) and CPU computation.

# Arguments
- `genotypes::GenotypeMatrix{T}`: `GenotypeMatrix` object.
- `method::Symbol = :hadamard`: Specifies the computation method. Currently, `:hadamard` is the main supported path. Other methods like `:symmetric_polynomial` might be routed to their respective modules or implemented here in the future.
- `use_gpu::Bool = true`: If `true` and CUDA is functional, uses GPU. Otherwise, CPU.

# Returns
- `Union{CuArray{T,2}, Matrix{T}}`: The computed epistatic GRM.

# Details
- Calls `update_allele_frequencies!` and `compute_standardized_genotypes`.
- Uses `epistatic_grm_kernel!` (from `gpu_kernels.jl`) for GPU computation.
- Includes a CPU fallback (`compute_epistatic_grm_cpu!`).
"""
function compute_epistatic_grm!(
    genotypes::GenotypeMatrix{T};
    method::Symbol = :hadamard,
    use_gpu::Bool = true
) where T <: AbstractFloat

    n_individuals = genotypes.n_individuals
    n_snps = genotypes.n_snps

    # Epistatic GRM usually requires standardized genotypes: W_std_ij = (X_ij - 2p_j) / sqrt(2p_j(1-p_j))
    # Ensure allele frequencies are up-to-date
    update_allele_frequencies!(genotypes)
    W_standardized = compute_standardized_genotypes(genotypes) # Returns a new CuArray

    local G_aa_matrix # Declare to be assigned

    if method == :hadamard
        # G_aa_ij = (1/NumPairs) * sum_{k<l} (W_std_ik * W_std_il) * (W_std_jk * W_std_jl)
        # NumPairs = n_snps * (n_snps - 1) / 2
        num_snp_pairs = T(n_snps * (n_snps - 1) / 2)

        if num_snp_pairs <= zero(T) && n_snps > 1 # Avoid division by zero if n_snps <= 1
            error("Number of SNP pairs is zero or negative, cannot compute epistatic GRM. Need at least 2 SNPs.")
        elseif n_snps <=1 # If only 1 SNP, no pairs, G_aa is zero
             G_aa_matrix = use_gpu && CUDA.functional() ? CUDA.zeros(T, n_individuals, n_individuals) : zeros(T, n_individuals, n_individuals)
             return G_aa_matrix
        end

        inv_num_snp_pairs = one(T) / num_snp_pairs

        if use_gpu && CUDA.functional()
            G_aa_gpu = CuArray{T}(undef, n_individuals, n_individuals)
            backend = KernelAbstractions.get_backend(G_aa_gpu)

            kernel! = epistatic_grm_kernel!(backend) # From gpu_kernels.jl
            kernel!(G_aa_gpu, W_standardized, inv_num_snp_pairs, ndrange=(n_individuals, n_individuals))
            KernelAbstractions.synchronize(backend)
            G_aa_matrix = G_aa_gpu
        else # CPU fallback
            # CPU fallback for Hadamard method
            G_aa_cpu_array = zeros(T, n_individuals, n_individuals)
            compute_epistatic_grm_cpu!(G_aa_cpu_array, W_standardized, inv_num_snp_pairs)
            G_aa_matrix = G_aa_cpu_array
        end
    # elseif method == :sparse
        # interactions = detect_sparse_epistasis(W_standardized, sparse_threshold)
        # compute_sparse_epistatic_grm!(G_aa_matrix, W_standardized, interactions)
    elseif method == :symmetric_polynomial
        # This method should be called from SymmetricPolynomials module or be implemented here
        # For now, error out if called directly without routing through the module.
        error("Symmetric polynomial method for epistatic GRM should be called via SymmetricPolynomials module.")
    else
        error("Epistatic GRM computation method '$method' not implemented.")
    end

    return G_aa_matrix
end


"""
    compute_centered_genotypes(genotypes::GenotypeMatrix{T}) where T <: AbstractFloat -> CuArray{T,2}

Computes the centered genotype matrix `W` on the GPU.
Each element `W_ij` is calculated as `X_ij - 2*p_j`, where `X_ij` is the raw genotype
(e.g., 0, 1, or 2 allele count) of individual `i` at SNP `j`, and `p_j` is the
allele frequency of SNP `j`.

This function can use allele frequencies from `genotypes.allele_freq` or an externally
provided `allele_freq_source`. If `allele_freq_source` is `nothing`, frequencies from
`genotypes.allele_freq` are used (it's assumed these are up-to-date, e.g., via
`update_allele_frequencies!`).

# Arguments
- `genotypes::GenotypeMatrix{T}`: The `GenotypeMatrix` containing raw GPU genotype data (`genotypes.data`).
- `allele_freq_source::Union{Nothing, CuVector{T}} = nothing`: Optional. If provided, these allele frequencies are used for centering. Otherwise, `genotypes.allele_freq` is used.

# Returns
- `CuArray{T,2}`: A new GPU array representing the centered genotype matrix `W`.

# GPU Kernel
- Uses `center_genotypes_kernel!` from `gpu_kernels.jl` for the computation.
"""
function compute_centered_genotypes(
    genotypes::GenotypeMatrix{T},
    allele_freq_source::Union{Nothing, CuVector{T}} = nothing
) where T <: AbstractFloat
    n_individuals = genotypes.n_individuals
    n_snps = genotypes.n_snps

    freqs_to_use = allele_freq_source === nothing ? genotypes.allele_freq : allele_freq_source
    if length(freqs_to_use) != n_snps
        error("Provided allele_freq_source length ($(length(freqs_to_use))) does not match number of SNPs ($n_snps).")
    end

    W_centered = CuArray{T}(undef, n_individuals, n_snps)

    backend = KernelAbstractions.get_backend(W_centered)
    # Kernel signature is now: W_out, data_in, allele_freq_vector, n_individuals, n_snps
    kernel! = Main.DynamicEpistasisGBLUP.center_genotypes_kernel!(backend)
    kernel!(W_centered, genotypes.data, freqs_to_use, n_individuals, n_snps, ndrange = n_individuals * n_snps)
    KernelAbstractions.synchronize(backend)

    return W_centered
end

"""
    compute_standardized_genotypes(genotypes::GenotypeMatrix{T}) where T <: AbstractFloat -> CuArray{T,2}

Computes the standardized genotype matrix `W_std` on the GPU.
Each element `W_std_ij` is `(X_ij - 2*p_j) / sqrt(2*p_j*(1-p_j))`.
`X_ij` is the raw genotype, `p_j` is the allele frequency.
Standardization accounts for differences in variance due to allele frequencies.
This is often required for methods like Hadamard product-based epistatic GRMs
or certain association study models.

It first calls `compute_centered_genotypes`. If `allele_freq_source` is not provided,
it uses `genotypes.allele_freq` (assumed to be up-to-date).

# Arguments
- `genotypes::GenotypeMatrix{T}`: `GenotypeMatrix` object.
- `allele_freq_source::Union{Nothing, CuVector{T}} = nothing`: Optional. Allele frequencies to use for both centering and standardization. If `nothing`, `genotypes.allele_freq` is used.

# Returns
- `CuArray{T,2}`: A new GPU array, the standardized genotype matrix `W_std`.

# GPU Kernel
- Uses `standardize_genotypes_kernel!` from `gpu_kernels.jl`.
"""
function compute_standardized_genotypes(
    genotypes::GenotypeMatrix{T},
    allele_freq_source::Union{Nothing, CuVector{T}} = nothing
) where T <: AbstractFloat
    n_individuals = genotypes.n_individuals
    n_snps = genotypes.n_snps

    freqs_to_use = allele_freq_source === nothing ? genotypes.allele_freq : allele_freq_source
    if length(freqs_to_use) != n_snps
        error("Provided allele_freq_source length ($(length(freqs_to_use))) does not match number of SNPs ($n_snps).")
    end

    # First, get centered genotypes using the specified frequencies
    W_centered = compute_centered_genotypes(genotypes, freqs_to_use)

    W_standardized = CuArray{T}(undef, n_individuals, n_snps)
    backend = KernelAbstractions.get_backend(W_standardized)
    # Kernel signature: W_std_out, W_centered_in, allele_freq_vector, n_individuals, n_snps
    kernel! = Main.DynamicEpistasisGBLUP.standardize_genotypes_kernel!(backend)
    kernel!(W_standardized, W_centered, freqs_to_use, n_individuals, n_snps, ndrange = n_individuals * n_snps)
    KernelAbstractions.synchronize(backend)

    return W_standardized
end


"""
    compute_sum_2pq_scaling_factor(genotypes::GenotypeMatrix{T}) where T <: AbstractFloat -> T

Computes the scaling factor `sum(2*p_j*(1-p_j))` across all SNPs `j`.
This factor is used as the denominator in VanRaden's Method 1 for calculating
the additive GRM, ensuring `avg(diag(G)) approx 1`.
Allele frequencies `p_j` are taken from `genotypes.allele_freq`.

# Arguments
- `genotypes::GenotypeMatrix{T}`: `GenotypeMatrix` containing updated allele frequencies.

# Returns
- `T`: The computed scaling factor.

# Implementation Notes
- If `genotypes.allele_freq` is a `CuVector`, the sum can be performed efficiently on GPU using `CUDA.mapreduce` or by copying to CPU if the vector is not excessively large. Current implementation copies to CPU for the sum.
"""
function compute_sum_2pq_scaling_factor(genotypes::GenotypeMatrix{T}) where T <: AbstractFloat
    # Allele frequencies are on GPU (genotypes.allele_freq)
    # Perform sum on GPU for efficiency, then transfer scalar result.
    # Or, if allele_freq is small, can transfer to CPU first.

    # Example: Sum on GPU using CUDA.jl mapreduce
    # sum_2pq = CUDA.mapreduce(p -> T(2) * p * (one(T) - p), +, genotypes.allele_freq; init=zero(T))

    # Simpler for now if n_snps is not excessively large: move to CPU for sum
    # For very large n_snps, GPU reduction is better.
    allele_freq_cpu = Array(genotypes.allele_freq)

    scale_sum = zero(T)
    @inbounds for p_val in allele_freq_cpu
        scale_sum += T(2) * p_val * (one(T) - p_val)
    end

    return scale_sum
end


# The original grm_computation.jl also had update_allele_frequencies!
# This function is now primarily in utils.jl to avoid circular dependencies if grm_computation
# needs to be used by other modules that don't need all of utils.
# However, it's tightly coupled with GenotypeMatrix and GRM computation.
# It might be better to keep it here or in types.jl if it modifies GenotypeMatrix directly.
# For now, assuming it's accessible from utils.jl.

# Export functions if this file were a module
# export compute_grm!, compute_epistatic_grm!, compute_centered_genotypes, compute_standardized_genotypes, compute_sum_2pq_scaling_factor

"""
    compute_epistatic_grm_cpu!(G_aa_output_cpu::Matrix{T}, W_standardized_input::CuArray{T,2}, inv_num_snp_pairs::T) where T

CPU implementation for computing the epistatic GRM using the Hadamard method logic.
`G_aa_output_cpu` is a pre-allocated CPU matrix that will be filled.
`W_standardized_input` is the standardized genotype matrix (can be CuArray, will be copied to CPU).
`inv_num_snp_pairs` is `1.0 / (n_snps * (n_snps - 1) / 2)`.
"""
function compute_epistatic_grm_cpu!(
    G_aa_output_cpu::Matrix{T},         # Output matrix (Individuals x Individuals) on CPU
    W_standardized_input::CuArray{T,2}, # Standardized genotypes (Indiv x SNPs) on GPU
    inv_num_snp_pairs::T               # Precomputed 1.0 / NumPairs
) where T <: AbstractFloat

    n_individuals, n_snps = size(W_standardized_input)
    if size(G_aa_output_cpu) != (n_individuals, n_individuals)
        error("Output G_aa CPU matrix dimensions do not match.")
    end
    if n_snps < 2 && inv_num_snp_pairs != 0 # If only 1 SNP, inv_num_snp_pairs might be Inf or NaN if not handled
        fill!(G_aa_output_cpu, zero(T)) # No pairs, G_aa is zero
        return
    end
    if inv_num_snp_pairs == 0 && n_snps >=2 # Should not happen if n_snps >= 2
        error("Inverse number of SNP pairs is zero, which is invalid for n_snps >= 2.")
    end


    # Copy standardized genotype matrix from GPU to CPU for processing
    W_std_cpu = Array(W_standardized_input)

    # Multi-threading over individuals (outer loop) for potential speedup on CPU
    Threads.@threads for r_idx in 1:n_individuals
        # Inner loop computes upper triangle including diagonal
        for c_idx in r_idx:n_individuals
            sum_rc_interactions = zero(T)
            # Loop over all unique pairs of SNPs (k1 < k2)
            for k1 in 1:(n_snps-1)
                # Pre-fetch W_std_cpu[r_idx, k1] and W_std_cpu[c_idx, k1] for the inner loop
                val_r_k1 = W_std_cpu[r_idx, k1]
                val_c_k1 = W_std_cpu[c_idx, k1]

                # Sum over k2 > k1
                # @simd for k2 in (k1+1):n_snps # SIMD might help inner loop
                for k2 in (k1+1):n_snps # SIMD might help inner loop
                    # Interaction term for individual r_idx, pair (k1,k2): W_r_k1 * W_r_k2
                    # Interaction term for individual c_idx, pair (k1,k2): W_c_k1 * W_c_k2
                    # Product of these interaction terms: (W_r_k1 * W_r_k2) * (W_c_k1 * W_c_k2)
                    sum_rc_interactions += (val_r_k1 * W_std_cpu[r_idx, k2]) * (val_c_k1 * W_std_cpu[c_idx, k2])
                end
            end
            G_aa_output_cpu[r_idx, c_idx] = sum_rc_interactions * inv_num_snp_pairs

            # Symmetrize if not on diagonal
            if r_idx != c_idx
                G_aa_output_cpu[c_idx, r_idx] = G_aa_output_cpu[r_idx, c_idx]
            end
        end
    end
    # G_aa_output_cpu is modified in-place
end

"""
    compute_grm_cross!(G_cross_output::Union{Matrix{T}, CuArray{T,2}}, genotypes1::GenotypeMatrix{T}, genotypes2::GenotypeMatrix{T}; use_gpu::Bool = true) where T

Computes the cross-genomic relationship matrix (G12) between two sets of individuals,
`genotypes1` (N1 individuals) and `genotypes2` (N2 individuals, often the reference/training set).
The resulting `G_cross_output` matrix will have dimensions N1 × N2.

The formula used is `G12 = W1 * W2' / scale`, where:
- `W1` is the centered genotype matrix for `genotypes1`.
- `W2` is the centered genotype matrix for `genotypes2`.
- Both `W1` and `W2` are centered using allele frequencies derived from `genotypes2` (the reference set).
- `scale` is `sum(2*p_j*(1-p_j))` calculated from the reference allele frequencies of `genotypes2`.

The output matrix `G_cross_output` is filled in-place.

# Arguments
- `G_cross_output::Union{Matrix{T}, CuArray{T,2}}`: Pre-allocated matrix (N1 × N2) to store the results. Must be `CuArray` if `use_gpu=true`.
- `genotypes1::GenotypeMatrix{T}`: Genotypes for the first set of individuals.
- `genotypes2::GenotypeMatrix{T}`: Genotypes for the second (reference) set of individuals. Allele frequencies from this set are used for centering.
- `use_gpu::Bool = true`: If `true` and CUDA is functional, computation is performed on GPU. Otherwise, on CPU.

# Details
- Ensures SNP counts match between `genotypes1` and `genotypes2`.
- Updates allele frequencies for `genotypes2` if not already current.
- Centers both genotype matrices using allele frequencies from `genotypes2`.
- Uses `grm_cross_kernel!` (from `gpu_kernels.jl`) for GPU path.
"""
function compute_grm_cross!(
    G_cross_output::Union{Matrix{T}, CuArray{T,2}},
    genotypes1::GenotypeMatrix{T}, # New individuals (N1 x M)
    genotypes2::GenotypeMatrix{T}; # Reference individuals (N2 x M)
    use_gpu::Bool = true
) where T <: AbstractFloat

    n1, m1 = genotypes1.n_individuals, genotypes1.n_snps
    n2, m2 = genotypes2.n_individuals, genotypes2.n_snps

    if m1 != m2
        error("SNP counts must match between the two genotype sets (got $m1 and $m2).")
    end
    if size(G_cross_output) != (n1, n2)
        error("Output G_cross matrix dimensions ($(size(G_cross_output))) do not match (expected ($n1, $n2)).")
    end

    n_snps_common = m1

    # Use allele frequencies from the reference set (genotypes2) for centering both
    update_allele_frequencies!(genotypes2) # Ensure genotypes2.allele_freq is current
    ref_allele_freqs = genotypes2.allele_freq # This is a CuVector

    # Center genotypes1 using reference allele frequencies from genotypes2
    W1_centered = compute_centered_genotypes(genotypes1, ref_allele_freqs)

    # Center genotypes2 using its own (reference) allele frequencies
    W2_centered = compute_centered_genotypes(genotypes2, ref_allele_freqs) # or just compute_centered_genotypes(genotypes2) if its internal freqs are already ref_allele_freqs

    # Scaling factor from reference set (genotypes2)
    scaling_factor_sum_2pq = compute_sum_2pq_scaling_factor(genotypes2) # This uses genotypes2.allele_freq which is ref_allele_freqs
    if scaling_factor_sum_2pq <= eps(T)
        error("Sum of 2*p_j*(1-p_j) for reference set is zero or negative.")
    end
    actual_scale_for_kernel = one(T) / scaling_factor_sum_2pq

    if use_gpu && CUDA.functional()
        if !(G_cross_output isa CuArray)
            error("G_cross_output must be a CuArray when use_gpu is true.")
        end
        backend = KernelAbstractions.get_backend(G_cross_output)
        kernel! = grm_cross_kernel!(backend) # From gpu_kernels.jl
        # Kernel needs: G_out, W1, W2, n_snps, scale_factor
        kernel!(G_cross_output, W1_centered, W2_centered, n_snps_common, actual_scale_for_kernel, ndrange=(n1, n2))
        KernelAbstractions.synchronize(backend)
    else # CPU fallback
        if G_cross_output isa CuArray # If output is GPU but use_gpu=false, need to decide behavior.
                                     # For now, assume G_cross_output is Matrix if use_gpu=false.
             error("G_cross_output must be a Matrix when use_gpu is false.")
        end
        W1_cpu = Array(W1_centered)
        W2_cpu = Array(W2_centered)

        # G_cross_output .= (W1_cpu * W2_cpu') .* actual_scale_for_kernel # Direct computation
        # Or loop version:
        Threads.@threads for i_g1 in 1:n1
            for j_g2 in 1:n2
                sum_prod = zero(T)
                for k_snp in 1:n_snps_common
                    sum_prod += W1_cpu[i_g1, k_snp] * W2_cpu[j_g2, k_snp]
                end
                G_cross_output[i_g1, j_g2] = sum_prod * actual_scale_for_kernel
            end
        end
    end
    # G_cross_output is modified in-place
end

"""
    compute_epistatic_grm_cross!(G_aa_cross_output::Union{Matrix{T}, CuArray{T,2}}, genotypes1::GenotypeMatrix{T}, genotypes2::GenotypeMatrix{T}; method::Symbol = :hadamard, snp_chunk_size_for_gpu::Int = 1000, use_gpu::Bool = true) where T

Computes the cross-epistatic Genomic Relationship Matrix (G_aa_12) between two sets of individuals,
`genotypes1` (N1 individuals) and `genotypes2` (N2 individuals, the reference set).
The output matrix `G_aa_cross_output` (N1 × N2) is filled in-place.

The computation follows the Hadamard-like product logic for epistatic interactions:
`G_aa_12(i,j) = (1/NumPairs) * sum_{k<l} (W1_std_ik * W1_std_il) * (W2_std_jk * W2_std_jl)`
where `W1_std` and `W2_std` are standardized genotype matrices for `genotypes1` and `genotypes2`,
respectively. Both are standardized using allele frequencies from `genotypes2`.
`NumPairs` is the total number of unique SNP pairs.

# Arguments
- `G_aa_cross_output::Union{Matrix{T}, CuArray{T,2}}`: Pre-allocated matrix (N1 × N2) for results. Must be `CuArray` if `use_gpu=true`.
- `genotypes1::GenotypeMatrix{T}`: Genotypes for the first set.
- `genotypes2::GenotypeMatrix{T}`: Genotypes for the second (reference) set.
- `method::Symbol = :hadamard`: Computation method. Currently, only `:hadamard` logic is implemented.
- `snp_chunk_size_for_gpu::Int = 1000`: Size of SNP chunks for GPU kernel processing to manage memory and workload for the outer loop of SNP pairs.
- `use_gpu::Bool = true`: If `true` and CUDA is functional, uses GPU. Otherwise, CPU.

# Returns
- The function modifies `G_aa_cross_output` in-place.

# Details
- Ensures SNP counts match.
- Updates allele frequencies for `genotypes2` and uses these for standardizing both genotype sets.
- Uses `epistatic_cross_chunk_kernel!` (from `gpu_kernels.jl`) for GPU computation, processing SNP pairs in chunks.
- Includes a CPU fallback.
"""
function compute_epistatic_grm_cross!(
    G_aa_cross_output::Union{Matrix{T}, CuArray{T,2}},
    genotypes1::GenotypeMatrix{T}, # New individuals (N1 x M)
    genotypes2::GenotypeMatrix{T}; # Reference individuals (N2 x M)
    method::Symbol = :hadamard,    # Assuming Hadamard product logic
    snp_chunk_size_for_gpu::Int = 1000, # Chunk size for processing SNPs on GPU
    use_gpu::Bool = true
) where T <: AbstractFloat

    n1, m1 = genotypes1.n_individuals, genotypes1.n_snps
    n2, m2 = genotypes2.n_individuals, genotypes2.n_snps

    if m1 != m2
        error("SNP counts must match between the two genotype sets (got $m1 and $m2).")
    end
    if size(G_aa_cross_output) != (n1, n2)
        error("Output G_aa_cross matrix dimensions ($(size(G_aa_cross_output))) do not match (expected ($n1, $n2)).")
    end

    n_snps_common = m1
    if n_snps_common < 2
        # No SNP pairs, so epistatic GRM is zero.
        G_aa_cross_output .= zero(T)
        return
    end

    # Use allele frequencies from reference set (genotypes2) for standardization
    update_allele_frequencies!(genotypes2) # Ensure freqs are current for ref set
    ref_allele_freqs = genotypes2.allele_freq

    # Standardize genotypes1 using reference allele frequencies from genotypes2
    W1_std = compute_standardized_genotypes(genotypes1, ref_allele_freqs)

    # Standardize genotypes2 using its own (reference) allele frequencies
    W2_std = compute_standardized_genotypes(genotypes2, ref_allele_freqs) # or just compute_standardized_genotypes(genotypes2)

    num_snp_pairs = T(n_snps_common * (n_snps_common - 1) / 2)
    inv_num_snp_pairs = one(T) / num_snp_pairs

    if use_gpu && CUDA.functional()
        if !(G_aa_cross_output isa CuArray)
            error("G_aa_cross_output must be a CuArray when use_gpu is true.")
        end
        fill!(G_aa_cross_output, zero(T)) # Initialize accumulator for GPU atomic adds

        backend = KernelAbstractions.get_backend(G_aa_cross_output)
        kernel! = epistatic_cross_chunk_kernel!(backend) # From gpu_kernels.jl

        # Process in chunks of SNPs for the first SNP in a pair (k1) to manage workload
        num_snp_chunks = cld(n_snps_common, snp_chunk_size_for_gpu)

        for chunk_idx in 1:num_snp_chunks
            k1_start_idx = (chunk_idx - 1) * snp_chunk_size_for_gpu + 1
            k1_end_idx = min(chunk_idx * snp_chunk_size_for_gpu, n_snps_common - 1) # k1 cannot be the last SNP

            if k1_start_idx > k1_end_idx continue end # Skip if chunk is empty or invalid

            # Kernel sums contributions for k1 in [k1_start_idx, k1_end_idx] and k2 from k1+1 to n_snps_common
            kernel!(G_aa_cross_output, W1_std, W2_std,
                    k1_start_idx, k1_end_idx, n_snps_common,
                    ndrange=(n1,n2)) # Launch N1*N2 threads
            KernelAbstractions.synchronize(backend) # Sync after each chunk's kernel
        end
        # Final scaling after all chunks are accumulated
        G_aa_cross_output .*= inv_num_snp_pairs

    else # CPU fallback
        if G_aa_cross_output isa CuArray
             error("G_aa_cross_output must be a Matrix when use_gpu is false.")
        end
        fill!(G_aa_cross_output, zero(T)) # Initialize CPU matrix

        W1_std_cpu = Array(W1_std)
        W2_std_cpu = Array(W2_std)

        Threads.@threads for i_g1 in 1:n1
            for j_g2 in 1:n2
                sum_interaction_prod = zero(T)
                for k1 in 1:(n_snps_common-1)
                    w1_std_ig1k1 = W1_std_cpu[i_g1, k1]
                    w2_std_jg2k1 = W2_std_cpu[j_g2, k1]
                    for k2 in (k1+1):n_snps_common
                        term1 = w1_std_ig1k1 * W1_std_cpu[i_g1, k2]
                        term2 = w2_std_jg2k1 * W2_std_cpu[j_g2, k2]
                        sum_interaction_prod += term1 * term2
                    end
                end
                G_aa_cross_output[i_g1, j_g2] = sum_interaction_prod * inv_num_snp_pairs
            end
        end
    end
    # G_aa_cross_output is modified in-place
end
