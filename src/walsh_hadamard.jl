# ===== src/walsh_hadamard.jl =====
"""
Walsh-Hadamard Transform (WHT) implementation for O(M log M) epistasis detection,
where M is the number of markers.
This is often used in specific epistasis models that leverage spectral analysis.
"""

module WalshHadamard

using CUDA
using KernelAbstractions # For GPU kernels
using LinearAlgebra # For norm if used
using Statistics # For mean if used

# Assuming types.jl is accessible for GenotypeMatrix if functions operate on it.
# Helper to get Float type, if defined in main module
_Float() = Main.DynamicEpistasisGBLUP.Float # Accessing main module's Float

export fast_walsh_hadamard_transform!, EpistaticInteractionsWHT, # Changed name to avoid conflict
       detect_epistasis_wht # Renamed sparse_epistasis_detection to be specific

# Structure to store detected epistatic interactions from WHT
struct EpistaticInteractionsWHT{T<:AbstractFloat} # Renamed from EpistaticInteractions
    indices::Vector{Tuple{Int32, Int32}} # Pairs of SNP indices
    scores::Vector{T}                   # Scores for these interactions (e.g., spectral power)
    # spectral_coefficients::CuArray{T, 1} # Full spectrum might be too large to store often
end

"""
    fast_walsh_hadamard_transform!(output::CuArray{T,1}, input::CuArray{T,1}) where T

Performs an in-place Fast Walsh-Hadamard Transform on the `output` array,
which is initialized with `input` data.
The length of `input` (and `output`) must be a power of 2.
The result is normalized by `1/sqrt(N)`.
"""
function fast_walsh_hadamard_transform!(
    data_array::CuArray{T, 1} # Acts as both input and output
) where T <: AbstractFloat
    n_elements = length(data_array)
    if !ispow2(n_elements)
        error("Input length for Fast Walsh-Hadamard Transform must be a power of 2. Got $n_elements.")
    end
    if n_elements == 0 return data_array end # Handle empty array

    # The WHT is performed in stages (log2(N) stages)
    # `h_stride` is the current "distance" or step size for butterfly operations
    h_stride = 1
    backend = KernelAbstractions.get_backend(data_array)
    kernel! = wht_butterfly_kernel!(backend) # From gpu_kernels.jl (needs to be defined there or here)

    while h_stride < n_elements
        # Each kernel call performs all butterfly operations for the current h_stride
        # The kernel needs to handle n_elements/2 butterfly operations.
        # ndrange should be n_elements/2.
        # The kernel itself needs `data_array`, `h_stride`, and `n_elements`.
        kernel!(data_array, h_stride, n_elements, ndrange = n_elements ÷ 2)
        KernelAbstractions.synchronize(backend)
        h_stride *= 2
    end

    # Normalize the output
    norm_factor = sqrt(T(n_elements))
    if norm_factor > eps(T)
        data_array ./= norm_factor
    end

    return data_array # Modified in-place
end

# The wht_butterfly_kernel! is assumed to be in gpu_kernels.jl
# If it's specific to this module, it should be defined here.
# For modularity, general kernels are better in gpu_kernels.jl.

"""
    detect_epistasis_wht(genotypes_gpu::CuArray{T,2}; k_top_interactions::Int=1000, power_threshold::T=T(0.1)) -> EpistaticInteractionsWHT

Detects sparse epistatic interactions using the Walsh-Hadamard Transform spectrum.
Applies WHT to each individual's genotype vector (or derived vector), then analyzes
the spectral coefficients to identify significant interactions.

`genotypes_gpu` should be individuals × SNPs. SNPs dimension must be padded to a power of 2.
The interpretation of `genotypes_gpu` here (e.g. 0/1 for allele presence, or -1/1 for centered)
is crucial for how WHT coefficients map to interactions. Often {-1, 1} coding is used.
"""
function detect_epistasis_wht(
    genotypes_gpu::CuArray{T,2}; # Individuals x SNPs (SNPs dim must be power of 2, or padded)
    k_top_interactions::Int = 1000,
    power_threshold::T = T(0.01) # Adjusted threshold, original was 0.1
) where T <: AbstractFloat

    n_individuals, n_snps_padded = size(genotypes_gpu)

    if !ispow2(n_snps_padded)
        error("Number of SNPs (columns in genotype matrix) must be a power of 2 for WHT. Got $n_snps_padded.")
    end
    if n_individuals == 0 || n_snps_padded == 0
        return EpistaticInteractionsWHT(Tuple{Int32,Int32}[], T[])
    end

    # Allocate array for spectral coefficients (individuals × spectral_coeffs)
    spectral_matrix = CuArray{T}(undef, n_individuals, n_snps_padded)
    backend = KernelAbstractions.get_backend(genotypes_gpu)

    # Apply WHT to each individual's genotype vector (each row)
    # This can be done by launching one kernel per individual if `apply_wht_per_individual!`
    # performs the full WHT for that row.
    # The original `apply_wht_per_individual!` kernel was a serial WHT per thread.
    # A more efficient approach for larger n_snps_padded would be to parallelize the WHT itself.
    # For now, using the provided structure.

    # If apply_wht_per_individual! is a KA kernel that does the full WHT for one row:
    # kernel_apply_wht! = apply_wht_per_individual!(backend)
    # kernel_apply_wht!(spectral_matrix, genotypes_gpu, n_individuals, n_snps_padded, ndrange = n_individuals)
    # KernelAbstractions.synchronize(backend)
    # This assumes `apply_wht_per_individual!` is correctly implemented in `gpu_kernels.jl`.

    # Alternative: loop and call `fast_walsh_hadamard_transform!` for each row (less efficient due to kernel launch overhead)
    # This is more robust if `apply_wht_per_individual!` is not a single KA kernel.
    for i in 1:n_individuals
        row_data = similar(genotypes_gpu, n_snps_padded) # Temp CuArray for one row
        copyto!(row_data, @view genotypes_gpu[i, :])
        fast_walsh_hadamard_transform!(row_data) # In-place WHT
        spectral_matrix[i, :] = row_data
    end

    # Analyze spectral coefficients to identify interactions
    # This involves averaging power across individuals for each coefficient
    # and then mapping spectral indices back to SNP pairs.

    # Compute average spectral power across individuals for each coefficient
    # spectral_power = vec(mean(abs2.(spectral_matrix), dims=1))
    # Using CUDA.jl for GPU-side reduction:
    if n_individuals > 0
        spectral_power_gpu = sum(abs2.(spectral_matrix), dims=1) ./ n_individuals
        spectral_power_cpu = Array(vec(spectral_power_gpu)) # Move to CPU for sorting and selection
    else
        spectral_power_cpu = zeros(T, n_snps_padded)
    end


    # Find top-k coefficients that exceed the threshold
    # Store as (original_spectral_index, power_value)
    candidate_coeffs = Tuple{Int, T}[]
    for idx in 1:n_snps_padded
        if spectral_power_cpu[idx] > power_threshold
            push!(candidate_coeffs, (idx, spectral_power_cpu[idx]))
        end
    end

    # Sort candidates by power and take top k
    sort!(candidate_coeffs, by = x -> x[2], rev=true)
    num_to_select = min(k_top_interactions, length(candidate_coeffs))

    selected_interactions_indices = Tuple{Int32, Int32}[]
    selected_interaction_scores = T[]

    for i in 1:num_to_select
        spectral_idx, score = candidate_coeffs[i]

        # Decode interaction from spectral index (1-based index from WHT output)
        # This mapping depends on the WHT definition (e.g., sequency/Walsh ordering).
        # A common mapping: spectral index `s` (0-based) corresponds to interaction between SNPs
        # whose indices have '1's at the same bit positions as `s` in their binary representation.
        # For pairwise interactions, `s` will have two '1's in its binary form.
        # E.g., if s = (binary ...1...1...), the 1s are at positions k1 and k2. Interaction is (k1, k2).
        # The `decode_interaction_index` function from original code attempts this.

        # Assuming `decode_interaction_index` takes 1-based spectral_idx and total (original) SNPs.
        # The WHT is on `n_snps_padded`. We need to map back to original SNP indices if padding occurred.
        # This detail is missing from the original `decode_interaction_index`.
        # For now, assume n_snps_padded is the actual number of SNPs we care about for indexing.

        snp1, snp2 = decode_interaction_index_wht(spectral_idx, n_snps_padded)

        # Ensure valid, distinct pair and within original SNP range if applicable
        # (original_n_snps would be needed if n_snps_padded includes padding)
        if snp1 > 0 && snp2 > 0 && snp1 != snp2 # And snp1 <= original_n_snps, snp2 <= original_n_snps
            # Avoid duplicates, e.g. (s1,s2) vs (s2,s1) by storing canonical form
            push!(selected_interactions_indices, (min(snp1,snp2), max(snp1,snp2)))
            push!(selected_interaction_scores, score)
        end
    end

    # Remove duplicate pairs that might arise from decoding if not careful
    unique_pairs_dict = Dict{Tuple{Int32,Int32}, T}()
    for (idx, pair) in enumerate(selected_interactions_indices)
        current_score = selected_interaction_scores[idx]
        if get(unique_pairs_dict, pair, -Inf) < current_score # Keep the one with higher score if duplicate
            unique_pairs_dict[pair] = current_score
        end
    end

    final_indices = collect(keys(unique_pairs_dict))
    final_scores = collect(values(unique_pairs_dict))

    return EpistaticInteractionsWHT(final_indices, final_scores)
end


"""
    decode_interaction_index_wht(spectral_idx_1_based::Int, n_coeffs::Int) -> Tuple{Int32, Int32}

Decodes a 1-based spectral coefficient index from WHT into a pair of interacting SNP indices (1-based).
This simplified version assumes the spectral index directly maps to a bitmask,
and we look for bitmasks with two bits set (for pairwise interactions).
The mapping from WHT coefficient order (e.g., sequency) to SNP interactions can be complex.
The "Gray code" mention in original `decode_interaction_index` implies a specific ordering.
This is a common but non-trivial mapping.

For a Walsh (Hadamard) ordered WHT:
- Index 0 (DC): Mean effect.
- Indices with one bit set (2^k): Main effects of SNPs.
- Indices with two bits set (2^k1 + 2^k2): Pairwise interaction between SNP k1 and SNP k2.
(This assumes SNPs are indexed 0 to N-1 corresponding to bit positions).
"""
function decode_interaction_index_wht(spectral_idx_1_based::Int, n_coeffs::Int) :: Tuple{Int32, Int32}
    # Convert to 0-based index for bitwise operations
    idx_0_based = spectral_idx_1_based - 1

    if idx_0_based < 0 || idx_0_based >= n_coeffs
        return (Int32(0), Int32(0)) # Invalid index
    end

    # We are looking for indices that correspond to pairwise interactions.
    # In Walsh (Hadamard) ordering, these are indices `s` where `s` has exactly two bits set in its binary representation.
    # The positions of these two bits (0-indexed) correspond to the 0-indexed SNP numbers.

    if count_set_bits(idx_0_based) == 2
        snp1_0idx = -1
        snp2_0idx = -1

        for bit_pos in 0:(sizeof(idx_0_based)*8 - 1) # Iterate through possible bit positions
            if (idx_0_based >> bit_pos) & 1 == 1 # If bit is set
                if snp1_0idx == -1
                    snp1_0idx = bit_pos
                else
                    snp2_0idx = bit_pos
                    break # Found the second bit
                end
            end
        end

        if snp1_0idx != -1 && snp2_0idx != -1
            # Convert 0-based SNP indices to 1-based
            return (Int32(snp1_0idx + 1), Int32(snp2_0idx + 1))
        end
    end

    return (Int32(0), Int32(0)) # Not a pairwise interaction index by this rule, or invalid
end

# Helper to count set bits (population count)
function count_set_bits(n::Int)
    count = 0
    while n > 0
        n &= (n - 1) # Clear the least significant set bit
        count += 1
    end
    return count
end


# The original `pad_genotypes` function:
"""
    pad_genotypes_wht(genotypes_in::CuArray{T,2}, target_n_snps_padded::Int) where T

Pads the genotype matrix (SNPs dimension) with zeros to reach `target_n_snps_padded`.
`target_n_snps_padded` must be a power of 2 and >= original number of SNPs.
"""
function pad_genotypes_wht(
    genotypes_in::CuArray{T,2},
    target_n_snps_padded::Int
) where T <: AbstractFloat
    n_individuals, n_snps_original = size(genotypes_in)

    if n_snps_original == target_n_snps_padded
        return genotypes_in # No padding needed
    elseif n_snps_original > target_n_snps_padded
        error("Target padded size ($target_n_snps_padded) is less than original SNP count ($n_snps_original).")
    end

    padded_genotypes = CUDA.zeros(T, n_individuals, target_n_snps_padded)
    # Copy original genotype data into the padded matrix
    # copyto!(padded_genotypes, CartesianIndices((1:n_individuals, 1:n_snps_original)),
    #         genotypes_in, CartesianIndices(genotypes_in)) -> This is not correct for submatrix copy.
    # Correct way for submatrix copy:
    padded_genotypes_view = @view padded_genotypes[:, 1:n_snps_original]
    copyto!(padded_genotypes_view, genotypes_in)

    return padded_genotypes
end

# Kernels `wht_butterfly_kernel!` and `apply_wht_per_individual!` are assumed to be in `gpu_kernels.jl`.

end # module WalshHadamard
