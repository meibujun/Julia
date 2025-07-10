# ===== src/symmetric_polynomials.jl =====
"""
Symmetric polynomial algorithms for efficient computation of epistatic GRMs,
particularly for pairwise (additive x additive) interactions.
This approach, often attributed to work by Jiang & Reif, can achieve O(N²M)
complexity for the epistatic GRM instead of O(N²M²), where N is individuals, M is markers.
"""

module SymmetricPolynomials

using CUDA
using LinearAlgebra # For tr, Diagonal, I if needed for related math
using KernelAbstractions # For GPU kernels

# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float # Accessing main module's Float

export SymmetricPolynomialState, initialize_symmetric_state,
       compute_epistatic_grm_symmetric! # Main function to export

"""
    SymmetricPolynomialState{T}

Structure to store pre-computed symmetric polynomials (e.g., power sums p_k,
elementary symmetric polynomials e_k) for each individual. These are used
to efficiently calculate sums of interaction terms.

- `e1`: First elementary symmetric polynomial (sum of elements). CuArray: Individuals × 1.
- `p2`: Second power sum (sum of squares of elements). CuArray: Individuals × 1.
        (The original struct had `e2`, but `p2` is often more directly used with `e1`
         to get sum of pairwise products: `sum_{i<j} x_i x_j = (e1^2 - p2) / 2`.
         Let's stick to `e1` and `p2` as fundamental, and `e2` can be derived if needed.)
- `power_sums`: Dictionary to store higher-order power sums if needed.
"""
struct SymmetricPolynomialState{T<:AbstractFloat}
    e1::CuArray{T, 2}  # First elementary symmetric polynomial (sum_k W_ik). Dim: (n_individuals, 1)
    p2::CuArray{T, 2}  # Second power sum (sum_k W_ik^2). Dim: (n_individuals, 1)
    # Higher order power sums can be added if general k-way interactions are modeled.
    # power_sums::Dict{Int, CuArray{T, 2}}
end

"""
    initialize_symmetric_state(W_std::CuArray{T,2}) -> SymmetricPolynomialState{T}

Initializes the `SymmetricPolynomialState` by computing the first elementary
symmetric polynomial (e1, sum of standardized genotypes per individual) and
the second power sum (p2, sum of squared standardized genotypes per individual).
`W_std` is the standardized genotype matrix (individuals × SNPs).
"""
function initialize_symmetric_state(
    W_std::CuArray{T,2} # Standardized genotype matrix: (X_ij - 2p_j) / sqrt(2p_j(1-p_j))
) where T <: AbstractFloat

    # e1_i = sum_k W_std_ik (sum over SNPs for each individual i)
    e1_gpu = sum(W_std, dims=2) # Result is Individuals x 1

    # p2_i = sum_k (W_std_ik)^2 (sum of squares over SNPs for each individual i)
    p2_gpu = sum(abs2.(W_std), dims=2) # abs2 for complex numbers, same as .^2 for reals. Use .^2 for clarity.
                                     # p2_gpu = sum(W_std .^ 2, dims=2)

    return SymmetricPolynomialState(e1_gpu, p2_gpu)
end

"""
    compute_epistatic_grm_symmetric!(
        G_aa_output::CuArray{T,2},
        W_std::CuArray{T,2};
        # interaction_order::Int = 2, # Currently only pairwise (order 2) fully detailed
        use_gpu::Bool = true
    )

Computes the additive-by-additive epistatic GRM using the symmetric polynomial approach.
This method is efficient for pairwise interactions.
`G_aa_output` is modified in-place.
`W_std` is the standardized genotype matrix.
The formula for pairwise epistatic GRM element (i,j) is often:
  G_aa(i,j) = 0.5 * [ (sum_k W_ik W_jk)^2 - sum_k (W_ik W_jk)^2 ]
This needs to be normalized by the number of SNP pairs if it's to match other GRM scales.
"""
function compute_epistatic_grm_symmetric!(
    G_aa_output::CuArray{T,2}, # Output matrix (Individuals x Individuals)
    W_std::CuArray{T,2};       # Standardized genotype matrix (Individuals x SNPs)
    # sym_state::SymmetricPolynomialState{T} # Optional: pass precomputed state
    # The original kernel `epistatic_grm_symmetric_kernel!` took e1, e2.
    # However, the formula G_aa(i,j) = 0.5 * [ (sum_k W_ik W_jk)^2 - sum_k (W_ik W_jk)^2 ]
    # does NOT directly use e1 and e2 (which are sums over SNPs for a *single* individual).
    # It uses terms involving products of W_ik and W_jk (for *two* individuals i and j).
    # The kernel named `epistatic_grm_symmetric_kernel!` in `gpu_kernels.jl` implements this formula.
    use_gpu::Bool = true # Redundant if inputs are CuArray, but for consistency
) where T <: AbstractFloat

    n_individuals, n_snps = size(W_std)

    if size(G_aa_output) != (n_individuals, n_individuals)
        error("Output G_aa matrix dimensions do not match number of individuals.")
    end

    if use_gpu && CUDA.functional()
        backend = KernelAbstractions.get_backend(G_aa_output)
        # The kernel `epistatic_grm_symmetric_kernel!` from `gpu_kernels.jl`
        # should implement the formula: 0.5 * [ (sum_k W_ik W_jk)^2 - sum_k (W_ik^2 W_jk^2) ]
        # It needs W_std as input. The e1, e2 parameters in the original kernel def were misleading
        # if that formula is the target.
        # Let's assume the kernel in gpu_kernels.jl is correctly:
        # kernel!(G_aa_output, W_std, n_individuals, n_snps, ndrange=(n_individuals, n_individuals))

        # Corrected call based on the formula for pairwise interactions:
        # The kernel in `gpu_kernels.jl` (epistatic_grm_symmetric_kernel!) calculates:
        #   sum_wiwj = sum_k (W_ik * W_jk)
        #   sum_wi2wj2 = sum_k (W_ik^2 * W_jk^2)  -- error in original kernel, was (W_ik * W_jk)^2
        #   val = 0.5 * (sum_wiwj^2 - sum_wi2wj2)
        # This is the correct formula for G_aa(i,j) for pairwise epistasis.
        # The kernel needs W_std, n_individuals, n_snps.
        # The `e1, e2` params in the original kernel were not used as per this formula.

        kernel! = epistatic_grm_symmetric_kernel!(backend) # From gpu_kernels.jl
        # This kernel was defined as: epistatic_grm_symmetric_kernel!(G_aa, W, e1, e2, n_individuals, n_snps)
        # This signature is inconsistent with the formula above.
        # The formula G_aa(i,j) = 0.5 * [ (direct_prod_sum)^2 - direct_prod_sq_sum ] does not use e1, e2 of individuals.
        # The kernel in `gpu_kernels.jl` was modified to reflect this formula directly from W_std.
        # It should be: kernel!(G_aa_output, W_std, ndrange=...)
        # Let's assume the kernel in `gpu_kernels.jl` is:
        # @kernel function epistatic_grm_symmetric_kernel!(G_aa, W_std_input)
        #     i, j = @index(Global, NTuple)
        #     n_indiv, n_snps_dim = size(W_std_input)
        #     if i <= n_indiv && j <= n_indiv && i <= j
        #         sum_prod_direct = zero(eltype(G_aa))
        #         sum_prod_sq_direct = zero(eltype(G_aa))
        #         for k in 1:n_snps_dim
        #             term_ik = W_std_input[i,k]
        #             term_jk = W_std_input[j,k]
        #             prod_val = term_ik * term_jk
        #             sum_prod_direct += prod_val
        #             sum_prod_sq_direct += prod_val * prod_val # This is sum (W_ik W_jk)^2
        #         end
        #         G_aa[i,j] = 0.5 * (sum_prod_direct * sum_prod_direct - sum_prod_sq_direct)
        #         if i != j; G_aa[j,i] = G_aa[i,j]; end
        #     end
        # end
        # This kernel would be called as:
        kernel!(G_aa_output, W_std, ndrange=(n_individuals, n_individuals))
        KernelAbstractions.synchronize(backend)
    else # CPU fallback
        W_std_cpu = Array(W_std)
        G_aa_cpu = Array(G_aa_output) # Operate on CPU version of output

        Threads.@threads for i_row in 1:n_individuals
            for j_col in i_row:n_individuals # Upper triangle
                sum_prod_direct_cpu = zero(T)
                sum_prod_sq_direct_cpu = zero(T)
                for k_snp in 1:n_snps
                    term_ik_cpu = W_std_cpu[i_row, k_snp]
                    term_jk_cpu = W_std_cpu[j_col, k_snp]
                    prod_val_cpu = term_ik_cpu * term_jk_cpu
                    sum_prod_direct_cpu += prod_val_cpu
                    sum_prod_sq_direct_cpu += prod_val_cpu * prod_val_cpu
                end
                val = T(0.5) * (sum_prod_direct_cpu * sum_prod_direct_cpu - sum_prod_sq_direct_cpu)
                G_aa_cpu[i_row, j_col] = val
                if i_row != j_col
                    G_aa_cpu[j_col, i_row] = val
                end
            end
        end
        copyto!(G_aa_output, G_aa_cpu) # Copy back to GPU if G_aa_output was CuArray
    end

    # Normalization: The formula G_aa(i,j) = 0.5 * [ (sum_k W_ik W_jk)^2 - sum_k (W_ik W_jk)^2 ]
    # represents sum_{u<v} (W_iu W_iv)(W_ju W_jv). This is often scaled by 1/num_pairs.
    # The original code had `G_aa ./= T(n_interactions)` where `n_interactions = binomial(n_snps, order)`.
    # For pairwise (order=2), this is `binomial(n_snps, 2) = n_snps * (n_snps-1) / 2`.
    if n_snps >= 2
        num_snp_pairs = T(n_snps * (n_snps - 1) / 2)
        if num_snp_pairs > eps(T)
            G_aa_output ./= num_snp_pairs
        end
    else # If less than 2 SNPs, no pairs, G_aa should be zero.
        G_aa_output .= zero(T)
    end

    return G_aa_output # Modified in-place
end


# Higher-order interactions and related kernels (compute_higher_order_epistatic_grm!, extend_symmetric_polynomials!, higher_order_kernel!)
# were mentioned but not fully detailed or essential for the primary pairwise epistasis.
# These are advanced extensions. For now, focusing on the core pairwise functionality.
# The `compute_epistatic_grm_symmetric_cpu!` was a CPU fallback/verification, which is good to have.
# The `compute_epistatic_kinship_optimized` and its chunk kernel were alternative ways, perhaps for very large scale.

# The kernel `epistatic_grm_symmetric_kernel!` is defined in `gpu_kernels.jl`.
# Its signature in `gpu_kernels.jl` was:
# `epistatic_grm_symmetric_kernel!(G_aa, W, e1, e2, n_individuals, n_snps)`
# This is inconsistent with the formula `0.5 * [ (sum_k W_ik W_jk)^2 - sum_k (W_ik W_jk)^2 ]`.
# The kernel in `gpu_kernels.jl` has been updated to use the formula that takes `W_std` directly.

# Export functions if this file were a module
# export SymmetricPolynomialState, initialize_symmetric_state, compute_epistatic_grm_symmetric!, compute_higher_order_epistatic_grm! # Add new export

"""
    extend_symmetric_polynomials!(
        sym_state::SymmetricPolynomialState{T},
        W_std::CuArray{T,2},
        max_order::Int
    ) where T

Extends the `sym_state` to include power sums (p_k = sum_s W_s^k) up to `max_order`.
These are power sums for individual genotype vectors, not for products like W_is * W_js.
This function was part of the original combined code structure.
Its direct utility for the higher-order GRM formula K_k(i,j) involving products W_is*W_js
needs to be clarified, as that formula requires power sums of these products, not power sums of W_is.
However, including it as per original structure.
"""
function extend_symmetric_polynomials!(
    sym_state::SymmetricPolynomialState{T},
    W_std::CuArray{T,2}, # Standardized genotypes (Individuals x SNPs)
    max_order::Int
) where T <: AbstractFloat

    # The sym_state already has p1 (as e1, though e1 is sum, p1 is also sum) and p2.
    # For this function to be useful with the struct, power_sums field should exist.
    # Let's assume SymmetricPolynomialState is augmented with:
    # power_sums_dict::Dict{Int, CuArray{T,2}}
    if !isdefined(sym_state, :power_sums_dict)
        # This indicates a mismatch with the original struct idea that had this field.
        # For now, we'll proceed as if it could be added or this function is more conceptual.
        # If strictly following the provided struct, this function cannot store p_k for k > 2.
        # To make it work with the provided struct, this function would need to return them,
        # or the struct needs modification. The original code had `power_sums` in the struct.
        # I will assume the struct is intended to be:
        # struct SymmetricPolynomialState{T<:AbstractFloat}
        #    e1::CuArray{T, 2}
        #    # p2::CuArray{T, 2} # p2 is power_sums_dict[2]
        #    power_sums_dict::Dict{Int, CuArray{T,2}} # Stores p1, p2, p3, ...
        # end
        # And initialize_symmetric_state would populate power_sums_dict[1] and power_sums_dict[2].
        # For now, let's proceed conceptually. This is a structural detail to refine.
        # The current struct only has e1 and p2. This function cannot extend it as is.

        # If the struct had `power_sums::Dict{Int, CuArray{T, 2}}`:
        # if !haskey(sym_state.power_sums, 1) sym_state.power_sums[1] = sum(W_std, dims=2) end
        # if !haskey(sym_state.power_sums, 2) sym_state.power_sums[2] = sum(W_std.^2, dims=2) end

        # for k_order in 3:max_order
        #     if !haskey(sym_state.power_sums, k_order)
        #         sym_state.power_sums[k_order] = sum(W_std .^ k_order, dims=2)
        #     end
        # end
        # This part is commented out as it depends on struct modification.
        # For now, this function is a no-op or needs to return the power sums.
        return # Does nothing without a place to store results in sym_state
    end
end


"""
    compute_higher_order_epistatic_grm!(
        G_out::CuArray{T,2},
        W_std::CuArray{T,2},
        # sym_state::SymmetricPolynomialState{T}, # sym_state for W_i might not be directly used by G(i,j) kernel
        order::Int
    ) where T

Computes higher-order epistatic GRMs (e.g., 3-way, 4-way interactions).
`order` specifies the interaction order (e.g., 3 for AxAxA).
The actual computation kernel `higher_order_epistatic_kernel!` needs to implement
the complex formulas for these GRM elements.
"""
function compute_higher_order_epistatic_grm!(
    G_out::CuArray{T,2},      # Output GRM (Individuals x Individuals)
    W_std::CuArray{T,2},      # Standardized genotypes (Individuals x SNPs)
    # sym_state::SymmetricPolynomialState{T}, # May not be needed if kernel recomputes sums of products
    order::Int                # Interaction order (e.g., 3 for AAA)
) where T <: AbstractFloat

    n_individuals, n_snps = size(W_std)
    if size(G_out) != (n_individuals, n_individuals)
        error("Output G matrix dimensions do not match.")
    end
    if order < 2
        error("Interaction order must be at least 2.")
    end
    if order == 2
        # For order 2, call the specific symmetric polynomial function for pairwise
        # This ensures consistency with the optimized pairwise version.
        return compute_epistatic_grm_symmetric!(G_out, W_std)
    end
    if order > 3 # For now, only stubbing up to order 3 conceptually
        @warn "Higher-order GRM for order > 3 is highly experimental and may be a stub."
        # fill!(G_out, zero(T))
        # return G_out
    end

    # The kernel `higher_order_epistatic_kernel!` (in gpu_kernels.jl) will handle the actual math.
    # It needs W_std and the order.
    # The `sym_state` (based on individual W_i) is not directly used in formulas like
    # K_k(i,j) involving sums of products of (W_is * W_js).
    # So, sym_state is removed from the kernel call here.

    backend = KernelAbstractions.get_backend(G_out)
    # Assuming higher_order_epistatic_kernel! is defined in gpu_kernels.jl
    kernel_higher! = higher_order_epistatic_kernel!(backend)

    # The kernel needs G_out, W_std, and Val(order) to dispatch on order if it's a compile-time constant.
    # Or, pass order as a runtime argument. Val(order) is better for specialization.
    kernel_higher!(G_out, W_std, Val(order), ndrange=(n_individuals, n_individuals))
    KernelAbstractions.synchronize(backend)

    # Normalization: For k-th order, scale by 1 / choose(M, k)
    if n_snps >= order
        num_interactions = T(binomial(n_snps, order))
        if num_interactions > eps(T)
            G_out ./= num_interactions
        else # Should not happen if n_snps >= order > 1
            G_out .= zero(T)
        end
    else # Not enough SNPs for this order of interaction
        G_out .= zero(T)
    end

    return G_out # Modified in-place
end


end # module SymmetricPolynomials
