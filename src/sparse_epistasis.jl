# ===== src/sparse_epistasis.jl =====
"""
Sparse Epistasis Detection and Modeling.
Implements algorithms for identifying a sparse set of significant epistatic interactions
from a vast number of potential pairs, and fitting models with these interactions.
Methods include screening, regularization (Elastic Net, Group LASSO), and adaptive learning.
"""

module SparseEpistasis

using CUDA
using SparseArrays # For sparse score matrices
using LinearAlgebra # For norm, dot, opnorm, I, cholesky, svd, qr
using Statistics  # For mean, std, quantile, cor
# using Wavelets # Mentioned in original, but not directly used in provided code for this module
# using FFTW   # Might be used by Wavelets or other signal processing if added
using IterativeSolvers # For lsqr or other iterative solves if MME becomes too large
using ProximalOperators # For Elastic Net / LASSO proximal gradient steps

# Assuming types.jl and gpu_kernels.jl are accessible.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export SparseEpistaticModel, detect_sparse_interactions_screening, # Renamed
       fit_elastic_net_epistasis, fit_group_lasso_epistasis, # Renamed
       adaptive_sparse_epistasis_learning # Renamed

"""
    SparseEpistaticModel{T}

Represents a sparse epistatic model, storing selected interactions and their coefficients.
Includes information about the regularization method used.
"""
struct SparseEpistaticModel{T<:AbstractFloat}
    interactions::Vector{Tuple{Int32, Int32}} # List of (SNP1_idx, SNP2_idx)
    coefficients::Vector{T}                 # Corresponding effect sizes
    lambda1_penalty::T                      # L1 penalty strength (for LASSO part)
    lambda2_penalty::T                      # L2 penalty strength (for Ridge part)
    achieved_sparsity_level::T              # Proportion of non-zero effects relative to candidates
    method_used::Symbol                     # e.g., :elastic_net, :group_lasso
end

"""
    detect_sparse_interactions_screening(...) -> Tuple{Vector{Tuple{Int32,Int32}}, SparseMatrixCSC{T}}

Performs an initial screening to detect potentially important sparse epistatic interactions.
Uses multiple criteria (e.g., mutual information, distance correlation, HSIC) if specified,
then combines scores and selects top candidates.
Returns a vector of selected interaction pairs and the ensemble score matrix.
"""
function detect_sparse_interactions_screening(
    genotypes_gpu::CuArray{T,2},    # Individuals x SNPs
    phenotypes_gpu::CuVector{T};  # Vector of phenotypes on GPU
    max_candidate_interactions::Int = 10000, # Max number of interaction pairs to return from screening
    screening_methods::Vector{Symbol} = [:correlation], # Default to simple correlation for now
                                                      # [:mutual_information, :dcor, :hsic] require more complex impl.
    score_aggregation_quantile::T = T(0.99), # Quantile for thresholding combined scores
    # parallel_chunks_screening::Int = 10 # For chunking computation if methods are per-chunk
    verbose::Bool = false
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_gpu)
    if n_snps < 2
        return Tuple{Int32,Int32}[], spzeros(T, Int, n_snps, n_snps)
    end

    # Store scores from different methods if multiple are used.
    # For now, focusing on one method for simplicity in structure.
    # interaction_scores_by_method = Dict{Symbol, SparseMatrixCSC{T, Int}}()

    final_interaction_scores_sparse = spzeros(T, Int, n_snps, n_snps) # Store as SNP_idx1 x SNP_idx2 -> score

    for method_symbol in screening_methods
        if verbose println("  Screening with method: $method_symbol...") end

        # `compute_interaction_scores_for_method` would dispatch to different score calculations.
        # This is a placeholder for calling specific scoring functions.
        # For :correlation, it would be correlation of (SNP_i * SNP_j) with phenotype.
        # The original `compute_interaction_scores` and `compute_chunk_scores` were complex.
        # Let's simplify for initial structure, assuming a direct scoring function.

        # Example for :correlation method (simplified, full pairwise is N^2 SNPs)
        if method_symbol == :correlation
            # This is computationally intensive. A full pairwise screen is often not done.
            # More common: screen main effects, then test interactions among top main effects.
            # Or, use random subset of pairs.
            # For now, conceptual loop for all pairs (inefficient for large n_snps).
            # This part needs a highly optimized kernel or a more selective strategy.

            # Placeholder for a more optimized pairwise scoring kernel:
            # pairwise_scores_gpu = CUDA.zeros(T, n_snps, n_snps) # Temp dense matrix for scores
            # kernel_pairwise_correlation_scores!(pairwise_scores_gpu, genotypes_gpu, phenotypes_gpu)
            # KernelAbstractions.synchronize(get_backend(pairwise_scores_gpu))
            # final_interaction_scores_sparse = sparse(pairwise_scores_gpu) # Convert dense to sparse if needed

            # Fallback to a very simplified CPU version for structure (NOT FOR PERFORMANCE)
            # This is just to illustrate where scores would be populated.
            # In reality, this must be a GPU kernel or a more advanced screening.
            geno_host = Array(genotypes_gpu)
            pheno_host = Array(phenotypes_gpu)
            temp_scores_dict = Dict{Tuple{Int,Int}, T}()

            # Limit number of pairs to check for this placeholder version to avoid timeout
            max_pairs_to_check_stub = min(n_snps * (n_snps-1) ÷ 2, 100000) # Limit for stub
            checked_pairs = 0

            for i in 1:n_snps
                if checked_pairs >= max_pairs_to_check_stub break end
                for j in (i+1):n_snps
                    if checked_pairs >= max_pairs_to_check_stub break end
                    interaction_term_host = geno_host[:,i] .* geno_host[:,j]
                    if var(interaction_term_host) > eps(T) && var(pheno_host) > eps(T)
                        score = cor(interaction_term_host, pheno_host)
                        if abs(score) > T(0.01) # Arbitrary small threshold for storing
                           temp_scores_dict[(i,j)] = score
                        end
                    end
                    checked_pairs += 1
                end
            end
            # Convert dict to sparse matrix
            I_inds, J_inds, V_vals = Int[], Int[], T[]
            for ((i,j), val) in temp_scores_dict
                push!(I_inds, i); push!(J_inds, j); push!(V_vals, val)
                # Add symmetric part if matrix is symmetric (scores often are)
                # push!(I_inds, j); push!(J_inds, i); push!(V_vals, val)
            end
            if !isempty(I_inds)
                final_interaction_scores_sparse = sparse(I_inds, J_inds, V_vals, n_snps, n_snps)
            end

        elseif method_symbol == :mutual_information
            # Call the chunked version of MI scores
            chunk_scores_dict = mutual_information_scores_chunked(
                genotypes_gpu, phenotypes_gpu,
                1, n_snps, # For full matrix pass, chunk is all snp_i
                n_snps
            )
            # Convert dict to sparse matrix and add to final_interaction_scores_sparse
            # This part needs to handle merging if multiple methods are used, or if method itself is chunked.
            # For now, assuming only one method and it populates final_interaction_scores_sparse directly or via temp dict.
            # This simplified loop assumes the method fills a compatible sparse matrix or can be converted.
            # The current MI stub returns a Dict.
            for ((i,j), val) in chunk_scores_dict
                if i < j final_interaction_scores_sparse[i,j] = val
                else final_interaction_scores_sparse[j,i] = val end
            end

        elseif method_symbol == :dcor
            # Call the chunked version of dCor scores
            chunk_scores_dict = distance_correlation_scores_for_chunk( # Renamed in previous step
                genotypes_gpu, phenotypes_gpu,
                1, n_snps, # For full matrix pass
                n_snps
            )
            for ((i,j), val) in chunk_scores_dict
                 if i < j final_interaction_scores_sparse[i,j] = val
                 else final_interaction_scores_sparse[j,i] = val end
            end
        elseif method_symbol == :hsic
            chunk_scores_dict = hilbert_schmidt_scores_chunked(
                genotypes_gpu, phenotypes_gpu,
                1, n_snps, # For full matrix pass
                n_snps
            )
            for ((i,j), val) in chunk_scores_dict
                if i < j final_interaction_scores_sparse[i,j] = val
                else final_interaction_scores_sparse[j,i] = val end
            end
        else
            # println("Warning: Screening method $method_symbol not implemented in this stub.")
        end
    end

    # If multiple methods were used, `final_interaction_scores_sparse` should be an aggregation.
    # For now, `final_interaction_scores_sparse` holds scores from the (single) method.

    # Select top interactions based on aggregated scores
    # Extract non-zero elements from sparse matrix (these are candidate interactions)
    rows, cols, scores_vec = findnz(final_interaction_scores_sparse)

    candidate_interaction_tuples = Tuple{Int32, Int32}[]
    candidate_scores_final = T[]

    for k_idx in 1:length(scores_vec)
        snp1 = rows[k_idx]
        snp2 = cols[k_idx]
        score_val = scores_vec[k_idx]
        # Ensure snp1 < snp2 if matrix was symmetric or only upper/lower triangle stored.
        # If findnz gives both (i,j) and (j,i), ensure unique pairs.
        # Assuming findnz on the (potentially symmetric) sparse matrix.
        if snp1 < snp2 # Process unique pairs from upper triangle
            push!(candidate_interaction_tuples, (Int32(snp1), Int32(snp2)))
            push!(candidate_scores_final, score_val)
        end
    end

    # Sort candidates by absolute score and take top N or threshold
    # Using absolute scores for ranking importance.
    sorted_indices = sortperm(abs.(candidate_scores_final), rev=true)

    num_to_select_final = min(max_candidate_interactions, length(sorted_indices))

    selected_interactions_final = Vector{Tuple{Int32,Int32}}(undef, num_to_select_final)
    # Also return their scores if needed by subsequent steps.
    # selected_scores_for_output = Vector{T}(undef, num_to_select_final)

    for i in 1:num_to_select_final
        original_idx = sorted_indices[i]
        selected_interactions_final[i] = candidate_interaction_tuples[original_idx]
        # selected_scores_for_output[i] = candidate_scores_final[original_idx]
    end

    # The ensemble_scores matrix is `final_interaction_scores_sparse` in this simplified version.
    return selected_interactions_final, final_interaction_scores_sparse
end


"""
    fit_elastic_net_epistasis(...) -> SparseEpistaticModel{T}

Fits an Elastic Net regularized model to estimate effects for a pre-selected set of interactions.
`candidate_interactions` is a list of (SNP1, SNP2) tuples.
"""
function fit_elastic_net_epistasis(
    genotypes_gpu::CuArray{T,2},    # Individuals x SNPs
    phenotypes_gpu::CuVector{T},  # Phenotypes on GPU
    candidate_interactions::Vector{Tuple{Int32, Int32}}; # From screening
    lambda1::T = T(0.01),  # L1 penalty (LASSO)
    lambda2::T = T(0.001), # L2 penalty (Ridge)
    max_iterations_en::Int = 1000,
    tolerance_en::T = T(1e-6)
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_gpu)
    n_candidate_interactions = length(candidate_interactions)

    if n_candidate_interactions == 0
        return SparseEpistaticModel(Tuple{Int32,Int32}[], T[], lambda1, lambda2, T(0.0), :elastic_net_no_candidates)
    end

    # 1. Build design matrix X_interactions (Individuals x n_candidate_interactions)
    # Each column of X is the product of genotypes for an interaction pair.
    X_interactions_gpu = CUDA.zeros(T, n_individuals, n_candidate_interactions)
    # This needs a kernel: `build_interaction_kernel!` from `gpu_kernels.jl`
    backend = KernelAbstractions.get_backend(X_interactions_gpu)
    kernel_build_X! = build_interaction_kernel!(backend)
    kernel_build_X!(X_interactions_gpu, genotypes_gpu, CuArray(candidate_interactions), # Pass interactions to GPU
                    n_individuals, n_candidate_interactions,
                    ndrange = n_individuals * n_candidate_interactions)
    KernelAbstractions.synchronize(backend)

    # 2. Solve Elastic Net problem: min_β { ||y - Xβ||² + λ1||β||₁ + λ2||β||₂² }
    # This is often done with Proximal Gradient Descent or Coordinate Descent.
    # The original code referred to `accelerated_proximal_gradient`.

    initial_beta_coeffs = CUDA.zeros(T, n_candidate_interactions)

    # `accelerated_proximal_gradient_en` is a placeholder for the actual solver.
    # It would use `soft_threshold` operator from ProximalOperators.jl or custom.
    final_beta_coeffs_gpu = accelerated_proximal_gradient_en(
        X_interactions_gpu, phenotypes_gpu, initial_beta_coeffs,
        lambda1, lambda2, max_iterations_en, tolerance_en
    )

    # 3. Extract non-zero coefficients and corresponding interactions
    final_beta_coeffs_host = Array(final_beta_coeffs_gpu)
    non_zero_indices = findall(x -> abs(x) > T(1e-8), final_beta_coeffs_host) # Small tolerance for floating point

    selected_model_interactions = candidate_interactions[non_zero_indices]
    selected_model_coefficients = final_beta_coeffs_host[non_zero_indices]

    achieved_sparsity = n_candidate_interactions > 0 ? T(length(non_zero_indices) / n_candidate_interactions) : T(0.0)

    return SparseEpistaticModel(
        selected_model_interactions,
        selected_model_coefficients,
        lambda1, lambda2,
        achieved_sparsity,
        :elastic_net
    )
end

"""
Placeholder for Accelerated Proximal Gradient for Elastic Net.
Solves: min_β { (1/2N) ||y - Xβ||² + λ1||β||₁ + (λ2/2)||β||₂² }
(Scaling of λ2 might differ, original had λ2 directly on ||β||₂²)
"""
function accelerated_proximal_gradient_en(
    X_gpu::CuArray{T,2}, y_gpu::CuVector{T}, beta_initial::CuVector{T},
    lambda1::T, lambda2::T, max_iter::Int, tol::T
) where T
    n_obs, n_features = size(X_gpu)

    beta_current = copy(beta_initial)
    beta_previous_iter = copy(beta_initial)
    v_momentum = copy(beta_initial) # For FISTA momentum

    # Lipschitz constant of the smooth part of the objective: L = ||X'X||/N + λ2
    # opnorm(X'X) can be expensive. Estimate or use backtracking line search for step size 1/L.
    # For simplicity, use a fixed sufficiently large L or estimate.
    # L_estimate = opnorm(Array(X_gpu)' * Array(X_gpu)) / n_obs + lambda2 # Expensive
    # Heuristic for L:
    L_heuristic = T(sum(abs2, X_gpu) / (n_obs * n_features) * n_features + lambda2) # Rough estimate
    if L_heuristic < eps(T) L_heuristic = one(T) end
    step_size = one(T) / L_heuristic

    t_fista = one(T)

    for iter_count in 1:max_iter
        beta_old_val_for_conv_check = copy(beta_current)

        # Gradient of the smooth part: (1/N)X'(Xv - y) + λ2*v
        grad_smooth = (X_gpu' * (X_gpu * v_momentum - y_gpu)) ./ n_obs .+ lambda2 .* v_momentum

        # Gradient descent step on smooth part
        z_intermediate = v_momentum .- step_size .* grad_smooth

        # Proximal operator for L1 norm (soft thresholding)
        # prox_L1(z, λ) = sign(z) * max(|z| - λ, 0)
        beta_current = sign.(z_intermediate) .* max.(abs.(z_intermediate) .- (step_size * lambda1), zero(T))

        # FISTA acceleration update
        t_fista_new = (one(T) + sqrt(one(T) + T(4) * t_fista^2)) / T(2)
        v_momentum = beta_current .+ ((t_fista - one(T)) / t_fista_new) .* (beta_current .- beta_previous_iter)

        beta_previous_iter = copy(beta_old_val_for_conv_check) # Store previous beta_current
        t_fista = t_fista_new

        # Convergence check
        if norm(beta_current .- beta_old_val_for_conv_check) < tol * norm(beta_old_val_for_conv_check)
            # println("Elastic Net (FISTA) converged at iteration $iter_count.")
            break
        end
        if iter_count == max_iter
            # println("Elastic Net (FISTA) reached max iterations.")
        end
    end
    return beta_current
end


"""
    fit_group_lasso_epistasis(...) -> SparseEpistaticModel{T}

Fits a Group LASSO model for epistasis. Interactions are grouped (e.g., by SNP),
and entire groups of effects are penalized.
`groups` should be a list of vectors, where each inner vector contains indices
of interactions belonging to that group.
"""
function fit_group_lasso_epistasis(
    genotypes_gpu::CuArray{T,2},
    phenotypes_gpu::CuVector{T},
    candidate_interactions::Vector{Tuple{Int32, Int32}};
    lambda_group::T = T(0.01), # Penalty for group L2 norms
    groups_definition::Union{Nothing, Vector{Vector{Int}}} = nothing, # List of interaction indices per group
    max_iterations_gl::Int = 1000,
    tolerance_gl::T = T(1e-6)
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_gpu)
    n_candidate_interactions = length(candidate_interactions)

    if n_candidate_interactions == 0
        return SparseEpistaticModel(Tuple{Int32,Int32}[], T[], lambda_group, T(0.0), T(0.0), :group_lasso_no_candidates)
    end

    # 1. Define groups if not provided (e.g., group by first SNP in interaction)
    actual_groups = groups_definition
    if actual_groups === nothing
        actual_groups = default_snp_interaction_groups(candidate_interactions, n_snps)
    end
    if isempty(actual_groups)
        # println("Warning: No groups defined for Group LASSO. Returning empty model.")
        return SparseEpistaticModel(Tuple{Int32,Int32}[], T[], lambda_group, T(0.0), T(0.0), :group_lasso_no_groups)
    end

    # 2. Build design matrix X_interactions (as in Elastic Net)
    X_interactions_gpu = CUDA.zeros(T, n_individuals, n_candidate_interactions)
    backend = KernelAbstractions.get_backend(X_interactions_gpu)
    kernel_build_X! = build_interaction_kernel!(backend) # from gpu_kernels.jl
    kernel_build_X!(X_interactions_gpu, genotypes_gpu, CuArray(candidate_interactions),
                    n_individuals, n_candidate_interactions,
                    ndrange = n_individuals * n_candidate_interactions)
    KernelAbstractions.synchronize(backend)

    # 3. Solve Group LASSO problem
    # Often uses Block Coordinate Descent with group-wise proximal operator.
    beta_coeffs_gpu = CUDA.zeros(T, n_candidate_interactions) # Initialize coefficients

    # Placeholder for Group LASSO solver
    # This is a complex optimization problem.
    # The original code had a sketch of block coordinate descent.
    # Iteratively update β for each group:
    #   β_g = prox_GroupL2( (X_g'X_g)^-1 * X_g' * (y - X_(-g)β_(-g) ), λ_group )
    # where prox_GroupL2(v, λ) = max(0, 1 - λ/||v||₂) * v

    # Simplified iterative approach (Block Coordinate Descent sketch):
    for iter_gl in 1:max_iterations_gl
        beta_old_iter_gl = copy(beta_coeffs_gpu)

        for group_indices in actual_groups
            if isempty(group_indices) continue end

            # Calculate partial residuals: r_g = y - sum_{j ∉ group} X_j β_j
            X_not_g_indices = setdiff(1:n_candidate_interactions, group_indices)

            residual_g = phenotypes_gpu # Start with y
            if !isempty(X_not_g_indices) # If there are other groups with non-zero effects
                 X_not_g = X_interactions_gpu[:, X_not_g_indices]
                 beta_not_g = beta_coeffs_gpu[X_not_g_indices]
                 residual_g = residual_g .- (X_not_g * beta_not_g)
            end

            X_g = X_interactions_gpu[:, group_indices] # Submatrix for current group

            # Solve unpenalized least squares for this group: β_g_ls = (X_g'X_g)^-1 X_g' r_g
            # (X_g'X_g) can be small. Add ridge for stability.
            XtX_g = X_g' * X_g
            ridge_val_gl = T(1e-5) * tr(XtX_g) / length(group_indices)
            XtX_g_reg = XtX_g + CuMatrix{T}(I, length(group_indices), length(group_indices)) * ridge_val_gl

            beta_g_ls = XtX_g_reg \ (X_g' * residual_g) # Least squares estimate for this group

            # Apply group proximal operator (group soft thresholding)
            norm_beta_g_ls = norm(beta_g_ls) # L2 norm of the LS estimate vector for this group

            if norm_beta_g_ls > eps(T)
                scale_factor = max(zero(T), one(T) - (lambda_group * sqrt(T(length(group_indices))) / norm_beta_g_ls) )
                beta_coeffs_gpu[group_indices] = scale_factor .* beta_g_ls
            else
                beta_coeffs_gpu[group_indices] .= zero(T)
            end
        end

        if norm(beta_coeffs_gpu .- beta_old_iter_gl) < tolerance_gl * norm(beta_old_iter_gl)
            # println("Group LASSO converged at iteration $iter_gl.")
            break
        end
         if iter_gl == max_iterations_gl
            # println("Group LASSO reached max iterations.")
        end
    end

    # 4. Extract results (similar to Elastic Net)
    final_beta_coeffs_host_gl = Array(beta_coeffs_gpu)
    non_zero_indices_gl = findall(x -> abs(x) > T(1e-8), final_beta_coeffs_host_gl)

    selected_model_interactions_gl = candidate_interactions[non_zero_indices_gl]
    selected_model_coefficients_gl = final_beta_coeffs_host_gl[non_zero_indices_gl]

    achieved_sparsity_gl = n_candidate_interactions > 0 ? T(length(non_zero_indices_gl) / n_candidate_interactions) : T(0.0)

    return SparseEpistaticModel(
        selected_model_interactions_gl,
        selected_model_coefficients_gl,
        lambda_group, # L1 penalty effectively (though applied to group norm)
        T(0.0),       # No separate L2 penalty in basic Group LASSO formulation here
        achieved_sparsity_gl,
        :group_lasso
    )
end

"""
Helper to define default SNP interaction groups for Group LASSO.
Groups interactions by the first SNP in each pair.
"""
function default_snp_interaction_groups(
    candidate_interactions::Vector{Tuple{Int32,Int32}},
    n_total_snps::Int # Max SNP index, for creating group map
)
    groups_map = Dict{Int32, Vector{Int}}() # SNP_index -> list of interaction_indices
    for (interaction_idx, (snp1, _)) in enumerate(candidate_interactions) # Group by snp1
        if !haskey(groups_map, snp1)
            groups_map[snp1] = Int[]
        end
        push!(groups_map[snp1], interaction_idx)
    end
    return collect(values(groups_map)) # Return list of lists of indices
end


"""
    adaptive_sparse_epistasis_learning(...) -> SparseEpistaticModel{T}

Implements an adaptive learning strategy, potentially multi-stage,
to refine the set of selected epistatic interactions and their effects.
E.g., iterative screening and penalized regression with updated weights.
"""
function adaptive_sparse_epistasis_learning(
    genotypes_gpu::CuArray{T,2},
    phenotypes_gpu::CuVector{T};
    initial_screening_size::Int = 50000, # Number of interactions after first broad screen
    final_model_target_size::Int = 1000, # Desired number of interactions in final model
    num_refinement_stages::Int = 3,
    verbose::Bool = false
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_gpu)
    current_candidate_interactions = Tuple{Int32,Int32}[]

    # Stage 1: Broad initial screening (e.g. using fast correlation or WHT-based scores)
    if verbose println("Adaptive Stage 1: Initial broad screening...") end
    # `fast_correlation_screening_interactions` is a placeholder for a fast screening method.
    # It should return a list of interaction tuples.
    current_candidate_interactions, _ = fast_correlation_screening_interactions(
        genotypes_gpu, phenotypes_gpu, initial_screening_size
    )
    if verbose println("  Found $(length(current_candidate_interactions)) candidates after initial screen.") end

    if isempty(current_candidate_interactions)
        return SparseEpistaticModel(Tuple{Int32,Int32}[], T[], T(0), T(0), T(0), :adaptive_learning_no_candidates)
    end

    # Multi-stage refinement using weighted penalized regression (e.g., Adaptive LASSO idea)
    # Weights are typically inversely proportional to initial effect estimates.
    # λ1, λ2 parameters for Elastic Net in refinement stages.
    refinement_lambda1 = T(0.005)
    refinement_lambda2 = T(0.0005)

    for stage_num in 1:num_refinement_stages
        if verbose println("Adaptive Stage $(stage_num+1) (Refinement): Refining $(length(current_candidate_interactions)) candidates...") end
        if isempty(current_candidate_interactions) break end

        # Fit Elastic Net on current candidates to get effect estimates for weighting
        # (or use weights from previous stage if iterative weighting)
        temp_model = fit_elastic_net_epistasis(
            genotypes_gpu, phenotypes_gpu, current_candidate_interactions,
            lambda1=refinement_lambda1, lambda2=refinement_lambda2,
            max_iterations_en=500 # Fewer iterations for intermediate stages
        )

        if isempty(temp_model.interactions)
            current_candidate_interactions = Tuple{Int32,Int32}[] # No interactions left
            break
        end

        # Adaptive step: Could re-weight, re-select, or adjust penalties.
        # For simplicity here, let's assume refinement means selecting a subset from temp_model.interactions
        # or re-running with different parameters if it was more complex.
        # The original `compute_importance_weights` and selection logic was more involved.
        # Here, just take the non-zero effects from this stage as candidates for next.
        current_candidate_interactions = temp_model.interactions

        # If we need to reduce to a target size:
        if length(current_candidate_interactions) > final_model_target_size && stage_num < num_refinement_stages
            # Example: reduce candidates by taking top N by effect size
            coeffs = abs.(temp_model.coefficients)
            sorted_idx_stage = sortperm(coeffs, rev=true)
            num_to_keep_stage = max(final_model_target_size, length(coeffs) ÷ 2) # Example reduction
            current_candidate_interactions = current_candidate_interactions[sorted_idx_stage[1:min(num_to_keep_stage, end)]]
        end
        if verbose println("  $(length(current_candidate_interactions)) candidates remaining after stage $(stage_num+1).") end
    end

    # Final model fitting on the refined set of interactions
    if verbose println("Adaptive Final Stage: Fitting final model on $(length(current_candidate_interactions)) interactions...") end
    if isempty(current_candidate_interactions)
        return SparseEpistaticModel(Tuple{Int32,Int32}[], T[], refinement_lambda1, refinement_lambda2, T(0), :adaptive_learning_empty_final_set)
    end

    final_model = fit_elastic_net_epistasis(
        genotypes_gpu, phenotypes_gpu, current_candidate_interactions,
        lambda1 = refinement_lambda1 / T(2), # Potentially less aggressive penalty for final model
        lambda2 = refinement_lambda2 / T(2),
        max_iterations_en = 2000 # More iterations for final model
    )

    return final_model # This is a SparseEpistaticModel struct
end


"""
Placeholder for a fast correlation screening method for interactions.
Returns top_k interactions based on correlation of interaction_term with phenotype.
This is a simplified version of what `fast_correlation_screening` in original `utils.jl` (for demo) did.
"""
function fast_correlation_screening_interactions(
    genotypes_gpu::CuArray{T,2},
    phenotypes_gpu::CuVector{T},
    top_k_interactions::Int
) where T
    n_individuals, n_snps = size(genotypes_gpu)
    if n_snps < 2 return Tuple{Int32,Int32}[], T[] end

    # This is computationally very expensive (O(N*M^2)) if done naively for all pairs.
    # Practical methods use approximations, random sampling of pairs, or multi-step approaches.
    # For this placeholder, simulate selecting some pairs.
    # A real implementation would involve optimized kernels.

    num_pairs_to_sample = min(top_k_interactions * 10, n_snps * (n_snps-1) ÷ 2, 200000) # Sample some pairs
    if num_pairs_to_sample <=0 return Tuple{Int32,Int32}[], T[] end

    sampled_interactions = Vector{Tuple{Int32,Int32}}(undef, num_pairs_to_sample)
    sampled_scores = Vector{T}(undef, num_pairs_to_sample)

    # Randomly sample pairs (not ideal, but a placeholder for a faster screen)
    # Better: screen main effects, then test interactions among top main effects.
    idx_checked = 0
    temp_pair_set = Set{Tuple{Int32,Int32}}()
    while idx_checked < num_pairs_to_sample && length(temp_pair_set) < num_pairs_to_sample
        snp1 = rand(1:n_snps)
        snp2 = rand(1:n_snps)
        if snp1 == snp2 continue end
        pair = (min(snp1,snp2), max(snp1,snp2))
        if pair in temp_pair_set continue end

        push!(temp_pair_set, pair)
        idx_checked += 1
        sampled_interactions[idx_checked] = pair

        # Compute interaction term and score (correlation)
        interaction_term_gpu = genotypes_gpu[:,pair[1]] .* genotypes_gpu[:,pair[2]]
        # Simple correlation (ensure no NaN if variance is zero)
        if var(Array(interaction_term_gpu)) > eps(T) && var(Array(phenotypes_gpu)) > eps(T)
            sampled_scores[idx_checked] = abs(cor(Array(interaction_term_gpu), Array(phenotypes_gpu)))
        else
            sampled_scores[idx_checked] = zero(T)
        end
    end
    # Resize if fewer unique pairs were found
    if idx_checked < num_pairs_to_sample
        resize!(sampled_interactions, idx_checked)
        resize!(sampled_scores, idx_checked)
    end

    # Select top_k from these sampled scores
    if isempty(sampled_scores) return Tuple{Int32,Int32}[], T[] end

    num_to_actually_select = min(top_k_interactions, length(sampled_scores))
    top_indices_sampled = partialsortperm(sampled_scores, 1:num_to_actually_select, rev=true)

    return sampled_interactions[top_indices_sampled], sampled_scores[top_indices_sampled]
end


# Helper functions like `build_interaction_matrix` (kernel in gpu_kernels.jl),
# `soft_threshold` (part of accelerated_proximal_gradient_en),
# `compute_importance_weights`, `define_snp_groups`,
# `gaussian_kernel`, `compute_hsic`, `discretize`, `mutual_information`, `randomized_svd`
# were part of the original structure. Some are complex and may need their own files or be part of a utils submodule.
# For now, key solver logic like `accelerated_proximal_gradient_en` is sketched within this module.

# Kernels related to these (e.g. `build_interaction_kernel!`) are in `gpu_kernels.jl`.

end # module SparseEpistasis
