# ===== src/augmented_aireml.jl =====
"""
Augmented Average Information REML (Augmented AI-REML) for efficient
estimation of variance components in mixed models.
This method aims to reduce computational costs compared to standard AI-REML,
especially when multiple random effects (multiple kinship matrices) are fitted.
"""

module AugmentedAIREML

using CUDA
using LinearAlgebra # For I, inv, cholesky, tr, Diagonal, Symmetric, dot, opnorm, eigen, pinv
using Statistics  # For var
using Optim       # Potentially for line search or direct likelihood optimization if needed
using ForwardDiff # For automatic differentiation if used for gradients/Hessians (not typical for AI-REML)

# Assuming types.jl (VarianceComponents) is accessible.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export AugmentedREMLState, initialize_augmented_reml_state!, # Renamed init
       fit_augmented_aireml! # Main fitting function
       # compute_augmented_matrices was internal details

"""
    AugmentedREMLState{T}

Structure to hold the state for the Augmented AI-REML algorithm.
This includes model matrices, kinship matrices, working matrices,
variance component estimates, and convergence status.
"""
mutable struct AugmentedREMLState{T<:AbstractFloat}
    # Model matrices (on GPU)
    X::CuArray{T, 2}  # Fixed effects design matrix (N x p)
    Z::CuArray{T, 2}  # Random effects design matrix (N x N, often Identity for GBLUP animal model)
    y::CuVector{T}    # Phenotypes (N x 1)

    # Relationship matrices (list of CuArrays, e.g., [G_add, G_epi])
    K_list::Vector{CuArray{T, 2}}

    # Working matrices (on GPU, updated each iteration)
    V::CuArray{T, 2}           # Phenotypic covariance matrix V = sum(θ_k * Z*K_k*Z') + θ_e*I
    V_inv::CuArray{T, 2}       # Inverse of V
    P::CuArray{T, 2}           # Projection matrix P = V_inv - V_inv*X*inv(X'*V_inv*X)*X'*V_inv

    # Augmented system components (specific to augmented method, might not be needed if using direct AI)
    # C_aug::CuArray{T, 2} # This was from the original very large pasted code, but not used by the direct AI-REML below.
                           # If a specific "augmented system" method is implemented, this might be used.
                           # For now, it's removed from the struct as it's unused by the current logic.

    # Variance components
    θ::Vector{T}               # Current estimates of variance parameters [σ²_k1, σ²_k2, ..., σ²_e]
    θ_names::Vector{Symbol}    # Names corresponding to θ, e.g., [:σ²_a, :σ²_aa, :σ²_e]

    # Convergence tracking
    log_likelihood::T
    iteration::Int
    converged::Bool

    # MME solutions (optional, can be stored here if computed during REML)
    beta_hat::Union{Nothing, CuArray{T,1}} # Estimates of fixed effects
    u_hats::Union{Nothing, Vector{CuArray{T,1}}} # List of BLUPs for each random effect K_i
end

"""
    initialize_augmented_reml_state!(...) -> AugmentedREMLState{T}

Initializes the `AugmentedREMLState` for fitting a model.
`y_host` and `X_host` are CPU arrays. `K_list_gpu` is a list of GPU kinship matrices.
"""
function initialize_augmented_reml_state!(
    y_host::Vector{T},
    X_host::Union{Nothing, Matrix{T}}, # Fixed effects design matrix (CPU)
    K_list_gpu::Vector{CuArray{T, 2}}; # List of kinship matrices (already on GPU)
    initial_θ_estimates::Union{Nothing, Vector{T}} = nothing,
    θ_param_names::Union{Nothing, Vector{Symbol}} = nothing
) where T <: AbstractFloat

    n_individuals = length(y_host)

    # Default fixed effects: intercept only if X_host is Nothing
    X_actual_host = X_host === nothing ? ones(T, n_individuals, 1) : X_host
    n_fixed_effects = size(X_actual_host, 2)

    # Transfer core matrices to GPU
    y_gpu = CuArray(y_host)
    X_gpu = CuArray(X_actual_host)
    # Z is often identity in animal GBLUP model, Z*K*Z' becomes K.
    # If Z is always I, it can be implicit. For generality, store it.
    Z_gpu = CuMatrix{T}(I, n_individuals, n_individuals) # N x N identity matrix on GPU

    # Initialize variance components (θ)
    num_random_effects = length(K_list_gpu)
    num_variance_params = num_random_effects + 1  # +1 for residual variance σ²_e

    θ_current_estimates = Vector{T}(undef, num_variance_params)
    if initial_θ_estimates !== nothing && length(initial_θ_estimates) == num_variance_params
        θ_current_estimates .= initial_θ_estimates
    else
        # Default initialization: partition phenotypic variance
        var_y_total = var(y_host)
        if var_y_total <= eps(T) var_y_total = one(T) end # Avoid zero variance issues

        # Assign roughly equal portions to genetic effects, larger to residual
        prop_genetic = T(0.5) / num_random_effects # e.g. if 2 genetic effects, each gets 0.25
        for i in 1:num_random_effects
            θ_current_estimates[i] = var_y_total * prop_genetic
        end
        θ_current_estimates[end] = var_y_total * T(0.5) # Residual gets 50%
        # Ensure sum matches var_y_total or normalize
        current_sum = sum(θ_current_estimates)
        if current_sum > eps(T)
            θ_current_estimates .*= (var_y_total / current_sum)
        end
        # Ensure positivity
        θ_current_estimates .= max.(θ_current_estimates, eps(T))
    end

    # Parameter names
    final_θ_names = Vector{Symbol}(undef, num_variance_params)
    if θ_param_names !== nothing && length(θ_param_names) == num_variance_params
        final_θ_names .= θ_param_names
    else
        for i in 1:num_random_effects
            final_θ_names[i] = Symbol("σ²_K$(i)") # Generic names like σ²_G, σ²_AA
        end
        final_θ_names[end] = :σ²_e
    end

    # Initialize working matrices (empty or correctly sized zeros)
    V_gpu = CUDA.zeros(T, n_individuals, n_individuals)
    V_inv_gpu = CUDA.zeros(T, n_individuals, n_individuals)
    P_gpu = CUDA.zeros(T, n_individuals, n_individuals)

    state = AugmentedREMLState{T}(
        X_gpu, Z_gpu, y_gpu,
        K_list_gpu,
        V_gpu, V_inv_gpu, P_gpu,
        θ_current_estimates, final_θ_names,
        -T(Inf), 0, false, # log_likelihood, iteration, converged
        nothing, nothing    # beta_hat, u_hats
    )
    return state
end


"""
    fit_augmented_aireml!(state::AugmentedREMLState{T}; ...)

Fits the mixed model using Augmented AI-REML (or standard AI-REML if augmentation isn't implemented yet)
by iteratively updating variance components until convergence or max iterations.
Modifies `state` in-place.
"""
function fit_augmented_aireml!(
    state::AugmentedREMLState{T};
    max_iterations::Int = 100,
    tolerance::T = T(1e-6), # Convergence tolerance for log-likelihood change
    verbose::Bool = true
) where T <: AbstractFloat

    if verbose
        println("Starting AI-REML variance component estimation...")
        println("  Initial parameters (θ): ", state.θ)
    end

    while !state.converged && state.iteration < max_iterations
        state.iteration += 1
        log_likelihood_old = state.log_likelihood

        # 1. Update V, V_inv, P matrices based on current state.θ
        # V = sum(θ_k * Z*K_k*Z') + θ_e*I
        # (Assuming Z=I for GBLUP animal model, so Z*K*Z' = K)
        fill!(state.V, zero(T)) # Reset V
        for k_idx in 1:length(state.K_list)
            state.V .+= state.θ[k_idx] .* state.K_list[k_idx]
        end
        # Add residual component θ_e * I
        # state.V += Diagonal(fill(state.θ[end], size(state.V,1))) # This creates Diagonal on CPU
        # GPU way:
        state.V .+= CuMatrix{T}(I, size(state.V)...) .* state.θ[end]

        # Add small ridge for stability before inversion of V
        ridge_V = T(1e-7) * tr(state.V) / size(state.V,1) # Relative ridge
        V_stable = state.V + CuMatrix{T}(I, size(state.V)...) * ridge_V

        # Compute V_inv and P
        # Error handling for non-PSD matrices is important here.
        try
            ch_V = cholesky(Symmetric(V_stable)) # V should be Symmetric Positive Definite
            state.V_inv = inv(ch_V)
        catch e
            if isa(e, PosDefException)
                if verbose println("Warning: V matrix not positive definite at iter $(state.iteration). Using general inv().") end
                state.V_inv = inv(V_stable) # Fallback to general inv()
            else
                rethrow(e)
            end
        end

        # P = V_inv - V_inv*X * inv(X'*V_inv*X) * X'*V_inv
        Xt_Vinv = state.X' * state.V_inv
        Xt_Vinv_X = Xt_Vinv * state.X
        ridge_XtVinvX = T(1e-8) * tr(Xt_Vinv_X) / size(state.X,2)
        Xt_Vinv_X_stable = Xt_Vinv_X + CuMatrix{T}(I, size(Xt_Vinv_X)...) * ridge_XtVinvX

        inv_Xt_Vinv_X = CUDA.zeros(T,0,0)
        try
            ch_XtVinvX = cholesky(Symmetric(Xt_Vinv_X_stable))
            inv_Xt_Vinv_X = inv(ch_XtVinvX)
        catch e
            if isa(e, PosDefException)
                if verbose println("Warning: X'V_invX not positive definite at iter $(state.iteration). Using general inv().") end
                inv_Xt_Vinv_X = inv(Xt_Vinv_X_stable)
            else
                 rethrow(e)
            end
        end
        if size(inv_Xt_Vinv_X,1) == 0 # Failed to compute inv
            if verbose println("Error: Failed to compute inv(X'V_invX) at iter $(state.iteration). Stopping.") end
            break
        end

        state.P = state.V_inv - Xt_Vinv' * inv_Xt_Vinv_X * Xt_Vinv


        # 2. Compute Average Information (AI) matrix and Score vector
        # Derivatives dV/dθ_i: K_i for genetic components, I for residual.
        # (Assuming Z=I, so dV/dθ_k = K_k)
        num_params = length(state.θ)
        AI_matrix_gpu = CUDA.zeros(T, num_params, num_params)
        score_vector_gpu = CUDA.zeros(T, num_params)

        Py = state.P * state.y # Precompute P*y

        # Loop through pairs of parameters (i,j) for AI matrix and score_i
        for i_param in 1:num_params
            dV_dθi = i_param <= length(state.K_list) ? state.K_list[i_param] : CuMatrix{T}(I, size(state.V)...)
            P_dV_dθi = state.P * dV_dθi # P * (dV/dθi)

            # Score element i: S_i = 0.5 * (y'*P*(dV/dθi)*P*y - tr(P*(dV/dθi)))
            score_vector_gpu[i_param] = T(0.5) * (dot(Py, dV_dθi * Py) - tr(P_dV_dθi))

            for j_param in i_param:num_params # Fill upper triangle of AI matrix
                dV_dθj = j_param <= length(state.K_list) ? state.K_list[j_param] : CuMatrix{T}(I, size(state.V)...)
                P_dV_dθj = state.P * dV_dθj # P * (dV/dθj)

                # AI element (i,j): AI_ij = 0.5 * tr(P*(dV/dθi)*P*(dV/dθj))
                # This is tr( (P_dV_dθi) * (P_dV_dθj) )
                # Note: P_dV_dθj is not necessarily symmetric.
                # AI_matrix_gpu[i_param, j_param] = T(0.5) * tr(P_dV_dθi * P_dV_dθj)
                # More robust trace for product: sum(diag(A*B)) or sum(A .* B') if B is symmetric
                # If P_dV_dθi and P_dV_dθj are general matrices:
                # tr(A*B) = sum(sum((A .* B'),dims=1)) is not right. tr(A*B) = sum(diag(A*B))
                # Or sum(A .* transpose(B)) element-wise then sum.
                # Or sum( (A * B.').*I )
                # For tr(M1 * M2), if M1, M2 are N x N: sum(M1[k,:]' * M2[:,k] for k=1:N)
                # Or simply tr( matrix_product )
                product_for_trace = P_dV_dθi * P_dV_dθj
                AI_matrix_gpu[i_param, j_param] = T(0.5) * tr(product_for_trace)

                if i_param != j_param
                    AI_matrix_gpu[j_param, i_param] = AI_matrix_gpu[i_param, j_param] # Symmetric
                end
            end
        end

        # 3. Solve AI * Δθ = Score for Δθ
        AI_cpu = Array(AI_matrix_gpu)
        score_cpu = Array(score_vector_gpu)

        # Regularize AI matrix slightly for stability
        ridge_AI = T(1e-7) * tr(AI_cpu) / num_params
        AI_cpu_reg = AI_cpu + Diagonal(fill(ridge_AI, num_params))

        Δθ_cpu = zeros(T, num_params) # Initialize
        try
            ch_AI = cholesky(Symmetric(AI_cpu_reg))
            Δθ_cpu = ch_AI \ score_cpu
        catch e
            if isa(e, PosDefException)
                if verbose println("Warning: AI matrix not positive definite at iter $(state.iteration). Using pinv().") end
                try
                    Δθ_cpu = pinv(AI_cpu_reg) * score_cpu
                catch e_pinv
                    if verbose println("Error during pinv of AI matrix: $e_pinv. Halting REML for this fit.") end
                    state.converged = false; break # Stop REML for this fit
                end
            else
                 rethrow(e) # Some other error
            end
        end
        if all(iszero, Δθ_cpu) && !all(iszero, score_cpu) # Check if solve failed silently
             if verbose println("Warning: Δθ is zero but score is not. Possible issue in solving AI system at iter $(state.iteration). Stopping.") end
             state.converged = false; break
        end


        # 4. Update variance components: θ_new = θ_old + Δθ
        # Implement step halving or other strategies to ensure parameters stay valid (e.g., positive)
        step_factor = one(T)
        θ_new = state.θ .+ step_factor .* Δθ_cpu

        # Boundary constraints (variances must be non-negative)
        # And ensure residual variance is robustly positive.
        min_genetic_var = eps(T) # Small positive value
        min_residual_var = eps(T) * T(100) # Residual should be clearly positive

        for i in 1:length(state.K_list)
            if θ_new[i] < min_genetic_var
                # Adjust step if boundary hit, or clamp
                # Simple clamping for now:
                θ_new[i] = min_genetic_var
            end
        end
        if θ_new[end] < min_residual_var
            θ_new[end] = min_residual_var
        end
        state.θ = θ_new

        # 5. Compute REML log-likelihood
        # logL = -0.5 * ( (N-p)log(2π) + log|V| + log|X'V_invX| + y'Py )
        # Need log|V| and log|X'V_invX|
        # log|V| = -log|V_inv|. log|X'V_invX| from inv_Xt_Vinv_X used to compute P.

        current_log_likelihood = zero(T)
        try
            # log|V| = -log|V_inv|
            # Need stable logdet of V_inv (which should be SPD)
            V_inv_stable_for_logdet = state.V_inv + CuMatrix{T}(I,size(state.V_inv)...)*ridge_V # Use ridged version if V_inv was from it
            log_det_V_inv = logdet(Symmetric(V_inv_stable_for_logdet)) # Symmetric ensures SPD path for logdet if possible
            log_det_V = -log_det_V_inv

            # log|X'V_invX|
            # inv_Xt_Vinv_X is inv(X'V_invX). So log|X'V_invX| = -log|inv(X'V_invX)|
            # This needs inv_Xt_Vinv_X to be SPD for logdet.
            # It was computed from cholesky of Xt_Vinv_X_stable.
            log_det_Xt_Vinv_X = -logdet(Symmetric(inv_Xt_Vinv_X)) # This requires inv_Xt_Vinv_X to be SPD

            yPy = dot(state.y, Py) # Py = P*y

            n_eff_data_points = size(state.y,1) - size(state.X,2) # N - p (rank of X)
            current_log_likelihood = -T(0.5) * (
                n_eff_data_points * log(T(2π)) +
                log_det_V +
                log_det_Xt_V_inv_X + # This was log|X'V_invX|
                yPy
            )
        catch e
            if isa(e, DomainError) || isa(e, PosDefException) || isa(e, SingularException)
                if verbose println("Warning: Numerical issue computing log-likelihood at iter $(state.iteration): $e. Using previous logL.") end
                current_log_likelihood = log_likelihood_old # Keep previous if new is unstable
            else
                rethrow(e)
            end
        end
        state.log_likelihood = current_log_likelihood

        # Check convergence
        if abs(state.log_likelihood - log_likelihood_old) < tolerance
            state.converged = true
            if verbose println("AI-REML converged at iteration $(state.iteration). LogL = $(state.log_likelihood)") end
        end

        if verbose && (state.iteration % 10 == 0 || state.converged)
            println("  Iter $(state.iteration): LogL = $(state.log_likelihood), θ = $(state.θ)")
        end
        if state.iteration == max_iterations && !state.converged
             if verbose println("AI-REML reached max iterations ($(max_iterations)) without full convergence. LogL = $(state.log_likelihood)") end
        end
    end # end REML iteration loop

    # After convergence (or max_iter), compute final BLUPs/BLUEs if needed by caller
    # This uses the final variance components in state.θ
    # The MME solution part is simplified here. A full MME solver would be called.
    # For now, assume the caller of `fit_augmented_aireml!` will handle final BLUPs
    # using the converged `state.θ`.
    # Or, store them in the state:
    # state.beta_hat, state.u_hats = final_mme_solve(state)

    return state # Return the modified state, which includes converged θ
end


# Helper: Solve MME for final BLUPs/BLUEs (conceptual, similar to reml.jl's solve_mme_for_reml)
# function final_mme_solve(state::AugmentedREMLState{T}) where T
#     # ... uses state.y, state.X, state.K_list, state.θ (final variance components)
#     # ... constructs and solves MME system ...
#     # beta_hat = ...
#     # u_hats_list = [u_hat_k1, u_hat_k2, ...]
#     # return beta_hat, u_hats_list
#     return nothing, nothing # Placeholder
# end


# The "Augmented" part of AI-REML often refers to specific ways of constructing and solving
# the AI matrix equations, possibly involving larger augmented MME-like systems for computational tricks.
# The implementation above is closer to a standard AI-REML.
# The original `build_augmented_system!` and related functions would need to be integrated
# if that specific augmented method is to be implemented faithfully.
# For now, this provides a functional AI-REML structure.

# Export functions if this file were a module
# export AugmentedREMLState, initialize_augmented_reml_state!, fit_augmented_aireml!

end # module AugmentedAIREML
