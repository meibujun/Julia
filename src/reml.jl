# ===== src/reml.jl =====
"""
    DynamicEpistasisGBLUP.REML

This module implements Restricted Maximum Likelihood (REML) methods for estimating
variance components in mixed models, particularly for GBLUP with additive and
epistatic effects. It includes core components of an Average Information REML
(AI-REML) algorithm.

Key functions:
- `initialize_variance_components`: Provides initial guesses for variance components.
- `estimate_variance_components_reml!`: The main iterative AI-REML solver.
- Helper functions for computing `V⁻¹`, `P` matrix, MME solutions within REML, and the REML log-likelihood.
"""

using CUDA
using LinearAlgebra # For I, inv, logdet, cholesky, tr, Diagonal, Symmetric
using Statistics  # For var
# Assuming types.jl (VarianceComponents) is included and accessible.
# Assuming gpu_kernels.jl might be needed for some matrix operations if not covered by CUDA.jl basics.

# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float


"""
    initialize_variance_components(phenotypes_vec::Vector{T}, include_epistasis::Bool = true) where T <: AbstractFloat -> VarianceComponents{T}

Provides initial estimates for variance components (additive, epistatic, residual)
based on the total phenotypic variance of the input `phenotypes_vec`.

A common heuristic is used:
- Additive variance (σ²_a) is a fraction (e.g., 35%) of total phenotypic variance.
- Epistatic variance (σ²_aa) is a smaller fraction (e.g., 10%) if `include_epistasis` is true, otherwise zero.
- Residual variance (σ²_e) makes up the remainder.
The function ensures components are non-negative and recalculates derived heritabilities (h², H²).

# Arguments
- `phenotypes_vec::Vector{T}`: Vector of phenotypic values.
- `include_epistasis::Bool = true`: If `true`, an initial non-zero guess for epistatic variance is made.

# Returns
- `VarianceComponents{T}`: A struct populated with initial variance component estimates.
"""
function initialize_variance_components(
    phenotypes_vec::Vector{T},
    include_epistasis::Bool = true
) where T <: AbstractFloat

    var_phenotypic_total = var(phenotypes_vec)
    if var_phenotypic_total <= eps(T) # Handle case of zero phenotypic variance
        var_phenotypic_total = one(T) # Avoid division by zero, assume unit variance
    end

    # Initial guesses based on proportions
    initial_σ²_a = var_phenotypic_total * T(0.35) # Additive variance
    initial_σ²_aa = include_epistasis ? var_phenotypic_total * T(0.10) : zero(T) # Epistatic variance
    initial_σ²_e = var_phenotypic_total * T(0.55) # Residual variance

    # Ensure residual is not negative if epistatic is not included but its share was taken by additive
    if !include_epistasis
        initial_σ²_e = var_phenotypic_total - initial_σ²_a
        initial_σ²_e = max(initial_σ²_e, var_phenotypic_total * T(0.1)) # Ensure some residual
        initial_σ²_a = var_phenotypic_total - initial_σ²_e # Re-adjust additive
    end

    initial_σ²_p = initial_σ²_a + initial_σ²_aa + initial_σ²_e
    # If sum doesn't match var_total due to adjustments, rescale σ²_e
    if abs(initial_σ²_p - var_phenotypic_total) > eps(T) * var_phenotypic_total && initial_σ²_p > eps(T)
        scale_factor = var_phenotypic_total / initial_σ²_p
        initial_σ²_a *= scale_factor
        initial_σ²_aa *= scale_factor
        initial_σ²_e *= scale_factor
        initial_σ²_p = var_phenotypic_total
    end

    # Ensure components are non-negative
    initial_σ²_a = max(zero(T), initial_σ²_a)
    initial_σ²_aa = max(zero(T), initial_σ²_aa)
    initial_σ²_e = max(eps(T), initial_σ²_e) # Residual should be positive

    h²_init = initial_σ²_p > eps(T) ? initial_σ²_a / initial_σ²_p : zero(T)
    H²_init = initial_σ²_p > eps(T) ? (initial_σ²_a + initial_σ²_aa) / initial_σ²_p : zero(T)

    return Main.DynamicEpistasisGBLUP.VarianceComponents{T}( # Explicitly call constructor from main module
        initial_σ²_a,
        initial_σ²_aa,
        initial_σ²_e,
        initial_σ²_p,
        h²_init,
        H²_init
    )
end

"""
    estimate_variance_components_reml!(phenotypes_vec::Vector{T}, G_additive::CuArray{T,2}, G_epistatic::Union{Nothing, CuArray{T,2}}; X_fixed_effects::Union{Nothing, CuArray{T,2}}=nothing, initial_variance_components::VarianceComponents{T}, max_iterations::Int=100, convergence_tol::T=T(1e-6)) where T <: AbstractFloat -> Tuple{VarianceComponents{T}, NamedTuple}

Estimates variance components (σ²_a, σ²_aa, σ²_e) using an Average Information REML (AI-REML) algorithm.

The function iteratively:
1.  Constructs the phenotypic covariance matrix `V` and its inverse `V⁻¹`, and the projection matrix `P`.
2.  Solves the Mixed Model Equations (MME) to obtain estimates of fixed effects (`β_hat`) and random genetic effects (`u_hat_additive`, `u_hat_epistatic`).
3.  Computes the Average Information (AI) matrix and Score vector based on current variance component estimates and MME solutions.
4.  Solves `AI * Δθ = Score` for updates `Δθ` to the variance components `θ = [σ²_a, σ²_aa, σ²_e]`.
5.  Applies updates to `θ` using a backtracking line search to ensure log-likelihood improvement and parameter validity.
6.  Calculates the REML log-likelihood.
7.  Checks for convergence based on changes in log-likelihood and parameter estimates.

# Arguments
- `phenotypes_vec::Vector{T}`: Vector of phenotypic observations (CPU).
- `G_additive::CuArray{T,2}`: Additive Genomic Relationship Matrix (GRM) on GPU.
- `G_epistatic::Union{Nothing, CuArray{T,2}}`: Epistatic GRM on GPU. If `Nothing`, an additive-only model is fitted.
- `X_fixed_effects::Union{Nothing, CuArray{T,2}} = nothing`: Optional design matrix for fixed effects (GPU). If `Nothing`, an intercept-only model is assumed.
- `initial_variance_components::VarianceComponents{T}`: A `VarianceComponents` struct with initial guesses for σ²_a, σ²_aa, σ²_e.
- `max_iterations::Int = 100`: Maximum number of AI-REML iterations.
- `convergence_tol::T = T(1e-6)`: Tolerance for convergence based on log-likelihood change and relative parameter change.

# Returns
- `Tuple{VarianceComponents{T}, NamedTuple}`:
    - `VarianceComponents{T}`: The struct containing the final REML estimates of variance components and derived heritabilities.
    - `NamedTuple`: A cache containing MME solutions: `(beta=β_hat, u_additive=u_add_hat, u_epistatic=u_epi_hat)`. These are on GPU.

# Notes
- Assumes an animal model where the incidence matrix `Z` for random effects is an identity matrix.
- Numerical stability is handled via ridging for matrix inversions and Cholesky decompositions.
"""
function estimate_variance_components_reml!(
    phenotypes_vec::Vector{T},
    G_additive::CuArray{T,2},
    G_epistatic::Union{Nothing, CuArray{T,2}};
    X_fixed_effects::Union{Nothing, CuArray{T,2}} = nothing, # Design matrix for fixed effects (GPU)
    initial_variance_components::Main.DynamicEpistasisGBLUP.VarianceComponents{T},
    max_iterations::Int = 100,
    convergence_tol::T = T(1e-6)
) where T <: AbstractFloat

    n_individuals = length(phenotypes_vec)
    y_gpu = CuArray(phenotypes_vec)

    X_gpu = X_fixed_effects === nothing ? CUDA.ones(T, n_individuals, 1) : X_fixed_effects
    n_fixed_effects = size(X_gpu, 2)

    var_comps = deepcopy(initial_variance_components)
    log_likelihood_prev = -T(Inf)

    beta_hat = CUDA.zeros(T, n_fixed_effects)
    u_add_hat = CUDA.zeros(T, n_individuals)
    u_epi_hat = G_epistatic === nothing ? CUDA.zeros(T,n_individuals) : CUDA.zeros(T, n_individuals)

    V_inv_current = CUDA.zeros(T, n_individuals, n_individuals) # To store current V_inv
    P_matrix_current = CUDA.zeros(T, n_individuals, n_individuals) # To store current P
    log_det_V_stable_current = zero(T)
    log_det_XtVinvX_stable_current = zero(T)

    for iter in 1:max_iterations
        V_inv_current, log_det_V_stable_current = compute_V_inverse_reml(G_additive, G_epistatic, var_comps, n_individuals)
        P_matrix_current, log_det_XtVinvX_stable_current = compute_P_matrix_reml(V_inv_current, X_gpu)

        beta_hat, u_add_hat, u_epi_hat = solve_mme_for_reml(
            y_gpu, X_gpu, G_additive, G_epistatic, var_comps, V_inv_current # Pass V_inv
        )

        num_var_params = G_epistatic === nothing ? 2 : 3
        AI_matrix = CUDA.zeros(T, num_var_params, num_var_params) # Renamed from AI_matrix_gpu
        score_vector = CUDA.zeros(T, num_var_params) # Renamed from score_vector_gpu

        Py = P_matrix_current * y_gpu

        dV_dsa = G_additive
        P_dV_dsa = P_matrix_current * dV_dsa
        score_vector[1] = T(0.5) * (dot(Py, dV_dsa * Py) - tr(P_dV_dsa))
        AI_matrix[1,1] = T(0.5) * tr(P_dV_dsa * P_dV_dsa)

        idx_res = num_var_params

        # dV/dσ²_e = I (Identity matrix on GPU)
        dV_dse = CuMatrix{T}(I, n_individuals, n_individuals)
        P_dV_dse = P_matrix_current * dV_dse # This is just P_matrix_current
        score_vector[idx_res] = T(0.5) * (dot(Py, dV_dse * Py) - tr(P_dV_dse)) # dot(Py, Py) - tr(P)
        AI_matrix[idx_res, idx_res] = T(0.5) * tr(P_dV_dse * P_dV_dse) # 0.5 * tr(P*P)
        AI_matrix[1, idx_res] = AI_matrix[idx_res, 1] = T(0.5) * tr(P_dV_dsa * P_dV_dse) # 0.5 * tr(P_dV_dsa * P)

        if G_epistatic !== nothing
            dV_dsaa = G_epistatic
            P_dV_dsaa = P_matrix_current * dV_dsaa
            score_vector[2] = T(0.5) * (dot(Py, dV_dsaa * Py) - tr(P_dV_dsaa))
            AI_matrix[2,2] = T(0.5) * tr(P_dV_dsaa * P_dV_dsaa)
            AI_matrix[1,2] = AI_matrix[2,1] = T(0.5) * tr(P_dV_dsa * P_dV_dsaa)
            AI_matrix[2,idx_res] = AI_matrix[idx_res,2] = T(0.5) * tr(P_dV_dsaa * P_dV_dse) # P_dV_dse is P
        end

        AI_cpu = Array(AI_matrix)
        score_cpu = Array(score_vector)
        ridge_AI = T(1e-6 * sum(diag(AI_cpu))/num_var_params)
        AI_cpu_reg = AI_cpu + Diagonal(fill(ridge_AI, num_var_params))

        Δθ_cpu = zeros(T, num_var_params)
        try
            Δθ_cpu = cholesky(Symmetric(AI_cpu_reg)) \ score_cpu
        catch e
            if isa(e, PosDefException); try Δθ_cpu = pinv(AI_cpu_reg) * score_cpu catch; break end
            else rethrow(e) end
        end
        if all(iszero, Δθ_cpu) && !all(iszero, score_cpu); break end

        step_factor = one(T)
        max_line_search_iter = 10
        line_search_factor = T(0.5)
        log_likelihood_at_theta_old = log_likelihood_prev

        accepted_step = false
        for ls_iter in 1:max_line_search_iter
            σ²_a_new = var_comps.σ²_a + step_factor * Δθ_cpu[1]
            σ²_aa_new_val = var_comps.σ²_aa # Keep old if not updated
            σ²_e_new = zero(T)

            param_idx_offset = 1
            if G_epistatic !== nothing
                σ²_aa_new_val = var_comps.σ²_aa + step_factor * Δθ_cpu[2]
                param_idx_offset = 2
            end
            σ²_e_new = var_comps.σ²_e + step_factor * Δθ_cpu[param_idx_offset+1]

            valid_step = true
            if σ²_a_new < eps(T); valid_step = false; end
            if G_epistatic !== nothing && σ²_aa_new_val < eps(T); valid_step = false; end
            if σ²_e_new < eps(T) * T(100); valid_step = false; end

            if valid_step
                var_comps_temp = Main.DynamicEpistasisGBLUP.VarianceComponents{T}(
                    σ²_a_new, G_epistatic !== nothing ? σ²_aa_new_val : zero(T), σ²_e_new,
                    zero(T), zero(T), zero(T) # p, h2, H2 are recalculated later
                )
                V_inv_temp, log_det_V_temp = compute_V_inverse_reml(G_additive, G_epistatic, var_comps_temp, n_individuals)
                P_matrix_temp, log_det_XtVinvX_temp = compute_P_matrix_reml(V_inv_temp, X_gpu)
                log_likelihood_new_candidate = compute_reml_loglikelihood_reml(y_gpu, X_gpu, P_matrix_temp, n_fixed_effects, log_det_V_temp, log_det_XtVinvX_temp, n_individuals)

                if !isnan(log_likelihood_new_candidate) && (log_likelihood_new_candidate > log_likelihood_at_theta_old || isinf(log_likelihood_at_theta_old) || ls_iter == max_line_search_iter)
                    var_comps.σ²_a = σ²_a_new
                    var_comps.σ²_aa = G_epistatic !== nothing ? σ²_aa_new_val : zero(T)
                    var_comps.σ²_e = σ²_e_new
                    log_likelihood_prev = log_likelihood_new_candidate
                    V_inv_current = V_inv_temp # Update current V_inv and P for next iteration's AI matrix
                    P_matrix_current = P_matrix_temp
                    log_det_V_stable_current = log_det_V_temp
                    log_det_XtVinvX_stable_current = log_det_XtVinvX_temp
                    accepted_step = true
                    break
                end
            end
            step_factor *= line_search_factor
        end # end line search

        if !accepted_step # Line search failed to find improvement, keep old var_comps
             # log_likelihood_prev remains from previous iteration
        end

        var_comps.σ²_p = var_comps.σ²_a + var_comps.σ²_aa + var_comps.σ²_e
        if var_comps.σ²_p > eps(T)
            var_comps.h² = var_comps.σ²_a / var_comps.σ²_p
            var_comps.H² = (var_comps.σ²_a + var_comps.σ²_aa) / var_comps.σ²_p
        else var_comps.h² = zero(T); var_comps.H² = zero(T) end

        current_log_likelihood = log_likelihood_prev # This is the logL after accepted step (or old if step failed)

        # Convergence checks
        converged_logL = false
        if iter > 1 # Need at least one previous logL to compare
            if abs(current_log_likelihood - log_likelihood_at_theta_old) < convergence_tol
                converged_logL = true
            end
        end

        # Check convergence of parameters (Δθ relative to θ)
        # Δθ_cpu contains the update vector for [σ²_a, (σ²_aa), σ²_e]
        # θ_current_for_conv_check = [var_comps.σ²_a, (G_epistatic !== nothing ? var_comps.σ²_aa : T[])..., var_comps.σ²_e]
        # Need to be careful with conditional inclusion of σ²_aa for norm calculation.
        # Let's use the Δθ before step factor was applied, and compare to current var_comps values.
        # Norm of update relative to norm of current parameters.
        param_norm = sqrt(var_comps.σ²_a^2 + (G_epistatic !== nothing ? var_comps.σ²_aa^2 : zero(T)) + var_comps.σ²_e^2)
        update_norm = norm(Δθ_cpu) # Δθ_cpu is the full update step for current parameters

        converged_params = false
        if param_norm > eps(T) && (update_norm / param_norm) < convergence_tol # Use same tolerance, or a different one
            converged_params = true
        elseif update_norm < convergence_tol # If params are near zero, relative change is tricky
            converged_params = true
        end

        if iter > 1 && converged_logL && converged_params # Require both logL and params to stabilize
             # println("REML converged at iteration $iter. LogL = $current_log_likelihood")
            break
        end

        # log_likelihood_prev = current_log_likelihood # This is already done if step was accepted.
                                                       # If step failed, log_likelihood_prev was restored.
        if iter == max_iterations
            # println("REML reached max iterations ($max_iterations) without full convergence. Final LogL = $current_log_likelihood")
        end
    end

    mme_cache = (beta = beta_hat, u_additive = u_add_hat, u_epistatic = u_epi_hat)
    return var_comps, mme_cache
end

"""
    compute_V_inverse_reml(G_add, G_epi, var_comps, n) -> Tuple{CuArray, T}

Computes V^-1 and log|V_stable|.
V_stable = (σ²_a*G_add + σ²_aa*G_epi + σ²_e*I) + ridge
"""
function compute_V_inverse_reml(
    G_additive::CuArray{T,2},
    G_epistatic::Union{Nothing, CuArray{T,2}},
    var_comps::Main.DynamicEpistasisGBLUP.VarianceComponents{T},
    n_individuals::Int
) where T <: AbstractFloat

    V = var_comps.σ²_e * CuMatrix{T}(I, n_individuals, n_individuals)
    V .+= var_comps.σ²_a .* G_additive
    if G_epistatic !== nothing && var_comps.σ²_aa > eps(T) # Check if σ²_aa is non-negligible
        V .+= var_comps.σ²_aa .* G_epistatic
    end

    ridge = T(1e-6) * (abs(tr(V))/n_individuals + T(1e-9)) # Ensure ridge is positive even if tr(V) is small/zero
    V_stable = V + CuMatrix{T}(I, n_individuals, n_individuals) * ridge

    log_det_V_stable_val = zero(T)
    V_inv = CUDA.zeros(T, n_individuals, n_individuals)

    try
        ch_V = cholesky(Symmetric(V_stable); check = true)
        V_inv = inv(ch_V)
        log_det_V_stable_val = logdet(ch_V)
    catch e
        if isa(e, PosDefException)
            V_inv = inv(V_stable)
            try log_det_V_stable_val = logdet(Symmetric(V_stable)) catch; log_det_V_stable_val = T(NaN) end
        else rethrow(e) end
    end
    return V_inv, log_det_V_stable_val
end

"""
    compute_P_matrix_reml(V_inv, X_gpu) -> Tuple{CuArray, T}

Computes P = V_inv - V_inv*X * inv(X'*V_inv*X_stable) * X'*V_inv and log|X'V_invX_stable|.
"""
function compute_P_matrix_reml(
    V_inv::CuArray{T,2},
    X_gpu::CuArray{T,2}
) where T <: AbstractFloat

    Xt_Vinv = X_gpu' * V_inv
    Xt_Vinv_X = Xt_Vinv * X_gpu

    ridge = T(1e-7) * (abs(tr(Xt_Vinv_X))/size(X_gpu,2) + T(1e-9))
    Xt_Vinv_X_stable = Xt_Vinv_X + CuMatrix{T}(I, size(X_gpu,2), size(X_gpu,2)) * ridge

    inv_Xt_Vinv_X = CUDA.zeros(T,0,0)
    log_det_XtVinvX_stable_val = zero(T)

    try
        ch_XtVinvX = cholesky(Symmetric(Xt_Vinv_X_stable); check=true)
        inv_Xt_Vinv_X = inv(ch_XtVinvX)
        log_det_XtVinvX_stable_val = logdet(ch_XtVinvX)
    catch e
         if isa(e, PosDefException)
            inv_Xt_Vinv_X = inv(Xt_Vinv_X_stable)
            try log_det_XtVinvX_stable_val = logdet(Symmetric(Xt_Vinv_X_stable)) catch; log_det_XtVinvX_stable_val = T(NaN) end
         else rethrow(e) end
    end

    if size(inv_Xt_Vinv_X,1) == 0 # If inversion failed
        # Return P as V_inv (approx if X has small effect or for numerical stability) or error
        # This case should be handled carefully, may indicate issues.
        # For now, returning V_inv and NaN logdet to signal problem.
        return V_inv, T(NaN)
    end

    P_matrix = V_inv - Xt_Vinv' * inv_Xt_Vinv_X * Xt_Vinv
    return P_matrix, log_det_XtVinvX_stable_val
end

"""
    solve_mme_for_reml(y_gpu, X_gpu, G_add, G_epi, var_comps, V_inv_current) -> beta, u_add, u_epi

Solves Mixed Model Equations using V_inv.
beta_hat = inv(X'V_invX) * X'V_inv*y
u_hat = G*Z'*V_inv*(y - X*beta_hat)
"""
function solve_mme_for_reml(
    y_gpu::CuArray{T,1},
    X_gpu::CuArray{T,2},
    G_additive::CuArray{T,2},
    G_epistatic::Union{Nothing, CuArray{T,2}},
    var_comps::Main.DynamicEpistasisGBLUP.VarianceComponents{T},
    V_inv::CuArray{T,2} # Pass current V_inv
) where T <: AbstractFloat

    n_individuals = length(y_gpu)
    n_fixed = size(X_gpu, 2)

    Xt_Vinv = X_gpu' * V_inv
    Xt_Vinv_X = Xt_Vinv * X_gpu
    ridge_mme = T(1e-7) * (abs(tr(Xt_Vinv_X))/n_fixed + T(1e-9))
    Xt_Vinv_X_stable = Xt_Vinv_X + CuMatrix{T}(I, n_fixed, n_fixed) * ridge_mme

    inv_Xt_Vinv_X_mme = CUDA.zeros(T,0,0)
    try
        ch_mme = cholesky(Symmetric(Xt_Vinv_X_stable); check=true)
        inv_Xt_Vinv_X_mme = inv(ch_mme)
    catch e
        if isa(e, PosDefException); inv_Xt_Vinv_X_mme = inv(Xt_Vinv_X_stable)
        else rethrow(e) end
    end
    if size(inv_Xt_Vinv_X_mme,1) == 0 # Solve failed
        return CUDA.zeros(T,n_fixed), CUDA.zeros(T,n_individuals), CUDA.zeros(T,n_individuals)
    end

    beta_hat = inv_Xt_Vinv_X_mme * Xt_Vinv * y_gpu

    y_minus_Xbeta = y_gpu - X_gpu * beta_hat

    u_add_hat = var_comps.σ²_a .* (G_additive * (V_inv * y_minus_Xbeta)) # Assuming Z=I

    u_epi_hat = CUDA.zeros(T, n_individuals)
    if G_epistatic !== nothing && var_comps.σ²_aa > eps(T)
        u_epi_hat = var_comps.σ²_aa .* (G_epistatic * (V_inv * y_minus_Xbeta)) # Assuming Z=I
    end

    return beta_hat, u_add_hat, u_epi_hat
end


"""
    compute_reml_loglikelihood_reml(y_gpu, X_gpu, P_matrix, n_fixed_effects, log_det_V_stable_val, log_det_XtVinvX_stable_val, n_individuals)

Computes the REML log-likelihood using precomputed log-determinants.
"""
function compute_reml_loglikelihood_reml(
    y_gpu::CuArray{T,1},
    X_gpu::CuArray{T,2},
    P_matrix::CuArray{T,2},
    n_fixed_effects::Int,
    log_det_V_stable_val::T,          # log|V_stable|
    log_det_XtVinvX_stable_val::T,  # log|X'V_invX_stable|
    n_individuals::Int
) where T <: AbstractFloat

    n_N = n_individuals
    n_p_rank = n_fixed_effects

    y_P_y = dot(y_gpu, P_matrix * y_gpu)

    current_log_likelihood = -T(0.5) * (
        (n_N - n_p_rank) * log(T(2π)) +
        log_det_V_stable_val +
        log_det_XtVinvX_stable_val +
        y_P_y
    )

    return current_log_likelihood
end


# Export functions if this file were a module
# export initialize_variance_components, estimate_variance_components_reml!

[end of src/reml.jl]
