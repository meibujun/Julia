# ===== src/reml.jl =====
"""
    DynamicEpistasisGBLUP.REML

This module implements the Average Information Restricted Maximum Likelihood (AI-REML)
algorithm for estimating variance components in mixed models. It is designed for
flexibility, accepting multiple Genomic Relationship Matrices (GRMs) to model various
random effects (e.g., additive, epistatic).
"""
module REML

export estimate_variance_components_reml!

using CUDA
using LinearAlgebra
using Statistics
using ..DynamicEpistasisGBLUP.Types

# Define a type alias for abstract matrices on CPU or GPU
const MaybeCuMatrix{T} = Union{Matrix{T}, CuMatrix{T}}

"""
    estimate_variance_components_reml!(
        y::AbstractVector{T},
        GRMs::Vector{<:MaybeCuMatrix{T}},
        X::MaybeCuMatrix{T},
        initial_variances::AbstractVector{T},
        reml_params::REMLParameters{T}
    ) where T <: AbstractFloat -> REMLResults{T}

Estimates variance components using the AI-REML algorithm.

The function iteratively refines estimates for the variance components associated with
each GRM and the residual variance.

# Arguments
- `y::AbstractVector{T}`: Phenotype vector. Must be on the same device as GRMs and X.
- `GRMs::Vector{<:MaybeCuMatrix{T}}`: A vector of GRMs. Each GRM corresponds to a random effect.
- `X::MaybeCuMatrix{T}`: Incidence matrix for fixed effects.
- `initial_variances::AbstractVector{T}`: Vector of initial guesses for variance components. The last element is the residual variance (`σ²_e`), and the preceding elements correspond to the GRMs in order.
- `reml_params::REMLParameters{T}`: Struct containing REML control parameters.

# Returns
- `REMLResults{T}`: A struct containing the final variance components, log-likelihood, convergence status, and final `V⁻¹` and `P` matrices.
"""
function estimate_variance_components_reml!(
    y::AbstractVector{T},
    GRMs::Vector{<:MaybeCuMatrix{T}},
    X::MaybeCuMatrix{T},
    initial_variances::AbstractVector{T},
    reml_params::REMLParameters{T}
) where T <: AbstractFloat

    num_g_effects = length(GRMs)
    num_components = num_g_effects + 1 # Plus residual
    n = length(y)
    device = reml_params.use_gpu ? CUDA.functional() ? CUDA.device() : nothing : nothing

    if length(initial_variances) != num_components
        error("Number of initial variances must equal number of GRMs + 1 (for residual).")
    end

    # --- Initialization ---
    θ = copy(initial_variances) # Current variance components [σ²_g1, σ²_g2, ..., σ²_e]
    log_likelihood = -Inf
    converged = false

    if reml_params.verbose
        println("--- Starting AI-REML ---")
        println("Initial variance components: ", round.(θ, digits=4))
    end

    # --- Main Iteration Loop ---
    iter = 0
    for i in 1:reml_params.max_iter
        iter = i
        log_likelihood_old = log_likelihood

        # 1. Construct V and compute V⁻¹
        V = construct_V(GRMs, θ, n, device)
        V_inv, log_det_V = stable_inv_logdet(V)

        if isnan(log_det_V)
            @warn "REML iter $i: V matrix is singular, stopping."
            break
        end

        # 2. Construct P matrix
        P, log_det_XtVinvX = construct_P(V_inv, X)
        if isnan(log_det_XtVinvX)
            @warn "REML iter $i: X'V⁻¹X matrix is singular, stopping."
            break
        end

        # 3. Calculate REML log-likelihood
        yPy = dot(y, P * y)
        log_likelihood = -0.5 * (log_det_V + log_det_XtVinvX + yPy + (n - size(X, 2)) * log(2π))

        if reml_params.verbose
             println("Iter $i: logL = ", round(log_likelihood, digits=4))
        end

        # Check for convergence based on log-likelihood
        if abs(log_likelihood - log_likelihood_old) < reml_params.tol
            converged = true
            if reml_params.verbose; println("Converged on log-likelihood tolerance."); end
            break
        end

        # If logL decreases, something is wrong, but line search should prevent this. Stop.
        if log_likelihood < log_likelihood_old && i > 1
             if reml_params.verbose; @warn "Log-likelihood decreased. Stopping."; end
             log_likelihood = log_likelihood_old # Revert
             break
        end

        # 4. Calculate Score vector and Average Information matrix
        scores = calculate_scores(y, P, GRMs, n, device)
        AI_matrix = calculate_ai_matrix(P, GRMs, n, device)

        # 5. Calculate update step Δθ
        # Regularize AI matrix slightly to ensure it's invertible
        ridge = T(1e-8) * tr(AI_matrix) / num_components
        AI_matrix_reg = AI_matrix + Diagonal(fill(ridge, num_components))

        Δθ = cholesky(Symmetric(AI_matrix_reg)) \ scores

        # 6. Backtracking line search to update θ
        step_factor = T(1.0)
        θ_new = θ
        accepted_step = false
        for _ in 1:10 # Max 10 backtracking steps
            θ_candidate = θ + step_factor * Δθ

            # Check if parameters are valid (positive)
            if all(θ_candidate .> reml_params.min_variance_value)
                V_cand = construct_V(GRMs, θ_candidate, n, device)
                V_inv_cand, log_det_V_cand = stable_inv_logdet(V_cand)

                if !isnan(log_det_V_cand)
                    P_cand, log_det_XtVinvX_cand = construct_P(V_inv_cand, X)
                    if !isnan(log_det_XtVinvX_cand)
                        yPy_cand = dot(y, P_cand * y)
                        log_likelihood_cand = -0.5 * (log_det_V_cand + log_det_XtVinvX_cand + yPy_cand + (n - size(X, 2)) * log(2π))

                        # If new logL is better, accept the step
                        if log_likelihood_cand > log_likelihood
                            θ_new = θ_candidate
                            accepted_step = true
                            if reml_params.verbose; println("   Accepted step with factor: ", round(step_factor, digits=3)); end
                            break
                        end
                    end
                end
            end
            step_factor *= 0.5 # Halve the step factor
        end

        if !accepted_step
            if reml_params.verbose; @warn "Line search failed to find an improvement. Stopping."; end
            break
        end

        # Check for convergence on parameter change
        if norm(θ_new - θ) / norm(θ) < reml_params.tol
            converged = true
            θ = θ_new
            if reml_params.verbose; println("Converged on parameter tolerance."); end
            break
        end

        θ = θ_new
    end # End of main iteration loop

    if iter == reml_params.max_iter && !converged
        @warn "REML reached maximum iterations ($(reml_params.max_iter)) without converging."
    end

    # Final calculations for results
    V_final = construct_V(GRMs, θ, n, device)
    V_inv_final, _ = stable_inv_logdet(V_final)
    P_final, _ = construct_P(V_inv_final, X)

    return REMLResults{T}(θ, log_likelihood, iter, converged, V_inv_final, P_final)
end

# --- Helper Functions ---

function construct_V(GRMs, θ, n, device)
    # V = G1*σ²_g1 + G2*σ²_g2 + ... + I*σ²_e
    V = device === nothing ?
        zeros(T, n, n) :
        CUDA.zeros(T, n, n)

    T = eltype(V)

    # Add genetic components
    for i in 1:length(GRMs)
        V .+= GRMs[i] .* θ[i]
    end

    # Add residual component
    eye = device === nothing ?
        Matrix{T}(I, n, n) :
        CuMatrix{T}(I, n, n)
    V .+= eye .* θ[end]

    return V
end

function stable_inv_logdet(A::MaybeCuMatrix{T}) where T
    # Add a small ridge for numerical stability before inversion
    ridge = T(1e-7) * (abs(tr(A))/size(A,1) + T(1e-9))
    A_stable = A + Diagonal(fill(ridge, size(A,1)))

    try
        ch = cholesky(Symmetric(A_stable))
        return inv(ch), logdet(ch)
    catch e
        if isa(e, PosDefException)
            @warn "Matrix is not positive definite, falling back to general inverse. Logdet may be inaccurate."
            return inv(A_stable), T(NaN)
        else
            rethrow(e)
        end
    end
end

function construct_P(V_inv::MaybeCuMatrix{T}, X::MaybeCuMatrix{T}) where T
    Xt_Vinv = X' * V_inv
    Xt_Vinv_X = Xt_Vinv * X

    inv_Xt_Vinv_X, log_det_XtVinvX = stable_inv_logdet(Xt_Vinv_X)

    P = V_inv - Xt_Vinv' * inv_Xt_Vinv_X * Xt_Vinv
    return P, log_det_XtVinvX
end

function calculate_scores(y, P, GRMs, n, device)
    num_g_effects = length(GRMs)
    num_components = num_g_effects + 1
    T = eltype(y)

    scores = device === nothing ?
        zeros(T, num_components) :
        CUDA.zeros(T, num_components)

    Py = P * y

    # Scores for genetic components
    for i in 1:num_g_effects
        PG_i = P * GRMs[i]
        scores[i] = 0.5 * (dot(Py, GRMs[i] * Py) - tr(PG_i))
    end

    # Score for residual component (G_i = I)
    scores[end] = 0.5 * (dot(Py, Py) - tr(P))

    return scores
end

function calculate_ai_matrix(P, GRMs, n, device)
    num_g_effects = length(GRMs)
    num_components = num_g_effects + 1
    T = eltype(P)

    AI = device === nothing ?
        zeros(T, num_components, num_components) :
        CUDA.zeros(T, num_components)

    # Pre-calculate P*G_i products
    PGs = [P * G for G in GRMs]

    # AI for genetic components
    for i in 1:num_g_effects
        for j in i:num_g_effects
            # AI_ij = 0.5 * tr(P*G_i*P*G_j)
            val = 0.5 * tr(PGs[i] * PGs[j])
            AI[i, j] = AI[j, i] = val
        end
    end

    # AI for residual component (G_i = I)
    # AI_ie = 0.5 * tr(P*G_i*P*I) = 0.5 * tr(P*G_i*P)
    for i in 1:num_g_effects
        val = 0.5 * tr(PGs[i] * P)
        AI[i, end] = AI[end, i] = val
    end

    # AI_ee = 0.5 * tr(P*I*P*I) = 0.5 * tr(P*P)
    AI[end, end] = 0.5 * tr(P * P)

    return AI
end

end # module REML
