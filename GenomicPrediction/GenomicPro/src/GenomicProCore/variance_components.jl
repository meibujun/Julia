# src/GenomicProPredict/variance_components.jl

using LinearAlgebra

"""
    estimate_variance_components(G::Matrix{Float64}, y::Vector{Float64}; kwargs...)

Estimate genetic and residual variance components using Average Information REML.

Solves the mixed linear model:
    y = Xβ + Zu + e
...
"""
function estimate_variance_components(G::Matrix{Float64},
                                     y::Vector{Float64};
                                     method::Symbol = :AIREML,
                                     X::Union{Matrix{Float64}, Nothing} = nothing,
                                     max_iterations::Int = 100,
                                     tolerance::Float64 = 1e-6,
                                     initial_h2::Float64 = 0.5,
                                     constrain_positive::Bool = true,
                                     compute_se::Bool = true)

    n = length(y)

    # Validate inputs
    @assert size(G) == (n, n) "G dimensions must match length of y"
    @assert issymmetric(G) "G must be symmetric"
    @assert 0.0 < initial_h2 < 1.0 "initial_h2 must be in (0,1)"

    # Default to intercept-only fixed effects
    if isnothing(X)
        X = ones(Float64, n, 1)
    end

    # Center phenotypes for numerical stability
    y_mean = mean(y)
    y_centered = y .- y_mean

    # Initialize variance components from phenotypic variance and initial h²
    var_y = var(y_centered)
    σ²_g = initial_h2 * var_y
    σ²_e = (1.0 - initial_h2) * var_y

    println("Estimating variance components via $(method)...")
    println("  Initial values:")
    println("    σ²_genetic:  $(round(σ²_g, digits=4))")
    println("    σ²_residual: $(round(σ²_e, digits=4))")
    println("    h²:          $(round(initial_h2, digits=3))")
    println()

    # Dispatch to appropriate estimation method
    if method == :AIREML
        result = aireml_iterate(G, y_centered, X, σ²_g, σ²_e,
                               max_iterations, tolerance, constrain_positive)
    elseif method == :EMREML
        result = emreml_iterate(G, y_centered, X, σ²_g, σ²_e,
                               max_iterations, tolerance)
    else
        throw(ArgumentError("Unknown method: $method"))
    end

    # Compute standard errors if requested
    if compute_se && result.converged
        # SE computation is complex and will be implemented later
    end

    return result
end

"""
    aireml_iterate(G, y, X, σ²_g_init, σ²_e_init, max_iter, tol, constrain)

Perform AI-REML iterations to estimate variance components.
"""
function aireml_iterate(G::Matrix{Float64},
                       y::Vector{Float64},
                       X::Matrix{Float64},
                       σ²_g_init::Float64,
                       σ²_e_init::Float64,
                       max_iter::Int,
                       tol::Float64,
                       constrain::Bool)

    σ²_g = σ²_g_init
    σ²_e = σ²_e_init

    converged = false

    for iter in 1:max_iter
        λ = σ²_e / σ²_g

        # Solve MME
        C = G + I * λ
        C_inv = inv(C)

        # REML estimates
        Py = C_inv * y
        u = G * Py
        e = y - u

        σ²_g_new = dot(u, u) / (size(G,1) - tr(C_inv * G))
        σ²_e_new = dot(e, e) / (size(G,1) - tr(C_inv * I))

        # Update
        change = abs(σ²_g_new - σ²_g) + abs(σ²_e_new - σ²_e)
        σ²_g = σ²_g_new
        σ²_e = σ²_e_new

        if change < tol
            converged = true
            break
        end
    end

    h² = σ²_g / (σ²_g + σ²_e)

    return (
        genetic_variance = σ²_g,
        residual_variance = σ²_e,
        heritability = h²,
        iterations = max_iter,
        converged = converged
    )
end

"""
    emreml_iterate(G, y, X, σ²_g_init, σ²_e_init, max_iter, tol)

Perform EM-REML iterations to estimate variance components.
"""
function emreml_iterate(G::Matrix{Float64},
                       y::Vector{Float64},
                       X::Matrix{Float64},
                       σ²_g_init::Float64,
                       σ²_e_init::Float64,
                       max_iter::Int,
                       tol::Float64)

    σ²_g = σ²_g_init
    σ²_e = σ²_e_init

    converged = false

    for iter in 1:max_iter
        λ = σ²_e / σ²_g

        # Solve MME
        C = G + I * λ
        C_inv = inv(C)

        Py = C_inv * y
        u = G * Py

        # Update variance components
        σ²_g_new = (dot(u, inv(G) * u) + tr(C_inv * G)) / size(G,1)
        σ²_e_new = (dot(y - u, y - u) + tr(C_inv * I) * σ²_e) / size(G,1)

        # Update
        change = abs(σ²_g_new - σ²_g) + abs(σ²_e_new - σ²_e)
        σ²_g = σ²_g_new
        σ²_e = σ²_e_new

        if change < tol
            converged = true
            break
        end
    end

    h² = σ²_g / (σ²_g + σ²_e)

    return (
        genetic_variance = σ²_g,
        residual_variance = σ²_e,
        heritability = h²,
        iterations = max_iter,
        converged = converged
    )
end
