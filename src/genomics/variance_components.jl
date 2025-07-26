# Variance component estimation module

module VarianceComponentEstimation

using LinearAlgebra
using Statistics
using Optim
using ..CoreTypes
using ..Validation
using ..Constants

export estimate_variance_components, AIREMLMethod

abstract type VarianceComponentMethod end
struct AIREMLMethod <: VarianceComponentMethod end

function estimate_variance_components(y::AbstractVector{<:Real},
                                    grms::GRMSet;
                                    method::VarianceComponentMethod=AIREMLMethod(),
                                    fixed_effects::Union{AbstractMatrix{<:Real}, Nothing}=nothing)

    n = length(y)
    X = isnothing(fixed_effects) ? ones(n, 1) : fixed_effects

    if isa(method, AIREMLMethod)
        result = aireml_estimation(y, X, grms)
    else
        error("Only AIREML is currently implemented.")
    end

    return result
end

function aireml_estimation(y::AbstractVector{<:Real},
                          X::AbstractMatrix{<:Real},
                          grms::GRMSet)

    n = length(y)

    kernels = [grms.G, grms.D, grms.G_AA, grms.G_AD, grms.G_DD, I(n)]
    n_components = length(kernels)

    var_y = var(y)
    θ = [0.3, 0.05, 0.05, 0.02, 0.02, 0.56] .* var_y

    for iter in 1:MAX_ITERATIONS
        θ_old = copy(θ)

        V = sum(θ[i] * kernels[i] for i in 1:n_components)
        V += RIDGE_LAMBDA * I

        L = cholesky(Symmetric(V))
        V_inv = inv(L)

        P = V_inv - V_inv * X * inv(X' * V_inv * X) * X' * V_inv
        Py = P * y

        s = zeros(n_components)
        AI = zeros(n_components, n_components)

        for i in 1:n_components
            PK_i = P * kernels[i]
            s[i] = -0.5 * tr(PK_i) + 0.5 * dot(y, PK_i * Py)
            for j in i:n_components
                AI[i,j] = 0.5 * dot(Py, kernels[i] * P * kernels[j] * Py)
                AI[j,i] = AI[i,j]
            end
        end

        try
            Δθ = AI \ s
            θ .+= Δθ
            θ = max.(θ, 0.0) # Ensure non-negative
        catch
            @warn "AI matrix singular at iteration \$iter, using gradient ascent."
            θ .+= 0.1 * s # Simple gradient step
            θ = max.(θ, 0.0)
        end

        if maximum(abs.(θ - θ_old)) < CONVERGENCE_TOL
            @info "AI-REML converged in \$iter iterations."
            break
        end
    end

    estimates = (σ²_a=θ[1], σ²_d=θ[2], σ²_aa=θ[3], σ²_ad=θ[4], σ²_dd=θ[5], σ²_e=θ[6])

    # Placeholder for SEs and heritabilities
    return VarianceComponents(estimates, NamedTuple(), 0.0, 0.0, 0.0, 0.0, 0.0, nothing, NamedTuple())
end

end
