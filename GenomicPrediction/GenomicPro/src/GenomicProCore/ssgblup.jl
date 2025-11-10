# src/GenomicProPredict/ssgblup.jl

using LinearAlgebra, SparseArrays

"""
    solve_ssgblup(G::Matrix{Float64}, ped::PedigreeData,
                  genotyped_indices::Vector{Int}, y::Vector{Float64},
                  λ::Float64; kwargs...)

Solve Single-Step Genomic BLUP combining pedigree and genomic information.
...
"""
function solve_ssgblup(G::Matrix{Float64},
                      ped::PedigreeData,
                      genotyped_indices::Vector{Int},
                      y::Vector{Float64},
                      λ::Float64;
                      method::Symbol = :pcg,
                      X::Union{Matrix{Float64}, Nothing} = nothing,
                      validate_compatibility::Bool = true,
                      blend_parameter::Float64 = 1.0,
                      tolerance::Float64 = 1e-6,
                      max_iterations::Int = 1000)

    n_total = length(y)
    n_genotyped = length(genotyped_indices)

    println("Solving Single-Step GBLUP...")

    # Compute A⁻¹ efficiently
    A_inv = compute_A_inverse(ped)

    # Extract A₂₂ submatrix for validation
    # This is inefficient and should be avoided in a production implementation
    A = inv(Matrix(A_inv))
    A22 = A[genotyped_indices, genotyped_indices]

    if validate_compatibility
        compat_metrics = assess_G_A22_compatibility(G, A22)
    else
        compat_metrics = (correlation = NaN,)
    end

    # Construct H⁻¹
    H_inv = construct_H_inverse(A_inv, G, A22, genotyped_indices, blend_parameter)

    if isnothing(X)
        X = ones(Float64, n_total, 1)
    end
    y_corrected, β = absorb_fixed_effects(X, y)

    # Solve MME
    if method == :pcg
        result = pcg_with_H_inverse(H_inv, y_corrected, λ, tolerance, max_iterations)
        u = result.solution
    else
        # Direct solve
        C = H_inv + I * (1/λ)
        u = C \ y_corrected
        result = (iterations=0, converged=true, residual_norm=0)
    end

    return (
        breeding_values = u,
        fixed_effects = β,
        compatibility_metrics = compat_metrics,
        iterations = result.iterations,
        converged = result.converged
    )
end

function assess_G_A22_compatibility(G::Matrix{Float64}, A22::Matrix{Float64})
    correlation = cor(vec(G), vec(A22))
    return (correlation = correlation,)
end

function construct_H_inverse(A_inv::SparseMatrixCSC,
                            G::Matrix{Float64},
                            A22::Matrix{Float64},
                            genotyped_indices::Vector{Int},
                            blend_parameter::Float64)

    # Efficiently compute G_inv_minus_A22_inv * augmentation
    # This avoids forming the inverse matrices explicitly

    # Placeholder for a more efficient implementation
    G_inv = inv(G)
    A22_inv = inv(A22)
    augmentation = blend_parameter .* (G_inv - A22_inv)

    H_inv = copy(A_inv)
    H_inv[genotyped_indices, genotyped_indices] .+= augmentation

    return H_inv
end

function pcg_with_H_inverse(H_inv::SparseMatrixCSC,
                           y::Vector{Float64},
                           λ::Float64,
                           tolerance::Float64,
                           max_iterations::Int)

    n = length(y)
    u = zeros(n)
    r = copy(y)

    M_diag_inv = 1.0 ./ (diag(H_inv) .+ 1/λ)

    z = r .* M_diag_inv
    p = copy(z)
    rz_old = dot(r, z)

    converged = false

    for k in 1:max_iterations
        Cp = (H_inv * p) .+ p .* (1/λ)

        α = rz_old / dot(p, Cp)
        u .+= α .* p
        r .-= α .* Cp

        if norm(r) < tolerance
            converged = true
            break
        end

        z = r .* M_diag_inv
        rz_new = dot(r, z)
        β = rz_new / rz_old
        p .= z .+ β .* p
        rz_old = rz_new
    end

    return (
        solution = u,
        iterations = max_iterations,
        residual_norm = norm(r),
        converged = converged
    )
end
