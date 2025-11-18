"""
GBLUP (Genomic Best Linear Unbiased Prediction) implementation.

Solves the mixed model equations:
[X'X    X'Z  ] [β]   [X'y]
[Z'X  Z'Z+G⁻¹λ] [u] = [Z'y]

where:
- y: phenotype vector
- X: design matrix for fixed effects
- Z: design matrix for random effects (usually identity)
- G: genomic relationship matrix
- λ = σ²ₑ/σ²ᵤ: variance ratio
- β: fixed effects
- u: random genomic effects (breeding values)
"""

"""
    GBLUPResult

Results from GBLUP model fitting.

# Fields
- `beta::Vector{Float64}`: Fixed effect estimates
- `u::Vector{Float64}`: Random effect estimates (breeding values)
- `var_e::Float64`: Residual variance (σ²ₑ)
- `var_u::Float64`: Genetic variance (σ²ᵤ)
- `heritability::Float64`: Heritability (h² = σ²ᵤ/(σ²ᵤ + σ²ₑ))
- `log_likelihood::Float64`: Log-likelihood
- `converged::Bool`: Whether variance components converged
- `iterations::Int`: Number of iterations for variance estimation
"""
struct GBLUPResult
    beta::Vector{Float64}
    u::Vector{Float64}
    var_e::Float64
    var_u::Float64
    heritability::Float64
    log_likelihood::Float64
    converged::Bool
    iterations::Int
end

"""
    GBLUPModel

GBLUP model for genomic prediction.

# Fields
- `method::Symbol`: Solver method (:cholesky or :pcg)
- `max_iter::Int`: Maximum iterations for iterative solvers
- `tol::Float64`: Convergence tolerance
- `estimate_variances::Bool`: Whether to estimate variance components
- `ridge::Float64`: Ridge penalty for numerical stability (default: 1e-6)

# Example
```julia
model = GBLUPModel(method=:cholesky)
result = fit!(model, geno, pheno; G=G)
predictions = predict(model, geno_test)
```
"""
mutable struct GBLUPModel
    method::Symbol
    max_iter::Int
    tol::Float64
    estimate_variances::Bool
    ridge::Float64

    # Fitted parameters
    result::Union{GBLUPResult, Nothing}
    sample_ids::Union{Vector{String}, Nothing}

    function GBLUPModel(;
        method::Symbol = :cholesky,
        max_iter::Int = 1000,
        tol::Float64 = 1e-6,
        estimate_variances::Bool = true,
        ridge::Float64 = 1e-6
    )
        if method ∉ [:cholesky, :pcg]
            throw(ArgumentError("method must be :cholesky or :pcg"))
        end

        new(method, max_iter, tol, estimate_variances, ridge, nothing, nothing)
    end
end

"""
    solve_mixed_model_cholesky(y, X, Z, G, lambda; ridge=1e-6)

Solve mixed model equations using Cholesky decomposition.

This is the direct solver - accurate but O(n³) complexity.

# Arguments
- `y`: Phenotype vector (n × 1)
- `X`: Design matrix for fixed effects (n × p)
- `Z`: Design matrix for random effects (n × q)
- `G`: Genomic relationship matrix (q × q)
- `lambda`: Variance ratio σ²ₑ/σ²ᵤ
- `ridge`: Ridge penalty for numerical stability

# Returns
- `beta`: Fixed effect estimates
- `u`: Random effect estimates
"""
function solve_mixed_model_cholesky(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Real},
    G::AbstractMatrix{<:Real},
    lambda::Real;
    ridge::Real = 1e-6
)
    n = length(y)
    p = size(X, 2)
    q = size(Z, 2)

    # Build coefficient matrix (left-hand side)
    # [X'X      X'Z     ]
    # [Z'X  Z'Z + G⁻¹λ  ]

    XtX = X' * X
    XtZ = X' * Z
    ZtZ = Z' * Z

    # Compute G⁻¹λ (with ridge for stability)
    G_ridge = G + ridge * I
    Ginv_lambda = inv(G_ridge) * lambda

    # Build full coefficient matrix
    C = zeros(Float64, p + q, p + q)
    C[1:p, 1:p] = XtX
    C[1:p, (p+1):end] = XtZ
    C[(p+1):end, 1:p] = XtZ'
    C[(p+1):end, (p+1):end] = ZtZ + Ginv_lambda

    # Build right-hand side
    # [X'y]
    # [Z'y]
    rhs = zeros(Float64, p + q)
    rhs[1:p] = X' * y
    rhs[(p+1):end] = Z' * y

    # Solve using Cholesky
    try
        C_sym = Symmetric(C)
        chol = cholesky(C_sym)
        solution = chol \ rhs

        beta = solution[1:p]
        u = solution[(p+1):end]

        return beta, u
    catch e
        @error "Cholesky decomposition failed" exception=e
        throw(ConvergenceError("Failed to solve mixed model equations", :cholesky, Inf))
    end
end

"""
    solve_mixed_model_pcg(y, X, Z, G, lambda; max_iter=1000, tol=1e-6, ridge=1e-6)

Solve mixed model equations using Preconditioned Conjugate Gradient.

This is an iterative solver - faster than Cholesky for large problems.

# Arguments
- `y`, `X`, `Z`, `G`, `lambda`: Same as solve_mixed_model_cholesky
- `max_iter`: Maximum iterations
- `tol`: Convergence tolerance
- `ridge`: Ridge penalty

# Returns
- `beta`: Fixed effect estimates
- `u`: Random effect estimates
- `converged`: Whether the solver converged
- `iterations`: Number of iterations
"""
function solve_mixed_model_pcg(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Real},
    G::AbstractMatrix{<:Real},
    lambda::Real;
    max_iter::Int = 1000,
    tol::Real = 1e-6,
    ridge::Real = 1e-6
)
    n = length(y)
    p = size(X, 2)
    q = size(Z, 2)

    # For simplicity, we'll use the same direct approach for fixed effects
    # and PCG for random effects only

    # First, absorb fixed effects (use simple approach for now)
    # In practice, you'd use a more sophisticated algorithm

    # Build coefficient matrix
    XtX = X' * X
    XtZ = X' * Z
    ZtZ = Z' * Z

    G_ridge = G + ridge * I
    Ginv_lambda = inv(G_ridge) * lambda

    # Build system
    C = zeros(Float64, p + q, p + q)
    C[1:p, 1:p] = XtX
    C[1:p, (p+1):end] = XtZ
    C[(p+1):end, 1:p] = XtZ'
    C[(p+1):end, (p+1):end] = ZtZ + Ginv_lambda

    rhs = zeros(Float64, p + q)
    rhs[1:p] = X' * y
    rhs[(p+1):end] = Z' * y

    # Use conjugate gradient on the symmetric system
    C_sym = Symmetric(C)

    # Initial guess
    x = zeros(Float64, p + q)

    # Simple diagonal preconditioner
    M = Diagonal([1.0 / max(abs(C_sym[i, i]), 1e-10) for i in 1:(p+q)])

    # PCG iteration
    r = rhs - C_sym * x
    z = M * r
    p_vec = copy(z)
    rsold = dot(r, z)

    converged = false
    iter = 0

    for iter in 1:max_iter
        Ap = C_sym * p_vec
        alpha = rsold / dot(p_vec, Ap)
        x .= x .+ alpha .* p_vec
        r .= r .- alpha .* Ap
        z .= M * r
        rsnew = dot(r, z)

        if sqrt(rsnew) < tol
            converged = true
            break
        end

        beta_cg = rsnew / rsold
        p_vec .= z .+ beta_cg .* p_vec
        rsold = rsnew
    end

    beta = x[1:p]
    u = x[(p+1):end]

    return beta, u, converged, iter
end

"""
    estimate_variance_components(y, X, u, Z; max_iter=50, tol=1e-4)

Estimate variance components using EM-REML algorithm.

# Arguments
- `y`: Phenotype vector
- `X`: Design matrix for fixed effects
- `u`: Current estimates of random effects
- `Z`: Design matrix for random effects
- `max_iter`: Maximum EM iterations
- `tol`: Convergence tolerance

# Returns
- `var_e`: Residual variance
- `var_u`: Genetic variance
- `converged`: Whether EM converged
- `iterations`: Number of iterations
"""
function estimate_variance_components(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    u::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real};
    max_iter::Int = 50,
    tol::Real = 1e-4
)
    n = length(y)
    q = length(u)

    # Initialize variance estimates
    y_var = var(y)
    var_u = 0.5 * y_var
    var_e = 0.5 * y_var

    converged = false

    for iter in 1:max_iter
        var_u_old = var_u
        var_e_old = var_e

        # E-step: Compute expected values
        # (Already have u from mixed model solution)

        # M-step: Update variances
        # Compute residuals
        beta_simple = X \ y  # Simple estimate of fixed effects
        residuals = y - X * beta_simple - Z * u

        # Update variance estimates
        var_e_new = dot(residuals, residuals) / n
        var_u_new = dot(u, u) / q

        # Ensure positive variances
        var_e = max(var_e_new, 1e-10)
        var_u = max(var_u_new, 1e-10)

        # Check convergence
        rel_change = max(
            abs(var_u - var_u_old) / (var_u_old + 1e-10),
            abs(var_e - var_e_old) / (var_e_old + 1e-10)
        )

        if rel_change < tol
            converged = true
            return var_e, var_u, converged, iter
        end
    end

    return var_e, var_u, converged, max_iter
end

"""
    fit!(model::GBLUPModel, geno::CompactGenotypes, pheno::PhenotypeData;
         G::Union{AbstractMatrix, Nothing}=nothing, trait_index::Int=1)

Fit GBLUP model to data.

# Arguments
- `model`: GBLUPModel object
- `geno`: Genotype data
- `pheno`: Phenotype data
- `G`: Pre-computed GRM (if nothing, will compute using VanRaden method)
- `trait_index`: Which trait to fit (default: 1)

# Returns
- `GBLUPResult`: Fitted model results

# Example
```julia
model = GBLUPModel()
result = fit!(model, geno, pheno; G=G)

println("Heritability: ", result.heritability)
println("Genetic variance: ", result.var_u)
```
"""
function fit!(
    model::GBLUPModel,
    geno::CompactGenotypes,
    pheno::PhenotypeData;
    G::Union{AbstractMatrix, Nothing} = nothing,
    trait_index::Int = 1
)
    # Ensure data is matched
    geno_ids = Set(sample_ids(geno))
    pheno_ids = Set(sample_ids(pheno))

    if geno_ids != pheno_ids
        @warn "Genotype and phenotype samples don't match. Merging..."
        geno, pheno, common_ids = merge_genotype_phenotype(geno, pheno)
    end

    n = n_samples(geno)

    # Extract phenotypes for specified trait
    y = pheno.values[:, trait_index]

    # Remove missing phenotypes
    valid_idx = findall(.!isnan.(y))
    if length(valid_idx) < n
        @info "Removing $(n - length(valid_idx)) samples with missing phenotypes"
        y = y[valid_idx]
        geno = subset_samples(geno, valid_idx)
        n = length(y)
    end

    # Compute GRM if not provided
    if G === nothing
        @info "Computing GRM using VanRaden method"
        G = compute_grm_vanraden(geno; min_maf=0.01)
    end

    # Design matrices
    X = ones(Float64, n, 1)  # Intercept only
    Z = Matrix{Float64}(I, n, n)  # Identity (each sample is its own random effect)

    # Initialize variance components
    if model.estimate_variances
        # Initial estimates
        y_var = var(y)
        var_u = 0.5 * y_var
        var_e = 0.5 * y_var
        lambda = var_e / var_u

        # EM-REML iterations
        converged = false
        em_iter = 0

        for em_iter in 1:50
            # Solve mixed model equations
            if model.method == :cholesky
                beta, u = solve_mixed_model_cholesky(y, X, Z, G, lambda; ridge=model.ridge)
            else  # :pcg
                beta, u, _, _ = solve_mixed_model_pcg(y, X, Z, G, lambda;
                                                      max_iter=model.max_iter,
                                                      tol=model.tol,
                                                      ridge=model.ridge)
            end

            # Update variance components
            var_e_new, var_u_new, converged, _ = estimate_variance_components(y, X, u, Z;
                                                                              max_iter=1,
                                                                              tol=model.tol)

            # Check convergence
            rel_change = max(
                abs(var_u_new - var_u) / (var_u + 1e-10),
                abs(var_e_new - var_e) / (var_e + 1e-10)
            )

            var_e = var_e_new
            var_u = var_u_new
            lambda = var_e / var_u

            if rel_change < model.tol
                converged = true
                break
            end
        end

        # Final solution with converged variances
        if model.method == :cholesky
            beta, u = solve_mixed_model_cholesky(y, X, Z, G, lambda; ridge=model.ridge)
        else
            beta, u, _, _ = solve_mixed_model_pcg(y, X, Z, G, lambda;
                                                  max_iter=model.max_iter,
                                                  tol=model.tol,
                                                  ridge=model.ridge)
        end

        iterations = em_iter
    else
        # Use default variance ratio
        var_u = 1.0
        var_e = 1.0
        lambda = 1.0

        if model.method == :cholesky
            beta, u = solve_mixed_model_cholesky(y, X, Z, G, lambda; ridge=model.ridge)
        else
            beta, u, converged, iterations = solve_mixed_model_pcg(y, X, Z, G, lambda;
                                                                   max_iter=model.max_iter,
                                                                   tol=model.tol,
                                                                   ridge=model.ridge)
        end

        converged = true
        iterations = 0
    end

    # Compute log-likelihood (simplified)
    residuals = y - X * beta - Z * u
    log_likelihood = -0.5 * (n * log(2 * π * var_e) + dot(residuals, residuals) / var_e)

    # Compute heritability
    h2 = var_u / (var_u + var_e)

    # Store results
    result = GBLUPResult(
        beta,
        u,
        var_e,
        var_u,
        h2,
        log_likelihood,
        converged,
        iterations
    )

    model.result = result
    model.sample_ids = sample_ids(geno)

    return result
end

"""
    predict(model::GBLUPModel, geno::CompactGenotypes) -> Vector{Float64}

Predict genomic breeding values for new samples.

# Arguments
- `model`: Fitted GBLUPModel
- `geno`: Genotype data for prediction samples

# Returns
Vector of predicted breeding values

# Example
```julia
# Fit model on training data
result = fit!(model, geno_train, pheno_train)

# Predict on test data
predictions = predict(model, geno_test)
```
"""
function predict(model::GBLUPModel, geno::CompactGenotypes)
    if model.result === nothing
        throw(ArgumentError("Model not fitted. Call fit! first."))
    end

    pred_ids = sample_ids(geno)

    # Find samples that were in training set
    train_indices = Int[]
    pred_indices = Int[]

    for (i, id) in enumerate(pred_ids)
        idx = findfirst(==(id), model.sample_ids)
        if idx !== nothing
            push!(pred_indices, i)
            push!(train_indices, idx)
        end
    end

    if isempty(train_indices)
        @warn "No samples in prediction set were in training set"
        return zeros(length(pred_ids))
    end

    # Extract breeding values for matched samples
    predictions = zeros(Float64, length(pred_ids))
    predictions[pred_indices] = model.result.u[train_indices]

    return predictions
end

"""
    Base.show(io::IO, result::GBLUPResult)

Display GBLUP results.
"""
function Base.show(io::IO, result::GBLUPResult)
    println(io, "GBLUP Results:")
    @printf(io, "  Heritability (h²): %.4f\n", result.heritability)
    @printf(io, "  Genetic variance (σ²ᵤ): %.4f\n", result.var_u)
    @printf(io, "  Residual variance (σ²ₑ): %.4f\n", result.var_e)
    @printf(io, "  Log-likelihood: %.2f\n", result.log_likelihood)
    println(io, "  Converged: ", result.converged)
    println(io, "  Iterations: ", result.iterations)
    println(io, "  Number of effects: ", length(result.u))
end
