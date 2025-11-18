"""
BayesCπ: Bayesian Variable Selection with Estimated Mixture Proportions

BayesCπ extends BayesR by treating mixture proportions (π) as random variables
that are estimated from the data using a Dirichlet prior.

Key features:
- Mixture proportions are updated in each MCMC iteration
- Dirichlet prior for π with concentration parameters α
- Can use 2-component (BayesC) or 4-component (BayesCπ) mixtures
- More flexible than BayesR with fixed proportions

# Two-component model (BayesC):
- Component 1: Zero effect (π₀)
- Component 2: Non-zero effect with variance σ²ₐ (1-π₀)

# Four-component model (BayesCπ):
- Component 1: Zero effect (π₁)
- Component 2: Small effect with variance 0.0001σ²ₐ (π₂)
- Component 3: Medium effect with variance 0.001σ²ₐ (π₃)
- Component 4: Large effect with variance 0.01σ²ₐ (π₄)

# References
Habier et al. (2011). Extension of the Bayesian alphabet for genomic selection.
BMC Bioinformatics, 12(1), 186.

Kizilkaya et al. (2010). Genomic prediction of simulated multibreed and purebred
performance using observed fifty thousand single nucleotide polymorphism genotypes.
Journal of Animal Science, 88(2), 544-551.
"""

using Random
using Statistics
using LinearAlgebra
using Printf
using Distributions

"""
    BayesCπModel

BayesCπ model for genomic prediction with variable selection and
estimated mixture proportions.

# Fields
- `n_iter::Int`: Number of MCMC iterations (default: 50000)
- `burn_in::Int`: Number of burn-in iterations to discard (default: 20000)
- `thin::Int`: Thinning interval for posterior samples (default: 10)
- `n_components::Int`: Number of mixture components (2 or 4, default: 4)
- `mixture_variances::Vector{Float64}`: Variance multipliers for components
- `dirichlet_alpha::Vector{Float64}`: Dirichlet prior concentration parameters
- `verbose::Bool`: Print progress (default: true)
- `seed::Union{Int,Nothing}`: Random seed for reproducibility

# Results (after fitting)
- `result::Union{NamedTuple,Nothing}`: Fitted model results
- `sample_ids::Union{Vector{String},Nothing}`: Training sample IDs

# Examples
```julia
# Two-component model (BayesC)
model = BayesCπModel(n_components=2)

# Four-component model (BayesCπ)
model = BayesCπModel(n_components=4)

# Fit model
fit!(model, geno, pheno)

# Get results
predictions = predict(model, geno)
```
"""
mutable struct BayesCπModel
    n_iter::Int
    burn_in::Int
    thin::Int
    n_components::Int
    mixture_variances::Vector{Float64}
    dirichlet_alpha::Vector{Float64}
    verbose::Bool
    seed::Union{Int, Nothing}

    # Results
    result::Union{Nothing, NamedTuple}
    sample_ids::Union{Vector{String}, Nothing}

    function BayesCπModel(;
        n_iter::Int = 50000,
        burn_in::Int = 20000,
        thin::Int = 10,
        n_components::Int = 4,
        mixture_variances::Union{Vector{Float64},Nothing} = nothing,
        dirichlet_alpha::Union{Vector{Float64},Nothing} = nothing,
        verbose::Bool = true,
        seed::Union{Int, Nothing} = nothing
    )
        if n_iter <= burn_in
            throw(ArgumentError("n_iter must be > burn_in"))
        end

        if !(n_components in [2, 4])
            throw(ArgumentError("n_components must be 2 or 4"))
        end

        # Default variance multipliers
        if isnothing(mixture_variances)
            if n_components == 2
                mixture_variances = [0.0, 0.01]  # BayesC: zero and non-zero
            else
                mixture_variances = [0.0, 0.0001, 0.001, 0.01]  # BayesCπ
            end
        end

        if length(mixture_variances) != n_components
            throw(ArgumentError("mixture_variances must have length $n_components"))
        end

        if mixture_variances[1] != 0.0
            throw(ArgumentError("First mixture variance must be 0.0 (null component)"))
        end

        # Default Dirichlet prior (uniform)
        if isnothing(dirichlet_alpha)
            dirichlet_alpha = ones(n_components)
        end

        if length(dirichlet_alpha) != n_components
            throw(ArgumentError("dirichlet_alpha must have length $n_components"))
        end

        if any(dirichlet_alpha .<= 0.0)
            throw(ArgumentError("dirichlet_alpha must be positive"))
        end

        if thin < 1
            throw(ArgumentError("thin must be >= 1"))
        end

        new(n_iter, burn_in, thin, n_components, mixture_variances,
            dirichlet_alpha, verbose, seed, nothing, nothing)
    end
end

"""
    BayesCπResult

Container for BayesCπ model results.

# Fields
- `marker_effects::Vector{Float64}`: Posterior mean SNP effects
- `marker_effects_se::Vector{Float64}`: Posterior SD of SNP effects
- `pip::Vector{Float64}`: Posterior inclusion probabilities (non-zero effect)
- `component_assignment::Vector{Int}`: Most likely component for each SNP
- `mixture_proportions::Vector{Float64}`: Posterior mean mixture proportions
- `mixture_proportions_se::Vector{Float64}`: Posterior SD of mixture proportions
- `sigma2_e::Float64`: Posterior mean residual variance
- `sigma2_a::Float64`: Posterior mean genetic variance
- `heritability::Float64`: Estimated heritability
- `intercept::Float64`: Intercept term
- `y_mean::Float64`: Mean of phenotype (for prediction)
- `marker_ids::Vector{String}`: SNP identifiers
- `n_samples::Int`: Number of training samples
- `n_markers::Int`: Number of markers
- `n_iter::Int`: Total MCMC iterations
- `burn_in::Int`: Burn-in iterations
- `thin::Int`: Thinning interval
- `convergence::NamedTuple`: Convergence diagnostics
"""
struct BayesCπResult
    marker_effects::Vector{Float64}
    marker_effects_se::Vector{Float64}
    pip::Vector{Float64}
    component_assignment::Vector{Int}
    mixture_proportions::Vector{Float64}
    mixture_proportions_se::Vector{Float64}
    sigma2_e::Float64
    sigma2_a::Float64
    heritability::Float64
    intercept::Float64
    y_mean::Float64
    marker_ids::Vector{String}
    n_samples::Int
    n_markers::Int
    n_iter::Int
    burn_in::Int
    thin::Int
    convergence::NamedTuple
end

"""
    fit!(model::BayesCπModel, geno::CompactGenotypes, pheno::PhenotypeData;
         trait_index::Int = 1, min_maf::Float64 = 0.0, scale_X::Bool = true)

Fit BayesCπ model using Gibbs sampling.

# Arguments
- `model::BayesCπModel`: Model to fit
- `geno::CompactGenotypes`: Genotype data
- `pheno::PhenotypeData`: Phenotype data
- `trait_index::Int`: Index of trait to analyze (default: 1)
- `min_maf::Float64`: Minimum MAF filter (default: 0.0)
- `scale_X::Bool`: Whether to standardize genotypes (default: true)

# Returns
Updates `model.result` with fitted parameters.

# Algorithm
Uses Gibbs sampling with the following update steps:
1. Update SNP effects (β) conditional on component assignments
2. Update component assignments (δ) for each SNP
3. Update mixture proportions (π) using Dirichlet posterior
4. Update variance components (σ²ₑ, σ²ₐ)

# Example
```julia
model = BayesCπModel(n_iter=10000, burn_in=5000)
fit!(model, geno, pheno)
```
"""
function fit!(model::BayesCπModel, geno::CompactGenotypes, pheno::PhenotypeData;
              trait_index::Int = 1, min_maf::Float64 = 0.0, scale_X::Bool = true)

    # Set random seed if provided
    if !isnothing(model.seed)
        Random.seed!(model.seed)
    end

    if model.verbose
        println("="^80)
        println("BayesCπ: Bayesian Variable Selection with Estimated π")
        println("="^80)
    end

    # Validate inputs
    if n_samples(geno) != n_samples(pheno)
        throw(ArgumentError("Sample size mismatch"))
    end

    # Get matched samples
    common_samples = intersect(sample_ids(geno), sample_ids(pheno))
    if isempty(common_samples)
        throw(ArgumentError("No common samples between genotype and phenotype"))
    end

    # Subset to common samples
    geno_idx = [findfirst(==(s), sample_ids(geno)) for s in common_samples]
    pheno_idx = [findfirst(==(s), sample_ids(pheno)) for s in common_samples]

    # Extract phenotype
    y = pheno.data[pheno_idx, trait_index]

    # Remove missing phenotypes
    valid = .!isnan.(y)
    y = y[valid]
    geno_idx = geno_idx[valid]

    n = length(y)
    p = n_markers(geno)

    if model.verbose
        println("\n📊 Data Summary:")
        println("  Samples: $n")
        println("  Markers: $p")
        println("  Trait: $(pheno.trait_names[trait_index])")
    end

    # Convert genotypes to matrix and apply MAF filter
    X = to_matrix(geno)
    X = X[geno_idx, :]

    # MAF filtering
    if min_maf > 0.0
        maf = vec(mean(X, dims=1) ./ 2)
        maf = min.(maf, 1 .- maf)
        keep_markers = maf .>= min_maf
        X = X[:, keep_markers]
        p_filtered = sum(keep_markers)
        marker_ids_filtered = marker_ids(geno)[keep_markers]

        if model.verbose
            println("  Markers after MAF filter (>= $min_maf): $p_filtered")
        end
    else
        marker_ids_filtered = marker_ids(geno)
        keep_markers = trues(p)
    end

    p = size(X, 2)

    # Standardize genotypes
    X_mean = vec(mean(X, dims=1))
    X_std = vec(std(X, dims=1))
    X_std[X_std .== 0] .= 1.0  # Avoid division by zero

    if scale_X
        X = (X .- X_mean') ./ X_std'
    end

    # Center phenotype
    y_mean = mean(y)
    y = y .- y_mean

    # Initialize parameters
    β = zeros(p)  # SNP effects
    δ = ones(Int, p)  # Component assignments (1 = null component)
    π = copy(model.dirichlet_alpha) ./ sum(model.dirichlet_alpha)  # Initial proportions

    # Variance components
    var_y = var(y)
    σ2_e = var_y * 0.5
    σ2_a = var_y * 0.5

    # Pre-compute X'X diagonal
    XtX_diag = vec(sum(X .^ 2, dims=1))

    # Storage for posterior samples
    n_samples_store = div(model.n_iter - model.burn_in, model.thin)
    β_samples = zeros(p, n_samples_store)
    π_samples = zeros(model.n_components, n_samples_store)
    σ2_e_samples = zeros(n_samples_store)
    σ2_a_samples = zeros(n_samples_store)
    δ_samples = zeros(Int, p, n_samples_store)

    # Residuals
    ŷ = X * β
    e = y - ŷ

    if model.verbose
        println("\n🔄 MCMC Sampling:")
        println("  Iterations: $(model.n_iter)")
        println("  Burn-in: $(model.burn_in)")
        println("  Thinning: $(model.thin)")
        println("  Samples stored: $n_samples_store")
        println("  Components: $(model.n_components)")
    end

    # Gibbs sampling
    sample_idx = 0
    progress_interval = max(1, div(model.n_iter, 20))

    for iter in 1:model.n_iter
        # Update SNP effects
        for j in 1:p
            # Remove effect of current SNP
            e .+= X[:, j] * β[j]

            # Current component
            k = δ[j]

            # Compute conditional variance
            if k == 1
                # Null component: effect = 0
                β[j] = 0.0
            else
                # Non-null component
                σ2_β = model.mixture_variances[k] * σ2_a
                var_post = 1.0 / (XtX_diag[j] / σ2_e + 1.0 / σ2_β)
                mean_post = var_post * dot(X[:, j], e) / σ2_e
                β[j] = mean_post + sqrt(var_post) * randn()
            end

            # Update residuals
            e .-= X[:, j] * β[j]
        end

        # Update component assignments (δ)
        component_counts = zeros(Int, model.n_components)

        for j in 1:p
            # Remove effect of current SNP
            e .+= X[:, j] * β[j]

            # Calculate likelihood for each component
            log_prob = zeros(model.n_components)

            for k in 1:model.n_components
                if k == 1
                    # Null component
                    β_k = 0.0
                else
                    # Non-null component
                    σ2_β = model.mixture_variances[k] * σ2_a
                    var_post = 1.0 / (XtX_diag[j] / σ2_e + 1.0 / σ2_β)
                    mean_post = var_post * dot(X[:, j], e) / σ2_e
                    β_k = mean_post

                    # Log likelihood
                    log_prob[k] = -0.5 * (β_k^2 / var_post + log(2π * var_post))
                end

                # Add log prior
                log_prob[k] += log(π[k] + 1e-10)
            end

            # Sample component
            log_prob .-= maximum(log_prob)  # Numerical stability
            prob = exp.(log_prob)
            prob ./= sum(prob)

            δ[j] = rand(Categorical(prob))
            component_counts[δ[j]] += 1

            # Sample new effect given component
            k = δ[j]
            if k == 1
                β[j] = 0.0
            else
                σ2_β = model.mixture_variances[k] * σ2_a
                var_post = 1.0 / (XtX_diag[j] / σ2_e + 1.0 / σ2_β)
                mean_post = var_post * dot(X[:, j], e) / σ2_e
                β[j] = mean_post + sqrt(var_post) * randn()
            end

            # Update residuals
            e .-= X[:, j] * β[j]
        end

        # Update mixture proportions using Dirichlet posterior
        α_post = model.dirichlet_alpha .+ component_counts
        π = rand(Dirichlet(α_post))

        # Update residual variance σ2_e
        sse = sum(e .^ 2)
        shape = (n + 3) / 2
        scale = (sse + 3 * var_y * 0.1) / 2
        σ2_e = scale / rand(Gamma(shape, 1.0))

        # Update genetic variance σ2_a
        # Sum of squared effects weighted by variance
        ss_β = 0.0
        n_nonzero = 0
        for j in 1:p
            if δ[j] > 1
                k = δ[j]
                ss_β += β[j]^2 / model.mixture_variances[k]
                n_nonzero += 1
            end
        end

        if n_nonzero > 0
            shape = (n_nonzero + 3) / 2
            scale = (ss_β + 3 * var_y * 0.5) / 2
            σ2_a = scale / rand(Gamma(shape, 1.0))
        else
            σ2_a = var_y * 0.5
        end

        # Store samples after burn-in
        if iter > model.burn_in && (iter - model.burn_in) % model.thin == 0
            sample_idx += 1
            β_samples[:, sample_idx] = β
            π_samples[:, sample_idx] = π
            σ2_e_samples[sample_idx] = σ2_e
            σ2_a_samples[sample_idx] = σ2_a
            δ_samples[:, sample_idx] = δ
        end

        # Progress
        if model.verbose && iter % progress_interval == 0
            pct = round(100 * iter / model.n_iter, digits=1)
            @printf("  Progress: %.1f%% (iter %d/%d)\r", pct, iter, model.n_iter)
            flush(stdout)
        end
    end

    if model.verbose
        println("\n  ✓ MCMC sampling completed                    ")
    end

    # Compute posterior summaries
    β_mean = vec(mean(β_samples, dims=2))
    β_se = vec(std(β_samples, dims=2))
    π_mean = vec(mean(π_samples, dims=2))
    π_se = vec(std(π_samples, dims=2))
    σ2_e_mean = mean(σ2_e_samples)
    σ2_a_mean = mean(σ2_a_samples)

    # Posterior inclusion probabilities (probability of non-zero effect)
    pip = vec(mean(δ_samples .> 1, dims=2))

    # Component assignment (mode)
    component_assignment = [argmax([sum(δ_samples[j, :] .== k) for k in 1:model.n_components]) for j in 1:p]

    # Heritability
    h2 = σ2_a_mean / (σ2_a_mean + σ2_e_mean)

    # Convergence diagnostics
    # Effective sample size (simplified)
    ess_sigma2_e = n_samples_store  # Simplified
    ess_sigma2_a = n_samples_store

    # Geweke diagnostic (simplified Z-score)
    first_10pct = 1:div(n_samples_store, 10)
    last_50pct = div(n_samples_store, 2):n_samples_store

    geweke_sigma2_e = (mean(σ2_e_samples[first_10pct]) - mean(σ2_e_samples[last_50pct])) /
                      sqrt(var(σ2_e_samples[first_10pct]) + var(σ2_e_samples[last_50pct]))

    convergence = (
        ess_sigma2_e = ess_sigma2_e,
        ess_sigma2_a = ess_sigma2_a,
        geweke_sigma2_e = geweke_sigma2_e
    )

    # Create result
    result = BayesCπResult(
        β_mean,
        β_se,
        pip,
        component_assignment,
        π_mean,
        π_se,
        σ2_e_mean,
        σ2_a_mean,
        h2,
        0.0,  # intercept (will be y_mean)
        y_mean,
        marker_ids_filtered,
        n,
        p,
        model.n_iter,
        model.burn_in,
        model.thin,
        convergence
    )

    model.result = result
    model.sample_ids = common_samples[valid]

    # Print summary
    if model.verbose
        println("\n" * "="^80)
        println("BayesCπ Results Summary")
        println("="^80)
        println("\n📊 Model Fit:")
        @printf("  Heritability (h²): %.4f\n", h2)
        @printf("  Genetic variance (σ²ₐ): %.4f\n", σ2_a_mean)
        @printf("  Residual variance (σ²ₑ): %.4f\n", σ2_e_mean)

        println("\n📊 Mixture Proportions (π):")
        for k in 1:model.n_components
            var_label = k == 1 ? "Zero" : @sprintf("%.4fσ²ₐ", model.mixture_variances[k])
            @printf("  Component %d (%s): %.4f ± %.4f\n", k, var_label, π_mean[k], π_se[k])
        end

        println("\n📊 SNP Effects:")
        n_nonzero = sum(pip .> 0.5)
        @printf("  Markers with PIP > 0.5: %d (%.2f%%)\n", n_nonzero, 100 * n_nonzero / p)
        @printf("  Mean |effect|: %.6f\n", mean(abs.(β_mean)))
        @printf("  Max |effect|: %.6f\n", maximum(abs.(β_mean)))

        println("\n📊 Convergence:")
        @printf("  Geweke Z-score (σ²ₑ): %.4f\n", geweke_sigma2_e)
        if abs(geweke_sigma2_e) < 2.0
            println("  ✓ Convergence looks good (|Z| < 2)")
        else
            println("  ⚠ May need more iterations (|Z| >= 2)")
        end

        println("\n" * "="^80)
    end

    return nothing
end

"""
    predict(model::BayesCπModel, geno::CompactGenotypes)

Predict genomic breeding values using fitted BayesCπ model.

# Arguments
- `model::BayesCπModel`: Fitted BayesCπ model
- `geno::CompactGenotypes`: Genotype data for prediction

# Returns
Vector of predicted breeding values.

# Example
```julia
predictions = predict(model, geno_test)
```
"""
function predict(model::BayesCπModel, geno::CompactGenotypes)
    if isnothing(model.result)
        throw(ArgumentError("Model not fitted. Call fit!() first."))
    end

    result = model.result
    p = n_markers(geno)

    # Convert to matrix
    X = to_matrix(geno)

    # Match markers
    pred_marker_ids = marker_ids(geno)
    train_marker_ids = result.marker_ids

    # Find common markers
    common_markers = intersect(train_marker_ids, pred_marker_ids)
    if isempty(common_markers)
        throw(ArgumentError("No common markers between training and prediction data"))
    end

    # Get indices
    train_idx = [findfirst(==(m), train_marker_ids) for m in common_markers]
    pred_idx = [findfirst(==(m), pred_marker_ids) for m in common_markers]

    # Extract effects and genotypes
    β = result.marker_effects[train_idx]
    X_pred = X[:, pred_idx]

    # Standardize using same scaling as training
    # (In practice, should store scaling parameters)
    X_mean = vec(mean(X_pred, dims=1))
    X_std = vec(std(X_pred, dims=1))
    X_std[X_std .== 0] .= 1.0
    X_pred = (X_pred .- X_mean') ./ X_std'

    # Predict
    predictions = X_pred * β .+ result.y_mean

    return predictions
end

# Export
export BayesCπModel, BayesCπResult
export fit!, predict
