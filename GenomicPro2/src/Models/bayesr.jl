"""
BayesR: Bayesian Variable Selection Model with Mixture Priors

BayesR uses a mixture of 4 normal distributions to model SNP effects:
- Component 1: Zero effect (proportion π₁)
- Component 2: Small effect with variance 0.0001σ²ₐ (proportion π₂)
- Component 3: Medium effect with variance 0.001σ²ₐ (proportion π₃)
- Component 4: Large effect with variance 0.01σ²ₐ (proportion π₄)

where σ²ₐ is the total genetic variance.

# References
Erbe et al. (2012). Improving accuracy of genomic predictions within and between dairy cattle breeds with imputed high-density single nucleotide polymorphism panels. Journal of Dairy Science, 95(7), 4114-4129.

Moser et al. (2015). Simultaneous discovery, estimation and prediction analysis of complex traits using a Bayesian mixture model. PLoS Genetics, 11(4), e1004969.
"""

using Random
using Statistics
using LinearAlgebra
using Printf
using Distributions

"""
    BayesRModel

BayesR model for genomic prediction with variable selection.

# Fields
- `n_iter::Int`: Number of MCMC iterations (default: 50000)
- `burn_in::Int`: Number of burn-in iterations to discard (default: 20000)
- `thin::Int`: Thinning interval for posterior samples (default: 10)
- `mixture_proportions::Vector{Float64}`: Initial π values for 4 components
- `mixture_variances::Vector{Float64}`: Variance multipliers [0.0, 0.0001, 0.001, 0.01]
- `update_pi::Bool`: Whether to update mixture proportions (default: true)
- `verbose::Bool`: Print progress (default: true)
- `seed::Union{Int,Nothing}`: Random seed for reproducibility

# Results (after fitting)
- `result::Union{BayesRResult,Nothing}`: Fitted model results
- `sample_ids::Union{Vector{String},Nothing}`: Training sample IDs
"""
mutable struct BayesRModel
    n_iter::Int
    burn_in::Int
    thin::Int
    mixture_proportions::Vector{Float64}
    mixture_variances::Vector{Float64}
    update_pi::Bool
    verbose::Bool
    seed::Union{Int, Nothing}

    # Results
    result::Union{Nothing, NamedTuple}
    sample_ids::Union{Vector{String}, Nothing}

    function BayesRModel(;
        n_iter::Int = 50000,
        burn_in::Int = 20000,
        thin::Int = 10,
        mixture_proportions::Vector{Float64} = [0.50, 0.30, 0.15, 0.05],
        mixture_variances::Vector{Float64} = [0.0, 0.0001, 0.001, 0.01],
        update_pi::Bool = true,
        verbose::Bool = true,
        seed::Union{Int, Nothing} = nothing
    )
        if n_iter <= burn_in
            throw(ArgumentError("n_iter must be > burn_in"))
        end

        if length(mixture_proportions) != 4
            throw(ArgumentError("mixture_proportions must have length 4"))
        end

        if !isapprox(sum(mixture_proportions), 1.0, atol=1e-6)
            throw(ArgumentError("mixture_proportions must sum to 1.0"))
        end

        if any(mixture_proportions .< 0.0)
            throw(ArgumentError("mixture_proportions must be non-negative"))
        end

        if length(mixture_variances) != 4
            throw(ArgumentError("mixture_variances must have length 4"))
        end

        if mixture_variances[1] != 0.0
            throw(ArgumentError("First mixture variance must be 0.0 (null component)"))
        end

        if thin < 1
            throw(ArgumentError("thin must be >= 1"))
        end

        new(n_iter, burn_in, thin, mixture_proportions, mixture_variances,
            update_pi, verbose, seed, nothing, nothing)
    end
end

"""
    BayesRResult

Results from fitting a BayesR model.

# Fields
- `marker_effects::Vector{Float64}`: Posterior mean SNP effects
- `marker_effects_sd::Vector{Float64}`: Posterior SD of SNP effects
- `marker_pip::Vector{Float64}`: Posterior inclusion probability (PIP) for each SNP
- `marker_components::Matrix{Float64}`: Posterior prob. for each component (m × 4)
- `genetic_variance::Float64`: Posterior mean genetic variance
- `residual_variance::Float64`: Posterior mean residual variance
- `heritability::Float64`: Posterior mean heritability
- `mixture_proportions::Vector{Float64}`: Posterior mean mixture proportions
- `intercept::Float64`: Model intercept
- `gebv_train::Vector{Float64}`: Genomic EBVs for training samples
- `n_samples::Int`: Number of training samples
- `n_markers::Int`: Number of markers
- `n_iter::Int`: Total MCMC iterations
- `burn_in::Int`: Burn-in iterations
- `n_saved::Int`: Number of posterior samples saved
"""
struct BayesRResult
    marker_effects::Vector{Float64}
    marker_effects_sd::Vector{Float64}
    marker_pip::Vector{Float64}
    marker_components::Matrix{Float64}
    genetic_variance::Float64
    residual_variance::Float64
    heritability::Float64
    mixture_proportions::Vector{Float64}
    intercept::Float64
    gebv_train::Vector{Float64}
    n_samples::Int
    n_markers::Int
    n_iter::Int
    burn_in::Int
    n_saved::Int
end

function Base.show(io::IO, result::BayesRResult)
    println(io, "BayesR Model Results")
    println(io, "=" ^60)
    println(io, "  Samples: $(result.n_samples)")
    println(io, "  Markers: $(result.n_markers)")
    println(io, "  MCMC iterations: $(result.n_iter)")
    println(io, "  Burn-in: $(result.burn_in)")
    println(io, "  Posterior samples: $(result.n_saved)")
    println(io, "")
    println(io, "Variance Components:")
    @printf(io, "  Genetic variance (σ²ₐ): %.4f\n", result.genetic_variance)
    @printf(io, "  Residual variance (σ²ₑ): %.4f\n", result.residual_variance)
    @printf(io, "  Heritability (h²): %.4f\n", result.heritability)
    println(io, "")
    println(io, "Mixture Proportions:")
    @printf(io, "  π₁ (zero): %.4f\n", result.mixture_proportions[1])
    @printf(io, "  π₂ (small): %.4f\n", result.mixture_proportions[2])
    @printf(io, "  π₃ (medium): %.4f\n", result.mixture_proportions[3])
    @printf(io, "  π₄ (large): %.4f\n", result.mixture_proportions[4])
    println(io, "")
    println(io, "SNP Effects:")
    n_nonzero = sum(result.marker_pip .> 0.5)
    @printf(io, "  Non-zero effects (PIP > 0.5): %d (%.1f%%)\n",
            n_nonzero, 100 * n_nonzero / result.n_markers)
    @printf(io, "  Mean |effect|: %.6f\n", mean(abs.(result.marker_effects)))
    @printf(io, "  Max |effect|: %.6f\n", maximum(abs.(result.marker_effects)))
    println(io, "=" ^60)
end

"""
    fit!(model::BayesRModel, geno::CompactGenotypes, pheno::PhenotypeData;
         trait_index::Int=1, min_maf::Float64=0.0, scale_X::Bool=true)

Fit BayesR model using Gibbs sampling.

# Arguments
- `model::BayesRModel`: BayesR model to fit
- `geno::CompactGenotypes`: Genotype data
- `pheno::PhenotypeData`: Phenotype data
- `trait_index::Int`: Index of trait to analyze (default: 1)
- `min_maf::Float64`: Minimum MAF threshold for marker filtering (default: 0.0)
- `scale_X::Bool`: Whether to scale genotypes to unit variance (default: true)

# Returns
Fitted BayesRModel with results stored in `model.result`

# Algorithm
Uses Gibbs sampling to iteratively sample from conditional distributions:
1. Sample SNP component assignments from mixture
2. Sample SNP effects given components
3. Sample variance components
4. Sample mixture proportions (if update_pi=true)

# Example
```julia
model = BayesRModel(n_iter=10000, burn_in=5000)
fit!(model, geno, pheno)
gebv = predict(model, geno)
```
"""
function fit!(model::BayesRModel, geno::CompactGenotypes, pheno::PhenotypeData;
              trait_index::Int = 1, min_maf::Float64 = 0.0, scale_X::Bool = true)

    # Set random seed if specified
    if !isnothing(model.seed)
        Random.seed!(model.seed)
    end

    if model.verbose
        println("\n" * "="^70)
        println("BayesR Model Fitting")
        println("="^70)
    end

    # Get matched data
    common_ids = intersect(geno.sample_ids, pheno.sample_ids)
    if isempty(common_ids)
        throw(ArgumentError("No common samples between genotype and phenotype data"))
    end

    geno_idx = indexin(common_ids, geno.sample_ids)
    pheno_idx = indexin(common_ids, pheno.sample_ids)

    geno_subset = subset_samples(geno, geno_idx)
    y = pheno.values[pheno_idx, trait_index]

    # Filter markers by MAF
    if min_maf > 0.0
        maf = minor_allele_frequency(geno_subset)
        keep_markers = findall(maf .>= min_maf)

        if isempty(keep_markers)
            throw(ArgumentError("No markers pass MAF threshold of $min_maf"))
        end

        geno_subset = subset_markers(geno_subset, keep_markers)
    end

    # Convert to matrix
    X = to_matrix(geno_subset; impute=true)
    n, m = size(X)

    if model.verbose
        println("  Samples: $n")
        println("  Markers: $m")
        println("  Trait: $(pheno.trait_names[trait_index])")
        println("  MAF threshold: $min_maf")
    end

    # Center and scale
    X_means = vec(mean(X, dims=1))
    X = X .- X_means'

    if scale_X
        X_sds = vec(std(X, dims=1))
        X_sds[X_sds .< 1e-10] .= 1.0  # Avoid division by zero
        X = X ./ X_sds'
    else
        X_sds = ones(m)
    end

    # Center phenotypes
    y_mean = mean(y)
    y = y .- y_mean

    # Initialize parameters
    β = zeros(m)  # SNP effects
    δ = ones(Int, m)  # Component assignments (1-4)
    π = copy(model.mixture_proportions)
    σ²ₐ = var(y) * 0.5  # Genetic variance
    σ²ₑ = var(y) * 0.5  # Residual variance

    # Compute column-wise variances for efficient sampling
    x_var = vec(sum(X.^2, dims=1))

    # Storage for posterior samples
    n_save = div(model.n_iter - model.burn_in, model.thin)
    β_samples = zeros(m, n_save)
    σ²ₐ_samples = zeros(n_save)
    σ²ₑ_samples = zeros(n_save)
    π_samples = zeros(4, n_save)
    δ_counts = zeros(Int, m, 4)

    # Current fitted values
    μ = X * β

    if model.verbose
        println("\n  MCMC Settings:")
        println("    Iterations: $(model.n_iter)")
        println("    Burn-in: $(model.burn_in)")
        println("    Thinning: $(model.thin)")
        println("    Posterior samples: $n_save")
        println("\n  Starting Gibbs sampling...")
    end

    # Gibbs sampler
    sample_idx = 1
    for iter in 1:model.n_iter
        # Update SNP effects and component assignments
        for j in 1:m
            # Remove current SNP effect
            μ .-= X[:, j] * β[j]

            # Residuals
            r = y - μ

            # Sample component assignment
            log_probs = zeros(4)
            for k in 1:4
                σ²_k = model.mixture_variances[k] * σ²ₐ

                if σ²_k == 0.0
                    # Component 1: zero effect
                    log_probs[k] = log(π[k]) - 0.5 * dot(r, r) / σ²ₑ
                else
                    # Components 2-4: non-zero effects
                    v = 1.0 / (1.0 / σ²_k + x_var[j] / σ²ₑ)
                    m_k = v * dot(X[:, j], r) / σ²ₑ

                    log_probs[k] = log(π[k]) + 0.5 * log(v) + 0.5 * m_k^2 / v - 0.5 * dot(r, r) / σ²ₑ
                end
            end

            # Normalize and sample
            log_probs .-= maximum(log_probs)  # Numerical stability
            probs = exp.(log_probs)
            probs ./= sum(probs)

            δ[j] = rand(Categorical(probs))

            # Sample effect given component
            σ²_k = model.mixture_variances[δ[j]] * σ²ₐ

            if σ²_k == 0.0
                β[j] = 0.0
            else
                v = 1.0 / (1.0 / σ²_k + x_var[j] / σ²ₑ)
                m_k = v * dot(X[:, j], r) / σ²ₑ
                β[j] = rand(Normal(m_k, sqrt(v)))
            end

            # Add back updated effect
            μ .+= X[:, j] * β[j]
        end

        # Update genetic variance
        β_nonzero = β[δ .!= 1]
        δ_nonzero = δ[δ .!= 1]

        if !isempty(β_nonzero)
            SS = sum(β_nonzero[i]^2 / model.mixture_variances[δ_nonzero[i]]
                     for i in 1:length(β_nonzero))
            df = length(β_nonzero) + 3  # Prior df
            scale = SS + 0.002 * var(y)  # Prior scale

            σ²ₐ = rand(InverseGamma(df/2, scale/2))
        end

        # Update residual variance
        r = y - μ
        SS_e = dot(r, r)
        df_e = n + 3
        scale_e = SS_e + 0.002 * var(y)

        σ²ₑ = rand(InverseGamma(df_e/2, scale_e/2))

        # Update mixture proportions
        if model.update_pi
            counts = [sum(δ .== k) for k in 1:4]
            α = counts .+ 1.0  # Dirichlet prior
            π = rand(Dirichlet(α))
        end

        # Store samples after burn-in
        if iter > model.burn_in && (iter - model.burn_in) % model.thin == 0
            β_samples[:, sample_idx] = β
            σ²ₐ_samples[sample_idx] = σ²ₐ
            σ²ₑ_samples[sample_idx] = σ²ₑ
            π_samples[:, sample_idx] = π

            for j in 1:m
                δ_counts[j, δ[j]] += 1
            end

            sample_idx += 1
        end

        # Progress
        if model.verbose && iter % 10000 == 0
            @printf("    Iteration %6d/%d  h² = %.4f  Non-zero: %d (%.1f%%)\n",
                    iter, model.n_iter, σ²ₐ / (σ²ₐ + σ²ₑ),
                    sum(δ .!= 1), 100 * sum(δ .!= 1) / m)
        end
    end

    # Compute posterior summaries
    β_mean = vec(mean(β_samples, dims=2))
    β_sd = vec(std(β_samples, dims=2))

    # Unscale effects
    if scale_X
        β_mean = β_mean ./ X_sds
        β_sd = β_sd ./ X_sds
    end

    # Component probabilities
    component_probs = δ_counts ./ n_save
    pip = 1.0 .- component_probs[:, 1]  # Posterior inclusion probability

    # Variance components
    σ²ₐ_mean = mean(σ²ₐ_samples)
    σ²ₑ_mean = mean(σ²ₑ_samples)
    h² = σ²ₐ_mean / (σ²ₐ_mean + σ²ₑ_mean)

    # Mixture proportions
    π_mean = vec(mean(π_samples, dims=2))

    # Compute GEBVs on training data
    X_original = to_matrix(geno_subset; impute=true)
    gebv_train = (X_original .- X_means') * β_mean

    if model.verbose
        println("\n  Gibbs sampling complete!")
        println("\n" * "="^70)
        println("Results Summary")
        println("="^70)
        @printf("  h² = %.4f\n", h²)
        @printf("  σ²ₐ = %.4f\n", σ²ₐ_mean)
        @printf("  σ²ₑ = %.4f\n", σ²ₑ_mean)
        println("\n  Mixture proportions:")
        @printf("    π₁ (zero): %.4f\n", π_mean[1])
        @printf("    π₂ (small): %.4f\n", π_mean[2])
        @printf("    π₃ (medium): %.4f\n", π_mean[3])
        @printf("    π₄ (large): %.4f\n", π_mean[4])
        println("\n  SNP effects:")
        @printf("    Non-zero (PIP > 0.5): %d (%.1f%%)\n",
                sum(pip .> 0.5), 100 * sum(pip .> 0.5) / m)
        println("="^70)
    end

    # Store results
    model.result = BayesRResult(
        β_mean, β_sd, pip, component_probs,
        σ²ₐ_mean, σ²ₑ_mean, h², π_mean,
        y_mean, gebv_train,
        n, m, model.n_iter, model.burn_in, n_save
    )

    model.sample_ids = common_ids

    return model
end

"""
    predict(model::BayesRModel, geno::CompactGenotypes) -> Vector{Float64}

Predict genomic breeding values using fitted BayesR model.

# Arguments
- `model::BayesRModel`: Fitted BayesR model
- `geno::CompactGenotypes`: Genotype data for prediction

# Returns
Vector of predicted breeding values

# Example
```julia
gebv = predict(model, geno_test)
```
"""
function predict(model::BayesRModel, geno::CompactGenotypes)
    if isnothing(model.result)
        throw(ArgumentError("Model must be fitted before prediction. Call fit!() first."))
    end

    result = model.result

    # Get marker effects in original order
    β = result.marker_effects

    # Match markers
    marker_idx = indexin(geno.marker_ids, geno.marker_ids)

    if any(isnothing.(marker_idx))
        @warn "Some markers in prediction data not found in training data"
    end

    # Convert to matrix
    X = to_matrix(geno; impute=true)

    # Compute breeding values
    gebv = X * β .+ result.intercept

    return gebv
end

# Export
export BayesRModel, BayesRResult, fit!, predict
