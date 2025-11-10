# src/GenomicProPredict/bayesr_mcmc.jl

"""
    run_bayesr_mcmc(model::BayesRModel, genotypes, phenotypes; kwargs...)

Perform Bayesian variable selection via BayesR MCMC sampling.

Implements Gibbs sampling algorithm for BayesR model, cycling through conditional
distributions for marker effects, component assignments, and variance components.
The algorithm produces samples from the joint posterior distribution enabling
full Bayesian inference including uncertainty quantification and hypothesis testing.

# MCMC Algorithm Details

The Gibbs sampler proceeds as follows for each iteration:

## Step 1: Sample Marker Effects
For each marker j, sample βⱼ from its full conditional distribution:

    βⱼ | rest ~ N(μⱼ, τ²ⱼ)

where:
- μⱼ = (X'ⱼXⱼ + σ²ₑ/σ²ₖ)⁻¹ X'ⱼr  (posterior mean)
- τ²ⱼ = σ²ₑ(X'ⱼXⱼ + σ²ₑ/σ²ₖ)⁻¹  (posterior variance)
- r is the residual vector with marker j effect removed
- σ²ₖ is the variance for current component assignment

This is computed efficiently by updating residuals incrementally rather than
recomputing from scratch.

## Step 2: Sample Component Assignments
For each marker j, sample δⱼ from its full conditional:

    P(δⱼ = k | rest) ∝ πₖ × N(βⱼ | 0, σ²ₖ)

The probability is proportional to the product of the prior mixing proportion and
the likelihood of the current effect under each component's distribution.

## Step 3: Update Variance Components
Sample genetic variance from inverse-gamma posterior:

    σ²ₐ | rest ~ InvGamma(shape', scale')

where shape and scale are updated from prior hyperparameters and current marker
effects. Similarly for residual variance using current residuals.

## Step 4: Update Mixing Proportions
Sample from Dirichlet posterior:

    π | rest ~ Dirichlet(α + n)

where n is the vector of marker counts in each component and α is the prior parameter.

# Arguments
- `model::BayesRModel`: Model specification with priors
- `genotypes::AbstractGenotypeData`: Genotype matrix (n × m)
- `phenotypes::Vector{Float64}`: Phenotype vector (n × 1)

# Keyword Arguments
- `n_iterations::Int = 50000`: Total MCMC iterations
- `burn_in::Int = 10000`: Initial iterations discarded as burn-in
- `thinning::Int = 10`: Keep every k-th sample to reduce autocorrelation
- `save_samples::Bool = false`: Store full MCMC samples (memory intensive)
- `compute_pip::Bool = true`: Calculate posterior inclusion probabilities
- `verbose::Bool = true`: Print progress information
- `convergence_check_interval::Int = 1000`: Iterations between diagnostics

# Returns
Named tuple containing:
- `marker_effects::Vector{Float64}`: Posterior mean marker effects
- `marker_effects_sd::Vector{Float64}`: Posterior standard deviations
- `component_assignments::Vector{Int}`: Modal component assignment per marker
- `pip::Vector{Float64}`: Posterior inclusion probability (P(δⱼ ≠ 1))
- `genetic_variance::Float64`: Posterior mean genetic variance
- `residual_variance::Float64`: Posterior mean residual variance
- `heritability::Float64`: Posterior mean heritability
- `mixing_proportions::Vector{Float64}`: Posterior mean mixing proportions
- `credible_intervals::Matrix{Float64}`: 95% credible intervals (m × 2)
- `mcmc_samples::Union{Nothing, MCMCSamples}`: Full samples if requested
- `diagnostics::MCMCDiagnostics`: Convergence diagnostics

# Convergence Assessment

Monitor convergence through multiple diagnostics:

**Visual Inspection**: Trace plots should show stable mixing without trends
**Effective Sample Size**: Target ESS > 1000 for reliable inference
**Autocorrelation**: Should decay to near-zero within 50 lags
**Gelman-Rubin**: Run multiple chains, R-hat near 1.0 indicates convergence

Typical convergence requires 10,000-20,000 burn-in iterations plus 30,000-80,000
post-burn-in samples depending on dataset size and genetic architecture complexity.

# Examples
```julia
# Standard BayesR analysis
model = BayesRModel()
results = run_bayesr_mcmc(model, genotypes, phenotypes,
                         n_iterations = 50000,
                         burn_in = 10000,
                         thinning = 10)

# Extract high-PIP markers
significant_markers = findall(results.pip .> 0.50)
println("Significant markers: ", length(significant_markers))

# Compute breeding values
gebvs = genotypes * results.marker_effects

# Run multiple chains for convergence assessment
chains = [run_bayesr_mcmc(model, genotypes, phenotypes,
                         n_iterations = 50000,
                         burn_in = 10000)
         for _ in 1:3]

# Check convergence across chains
rhat = compute_gelman_rubin(chains)
println("R-hat for genetic variance: ", rhat.genetic_variance)

# Save full samples for detailed posterior analysis
results_full = run_bayesr_mcmc(model, genotypes, phenotypes,
                              n_iterations = 100000,
                              burn_in = 20000,
                              thinning = 20,
                              save_samples = true)

# Analyze posterior distribution
using StatsPlots
histogram(results_full.mcmc_samples.genetic_variance,
         title = "Posterior Distribution of Genetic Variance",
         xlabel = "σ²ₐ", ylabel = "Frequency")
```

# Performance Optimization

Computational bottleneck is the residual update operation executed m × n_iter times.
Optimization strategies include:

**Vectorization**: Process marker groups simultaneously when possible
**Incremental Updates**: Update residuals rather than recomputing from scratch
**Sparse Operations**: Exploit sparsity in genotype matrices
**GPU Acceleration**: Parallelize residual updates across individuals
**Memory Layout**: Store genotypes in row-major format for coalesced access

With these optimizations, throughput reaches 100-1000 markers/second depending on
sample size and hardware configuration.

# References
- Erbe et al. (2012) BMC Bioinformatics 13:186 (Original BayesR)
- Moser et al. (2015) Nat Genet 47:1385-1392 (BayesR applications)
- Gelman & Rubin (1992) Stat Sci 7:457-472 (Convergence diagnostics)

# See Also
- [`MCMCDiagnostics`](@ref): Convergence assessment tools
- [`posterior_summary`](@ref): Extract posterior statistics
- [`plot_trace`](@ref): Visualize MCMC trajectories
"""
function run_bayesr_mcmc(model::BayesRModel,
                        genotypes::AbstractGenotypeData,
                        phenotypes::Vector{Float64};
                        n_iterations::Int = 50000,
                        burn_in::Int = 10000,
                        thinning::Int = 10,
                        save_samples::Bool = false,
                        compute_pip::Bool = true,
                        verbose::Bool = true,
                        convergence_check_interval::Int = 1000)

    n_individuals, n_markers = size(genotypes)

    verbose && println("="^70)
    verbose && println("BayesR MCMC Analysis")
    verbose && println("="^70)
    verbose && println("Dataset:")
    verbose && println("  Individuals: $n_individuals")
    verbose && println("  Markers: $n_markers")
    verbose && println("MCMC Configuration:")
    verbose && println("  Total iterations: $n_iterations")
    verbose && println("  Burn-in: $burn_in")
    verbose && println("  Thinning: $thinning")
    verbose && println("  Components: $(model.n_components)")
    verbose && println()

    # Initialize MCMC state
    state = initialize_mcmc_state(model, genotypes, phenotypes)

    # Pre-compute X'X for each marker (for efficient sampling)
    verbose && println("Pre-computing marker statistics...")
    XtX = Vector{Float64}(undef, n_markers)
    for j in 1:n_markers
        XtX[j] = compute_marker_self_product(genotypes, j)
    end
    verbose && println("  ✓ Complete")
    verbose && println()

    # Storage for posterior samples
    n_saved_samples = div(n_iterations - burn_in, thinning)

    # Accumulators for posterior statistics
    β_sum = zeros(Float64, n_markers)
    β_sq_sum = zeros(Float64, n_markers)
    inclusion_count = zeros(Int, n_markers)
    σ²_g_sum = 0.0
    σ²_e_sum = 0.0
    π_sum = zeros(Float64, model.n_components)
    component_count = zeros(Int, model.n_components, n_markers)

    # Full sample storage if requested
    if save_samples
        β_samples = zeros(Float64, n_saved_samples, min(n_markers, 1000))
        σ²_g_samples = zeros(Float64, n_saved_samples)
        σ²_e_samples = zeros(Float64, n_saved_samples)
        π_samples = zeros(Float64, n_saved_samples, model.n_components)
    else
        β_samples = nothing
        σ²_g_samples = nothing
        σ²_e_samples = nothing
        π_samples = nothing
    end

    sample_idx = 0

    # MCMC sampling
    verbose && println("Starting MCMC sampling...")
    verbose && println("-"^70)

    for iter in 1:n_iterations
        state.iteration = iter

        # Step 1: Sample marker effects and component assignments
        sample_marker_effects_and_components!(
            state, model, genotypes, XtX
        )

        # Step 2: Sample variance components
        sample_variance_components!(state, model, n_markers)

        # Step 3: Sample mixing proportions
        sample_mixing_proportions!(state, model)

        # Store samples after burn-in
        if iter > burn_in && (iter - burn_in) % thinning == 0
            sample_idx += 1

            # Accumulate for posterior means
            β_sum .+= state.marker_effects
            β_sq_sum .+= state.marker_effects.^2
            inclusion_count .+= (state.component_assignments .!= 1)
            σ²_g_sum += state.genetic_variance
            σ²_e_sum += state.residual_variance
            π_sum .+= state.mixing_proportions

            # Track component assignments
            for j in 1:n_markers
                component_count[state.component_assignments[j], j] += 1
            end

            # Save full samples if requested
            if save_samples && sample_idx <= size(β_samples, 1)
                β_samples[sample_idx, :] .= state.marker_effects[1:min(n_markers, 1000)]
                σ²_g_samples[sample_idx] = state.genetic_variance
                σ²_e_samples[sample_idx] = state.residual_variance
                π_samples[sample_idx, :] .= state.mixing_proportions
            end
        end

        # Progress reporting
        if verbose && (iter % convergence_check_interval == 0 || iter <= 10)
            h² = state.genetic_variance / (state.genetic_variance + state.residual_variance)
            n_nonzero = sum(state.component_assignments .!= 1)

            println("  Iteration $iter:")
            println("    h²: $(round(h², digits=3))")
            println("    σ²_genetic: $(round(state.genetic_variance, digits=2))")
            println("    σ²_residual: $(round(state.residual_variance, digits=2))")
            println("    Non-null markers: $n_nonzero")
            println("    Mixing proportions: [$(join([@sprintf("%.3f", p) for p in state.mixing_proportions], ", "))]")
        end
    end

    verbose && println("-"^70)
    verbose && println("MCMC sampling complete")
    verbose && println()

    # Compute posterior summaries
    β_mean = β_sum ./ n_saved_samples
    β_var = (β_sq_sum ./ n_saved_samples) .- β_mean.^2
    β_sd = sqrt.(max.(β_var, 0.0))

    σ²_g_mean = σ²_g_sum / n_saved_samples
    σ²_e_mean = σ²_e_sum / n_saved_samples
    h²_mean = σ²_g_mean / (σ²_g_mean + σ²_e_mean)
    π_mean = π_sum ./ n_saved_samples

    # Posterior inclusion probabilities
    pip = compute_pip ? (inclusion_count ./ n_saved_samples) : zeros(Float64, n_markers)

    # Modal component assignment
    component_modal = [argmax(component_count[:, j]) for j in 1:n_markers]

    # 95% credible intervals (approximate from mean and SD)
    ci_lower = β_mean .- 1.96 .* β_sd
    ci_upper = β_mean .+ 1.96 .* β_sd
    credible_intervals = hcat(ci_lower, ci_upper)

    verbose && println("Posterior Summaries:")
    verbose && println("  Genetic variance: $(round(σ²_g_mean, digits=2))")
    verbose && println("  Residual variance: $(round(σ²_e_mean, digits=2))")
    verbose && println("  Heritability: $(round(h²_mean, digits=3))")
    verbose && println("  Mean mixing proportions: [$(join([@sprintf("%.3f", p) for p in π_mean], ", "))]")
    verbose && println()
    verbose && println("Variable Selection Results:")
    verbose && println("  Markers with PIP > 0.95: $(sum(pip .> 0.95))")
    verbose && println("  Markers with PIP > 0.50: $(sum(pip .> 0.50))")
    verbose && println("  Markers with PIP > 0.10: $(sum(pip .> 0.10))")
    verbose && println()

    # Construct diagnostics object
    diagnostics = compute_mcmc_diagnostics(
        save_samples ? β_samples : nothing,
        σ²_g_samples,
        σ²_e_samples,
        n_saved_samples
    )

    # Construct samples object if requested
    samples = save_samples ? MCMCSamples(β_samples, σ²_g_samples, σ²_e_samples, π_samples) : nothing

    return (
        marker_effects = β_mean,
        marker_effects_sd = β_sd,
        component_assignments = component_modal,
        pip = pip,
        genetic_variance = σ²_g_mean,
        residual_variance = σ²_e_mean,
        heritability = h²_mean,
        mixing_proportions = π_mean,
        credible_intervals = credible_intervals,
        mcmc_samples = samples,
        diagnostics = diagnostics
    )
end


"""
    sample_marker_effects_and_components!(state, model, genotypes, XtX)

Sample marker effects and component assignments for one MCMC iteration.

This is the computational bottleneck of BayesR, executed once per iteration for each
marker. Efficient implementation is critical for practical application to large datasets.
"""
function sample_marker_effects_and_components!(state::MCMCState,
                                              model::BayesRModel,
                                              genotypes::AbstractGenotypeData,
                                              XtX::Vector{Float64})

    n_markers = length(state.marker_effects)

    for j in 1:n_markers
        # Remove current marker contribution from residuals
        β_old = state.marker_effects[j]
        if abs(β_old) > 1e-10
            add_marker_contribution_to_residuals!(
                state.residuals, genotypes, j, -β_old
            )
        end

        # Compute X'r for marker j
        Xtr = compute_marker_residual_product(genotypes, state.residuals, j)

        # Sample component assignment
        component_probs = compute_component_probabilities(
            Xtr, XtX[j], state.mixing_proportions,
            model.component_variances, state.genetic_variance,
            state.residual_variance
        )

        state.component_assignments[j] = sample_categorical(component_probs)

        # Sample marker effect given component
        k = state.component_assignments[j]
        σ²_k = model.component_variances[k] * state.genetic_variance

        if σ²_k > 1e-10
            # Posterior parameters
            inv_τ² = XtX[j] / state.residual_variance + 1.0 / σ²_k
            τ² = 1.0 / inv_τ²
            μ = τ² * Xtr / state.residual_variance

            # Sample from N(μ, τ²)
            state.marker_effects[j] = μ + sqrt(τ²) * randn()
        else
            state.marker_effects[j] = 0.0
        end

        # Add new marker contribution to residuals
        if abs(state.marker_effects[j]) > 1e-10
            add_marker_contribution_to_residuals!(
                state.residuals, genotypes, j, state.marker_effects[j]
            )
        end
    end
end


"""
    compute_component_probabilities(Xtr, XtX, π, σ²_components, σ²_g, σ²_e)

Compute posterior probabilities for component assignment.

Returns probability vector over components for given marker.
"""
function compute_component_probabilities(Xtr::Float64,
                                        XtX::Float64,
                                        π::Vector{Float64},
                                        σ²_components::Vector{Float64},
                                        σ²_g::Float64,
                                        σ²_e::Float64)

    n_components = length(π)
    log_probs = Vector{Float64}(undef, n_components)

    for k in 1:n_components
        σ²_k = σ²_components[k] * σ²_g

        if σ²_k > 1e-10
            # Log probability for non-zero component
            inv_C = XtX / σ²_e + 1.0 / σ²_k
            C = 1.0 / inv_C

            log_probs[k] = log(π[k]) + 0.5 * log(C) + 0.5 * (Xtr^2 * C / σ²_e)
        else
            # Log probability for zero component
            log_probs[k] = log(π[k])
        end
    end

    # Convert to probabilities (numerically stable)
    max_log_prob = maximum(log_probs)
    probs = exp.(log_probs .- max_log_prob)
    probs ./= sum(probs)

    return probs
end


"""
    sample_variance_components!(state, model, n_markers)

Sample genetic and residual variance from their posterior distributions.
"""
function sample_variance_components!(state::MCMCState,
                                    model::BayesRModel,
                                    n_markers::Int)

    # Sample genetic variance
    # Posterior: InvGamma(shape', scale')
    shape_g = model.prior_shape_genetic + n_markers / 2.0

    scale_g = model.prior_scale_genetic
    for j in 1:n_markers
        k = state.component_assignments[j]
        if model.component_variances[k] > 1e-10
            scale_g += state.marker_effects[j]^2 / (2.0 * model.component_variances[k])
        end
    end

    state.genetic_variance = rand(InverseGamma(shape_g, scale_g))

    # Sample residual variance
    n_individuals = length(state.residuals)
    shape_e = model.prior_shape_residual + n_individuals / 2.0
    scale_e = model.prior_scale_residual + sum(state.residuals.^2) / 2.0

    state.residual_variance = rand(InverseGamma(shape_e, scale_e))
end


"""
    sample_mixing_proportions!(state, model)

Sample mixing proportions from Dirichlet posterior distribution.
"""
function sample_mixing_proportions!(state::MCMCState,
                                   model::BayesRModel)

    # Count markers in each component
    component_counts = [sum(state.component_assignments .== k)
                       for k in 1:model.n_components]

    # Dirichlet posterior parameters
    α_post = model.prior_alpha .+ component_counts

    # Sample from Dirichlet
    state.mixing_proportions = rand(Dirichlet(α_post))
end


# Helper functions for efficient residual updates
function add_marker_contribution_to_residuals!(residuals::Vector{Float64},
                                              genotypes::AbstractGenotypeData,
                                              marker_idx::Int,
                                              β_change::Float64)
    n = length(residuals)
    for i in 1:n
        g = genotypes[i, marker_idx]
        if !ismissing(g)
            residuals[i] -= Float64(g) * β_change
        end
    end
end

function compute_marker_self_product(genotypes::AbstractGenotypeData, marker_idx::Int)
    n = size(genotypes, 1)
    sum_sq = 0.0
    for i in 1:n
        g = genotypes[i, marker_idx]
        if !ismissing(g)
            sum_sq += Float64(g)^2
        end
    end
    return sum_sq
end

function compute_marker_residual_product(genotypes::AbstractGenotypeData,
                                        residuals::Vector{Float64},
                                        marker_idx::Int)
    n = length(residuals)
    sum_prod = 0.0
    for i in 1:n
        g = genotypes[i, marker_idx]
        if !ismissing(g)
            sum_prod += Float64(g) * residuals[i]
        end
    end
    return sum_prod
end

function sample_categorical(probs::Vector{Float64})
    u = rand()
    cumsum_prob = 0.0
    for (k, p) in enumerate(probs)
        cumsum_prob += p
        if u <= cumsum_prob
            return k
        end
    end
    return length(probs)
end