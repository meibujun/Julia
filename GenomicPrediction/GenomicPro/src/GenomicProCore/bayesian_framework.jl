# src/GenomicProPredict/bayesian_framework.jl

"""
    AbstractBayesianModel

Abstract base type for Bayesian genomic prediction models.

Bayesian methods for genomic prediction offer several advantages over frequentist
approaches such as GBLUP. They provide full posterior distributions enabling principled
uncertainty quantification, perform automatic variable selection identifying markers
with large effects, incorporate prior biological knowledge through informative priors,
and naturally handle complex genetic architectures through mixture models.

# Bayesian Model Philosophy

Traditional GBLUP assumes all markers contribute equally to genetic variance, which
conflicts with biological reality where most markers have negligible effects while
a small proportion harbor QTLs with substantial impact. Bayesian variable selection
addresses this through mixture priors that explicitly model heterogeneous effect
distributions, enabling data-driven identification of important genomic regions.

# Common Bayesian Genomic Prediction Models

## BayesA
Assumes all markers have non-zero effects with marker-specific variances following
scaled inverse chi-square distributions. Appropriate for traits with many moderate-
effect loci but may overfit with highly polygenic architectures.

## BayesB
Introduces point mass at zero allowing markers to have exactly zero effect with
probability π. The remaining 1-π proportion follows heavy-tailed distribution.
Effective for traits with sparse genetic architecture (few large-effect QTLs).

## BayesC and BayesCπ
Similar to BayesB but uses common variance for non-zero effects rather than marker-
specific variances. BayesCπ estimates π from data rather than fixing a priori.

## BayesR
Mixture of four normal distributions representing zero, small, medium, and large
effects. Provides flexible modeling of complex architectures without requiring
pre-specification of genetic architecture sparsity.

## BayesRC
Extends BayesR by incorporating functional genomic annotations. Annotation information
(regulatory regions, conservation scores, eQTL evidence) influences mixing proportions
through logistic regression, improving power for detecting causal variants.

# Interface Requirements

Concrete Bayesian model types must implement:
- `initialize_parameters(model, data)`: Set starting values for MCMC
- `gibbs_sample!(model, data, state)`: Perform one MCMC iteration
- `update_hyperparameters!(model, state)`: Update variance components and priors
- `compute_posterior_summaries(model, samples)`: Calculate posterior statistics

# Computational Challenges

Bayesian methods are computationally intensive, typically requiring 50,000-100,000
MCMC iterations for convergence. The primary bottleneck is the residual update
operation executed millions of times. Efficient implementation requires careful
algorithm design, vectorized operations where possible, and GPU acceleration for
large datasets.

# Examples
```julia
# Define BayesR model
model = BayesRModel(
    n_components = 4,
    component_variances = [0.0, 0.0001, 0.001, 0.01]
)

# Run MCMC sampling
results = run_mcmc(model, genotypes, phenotypes,
                  n_iterations = 50000,
                  burn_in = 10000,
                  thinning = 10)

# Extract posterior summaries
marker_effects = results.posterior_means
pip = results.posterior_inclusion_probabilities
credible_intervals = results.credible_intervals
```

# See Also
- [`BayesRModel`](@ref): Mixture model with four components
- [`BayesRCModel`](@ref): Annotation-informed variable selection
- [`MCMCDiagnostics`](@ref): Convergence assessment tools
"""
abstract type AbstractBayesianModel end


"""
    BayesRModel <: AbstractBayesianModel

BayesR model implementing mixture-of-normals variable selection.

BayesR assumes each marker effect follows a mixture of four normal distributions
representing different effect size categories. This flexible framework accommodates
diverse genetic architectures without requiring prior specification of architecture
sparsity or effect size distribution.

# Mathematical Model

For marker j, the effect βⱼ follows:

    βⱼ | δⱼ=k ~ N(0, σ²ₖ)
    P(δⱼ = k) = πₖ

where δⱼ indicates the mixture component (k ∈ {1,2,3,4}), σ²ₖ is the variance for
component k, and πₖ is the mixing proportion. Standard parameterization uses:

- Component 1: σ²₁ = 0 (no effect)
- Component 2: σ²₂ = 0.0001 × σ²ₐ (small effect)
- Component 3: σ²₃ = 0.001 × σ²ₐ (medium effect)
- Component 4: σ²₄ = 0.01 × σ²ₐ (large effect)

where σ²ₐ is the total additive genetic variance.

# Prior Distributions

Mixing proportions: π ~ Dirichlet(α) with α = [1, 1, 1, 1] (uniform prior)
Genetic variance: σ²ₐ ~ InvGamma(shape, scale) (weakly informative)
Residual variance: σ²ₑ ~ InvGamma(shape, scale) (weakly informative)

# Gibbs Sampling Algorithm

The MCMC algorithm alternates between sampling marker effects, component assignments,
and variance components:

1. **Sample marker effects** given current component assignments and variances
2. **Sample component assignments** given current effects and mixing proportions
3. **Update genetic variance** from marker effects and component assignments
4. **Update residual variance** from current residuals
5. **Update mixing proportions** from component assignment counts

Each step uses conjugate priors enabling efficient Gibbs sampling without requiring
Metropolis-Hastings acceptance steps.

# Fields
- `n_components::Int`: Number of mixture components (typically 4)
- `component_variances::Vector{Float64}`: Relative variance for each component
- `prior_alpha::Vector{Float64}`: Dirichlet prior hyperparameters
- `prior_shape_genetic::Float64`: Genetic variance prior shape parameter
- `prior_scale_genetic::Float64`: Genetic variance prior scale parameter
- `prior_shape_residual::Float64`: Residual variance prior shape parameter
- `prior_scale_residual::Float64`: Residual variance prior scale parameter

# Examples
```julia
# Standard BayesR with default settings
model = BayesRModel(
    n_components = 4,
    component_variances = [0.0, 0.0001, 0.001, 0.01]
)

# Custom prior specification for highly polygenic trait
model_polygenic = BayesRModel(
    n_components = 4,
    component_variances = [0.0, 0.0001, 0.001, 0.005],
    prior_alpha = [2.0, 1.0, 1.0, 0.5]  # Favor smaller effects
)

# Run MCMC
results = run_bayesr_mcmc(model, genotypes, phenotypes,
                         n_iterations = 50000,
                         burn_in = 10000)

# Posterior analysis
println("Genetic variance: ", results.genetic_variance)
println("Heritability: ", results.heritability)
println("Markers in each component:")
for k in 1:4
    n_markers = sum(results.component_assignments .== k)
    println("  Component $k: $n_markers")
end
```

# Interpretation Guidelines

## Mixing Proportions
The posterior mean mixing proportions reveal trait genetic architecture:
- High π₁ (>70%): Sparse architecture with few causal variants
- High π₂ or π₃ (>20%): Polygenic with many small-to-moderate effects
- High π₄ (>5%): Presence of large-effect QTLs

## Posterior Inclusion Probability (PIP)
PIP quantifies evidence for non-zero marker effect:
- PIP > 0.95: Strong evidence for association
- PIP 0.50-0.95: Moderate evidence
- PIP < 0.10: Weak evidence

## Effect Size Estimates
Posterior mean effects represent expected marker contributions accounting for
uncertainty in component assignment. Large effects with high PIP identify
priority candidates for validation and functional characterization.

# Computational Performance

Typical MCMC performance on modern hardware:

| Individuals | Markers | Iterations | CPU Time | GPU Time |
|-------------|---------|------------|----------|----------|
| 5,000       | 50,000  | 50,000     | 6 hours  | 30 min   |
| 10,000      | 50,000  | 50,000     | 20 hours | 90 min   |
| 20,000      | 100,000 | 50,000     | 4 days   | 6 hours  |

GPU acceleration provides 10-50× speedup by parallelizing residual updates,
the primary computational bottleneck.

# References
- Erbe et al. (2012) BMC Bioinformatics 13:186
- Moser et al. (2015) Nat Genet 47:1385-1392
- MacLeod et al. (2016) Genetics 203:973-983

# See Also
- [`run_bayesr_mcmc`](@ref): MCMC sampling implementation
- [`BayesRCModel`](@ref): Annotation-informed extension
- [`posterior_summary`](@ref): Extract posterior statistics
"""
struct BayesRModel <: AbstractBayesianModel
    n_components::Int
    component_variances::Vector{Float64}
    prior_alpha::Vector{Float64}
    prior_shape_genetic::Float64
    prior_scale_genetic::Float64
    prior_shape_residual::Float64
    prior_scale_residual::Float64

    function BayesRModel(;
                        n_components::Int = 4,
                        component_variances::Vector{Float64} = [0.0, 0.0001, 0.001, 0.01],
                        prior_alpha::Vector{Float64} = ones(4),
                        prior_shape_genetic::Float64 = 2.0,
                        prior_scale_genetic::Float64 = 1.0,
                        prior_shape_residual::Float64 = 2.0,
                        prior_scale_residual::Float64 = 1.0)

        @assert n_components == length(component_variances) "n_components must match length of component_variances"
        @assert n_components == length(prior_alpha) "n_components must match length of prior_alpha"
        @assert all(component_variances .>= 0.0) "component_variances must be non-negative"
        @assert component_variances[1] == 0.0 "First component must have zero variance"

        new(n_components, component_variances, prior_alpha,
            prior_shape_genetic, prior_scale_genetic,
            prior_shape_residual, prior_scale_residual)
    end
end


"""
    MCMCState

Container for MCMC sampling state across iterations.

Maintains current parameter values, sufficient statistics for parameter updates,
and diagnostic information for convergence monitoring. Separating state from model
definition enables multiple chains and parallel sampling strategies.

# Fields
- `marker_effects::Vector{Float64}`: Current marker effect estimates (m × 1)
- `component_assignments::Vector{Int}`: Mixture component per marker (m × 1)
- `genetic_variance::Float64`: Current additive genetic variance σ²ₐ
- `residual_variance::Float64`: Current residual variance σ²ₑ
- `mixing_proportions::Vector{Float64}`: Current mixing proportions π (k × 1)
- `residuals::Vector{Float64}`: Current residuals y - Xβ (n × 1)
- `iteration::Int`: Current MCMC iteration number
"""
mutable struct MCMCState
    marker_effects::Vector{Float64}
    component_assignments::Vector{Int}
    genetic_variance::Float64
    residual_variance::Float64
    mixing_proportions::Vector{Float64}
    residuals::Vector{Float64}
    iteration::Int
end


"""
    initialize_mcmc_state(model::BayesRModel, genotypes, phenotypes)

Initialize MCMC sampling state with sensible starting values.

Starting values significantly impact MCMC efficiency. Poor initialization may require
extended burn-in or cause convergence to local modes. This function uses moment-based
estimates providing reasonable starting points across diverse datasets.

# Initialization Strategy

1. **Marker effects**: Initialize all to zero (β = 0)
2. **Component assignments**: All markers in null component (δ = 1)
3. **Residuals**: r = y - ȳ (centered phenotypes)
4. **Genetic variance**: σ²ₐ = 0.5 × var(y)
5. **Residual variance**: σ²ₑ = 0.5 × var(y)
6. **Mixing proportions**: π = uniform across components

These conservative starting values allow the MCMC to explore the posterior without
strong influence from potentially mis-specified initial conditions.

# Arguments
- `model::BayesRModel`: Bayesian model specification
- `genotypes::AbstractGenotypeData`: Genotype matrix (n × m)
- `phenotypes::Vector{Float64}`: Phenotype vector (n × 1)

# Returns
- `MCMCState`: Initialized state object ready for sampling
"""
function initialize_mcmc_state(model::BayesRModel,
                              genotypes::AbstractGenotypeData,
                              phenotypes::Vector{Float64})

    n_individuals, n_markers = size(genotypes)

    # Initialize marker effects to zero
    marker_effects = zeros(Float64, n_markers)

    # All markers start in null component
    component_assignments = ones(Int, n_markers)

    # Initialize variances from phenotypic variance
    phenotypic_var = var(phenotypes)
    genetic_variance = 0.5 * phenotypic_var
    residual_variance = 0.5 * phenotypic_var

    # Uniform mixing proportions
    mixing_proportions = ones(Float64, model.n_components) ./ model.n_components

    # Initialize residuals as centered phenotypes
    residuals = phenotypes .- mean(phenotypes)

    return MCMCState(
        marker_effects,
        component_assignments,
        genetic_variance,
        residual_variance,
        mixing_proportions,
        residuals,
        0
    )
end