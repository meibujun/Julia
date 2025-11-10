# src/GenomicProPredict/selection_index.jl

"""
    SelectionIndex

Structure representing optimal linear combination of traits for breeding decisions.

Selection indices provide a rigorous framework for optimizing genetic progress across
multiple traits simultaneously, accounting for genetic correlations, relative economic
importance, and heritability differences. The theory was developed to address the
fundamental challenge in breeding programs where improvement is desired across multiple
traits that may exhibit genetic correlations and require balancing competing objectives
to maximize overall economic merit.

# Mathematical Foundation

The selection index constructs a linear combination of breeding values that maximizes
the correlation between the index and a breeding objective, often termed the aggregate
genotype. For traits with breeding values u₁, u₂, ..., uₙ and economic weights a₁,
a₂, ..., aₙ, the aggregate genotype represents total genetic merit:

    H = a₁u₁ + a₂u₂ + ... + aₙuₙ

The selection index I combines observed or predicted trait values to approximate H:

    I = b₁x₁ + b₂x₂ + ... + bₙxₙ

where b represents index weights and x denotes trait observations or genomic predictions.
The optimal index weights maximize the correlation between I and H, derived by solving:

    Pb = Ga

where P represents the phenotypic covariance matrix among information sources, G
denotes the genetic covariance matrix among traits in the breeding objective, and a
contains the economic weights. This system yields index weights that optimally combine
available information to predict genetic merit.

# Economic Weight Specification

Economic weights quantify the change in profit per unit change in each trait, holding
all other traits constant. These weights require careful derivation from bioeconomic
models that simulate production systems and calculate marginal economic values. For
livestock production, economic weights account for revenue from product sales, costs
of feed and management, and impacts on production efficiency and product quality.

The specification of economic weights profoundly influences selection outcomes and
breeding program success. Weights that accurately reflect economic realities ensure
that genetic progress translates into improved profitability for producers. Conversely,
misspecified weights lead to suboptimal selection decisions that may improve traits
with limited economic benefit while neglecting traits with substantial value. Regular
updating of economic weights maintains alignment between breeding objectives and
evolving market conditions, production technologies, and consumer preferences.

# Index Properties and Interpretation

The selection index provides several useful properties for breeding program management.
The correlation between the index and aggregate genotype, termed index accuracy,
quantifies how effectively the index predicts true genetic merit. Higher accuracy
values approaching unity indicate that selection on the index will efficiently improve
the breeding objective. The expected response in each component trait per unit of
selection intensity on the index enables prediction of correlated responses and
verification that selection will move traits in desired directions.

Selection indices naturally account for genetic correlations between traits through
the genetic covariance matrix in the index equations. When traits exhibit positive
genetic correlation, the index assigns weights that leverage this correlation to
improve both traits simultaneously. For negatively correlated traits representing
biological trade-offs, the index balances competing objectives based on their relative
economic importance and the strength of the genetic antagonism.

# Fields
- `traits::Vector{String}`: Trait names included in the index
- `index_weights::Vector{Float64}`: Optimal weights for combining traits
- `economic_weights::Vector{Float64}`: Economic values per trait
- `accuracy::Float64`: Correlation between index and aggregate genotype
- `expected_responses::Vector{Float64}`: Predicted response per trait
- `genetic_covariance::Matrix{Float64}`: Genetic covariances among traits

# Examples
```julia
# Construct selection index for dairy cattle
traits = ["milk_yield", "protein_percent", "fat_percent", "fertility", "longevity"]

economic_weights = [
    0.20,   # Milk yield (dollars per kg)
    4.50,   # Protein percent (dollars per percentage point)
    2.80,   # Fat percent (dollars per percentage point)
    15.00,  # Fertility (dollars per percentage point conception rate)
    100.00  # Longevity (dollars per lactation)
]

# Create index from multi-trait model results
index = construct_selection_index(
    genetic_covariance = results.genetic_covariance,
    phenotypic_covariance = results.phenotypic_covariance,
    economic_weights = economic_weights,
    traits = traits
)

println("Selection Index Weights:")
for (trait, weight) in zip(traits, index.index_weights)
    println("  $trait: $(round(weight, digits=3))")
end

println("\nIndex Accuracy: $(round(index.accuracy, digits=3))")

# Apply index to rank selection candidates
candidates_index_values = candidates_gebvs * index.index_weights
top_selections = sortperm(candidates_index_values, rev=true)[1:100]

# Predict correlated responses
println("\nExpected Responses (per unit selection intensity):")
for (trait, response) in zip(traits, index.expected_responses)
    println("  $trait: $(round(response, digits=2))")
end
```

# Restricted Selection Indices

Standard selection indices optimize progress in the aggregate genotype without
constraints on individual trait responses. However, breeding programs sometimes require
restricting change in specific traits to zero or to desired levels. Restricted indices
incorporate constraints through Lagrange multipliers, solving a modified system that
maximizes progress in the breeding objective subject to specified restrictions.

Common applications include maintaining constant levels of traits that have reached
optimal values, such as mature body size in some species, or ensuring that selection
for production traits does not compromise welfare-related characteristics like disease
resistance or structural soundness. The restricted index framework provides mathematical
rigor for balancing competing objectives while respecting biological or market constraints.

# References
- Smith (1936) Ann Eugenics 7:240-250 (Original index theory)
- Hazel (1943) Genetics 28:476-490 (Economic foundation)
- Cunningham (1969) Anim Prod 11:9-16 (Restricted indices)

# See Also
- [`construct_selection_index`](@ref): Build index from genetic parameters
- [`apply_selection_index`](@ref): Rank individuals using index
- [`predict_correlated_responses`](@ref): Expected change per trait
"""
struct SelectionIndex
    traits::Vector{String}
    index_weights::Vector{Float64}
    economic_weights::Vector{Float64}
    accuracy::Float64
    expected_responses::Vector{Float64}
    genetic_covariance::Matrix{Float64}
    phenotypic_covariance::Matrix{Float64}
end


"""
    construct_selection_index(genetic_covariance, phenotypic_covariance,
                             economic_weights, traits; restrictions)

Construct optimal selection index for multi-trait breeding objectives.

This function implements the classical selection index theory to derive weights that
optimally combine trait information for maximizing genetic progress in an aggregate
breeding objective. The procedure solves the index equations that balance information
content, genetic relationships, and economic importance to produce weights that maximize
the correlation between the selection criterion and true genetic merit.

# Mathematical Derivation

The index weight vector b is obtained by solving the normal equations:

    Pb = Ga

where P denotes the phenotypic covariance matrix containing variances and covariances
among information sources used for selection, G represents the genetic covariance matrix
among traits in the breeding objective, and a contains economic weights expressing the
relative value of genetic change in each trait.

For genomic prediction applications, the information source matrix P corresponds to the
covariance among genomic estimated breeding values, which can be derived from the
reliability of predictions and genetic covariances. When prediction reliabilities differ
among traits, the index automatically assigns greater weight to traits with more accurate
genomic predictions, optimally exploiting differences in information quality.

The expected genetic gain per unit of selection intensity in trait i under index
selection is computed as:

    ΔG_i = (i_s / σ_I) × Cov(I, u_i)

where i_s represents selection intensity, σ_I denotes the index standard deviation, and
Cov(I, u_i) measures the covariance between the index and breeding values for trait i.
These expected responses enable breeders to verify that selection will produce desired
changes across all traits and identify situations where genetic correlations may cause
unfavorable correlated responses.

# Arguments
- `genetic_covariance::Matrix{Float64}`: Genetic covariance matrix (n_traits × n_traits)
- `phenotypic_covariance::Matrix{Float64}`: Phenotypic or GEBV covariance matrix
- `economic_weights::Vector{Float64}`: Economic values per unit genetic change
- `traits::Vector{String}`: Trait names for interpretation

# Keyword Arguments
- `restrictions::Union{Nothing, Dict}`: Constraints on trait responses
- `standardize_weights::Bool = false`: Express weights per genetic standard deviation

# Returns
Selection index structure containing optimal weights for combining trait information,
index accuracy measuring correlation with aggregate genotype, expected responses
quantifying predicted genetic change per trait under index selection, and component
matrices enabling further analysis of selection outcomes.

# Examples
```julia
# Basic selection index construction
index = construct_selection_index(
    genetic_covariance = Σ_g,
    phenotypic_covariance = Σ_p,
    economic_weights = [1.0, 5.0, 3.0],
    traits = ["trait1", "trait2", "trait3"]
)

# Restricted index maintaining trait 2 constant
restrictions = Dict("trait2" => 0.0)
restricted_index = construct_selection_index(
    genetic_covariance = Σ_g,
    phenotypic_covariance = Σ_p,
    economic_weights = [1.0, 5.0, 3.0],
    traits = ["trait1", "trait2", "trait3"],
    restrictions = restrictions
)

# Index with standardized weights
standardized_index = construct_selection_index(
    genetic_covariance = Σ_g,
    phenotypic_covariance = Σ_p,
    economic_weights = [1.0, 5.0, 3.0],
    traits = ["trait1", "trait2", "trait3"],
    standardize_weights = true
)
```
"""
function construct_selection_index(;
                                  genetic_covariance::Matrix{Float64},
                                  phenotypic_covariance::Matrix{Float64},
                                  economic_weights::Vector{Float64},
                                  traits::Vector{String},
                                  restrictions::Union{Nothing, Dict{String, Float64}} = nothing,
                                  standardize_weights::Bool = false)

    n_traits = length(traits)

    @assert size(genetic_covariance) == (n_traits, n_traits) "Genetic covariance dimension mismatch"
    @assert size(phenotypic_covariance) == (n_traits, n_traits) "Phenotypic covariance dimension mismatch"
    @assert length(economic_weights) == n_traits "Economic weights length mismatch"

    # Solve index equations: Pb = Ga
    if isnothing(restrictions)
        # Standard unrestricted index
        index_weights = phenotypic_covariance \ (genetic_covariance * economic_weights)
    else
        # Restricted index with constraints
        index_weights = solve_restricted_index(
            phenotypic_covariance,
            genetic_covariance,
            economic_weights,
            traits,
            restrictions
        )
    end

    # Standardize weights if requested
    if standardize_weights
        genetic_sd = sqrt.(diag(genetic_covariance))
        index_weights = index_weights .* genetic_sd
    end

    # Calculate index variance
    index_variance = index_weights' * phenotypic_covariance * index_weights

    # Calculate index-aggregate genotype covariance
    aggregate_genotype_variance = economic_weights' * genetic_covariance * economic_weights
    index_aggregate_covariance = index_weights' * genetic_covariance * economic_weights

    # Index accuracy
    accuracy = index_aggregate_covariance / sqrt(index_variance * aggregate_genotype_variance)

    # Expected responses per trait
    expected_responses = (genetic_covariance * index_weights) ./ sqrt(index_variance)

    return SelectionIndex(
        traits,
        index_weights,
        economic_weights,
        accuracy,
        expected_responses,
        genetic_covariance,
        phenotypic_covariance
    )
end


function solve_restricted_index(P::Matrix{Float64},
                                G::Matrix{Float64},
                                a::Vector{Float64},
                                traits::Vector{String},
                                restrictions::Dict{String, Float64})

    n_traits = length(traits)

    # Identify restricted traits
    restriction_indices = Int[]
    restriction_values = Float64[]

    for (trait, value) in restrictions
        idx = findfirst(==(trait), traits)
        if !isnothing(idx)
            push!(restriction_indices, idx)
            push!(restriction_values, value)
        end
    end

    n_restrictions = length(restriction_indices)

    if n_restrictions == 0
        return P \ (G * a)
    end

    # Build constraint matrix
    C = zeros(Float64, n_restrictions, n_traits)
    for (i, idx) in enumerate(restriction_indices)
        C[i, :] = G[idx, :]
    end

    # Solve constrained optimization using Lagrange multipliers
    # [P  C'] [b]   [Ga]
    # [C  0 ] [λ] = [r ]

    system_matrix = [P C'; C zeros(n_restrictions, n_restrictions)]
    rhs = [G * a; restriction_values]

    solution = system_matrix \ rhs

    index_weights = solution[1:n_traits]

    return index_weights
end