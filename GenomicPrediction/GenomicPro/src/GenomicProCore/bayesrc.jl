# src/GenomicProPredict/bayesrc.jl

using Distributions, LinearAlgebra

struct BayesRCModel <: AbstractBayesianModel
    base_model::BayesRModel
    n_annotations::Int
    annotation_names::Vector{String}
    prior_annotation_effects::Vector{Float64}

    function BayesRCModel(;
                         n_annotations::Int,
                         annotation_names::Vector{String},
                         prior_annotation_effects::Vector{Float64} = zeros(n_annotations),
                         base_model::BayesRModel = BayesRModel())

        @assert n_annotations == length(annotation_names) "Mismatch in annotation count"
        @assert n_annotations == length(prior_annotation_effects) "Mismatch in prior effects length"

        new(base_model, n_annotations, annotation_names, prior_annotation_effects)
    end
end

function run_bayesrc_mcmc(model::BayesRCModel,
                        genotypes::AbstractGenotypeData,
                        phenotypes::Vector{Float64},
                        annotations::Matrix{Float64};
                        n_iterations::Int = 50000,
                        burn_in::Int = 10000,
                        thinning::Int = 10)

    n_individuals, n_markers = size(genotypes)

    state = initialize_mcmc_state(model.base_model, genotypes, phenotypes)
    alpha = zeros(model.n_annotations, model.base_model.n_components)

    # Pre-compute XtX for efficiency
    XtX = [dot(genotypes[:, j], genotypes[:, j]) for j in 1:n_markers]

    # MCMC Loop
    for iter in 1:n_iterations
        # Update mixing proportions based on annotations
        state.mixing_proportions = compute_mixing_proportions(alpha, annotations)

        # Sample marker effects and component assignments (from BayesR)
        sample_marker_effects_and_components!(state, model.base_model, genotypes, XtX)

        # Sample variance components (from BayesR)
        sample_variance_components!(state, model.base_model, n_markers)

        # Sample annotation effects (alpha) using Metropolis-Hastings
        sample_annotation_effects!(alpha, state, model, annotations)

        if iter > burn_in && (iter - burn_in) % thinning == 0
            # Store samples
        end
    end

    # Summarize results
    return (
        marker_effects = state.marker_effects,
        pip = ones(n_markers) .- state.mixing_proportions[1, :],
        annotation_effects = alpha
    )
end

function compute_mixing_proportions(alpha::Matrix{Float64}, annotations::Matrix{Float64})
    logits = annotations * alpha
    π = exp.(logits) ./ sum(exp.(logits), dims=2)
    return π'
end

function sample_annotation_effects!(alpha::Matrix{Float64}, state::MCMCState, model::BayesRCModel, annotations::Matrix{Float64})
    # Metropolis-Hastings for each alpha parameter
    for l in 1:model.n_annotations
        for k in 1:model.base_model.n_components
            # Propose a new value for alpha[l, k]
            proposal = alpha[l, k] + randn() * 0.1

            # Calculate acceptance probability
            current_log_lik = log_likelihood_alpha(alpha, state, annotations)

            alpha_proposal = copy(alpha)
            alpha_proposal[l, k] = proposal
            proposal_log_lik = log_likelihood_alpha(alpha_proposal, state, annotations)

            acceptance_prob = min(1, exp(proposal_log_lik - current_log_lik))

            if rand() < acceptance_prob
                alpha[l, k] = proposal
            end
        end
    end
end

function log_likelihood_alpha(alpha::Matrix{Float64}, state::MCMCState, annotations::Matrix{Float64})
    π = compute_mixing_proportions(alpha, annotations)
    log_lik = 0.0
    for j in 1:length(state.component_assignments)
        log_lik += log(π[state.component_assignments[j], j])
    end
    return log_lik
end
