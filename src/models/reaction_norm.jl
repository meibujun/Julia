# Reaction norm model for G×E interactions

module ReactionNormModel

using LinearAlgebra
using ..CoreTypes
using ..GRMConstruction

export fit_reaction_norm

function fit_reaction_norm(population::Population)

    # This is a placeholder for the reaction norm model.
    # A full implementation would involve fitting a random regression model.

    @info "Fitting reaction norm model (placeholder)..."

    n = population.genotype_data.n_individuals

    # Simplified model: just return zeros
    return ReactionNormGBLUP(
        Matrix(1.0I, n, n), # G
        ones(n,1), # fixed_effects
        [0.0], # fixed_coef
        zeros(n,1), # intercepts
        zeros(n,1), # slopes
        [0.0], [0.0], [0.0], [1.0], # variances
        (0.0, 1.0), # env_range
        nothing # model_info
    )
end

end
