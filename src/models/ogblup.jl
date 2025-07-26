# Orthogonal GBLUP model implementation

module OGBLUPModel

using LinearAlgebra
using ..CoreTypes
using ..GRMConstruction
using ..VarianceComponentEstimation
using ..Constants

export fit_ogblup, predict_gebv

function fit_ogblup(population::Population;
                   fixed_effects::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
                   include_dominance::Bool=true,
                   include_epistasis::Bool=true)

    grms = compute_grm_set(population.genotype_data,
                           include_dominance=include_dominance,
                           include_epistasis=include_epistasis)

    y = vec(population.phenotypes)
    X = isnothing(fixed_effects) ? ones(length(y), 1) : fixed_effects

    var_comp = estimate_variance_components(y, grms, fixed_effects=X)

    # Build V matrix
    V = var_comp.estimates.σ²_a * grms.G
    if include_dominance V += var_comp.estimates.σ²_d * grms.D end
    if include_epistasis
        V += var_comp.estimates.σ²_aa * grms.G_AA
        V += var_comp.estimates.σ²_ad * grms.G_AD
        V += var_comp.estimates.σ²_dd * grms.G_DD
    end
    V += var_comp.estimates.σ²_e * I

    # Solve MME
    V_inv = inv(V)
    β = inv(X' * V_inv * X) * X' * V_inv * y
    resid = y - X * β

    # Get BLUPs
    u = Dict{Symbol, Matrix{Float64}}()
    u[:additive] = reshape(var_comp.estimates.σ²_a * grms.G * V_inv * resid, :, 1)
    if include_dominance
        u[:dominance] = reshape(var_comp.estimates.σ²_d * grms.D * V_inv * resid, :, 1)
    end
    if include_epistasis
        u[:epistasis_aa] = reshape(var_comp.estimates.σ²_aa * grms.G_AA * V_inv * resid, :, 1)
        u[:epistasis_ad] = reshape(var_comp.estimates.σ²_ad * grms.G_AD * V_inv * resid, :, 1)
        u[:epistasis_dd] = reshape(var_comp.estimates.σ²_dd * grms.G_DD * V_inv * resid, :, 1)
    end

    fitted = X * β + sum(values(u))

    return OGBLUP(grms, var_comp, X, reshape(β, :, 1), u, fitted, y - fitted, nothing)
end

function predict_gebv(model::OGBLUP; components::Symbol=:all)
    if components == :additive
        return model.random_effects[:additive]
    elseif components == :all
        return sum(values(model.random_effects))
    else
        error("Unknown component selection")
    end
end

end
