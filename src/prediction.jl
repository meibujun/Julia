# ===== src/prediction.jl =====
"""
    DynamicEpistasisGBLUP.Prediction

This module provides functions for predicting Genomic Estimated Breeding Values (GEBVs)
for new individuals using a previously fitted GBLUP model.
"""
module Prediction

export genomic_prediction

using CUDA
using LinearAlgebra
using ..DynamicEpistasisGBLUP.Types
using ..DynamicEpistasisGBLUP.GRMComputation

"""
    genomic_prediction(
        y_train::AbstractVector{T},
        X_train::MaybeCuMatrix{T},
        X_val::MaybeCuMatrix{T},
        train_GRMs::Vector{<:MaybeCuMatrix{T}},
        cross_GRMs::Vector{<:MaybeCuMatrix{T}},
        reml_results::REMLResults{T}
    ) where T <: AbstractFloat -> AbstractVector{T}

Predicts Genomic Estimated Breeding Values (GEBVs) for a set of validation
individuals using a fitted REML model.

The prediction formula is:
  `GEBV_val = G_total_val_train * V_inv * (y_train - X_train * b_hat)`

# Arguments
- `y_train::AbstractVector{T}`: Phenotype vector for the training set.
- `X_train::MaybeCuMatrix{T}`: Incidence matrix for fixed effects of the training set.
- `X_val::MaybeCuMatrix{T}`: Incidence matrix for fixed effects of the validation set.
- `train_GRMs::Vector{<:MaybeCuMatrix{T}}`: Vector of GRMs for the training set (e.g., [G_a, G_aa]).
- `cross_GRMs::Vector{<:MaybeCuMatrix{T}}`: Vector of cross-GRMs between validation and training sets. Must be in the same order as `train_GRMs`.
- `reml_results::REMLResults{T}`: The results object from a `estimate_variance_components_reml!` call, containing estimated variance components and `V_inv`.

# Returns
- `AbstractVector{T}`: A vector of predicted GEBVs for the validation individuals, on the same device as the input.
"""
function genomic_prediction(
    y_train::AbstractVector{T},
    X_train::MaybeCuMatrix{T},
    X_val::MaybeCuMatrix{T},
    train_GRMs::Vector{<:MaybeCuMatrix{T}},
    cross_GRMs::Vector{<:MaybeCuMatrix{T}},
    reml_results::REMLResults{T}
) where T <: AbstractFloat

    # Extract estimated variance components and V_inv from REML results
    var_components = reml_results.var_components
    V_inv = reml_results.V_inv

    # 1. Calculate b_hat (fixed effects estimates)
    # b_hat = (X' * V⁻¹ * X)⁻¹ * X' * V⁻¹ * y
    Xt_Vinv = X_train' * V_inv
    Xt_Vinv_X = Xt_Vinv * X_train

    # Use stable inverse for Xt_Vinv_X
    inv_Xt_Vinv_X, _ = GRMComputation.stable_inv_logdet(Xt_Vinv_X)
    b_hat = inv_Xt_Vinv_X * Xt_Vinv * y_train

    # 2. Calculate the total genetic covariance matrix between validation and training sets
    # G_total_val_train = G_a_vt * σ²_a + G_aa_vt * σ²_aa + ...
    if isempty(cross_GRMs)
        error("cross_GRMs vector cannot be empty.")
    end
    G_total_val_train = similar(cross_GRMs[1]) # Create output matrix of same type and size
    fill!(G_total_val_train, zero(T))

    for i in 1:length(cross_GRMs)
        G_total_val_train .+= cross_GRMs[i] .* var_components[i]
    end

    # 3. Calculate residuals adjusted for fixed effects
    y_adj = y_train - (X_train * b_hat)

    # 4. Calculate GEBVs for the validation set
    # GEBV_val = G_total_val_train * V⁻¹ * y_adj
    gebv_val = G_total_val_train * (V_inv * y_adj)

    # 5. Add fixed effect contribution for the validation set
    # This assumes X_val is the design matrix for the validation set
    final_gebv = gebv_val + (X_val * b_hat)

    return final_gebv
end

end # module Prediction
