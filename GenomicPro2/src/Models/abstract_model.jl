"""
# Abstract Model Interface

Defines the common interface for all genomic prediction models in GenomicPro2.

All models should inherit from `AbstractGenomicModel` and implement:
- `fit!(model, genotypes, phenotypes; kwargs...)`
- `predict(model, genotypes; kwargs...)`
- `model_name(model)`
- `model_type(model)`

Optional methods:
- `get_hyperparameters(model)`
- `set_hyperparameters!(model, params)`
- `is_fitted(model)`
- `score(model, genotypes, phenotypes)`
- `feature_importance(model)`

## Benefits
- Consistent API across all models
- Easy to add new models
- Support for model comparison and ensembles
- Simplified cross-validation and benchmarking

## Example
```julia
# All models share the same interface
models = [
    GBLUPModel(),
    BayesRModel(),
    RKHSModel(kernel=GaussianKernel()),
    DeepGBLUPModel()
]

for model in models
    fit!(model, genotypes, phenotypes)
    predictions = predict(model, genotypes)
    accuracy = score(model, genotypes, phenotypes)
    println("\$(model_name(model)): accuracy = \$accuracy")
end
```
"""

using LinearAlgebra
using Statistics

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractGenomicModel

Base type for all genomic prediction models.

All models must implement:
- `fit!(model, genotypes, phenotypes; kwargs...)`
- `predict(model, genotypes; kwargs...)`
"""
abstract type AbstractGenomicModel end

"""
    ModelType

Enum for model categories.
"""
@enum ModelType begin
    LINEAR_MODEL       # Linear models (GBLUP, Ridge)
    BAYESIAN_MODEL     # Bayesian models (BayesR, BayesCπ)
    KERNEL_MODEL       # Kernel methods (RKHS)
    DEEP_LEARNING      # Neural networks (Deep GBLUP)
    ENSEMBLE_MODEL     # Ensemble methods
end

"""
    ModelStatus

Enum for model fitting status.
"""
@enum ModelStatus begin
    NOT_FITTED
    FITTING
    FITTED
    FAILED
end

# ============================================================================
# Core Interface Methods (must be implemented)
# ============================================================================

"""
    fit!(model::AbstractGenomicModel, genotypes, phenotypes; kwargs...)

Train the model on the provided genotypes and phenotypes.

# Arguments
- `model`: Model instance
- `genotypes`: Genotype data
- `phenotypes`: Phenotype data
- `kwargs...`: Model-specific parameters

# Returns
- Fitted model (modifies in place)

# Example
```julia
model = GBLUPModel()
fit!(model, genotypes, phenotypes)
```
"""
function fit!(model::AbstractGenomicModel, genotypes, phenotypes; kwargs...)
    error("fit! not implemented for $(typeof(model))")
end

"""
    predict(model::AbstractGenomicModel, genotypes; kwargs...)

Make predictions using the fitted model.

# Arguments
- `model`: Fitted model
- `genotypes`: Genotype data for prediction
- `kwargs...`: Model-specific parameters

# Returns
- Vector of predictions

# Example
```julia
predictions = predict(model, new_genotypes)
```
"""
function predict(model::AbstractGenomicModel, genotypes; kwargs...)
    error("predict not implemented for $(typeof(model))")
end

"""
    model_name(model::AbstractGenomicModel)

Get the name of the model.

# Returns
- String with model name

# Example
```julia
name = model_name(model)  # "GBLUP"
```
"""
function model_name(model::AbstractGenomicModel)
    return string(typeof(model))
end

"""
    model_type(model::AbstractGenomicModel)

Get the category/type of the model.

# Returns
- ModelType enum value
"""
function model_type(model::AbstractGenomicModel)
    error("model_type not implemented for $(typeof(model))")
end

# ============================================================================
# Optional Interface Methods (have default implementations)
# ============================================================================

"""
    is_fitted(model::AbstractGenomicModel)

Check if the model has been fitted.

# Returns
- Boolean indicating if model is fitted
"""
function is_fitted(model::AbstractGenomicModel)
    return hasfield(typeof(model), :status) && model.status == FITTED
end

"""
    score(model::AbstractGenomicModel, genotypes, phenotypes; metric=:correlation)

Evaluate model performance.

# Arguments
- `model`: Fitted model
- `genotypes`: Test genotype data
- `phenotypes`: True phenotype values
- `metric`: Evaluation metric (:correlation, :mse, :mae, :r2)

# Returns
- Numeric score
"""
function score(model::AbstractGenomicModel, genotypes, phenotypes; metric=:correlation)
    predictions = predict(model, genotypes)

    if metric == :correlation
        return cor(predictions, phenotypes)
    elseif metric == :mse
        return mean((predictions .- phenotypes).^2)
    elseif metric == :mae
        return mean(abs.(predictions .- phenotypes))
    elseif metric == :r2
        ss_res = sum((phenotypes .- predictions).^2)
        ss_tot = sum((phenotypes .- mean(phenotypes)).^2)
        return 1.0 - ss_res / ss_tot
    else
        error("Unknown metric: $metric")
    end
end

"""
    get_hyperparameters(model::AbstractGenomicModel)

Get model hyperparameters as a dictionary.

# Returns
- Dict{Symbol, Any} with hyperparameters
"""
function get_hyperparameters(model::AbstractGenomicModel)
    return Dict{Symbol, Any}()
end

"""
    set_hyperparameters!(model::AbstractGenomicModel, params::Dict)

Set model hyperparameters from a dictionary.

# Arguments
- `model`: Model instance
- `params`: Dictionary of hyperparameters
"""
function set_hyperparameters!(model::AbstractGenomicModel, params::Dict)
    @warn "set_hyperparameters! not implemented for $(typeof(model))"
    return model
end

"""
    feature_importance(model::AbstractGenomicModel)

Get feature (SNP) importance scores if available.

# Returns
- Vector of importance scores, or nothing if not applicable
"""
function feature_importance(model::AbstractGenomicModel)
    return nothing
end

"""
    get_fitted_values(model::AbstractGenomicModel)

Get the fitted values (predictions on training data).

# Returns
- Vector of fitted values, or nothing if not available
"""
function get_fitted_values(model::AbstractGenomicModel)
    return nothing
end

"""
    get_variance_components(model::AbstractGenomicModel)

Get variance components (genetic, environmental) if available.

# Returns
- Named tuple with variance components, or nothing if not applicable
"""
function get_variance_components(model::AbstractGenomicModel)
    return nothing
end

"""
    heritability(model::AbstractGenomicModel)

Estimate heritability from the fitted model.

# Returns
- Float64 heritability estimate (0-1), or nothing if not applicable
"""
function heritability(model::AbstractGenomicModel)
    vc = get_variance_components(model)
    if vc !== nothing && haskey(vc, :genetic) && haskey(vc, :environmental)
        vg = vc.genetic
        ve = vc.environmental
        return vg / (vg + ve)
    end
    return nothing
end

# ============================================================================
# Model Comparison and Selection
# ============================================================================

"""
    compare_models(models::Vector{<:AbstractGenomicModel}, genotypes, phenotypes;
                   cv_folds=5, metrics=[:correlation, :mse])

Compare multiple models using cross-validation.

# Arguments
- `models`: Vector of model instances
- `genotypes`: Genotype data
- `phenotypes`: Phenotype data
- `cv_folds`: Number of cross-validation folds
- `metrics`: Evaluation metrics to compute

# Returns
- DataFrame with comparison results
"""
function compare_models(models::Vector{<:AbstractGenomicModel}, genotypes, phenotypes;
                        cv_folds=5, metrics=[:correlation, :mse])

    results = []

    for model in models
        model_results = Dict(
            :model_name => model_name(model),
            :model_type => model_type(model)
        )

        # Cross-validation
        cv_scores = Dict{Symbol, Vector{Float64}}()
        for metric in metrics
            cv_scores[metric] = Float64[]
        end

        # Perform k-fold CV
        n = length(phenotypes)
        fold_size = div(n, cv_folds)

        for fold in 1:cv_folds
            # Split data
            test_idx = ((fold-1)*fold_size+1):(fold*fold_size)
            train_idx = setdiff(1:n, test_idx)

            # Create a copy of the model
            model_copy = deepcopy(model)

            # Train
            fit!(model_copy, genotypes[train_idx, :], phenotypes[train_idx])

            # Evaluate
            for metric in metrics
                score_val = score(model_copy, genotypes[test_idx, :], phenotypes[test_idx]; metric=metric)
                push!(cv_scores[metric], score_val)
            end
        end

        # Aggregate scores
        for metric in metrics
            model_results[Symbol("$(metric)_mean")] = mean(cv_scores[metric])
            model_results[Symbol("$(metric)_std")] = std(cv_scores[metric])
        end

        push!(results, model_results)
    end

    return results
end

"""
    select_best_model(models::Vector{<:AbstractGenomicModel}, genotypes, phenotypes;
                      metric=:correlation, maximize=true)

Select the best model based on cross-validation performance.

# Arguments
- `models`: Vector of model instances
- `genotypes`: Genotype data
- `phenotypes`: Phenotype data
- `metric`: Metric to use for selection
- `maximize`: Whether higher is better (true) or lower is better (false)

# Returns
- Best model instance
"""
function select_best_model(models::Vector{<:AbstractGenomicModel}, genotypes, phenotypes;
                           metric=:correlation, maximize=true)

    comparison = compare_models(models, genotypes, phenotypes; metrics=[metric])

    metric_col = Symbol("$(metric)_mean")
    best_idx = if maximize
        argmax([r[metric_col] for r in comparison])
    else
        argmin([r[metric_col] for r in comparison])
    end

    return models[best_idx]
end

# ============================================================================
# Ensemble Methods
# ============================================================================

"""
    EnsembleModel <: AbstractGenomicModel

Ensemble of multiple models with weighted averaging.
"""
mutable struct EnsembleModel <: AbstractGenomicModel
    models::Vector{AbstractGenomicModel}
    weights::Vector{Float64}
    status::ModelStatus

    function EnsembleModel(models::Vector{<:AbstractGenomicModel})
        n = length(models)
        weights = ones(n) ./ n  # Equal weights by default
        new(models, weights, NOT_FITTED)
    end
end

model_name(::EnsembleModel) = "Ensemble"
model_type(::EnsembleModel) = ENSEMBLE_MODEL

function fit!(model::EnsembleModel, genotypes, phenotypes; optimize_weights=true, kwargs...)
    model.status = FITTING

    # Fit all models
    for m in model.models
        fit!(m, genotypes, phenotypes; kwargs...)
    end

    # Optimize weights if requested
    if optimize_weights
        # Simple approach: use correlation as weight
        correlations = [score(m, genotypes, phenotypes) for m in model.models]
        model.weights = correlations ./ sum(correlations)
    end

    model.status = FITTED
    return model
end

function predict(model::EnsembleModel, genotypes; kwargs...)
    if !is_fitted(model)
        error("Model not fitted yet")
    end

    # Weighted average of predictions
    predictions = zeros(size(genotypes, 1))

    for (i, m) in enumerate(model.models)
        pred = predict(m, genotypes; kwargs...)
        predictions .+= model.weights[i] .* pred
    end

    return predictions
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    print_model_summary(model::AbstractGenomicModel)

Print a summary of the model.
"""
function print_model_summary(model::AbstractGenomicModel)
    println("Model: ", model_name(model))
    println("Type: ", model_type(model))
    println("Status: ", is_fitted(model) ? "Fitted" : "Not fitted")

    h2 = heritability(model)
    if h2 !== nothing
        println("Heritability: ", round(h2, digits=3))
    end

    vc = get_variance_components(model)
    if vc !== nothing
        println("Variance Components:")
        for (k, v) in pairs(vc)
            println("  $k: ", round(v, digits=3))
        end
    end

    hp = get_hyperparameters(model)
    if !isempty(hp)
        println("Hyperparameters:")
        for (k, v) in hp
            println("  $k: $v")
        end
    end
end

# Export types
export AbstractGenomicModel, ModelType, ModelStatus
export LINEAR_MODEL, BAYESIAN_MODEL, KERNEL_MODEL, DEEP_LEARNING, ENSEMBLE_MODEL
export NOT_FITTED, FITTING, FITTED, FAILED

# Export interface methods
export fit!, predict, model_name, model_type
export is_fitted, score, get_hyperparameters, set_hyperparameters!
export feature_importance, get_fitted_values, get_variance_components, heritability

# Export comparison and selection
export compare_models, select_best_model

# Export ensemble
export EnsembleModel

# Export utilities
export print_model_summary
