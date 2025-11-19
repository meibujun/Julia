"""
Cross-validation functionality for model evaluation.

Provides various cross-validation strategies:
- k-fold cross-validation
- Leave-one-out cross-validation (LOO)
- Random sub-sampling validation
- Stratified sampling

Includes comprehensive evaluation metrics and result reporting.
"""

"""
    CVResult

Results from cross-validation.

# Fields
- `predictions::Vector{Float64}`: Predicted values
- `observed::Vector{Float64}`: Observed values
- `fold_results::Vector{NamedTuple}`: Per-fold results
- `metrics::NamedTuple`: Overall metrics (correlation, MSE, R², accuracy, bias)
- `fold_assignments::Vector{Int}`: Fold assignment for each sample
- `cv_method::Symbol`: Cross-validation method used
"""
struct CVResult
    predictions::Vector{Float64}
    observed::Vector{Float64}
    fold_results::Vector{NamedTuple}
    metrics::NamedTuple
    fold_assignments::Vector{Int}
    cv_method::Symbol
end

"""
    create_folds(n::Int, k::Int; shuffle::Bool=true, seed::Union{Int,Nothing}=nothing) -> Vector{Vector{Int}}

Create k folds for cross-validation.

# Arguments
- `n::Int`: Number of samples
- `k::Int`: Number of folds
- `shuffle::Bool`: Whether to shuffle samples (default: true)
- `seed::Union{Int,Nothing}`: Random seed for reproducibility

# Returns
Vector of k vectors, each containing indices for that fold

# Example
```julia
folds = create_folds(100, 5)
for fold in folds
    println("Fold size: ", length(fold))
end
```
"""
function create_folds(n::Int, k::Int; shuffle::Bool=true, seed::Union{Int,Nothing}=nothing)
    if k < 2
        throw(ArgumentError("k must be >= 2"))
    end
    if k > n
        throw(ArgumentError("k cannot be larger than n"))
    end

    # Set seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Create indices
    indices = collect(1:n)
    if shuffle
        Random.shuffle!(indices)
    end

    # Split into folds
    folds = Vector{Vector{Int}}(undef, k)
    fold_size = n ÷ k
    remainder = n % k

    start_idx = 1
    for i in 1:k
        # Add one extra to early folds if there's a remainder
        current_size = fold_size + (i <= remainder ? 1 : 0)
        end_idx = start_idx + current_size - 1

        folds[i] = indices[start_idx:end_idx]
        start_idx = end_idx + 1
    end

    return folds
end

"""
    kfold_cv(model_fn::Function, geno::CompactGenotypes, pheno::PhenotypeData;
             k::Int=5, trait_index::Int=1, shuffle::Bool=true, seed::Union{Int,Nothing}=nothing,
             compute_grm::Bool=true, grm_options::NamedTuple=NamedTuple(),
             verbose::Bool=true) -> CVResult

Perform k-fold cross-validation.

# Arguments
- `model_fn::Function`: Function that creates a model (e.g., `() -> GBLUPModel()`)
- `geno::CompactGenotypes`: Genotype data
- `pheno::PhenotypeData`: Phenotype data
- `k::Int`: Number of folds (default: 5)
- `trait_index::Int`: Which trait to validate (default: 1)
- `shuffle::Bool`: Whether to shuffle samples (default: true)
- `seed::Union{Int,Nothing}`: Random seed
- `compute_grm::Bool`: Whether to recompute GRM for each fold (default: true)
- `grm_options::NamedTuple`: Options for GRM computation
- `verbose::Bool`: Print progress (default: true)

# Returns
CVResult with predictions, metrics, and fold results

# Example
```julia
# 5-fold CV for GBLUP
result = kfold_cv(() -> GBLUPModel(), geno, pheno; k=5)
println("CV Accuracy: ", result.metrics.correlation)

# 10-fold CV with custom GRM options
result = kfold_cv(() -> GBLUPModel(), geno, pheno;
    k = 10,
    grm_options = (min_maf = 0.05, method = :vanraden)
)
```
"""
function kfold_cv(
    model_fn::Function,
    geno::CompactGenotypes,
    pheno::PhenotypeData;
    k::Int = 5,
    trait_index::Int = 1,
    shuffle::Bool = true,
    seed::Union{Int,Nothing} = nothing,
    compute_grm::Bool = true,
    grm_options::NamedTuple = NamedTuple(),
    verbose::Bool = true
)
    # Merge genotype and phenotype data
    geno_matched, pheno_matched, common_ids = merge_genotype_phenotype(geno, pheno)

    n = n_samples(geno_matched)
    y_all = pheno_matched.values[:, trait_index]

    # Remove missing phenotypes
    valid_idx = findall(.!isnan.(y_all))
    if length(valid_idx) < n
        if verbose
            @info "Removing $(n - length(valid_idx)) samples with missing phenotypes"
        end
        geno_matched = subset_samples(geno_matched, valid_idx)
        y_all = y_all[valid_idx]
        n = length(y_all)
    end

    if verbose
        println("\n" * "="^70)
        println("$k-Fold Cross-Validation")
        println("="^70)
        println("  Samples: $n")
        println("  Folds: $k")
        println("  Trait: $(pheno_matched.trait_names[trait_index])")
    end

    # Create folds
    folds = create_folds(n, k; shuffle=shuffle, seed=seed)

    # Storage for results
    predictions = zeros(Float64, n)
    fold_results = NamedTuple[]
    fold_assignments = zeros(Int, n)

    # Assign fold numbers
    for (fold_num, fold_idx) in enumerate(folds)
        fold_assignments[fold_idx] .= fold_num
    end

    # Run CV
    for (fold_num, test_idx) in enumerate(folds)
        if verbose
            print("\r  Processing fold $fold_num/$k...")
            flush(stdout)
        end

        # Split data
        train_idx = setdiff(1:n, test_idx)

        geno_train = subset_samples(geno_matched, train_idx)
        geno_test = subset_samples(geno_matched, test_idx)

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # Create phenotype data for training
        pheno_train = PhenotypeData(
            sample_ids(geno_train),
            [pheno_matched.trait_names[trait_index]],
            reshape(y_train, length(y_train), 1)
        )

        # Compute GRM if requested
        G_train = if compute_grm
            compute_grm(geno_train; grm_options...)
        else
            nothing
        end

        # Train model
        model = model_fn()
        fit!(model, geno_train, pheno_train; G=G_train, trait_index=1)

        # Predict test set
        pred_test = predict(model, geno_test)

        # Store predictions
        predictions[test_idx] = pred_test

        # Compute fold metrics
        fold_cor = cor(pred_test, y_test)
        fold_mse = mean((pred_test .- y_test).^2)
        fold_mae = mean(abs.(pred_test .- y_test))

        push!(fold_results, (
            fold = fold_num,
            n_train = length(train_idx),
            n_test = length(test_idx),
            correlation = fold_cor,
            mse = fold_mse,
            mae = fold_mae,
            r_squared = fold_cor^2
        ))
    end

    if verbose
        println("\r  ✓ All folds completed" * " "^20)
    end

    # Compute overall metrics
    overall_cor = cor(predictions, y_all)
    overall_mse = mean((predictions .- y_all).^2)
    overall_mae = mean(abs.(predictions .- y_all))
    overall_r2 = overall_cor^2
    overall_bias = mean(predictions .- y_all)

    # Regression slope (for accuracy assessment)
    # y_test = slope * y_pred + intercept
    # Ideal slope = 1.0
    X_reg = hcat(ones(n), predictions)
    beta_reg = X_reg \ y_all
    regression_slope = beta_reg[2]

    metrics = (
        correlation = overall_cor,
        mse = overall_mse,
        mae = overall_mae,
        r_squared = overall_r2,
        bias = overall_bias,
        regression_slope = regression_slope,
        mean_fold_correlation = mean(fr.correlation for fr in fold_results),
        std_fold_correlation = std(fr.correlation for fr in fold_results)
    )

    if verbose
        println("\n" * "="^70)
        println("Cross-Validation Results")
        println("="^70)
        @printf("  Correlation:     %.4f ± %.4f\n",
                metrics.mean_fold_correlation,
                metrics.std_fold_correlation)
        @printf("  R²:              %.4f\n", metrics.r_squared)
        @printf("  MSE:             %.4f\n", metrics.mse)
        @printf("  MAE:             %.4f\n", metrics.mae)
        @printf("  Bias:            %.4f\n", metrics.bias)
        @printf("  Regression slope: %.4f\n", metrics.regression_slope)
        println("="^70)
    end

    return CVResult(
        predictions,
        y_all,
        fold_results,
        metrics,
        fold_assignments,
        :kfold
    )
end

"""
    loo_cv(model_fn::Function, geno::CompactGenotypes, pheno::PhenotypeData;
           trait_index::Int=1, verbose::Bool=true) -> CVResult

Perform leave-one-out cross-validation.

More expensive than k-fold but gives unbiased estimates for small datasets.

# Example
```julia
result = loo_cv(() -> GBLUPModel(), geno, pheno)
```
"""
function loo_cv(
    model_fn::Function,
    geno::CompactGenotypes,
    pheno::PhenotypeData;
    trait_index::Int = 1,
    verbose::Bool = true
)
    # LOO is equivalent to n-fold CV
    n = n_samples(geno)

    if n > 1000
        @warn "LOO CV with $n samples will be very slow. Consider k-fold CV instead."
    end

    return kfold_cv(
        model_fn,
        geno,
        pheno;
        k = n,
        trait_index = trait_index,
        shuffle = false,
        verbose = verbose
    )
end

"""
    random_cv(model_fn::Function, geno::CompactGenotypes, pheno::PhenotypeData;
              n_reps::Int=10, test_fraction::Float64=0.2, trait_index::Int=1,
              seed::Union{Int,Nothing}=nothing, verbose::Bool=true) -> CVResult

Perform random sub-sampling cross-validation.

Randomly split data into train/test sets multiple times.

# Arguments
- `n_reps::Int`: Number of random splits (default: 10)
- `test_fraction::Float64`: Fraction of data for testing (default: 0.2)

# Example
```julia
# 10 random 80/20 splits
result = random_cv(() -> GBLUPModel(), geno, pheno; n_reps=10, test_fraction=0.2)
```
"""
function random_cv(
    model_fn::Function,
    geno::CompactGenotypes,
    pheno::PhenotypeData;
    n_reps::Int = 10,
    test_fraction::Float64 = 0.2,
    trait_index::Int = 1,
    seed::Union{Int,Nothing} = nothing,
    compute_grm::Bool = true,
    grm_options::NamedTuple = NamedTuple(),
    verbose::Bool = true
)
    if !(0 < test_fraction < 1)
        throw(ArgumentError("test_fraction must be in (0, 1)"))
    end

    # Merge data
    geno_matched, pheno_matched, _ = merge_genotype_phenotype(geno, pheno)

    n = n_samples(geno_matched)
    y_all = pheno_matched.values[:, trait_index]

    # Remove missing
    valid_idx = findall(.!isnan.(y_all))
    if length(valid_idx) < n
        geno_matched = subset_samples(geno_matched, valid_idx)
        y_all = y_all[valid_idx]
        n = length(y_all)
    end

    n_test = round(Int, n * test_fraction)

    if verbose
        println("\n" * "="^70)
        println("Random Sub-sampling Cross-Validation")
        println("="^70)
        println("  Samples: $n")
        println("  Repetitions: $n_reps")
        println("  Test fraction: $(test_fraction) ($n_test samples)")
    end

    # Set seed
    if seed !== nothing
        Random.seed!(seed)
    end

    # Storage
    all_predictions = Vector{Float64}[]
    all_observed = Vector{Float64}[]
    fold_results = NamedTuple[]

    for rep in 1:n_reps
        if verbose
            print("\r  Processing repetition $rep/$n_reps...")
            flush(stdout)
        end

        # Random split
        test_idx = sort(randperm(n)[1:n_test])
        train_idx = setdiff(1:n, test_idx)

        geno_train = subset_samples(geno_matched, train_idx)
        geno_test = subset_samples(geno_matched, test_idx)

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        pheno_train = PhenotypeData(
            sample_ids(geno_train),
            [pheno_matched.trait_names[trait_index]],
            reshape(y_train, length(y_train), 1)
        )

        # GRM
        G_train = if compute_grm
            compute_grm(geno_train; grm_options...)
        else
            nothing
        end

        # Train and predict
        model = model_fn()
        fit!(model, geno_train, pheno_train; G=G_train, trait_index=1)
        pred_test = predict(model, geno_test)

        push!(all_predictions, pred_test)
        push!(all_observed, y_test)

        # Metrics
        rep_cor = cor(pred_test, y_test)
        rep_mse = mean((pred_test .- y_test).^2)
        rep_mae = mean(abs.(pred_test .- y_test))

        push!(fold_results, (
            fold = rep,
            n_train = length(train_idx),
            n_test = length(test_idx),
            correlation = rep_cor,
            mse = rep_mse,
            mae = rep_mae,
            r_squared = rep_cor^2
        ))
    end

    if verbose
        println("\r  ✓ All repetitions completed" * " "^20)
    end

    # Aggregate results
    predictions = vcat(all_predictions...)
    observed = vcat(all_observed...)

    overall_cor = cor(predictions, observed)
    overall_mse = mean((predictions .- observed).^2)
    overall_mae = mean(abs.(predictions .- observed))
    overall_r2 = overall_cor^2
    overall_bias = mean(predictions .- observed)

    metrics = (
        correlation = overall_cor,
        mse = overall_mse,
        mae = overall_mae,
        r_squared = overall_r2,
        bias = overall_bias,
        regression_slope = NaN,  # Not meaningful for random CV
        mean_fold_correlation = mean(fr.correlation for fr in fold_results),
        std_fold_correlation = std(fr.correlation for fr in fold_results)
    )

    if verbose
        println("\n" * "="^70)
        println("Cross-Validation Results")
        println("="^70)
        @printf("  Correlation:     %.4f ± %.4f\n",
                metrics.mean_fold_correlation,
                metrics.std_fold_correlation)
        @printf("  R²:              %.4f\n", metrics.r_squared)
        @printf("  MSE:             %.4f\n", metrics.mse)
        @printf("  MAE:             %.4f\n", metrics.mae)
        println("="^70)
    end

    return CVResult(
        predictions,
        observed,
        fold_results,
        metrics,
        zeros(Int, length(predictions)),  # No fold assignments for random CV
        :random
    )
end

"""
    Base.show(io::IO, result::CVResult)

Display cross-validation results.
"""
function Base.show(io::IO, result::CVResult)
    println(io, "Cross-Validation Result ($(result.cv_method))")
    println(io, "─"^60)
    @printf(io, "  Samples:         %d\n", length(result.observed))

    if result.cv_method == :kfold
        n_folds = maximum(result.fold_assignments)
        @printf(io, "  Folds:           %d\n", n_folds)
    elseif result.cv_method == :random
        @printf(io, "  Repetitions:     %d\n", length(result.fold_results))
    end

    println(io, "\nMetrics:")
    @printf(io, "  Correlation:     %.4f ± %.4f\n",
            result.metrics.mean_fold_correlation,
            result.metrics.std_fold_correlation)
    @printf(io, "  R²:              %.4f\n", result.metrics.r_squared)
    @printf(io, "  MSE:             %.4f\n", result.metrics.mse)
    @printf(io, "  MAE:             %.4f\n", result.metrics.mae)
    @printf(io, "  Bias:            %.4f\n", result.metrics.bias)

    if !isnan(result.metrics.regression_slope)
        @printf(io, "  Regression slope: %.4f\n", result.metrics.regression_slope)
    end
end
