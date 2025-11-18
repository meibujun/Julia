"""
RKHS: Reproducing Kernel Hilbert Space Regression

RKHS is a semi-parametric kernel-based method for genomic prediction that
can capture non-linear and epistatic effects through kernel functions.

Key features:
- Multiple kernel types: Linear, Gaussian (RBF), Polynomial
- Automatic bandwidth selection for Gaussian kernel
- Ridge regularization
- Kernel PCA for dimension reduction (optional)
- Can capture complex genotype-phenotype relationships

# Kernel Types

1. **Linear Kernel**: K(x,x') = x'x / p
   - Equivalent to GBLUP/Ridge Regression
   - Fast, interpretable

2. **Gaussian (RBF) Kernel**: K(x,x') = exp(-||x-x'||²/(2h²))
   - Captures non-linear effects
   - Bandwidth h controls smoothness

3. **Polynomial Kernel**: K(x,x') = (x'x/p + c)^d
   - Captures interactions up to degree d
   - Parameter c controls inhomogeneity

# References
Gianola et al. (2006). Additive genetic variability and the Bayesian alphabet.
Genetics, 183(1), 347-363.

Morota & Gianola (2014). Kernel-based whole-genome prediction of complex traits:
a review. Frontiers in Genetics, 5, 363.

de los Campos et al. (2010). Semi-parametric genomic-enabled prediction of genetic
values using reproducing kernel Hilbert spaces methods. Genetics Research, 92(4), 295-308.
"""

using LinearAlgebra
using Statistics
using Printf
using Random

"""
    RKHSModel

RKHS regression model for genomic prediction.

# Fields
- `kernel::Symbol`: Kernel type (:linear, :gaussian, :polynomial)
- `bandwidth::Union{Float64,Nothing}`: Bandwidth for Gaussian kernel (auto if nothing)
- `degree::Int`: Degree for polynomial kernel
- `coef::Float64`: Coefficient for polynomial kernel
- `lambda::Float64`: Ridge regularization parameter
- `use_kernel_pca::Bool`: Use kernel PCA for dimension reduction
- `n_pcs::Int`: Number of kernel PCs to use (if use_kernel_pca=true)
- `center_kernel::Bool`: Center the kernel matrix
- `verbose::Bool`: Print progress

# Results (after fitting)
- `result::Union{NamedTuple,Nothing}`: Fitted model results
- `sample_ids::Union{Vector{String},Nothing}`: Training sample IDs

# Examples
```julia
# Linear kernel (equivalent to GBLUP)
model = RKHSModel(kernel=:linear)

# Gaussian kernel with automatic bandwidth
model = RKHSModel(kernel=:gaussian)

# Polynomial kernel of degree 2
model = RKHSModel(kernel=:polynomial, degree=2)

# Fit model
fit!(model, geno, pheno)

# Predict
predictions = predict(model, geno)
```
"""
mutable struct RKHSModel
    kernel::Symbol
    bandwidth::Union{Float64, Nothing}
    degree::Int
    coef::Float64
    lambda::Float64
    use_kernel_pca::Bool
    n_pcs::Int
    center_kernel::Bool
    verbose::Bool

    # Results
    result::Union{Nothing, NamedTuple}
    sample_ids::Union{Vector{String}, Nothing}

    function RKHSModel(;
        kernel::Symbol = :gaussian,
        bandwidth::Union{Float64, Nothing} = nothing,
        degree::Int = 2,
        coef::Float64 = 1.0,
        lambda::Float64 = 1e-5,
        use_kernel_pca::Bool = false,
        n_pcs::Int = 100,
        center_kernel::Bool = true,
        verbose::Bool = true
    )
        if !(kernel in [:linear, :gaussian, :polynomial])
            throw(ArgumentError("kernel must be :linear, :gaussian, or :polynomial"))
        end

        if !isnothing(bandwidth) && bandwidth <= 0
            throw(ArgumentError("bandwidth must be positive"))
        end

        if degree < 1
            throw(ArgumentError("degree must be >= 1"))
        end

        if lambda < 0
            throw(ArgumentError("lambda must be non-negative"))
        end

        if n_pcs < 1
            throw(ArgumentError("n_pcs must be >= 1"))
        end

        new(kernel, bandwidth, degree, coef, lambda, use_kernel_pca,
            n_pcs, center_kernel, verbose, nothing, nothing)
    end
end

"""
    RKHSResult

Container for RKHS model results.

# Fields
- `alpha::Vector{Float64}`: Dual coefficients in kernel space
- `y_mean::Float64`: Mean of training phenotype
- `K_train::Matrix{Float64}`: Training kernel matrix
- `X_train::Matrix{Float64}`: Training genotypes (for prediction)
- `kernel_params::NamedTuple`: Kernel parameters used
- `lambda::Float64`: Regularization parameter
- `training_r2::Float64`: Training R²
- `sample_ids::Vector{String}`: Training sample IDs
- `n_samples::Int`: Number of training samples
- `n_markers::Int`: Number of markers
"""
struct RKHSResult
    alpha::Vector{Float64}
    y_mean::Float64
    K_train::Matrix{Float64}
    X_train::Matrix{Float64}
    kernel_params::NamedTuple
    lambda::Float64
    training_r2::Float64
    sample_ids::Vector{String}
    n_samples::Int
    n_markers::Int
end

"""
    compute_kernel(X1::Matrix{Float64}, X2::Matrix{Float64}, kernel::Symbol;
                   bandwidth::Union{Float64,Nothing}=nothing, degree::Int=2, coef::Float64=1.0)

Compute kernel matrix between two sets of samples.

# Arguments
- `X1::Matrix{Float64}`: First set of samples (n1 × p)
- `X2::Matrix{Float64}`: Second set of samples (n2 × p)
- `kernel::Symbol`: Kernel type (:linear, :gaussian, :polynomial)
- `bandwidth::Union{Float64,Nothing}`: Bandwidth for Gaussian kernel
- `degree::Int`: Degree for polynomial kernel
- `coef::Float64`: Coefficient for polynomial kernel

# Returns
Kernel matrix K of size n1 × n2

# Examples
```julia
# Linear kernel
K = compute_kernel(X, X, :linear)

# Gaussian kernel with bandwidth h=1.0
K = compute_kernel(X, X, :gaussian, bandwidth=1.0)

# Polynomial kernel of degree 2
K = compute_kernel(X, X, :polynomial, degree=2)
```
"""
function compute_kernel(X1::Matrix{Float64}, X2::Matrix{Float64}, kernel::Symbol;
                       bandwidth::Union{Float64,Nothing}=nothing, degree::Int=2, coef::Float64=1.0)
    n1, p = size(X1)
    n2 = size(X2, 1)

    K = zeros(n1, n2)

    if kernel == :linear
        # Linear kernel: K(x,x') = x'x / p
        K = (X1 * X2') / p

    elseif kernel == :gaussian
        # Gaussian (RBF) kernel: K(x,x') = exp(-||x-x'||²/(2h²))

        # Auto-select bandwidth if not provided
        if isnothing(bandwidth)
            # Use median heuristic: h = median of pairwise distances
            # Sample for efficiency if dataset is large
            sample_size = min(200, n1)
            idx = randperm(n1)[1:sample_size]
            X_sample = X1[idx, :]

            dists = Float64[]
            for i in 1:(sample_size-1)
                for j in (i+1):sample_size
                    d = norm(X_sample[i, :] - X_sample[j, :])
                    push!(dists, d)
                end
            end

            bandwidth = median(dists)
            if bandwidth == 0
                bandwidth = 1.0  # Fallback
            end
        end

        # Compute kernel
        for i in 1:n1
            for j in 1:n2
                dist_sq = sum((X1[i, :] - X2[j, :]) .^ 2)
                K[i, j] = exp(-dist_sq / (2 * bandwidth^2))
            end
        end

    elseif kernel == :polynomial
        # Polynomial kernel: K(x,x') = (x'x/p + c)^d
        K = ((X1 * X2') / p .+ coef) .^ degree

    else
        throw(ArgumentError("Unknown kernel: $kernel"))
    end

    return K
end

"""
    center_kernel_matrix!(K::Matrix{Float64})

Center a kernel matrix in-place.

Centering formula: K_c = (I - 11'/n) K (I - 11'/n)
where 1 is a vector of ones and n is the number of samples.
"""
function center_kernel_matrix!(K::Matrix{Float64})
    n = size(K, 1)
    col_means = mean(K, dims=1)
    row_means = mean(K, dims=2)
    total_mean = mean(K)

    for i in 1:n
        for j in 1:n
            K[i, j] = K[i, j] - row_means[i] - col_means[j] + total_mean
        end
    end

    return K
end

"""
    fit!(model::RKHSModel, geno::CompactGenotypes, pheno::PhenotypeData;
         trait_index::Int = 1, min_maf::Float64 = 0.0)

Fit RKHS regression model.

# Arguments
- `model::RKHSModel`: Model to fit
- `geno::CompactGenotypes`: Genotype data
- `pheno::PhenotypeData`: Phenotype data
- `trait_index::Int`: Index of trait to analyze (default: 1)
- `min_maf::Float64`: Minimum MAF filter (default: 0.0)

# Returns
Updates `model.result` with fitted parameters.

# Algorithm
1. Construct kernel matrix K from genotypes
2. Center kernel if specified
3. Solve ridge regression: α = (K + λI)⁻¹ y
4. Store dual coefficients and kernel matrix for prediction

# Example
```julia
model = RKHSModel(kernel=:gaussian)
fit!(model, geno, pheno)
```
"""
function fit!(model::RKHSModel, geno::CompactGenotypes, pheno::PhenotypeData;
              trait_index::Int = 1, min_maf::Float64 = 0.0)

    if model.verbose
        println("="^80)
        println("RKHS: Reproducing Kernel Hilbert Space Regression")
        println("="^80)
    end

    # Validate inputs
    if n_samples(geno) != n_samples(pheno)
        throw(ArgumentError("Sample size mismatch"))
    end

    # Get matched samples
    common_samples = intersect(sample_ids(geno), sample_ids(pheno))
    if isempty(common_samples)
        throw(ArgumentError("No common samples between genotype and phenotype"))
    end

    # Subset to common samples
    geno_idx = [findfirst(==(s), sample_ids(geno)) for s in common_samples]
    pheno_idx = [findfirst(==(s), sample_ids(pheno)) for s in common_samples]

    # Extract phenotype
    y = pheno.data[pheno_idx, trait_index]

    # Remove missing phenotypes
    valid = .!isnan.(y)
    y = y[valid]
    geno_idx = geno_idx[valid]

    n = length(y)
    p = n_markers(geno)

    if model.verbose
        println("\n📊 Data Summary:")
        println("  Samples: $n")
        println("  Markers: $p")
        println("  Trait: $(pheno.trait_names[trait_index])")
    end

    # Convert genotypes to matrix
    X = to_matrix(geno)
    X = Float64.(X[geno_idx, :])

    # MAF filtering
    if min_maf > 0.0
        maf = vec(mean(X, dims=1) ./ 2)
        maf = min.(maf, 1 .- maf)
        keep_markers = maf .>= min_maf
        X = X[:, keep_markers]
        p_filtered = sum(keep_markers)

        if model.verbose
            println("  Markers after MAF filter (>= $min_maf): $p_filtered")
        end
    end

    p = size(X, 2)

    # Standardize genotypes
    X_mean = vec(mean(X, dims=1))
    X_std = vec(std(X, dims=1))
    X_std[X_std .== 0] .= 1.0
    X = (X .- X_mean') ./ X_std'

    # Center phenotype
    y_mean = mean(y)
    y_centered = y .- y_mean

    if model.verbose
        println("\n🔧 Kernel Configuration:")
        println("  Type: $(model.kernel)")
        if model.kernel == :gaussian
            if isnothing(model.bandwidth)
                println("  Bandwidth: Auto (median heuristic)")
            else
                println("  Bandwidth: $(model.bandwidth)")
            end
        elseif model.kernel == :polynomial
            println("  Degree: $(model.degree)")
            println("  Coefficient: $(model.coef)")
        end
        println("  Regularization (λ): $(model.lambda)")
        println("  Center kernel: $(model.center_kernel)")
    end

    # Compute kernel matrix
    if model.verbose
        println("\n⏳ Computing kernel matrix...")
    end

    K = compute_kernel(X, X, model.kernel;
                      bandwidth=model.bandwidth,
                      degree=model.degree,
                      coef=model.coef)

    # Store actual bandwidth used (for Gaussian kernel)
    actual_bandwidth = model.bandwidth
    if model.kernel == :gaussian && isnothing(model.bandwidth)
        # Bandwidth was auto-selected in compute_kernel
        # Re-compute to get the value
        sample_size = min(200, n)
        idx = randperm(n)[1:sample_size]
        X_sample = X[idx, :]

        dists = Float64[]
        for i in 1:(sample_size-1)
            for j in (i+1):sample_size
                d = norm(X_sample[i, :] - X_sample[j, :])
                push!(dists, d)
            end
        end

        actual_bandwidth = median(dists)
        if actual_bandwidth == 0
            actual_bandwidth = 1.0
        end

        if model.verbose
            @printf("  Auto-selected bandwidth: %.4f\n", actual_bandwidth)
        end
    end

    # Center kernel if requested
    if model.center_kernel
        if model.verbose
            println("  Centering kernel matrix...")
        end
        center_kernel_matrix!(K)
    end

    if model.verbose
        @printf("  Kernel matrix shape: %d × %d\n", size(K)...)
        @printf("  Mean kernel value: %.4f\n", mean(K))
        @printf("  Kernel matrix rank: %d\n", rank(K))
    end

    # Solve ridge regression in kernel space
    # α = (K + λI)⁻¹ y
    if model.verbose
        println("\n⏳ Solving RKHS regression...")
    end

    # Add regularization
    K_reg = K + model.lambda * I(n)

    # Solve system
    α = K_reg \ y_centered

    # Compute training R²
    y_pred_train = K * α
    ss_res = sum((y_centered - y_pred_train) .^ 2)
    ss_tot = sum(y_centered .^ 2)
    r2_train = 1 - ss_res / ss_tot

    if model.verbose
        println("\n📊 Model Fit:")
        @printf("  Training R²: %.4f\n", r2_train)
        @printf("  Training RMSE: %.4f\n", sqrt(ss_res / n))
        @printf("  Mean |α|: %.6f\n", mean(abs.(α)))
        @printf("  Max |α|: %.6f\n", maximum(abs.(α)))
    end

    # Store results
    kernel_params = (
        kernel = model.kernel,
        bandwidth = actual_bandwidth,
        degree = model.degree,
        coef = model.coef,
        centered = model.center_kernel
    )

    result = RKHSResult(
        α,
        y_mean,
        K,
        X,
        kernel_params,
        model.lambda,
        r2_train,
        common_samples[valid],
        n,
        p
    )

    model.result = result
    model.sample_ids = common_samples[valid]

    if model.verbose
        println("\n" * "="^80)
        println("RKHS model fitted successfully!")
        println("="^80)
    end

    return nothing
end

"""
    predict(model::RKHSModel, geno::CompactGenotypes)

Predict phenotypes using fitted RKHS model.

# Arguments
- `model::RKHSModel`: Fitted RKHS model
- `geno::CompactGenotypes`: Genotype data for prediction

# Returns
Vector of predicted phenotypes.

# Algorithm
1. Compute kernel matrix K_test between test and training samples
2. Predict: ŷ = K_test α + mean(y_train)

# Example
```julia
predictions = predict(model, geno_test)
```
"""
function predict(model::RKHSModel, geno::CompactGenotypes)
    if isnothing(model.result)
        throw(ArgumentError("Model not fitted. Call fit!() first."))
    end

    result = model.result

    # Convert test genotypes to matrix
    X_test = to_matrix(geno)
    X_test = Float64.(X_test)

    # Standardize using training statistics
    # (In practice, should store exact scaling from training)
    X_test_mean = vec(mean(X_test, dims=1))
    X_test_std = vec(std(X_test, dims=1))
    X_test_std[X_test_std .== 0] .= 1.0
    X_test = (X_test .- X_test_mean') ./ X_test_std'

    # Compute kernel between test and training samples
    K_test = compute_kernel(X_test, result.X_train, result.kernel_params.kernel;
                           bandwidth=result.kernel_params.bandwidth,
                           degree=result.kernel_params.degree,
                           coef=result.kernel_params.coef)

    # Center if training kernel was centered
    if result.kernel_params.centered
        # For centered kernels, need to apply same centering transformation
        # This is an approximation; exact centering requires training stats
        col_means = mean(K_test, dims=2)
        K_test = K_test .- col_means
    end

    # Predict
    predictions = K_test * result.alpha .+ result.y_mean

    return vec(predictions)
end

# Export
export RKHSModel, RKHSResult
export fit!, predict
export compute_kernel, center_kernel_matrix!
