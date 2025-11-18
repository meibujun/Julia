"""
Population Structure Analysis Module

Principal Component Analysis (PCA) and population structure tools for
detecting and correcting population stratification in genomic data.

# Features
- PCA on genomic relationship matrix
- Scree plots for variance explained
- Population assignment
- Outlier detection based on PCs
- ADMIXTURE-style analysis

# References
- Price et al. (2006). Principal components analysis corrects for stratification
  in genome-wide association studies. Nature Genetics, 38(8), 904-909.
- Patterson et al. (2006). Population structure and eigenanalysis. PLoS Genetics,
  2(12), e190.
"""

using LinearAlgebra
using Statistics
using Printf

"""
    PCAResult

Results from principal component analysis on genomic data.

# Fields
- `eigenvalues::Vector{Float64}`: Eigenvalues (variances of PCs)
- `eigenvectors::Matrix{Float64}`: Eigenvectors (PC loadings)
- `pcs::Matrix{Float64}`: Principal component scores (n_samples × n_pcs)
- `variance_explained::Vector{Float64}`: Proportion of variance explained by each PC
- `cumulative_variance::Vector{Float64}`: Cumulative variance explained
- `sample_ids::Vector{String}`: Sample IDs
- `n_pcs::Int`: Number of PCs computed
"""
struct PCAResult
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}
    pcs::Matrix{Float64}
    variance_explained::Vector{Float64}
    cumulative_variance::Vector{Float64}
    sample_ids::Vector{String}
    n_pcs::Int
end

function Base.show(io::IO, result::PCAResult)
    println(io, "="^70)
    println(io, "PCA Results")
    println(io, "="^70)
    println(io, "\nSamples: $(length(result.sample_ids))")
    println(io, "Principal Components: $(result.n_pcs)")

    println(io, "\nVariance Explained:")
    println(io, "-"^70)
    @printf(io, "%-5s %15s %20s\n", "PC", "Variance (%)", "Cumulative (%)")
    println(io, "-"^70)

    for i in 1:min(10, result.n_pcs)
        @printf(io, "%-5d %15.2f %20.2f\n",
                i,
                result.variance_explained[i] * 100,
                result.cumulative_variance[i] * 100)
    end

    if result.n_pcs > 10
        println(io, "  ... (showing first 10 PCs)")
    end

    println(io, "-"^70)
    @printf(io, "Total variance explained (PC1-PC%d): %.2f%%\n",
            result.n_pcs,
            result.cumulative_variance[result.n_pcs] * 100)
    println(io, "="^70)
end

"""
    pca(geno::CompactGenotypes;
        n_pcs::Int=10,
        min_maf::Float64=0.01,
        ld_prune::Bool=true,
        ld_threshold::Float64=0.2,
        center::Bool=true,
        scale::Bool=false,
        method::Symbol=:grm,
        verbose::Bool=true) -> PCAResult

Perform principal component analysis on genomic data.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `n_pcs::Int`: Number of principal components to compute (default: 10)
- `min_maf::Float64`: Minimum MAF for filtering (default: 0.01)
- `ld_prune::Bool`: Perform LD pruning before PCA (default: true, recommended)
- `ld_threshold::Float64`: LD r² threshold if ld_prune=true (default: 0.2)
- `center::Bool`: Center genotypes (default: true, recommended)
- `scale::Bool`: Scale genotypes to unit variance (default: false)
- `method::Symbol`: Method to use (:grm for GRM-based, :svd for direct SVD)
- `verbose::Bool`: Print progress (default: true)

# Returns
- `PCAResult`: PCA results with eigenvalues, eigenvectors, and PC scores

# Example
```julia
# Standard PCA for population structure
pca_result = pca(geno; n_pcs=10, ld_prune=true)

# Get PC scores
pc1 = pca_result.pcs[:, 1]
pc2 = pca_result.pcs[:, 2]

# Variance explained
println("PC1 explains: \$(pca_result.variance_explained[1] * 100)%")
```

# Notes
- LD pruning is strongly recommended to avoid correlation structure
- GRM-based method is more robust for missing data
- First few PCs capture population structure
- Later PCs may reflect family structure or technical artifacts
"""
function pca(geno::CompactGenotypes;
            n_pcs::Int = 10,
            min_maf::Float64 = 0.01,
            ld_prune::Bool = true,
            ld_threshold::Float64 = 0.2,
            center::Bool = true,
            scale::Bool = false,
            method::Symbol = :grm,
            verbose::Bool = true)

    if verbose
        println("\n" * "="^70)
        println("Principal Component Analysis")
        println("="^70)
        println("  Original data: $(geno.n_samples) samples × $(geno.n_markers) markers")
    end

    # Filter by MAF
    if min_maf > 0.0
        maf = minor_allele_frequency(geno)
        keep_maf = findall(maf .>= min_maf)
        geno_filtered = subset_markers(geno, keep_maf)

        if verbose
            println("  After MAF filter: $(geno_filtered.n_markers) markers")
        end
    else
        geno_filtered = geno
    end

    # LD pruning
    if ld_prune
        if verbose
            println("  Performing LD pruning (r² > $ld_threshold)...")
        end

        keep_ld = ld_prune_window(
            geno_filtered;
            window_size = 50,
            step_size = 10,
            r2_threshold = ld_threshold,
            respect_chromosomes = true,
            verbose = false
        )

        geno_pruned = subset_markers(geno_filtered, keep_ld)

        if verbose
            println("  After LD pruning: $(geno_pruned.n_markers) markers")
        end
    else
        geno_pruned = geno_filtered
    end

    if geno_pruned.n_markers < n_pcs
        throw(ArgumentError("Number of markers ($(geno_pruned.n_markers)) < n_pcs ($n_pcs). " *
                          "Reduce n_pcs or disable LD pruning."))
    end

    # Compute PCA
    if verbose
        println("  Computing PCA ($method method)...")
    end

    if method == :grm
        # GRM-based PCA (more robust)
        G = compute_grm(geno_pruned; method=:vanraden, scale=true, min_maf=0.0)

        # Eigendecomposition of GRM
        eigen_result = eigen(Symmetric(G))

        # Sort by eigenvalue (descending)
        idx = sortperm(eigen_result.values, rev=true)
        eigenvalues = eigen_result.values[idx]
        eigenvectors = eigen_result.vectors[:, idx]

        # PC scores are the eigenvectors
        pcs = eigenvectors[:, 1:n_pcs]

    elseif method == :svd
        # Direct SVD on genotype matrix
        X = to_matrix(geno_pruned; impute=true)

        # Center
        if center
            X = X .- mean(X, dims=1)
        end

        # Scale
        if scale
            X = X ./ std(X, dims=1)
        end

        # SVD
        U, S, V = svd(X)

        # PC scores
        pcs = U[:, 1:n_pcs] .* S[1:n_pcs]'

        # Eigenvalues from singular values
        eigenvalues = (S.^2) ./ (size(X, 1) - 1)
        eigenvectors = V[:, 1:n_pcs]

    else
        throw(ArgumentError("Unknown method: $method. Use :grm or :svd"))
    end

    # Compute variance explained
    total_variance = sum(eigenvalues)
    variance_explained = eigenvalues[1:n_pcs] ./ total_variance
    cumulative_variance = cumsum(variance_explained)

    if verbose
        println("\n  Variance Explained:")
        println("  " * "-"^66)
        @printf("  %-5s %15s %20s\n", "PC", "Variance (%)", "Cumulative (%)")
        println("  " * "-"^66)

        for i in 1:min(5, n_pcs)
            @printf("  %-5d %15.2f %20.2f\n",
                    i,
                    variance_explained[i] * 100,
                    cumulative_variance[i] * 100)
        end

        if n_pcs > 5
            println("  ... (showing first 5 PCs)")
        end

        println("  " * "-"^66)
        println("\n  ✓ PCA complete")
        println("="^70)
    end

    return PCAResult(
        eigenvalues[1:n_pcs],
        eigenvectors,
        pcs,
        variance_explained,
        cumulative_variance,
        geno.sample_ids,
        n_pcs
    )
end

"""
    detect_outliers_pca(pca_result::PCAResult;
                        n_pcs::Int=2,
                        method::Symbol=:mahalanobis,
                        threshold::Float64=6.0) -> Vector{Int}

Detect outlier samples based on principal components.

# Arguments
- `pca_result::PCAResult`: PCA results
- `n_pcs::Int`: Number of PCs to use for outlier detection (default: 2)
- `method::Symbol`: Method (:mahalanobis or :euclidean)
- `threshold::Float64`: Threshold for outlier detection (default: 6.0 for Mahalanobis distance)

# Returns
- `Vector{Int}`: Indices of outlier samples

# Example
```julia
pca_result = pca(geno)
outliers = detect_outliers_pca(pca_result; n_pcs=4, threshold=6.0)
println("Outlier samples: \$(pca_result.sample_ids[outliers])")
```
"""
function detect_outliers_pca(pca_result::PCAResult;
                            n_pcs::Int = 2,
                            method::Symbol = :mahalanobis,
                            threshold::Float64 = 6.0)

    if n_pcs > pca_result.n_pcs
        throw(ArgumentError("n_pcs ($n_pcs) > available PCs ($(pca_result.n_pcs))"))
    end

    pcs = pca_result.pcs[:, 1:n_pcs]
    n_samples = size(pcs, 1)

    if method == :mahalanobis
        # Mahalanobis distance from centroid
        μ = mean(pcs, dims=1)
        Σ = cov(pcs)

        # Regularize if needed
        if any(diag(Σ) .< 1e-10)
            Σ = Σ + I * 1e-6
        end

        Σ_inv = inv(Σ)

        distances = zeros(n_samples)
        for i in 1:n_samples
            diff = pcs[i, :] - μ[:]
            distances[i] = sqrt(diff' * Σ_inv * diff)
        end

    elseif method == :euclidean
        # Euclidean distance from centroid
        μ = mean(pcs, dims=1)

        distances = zeros(n_samples)
        for i in 1:n_samples
            distances[i] = norm(pcs[i, :] - μ[:])
        end

    else
        throw(ArgumentError("Unknown method: $method"))
    end

    # Identify outliers
    outliers = findall(distances .> threshold)

    return outliers
end

"""
    cluster_samples(pca_result::PCAResult;
                   n_clusters::Int=3,
                   n_pcs::Int=2,
                   method::Symbol=:kmeans,
                   max_iter::Int=100) -> Vector{Int}

Cluster samples based on principal components.

# Arguments
- `pca_result::PCAResult`: PCA results
- `n_clusters::Int`: Number of clusters
- `n_pcs::Int`: Number of PCs to use
- `method::Symbol`: Clustering method (:kmeans only for now)
- `max_iter::Int`: Maximum iterations

# Returns
- `Vector{Int}`: Cluster assignments (1 to n_clusters)

# Example
```julia
pca_result = pca(geno)
clusters = cluster_samples(pca_result; n_clusters=3, n_pcs=2)

# Count samples per cluster
for k in 1:3
    n = sum(clusters .== k)
    println("Cluster \$k: \$n samples")
end
```
"""
function cluster_samples(pca_result::PCAResult;
                        n_clusters::Int = 3,
                        n_pcs::Int = 2,
                        method::Symbol = :kmeans,
                        max_iter::Int = 100)

    if method != :kmeans
        throw(ArgumentError("Only :kmeans supported currently"))
    end

    pcs = pca_result.pcs[:, 1:n_pcs]
    n_samples = size(pcs, 1)

    # K-means clustering
    # Initialize centroids randomly
    centroid_idx = randperm(n_samples)[1:n_clusters]
    centroids = pcs[centroid_idx, :]

    assignments = zeros(Int, n_samples)

    for iter in 1:max_iter
        old_assignments = copy(assignments)

        # Assignment step
        for i in 1:n_samples
            distances = [norm(pcs[i, :] - centroids[k, :]) for k in 1:n_clusters]
            assignments[i] = argmin(distances)
        end

        # Update step
        for k in 1:n_clusters
            cluster_points = pcs[assignments .== k, :]
            if !isempty(cluster_points)
                centroids[k, :] = vec(mean(cluster_points, dims=1))
            end
        end

        # Check convergence
        if assignments == old_assignments
            break
        end
    end

    return assignments
end

# Export
export PCAResult
export pca, detect_outliers_pca, cluster_samples
