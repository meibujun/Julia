# ===== src/sparse_epistasis.jl =====
"""
Sparse Epistasis Detection and Modeling
Implements cutting-edge sparse learning algorithms for epistasis
"""

module SparseEpistasis

using CUDA
using SparseArrays
using LinearAlgebra
using Wavelets
using FFTW
using IterativeSolvers
using ProximalOperators

export SparseEpistaticModel, detect_sparse_interactions, 
       elastic_net_epistasis, group_lasso_epistasis,
       adaptive_sparse_learning

"""
Sparse epistatic model with regularization
"""
struct SparseEpistaticModel{T<:AbstractFloat}
    interactions::Vector{Tuple{Int32, Int32}}
    coefficients::Vector{T}
    λ1::T  # L1 penalty
    λ2::T  # L2 penalty
    sparsity_level::T
    method::Symbol
end

"""
Advanced sparse interaction detection using multiple criteria
"""
function detect_sparse_interactions(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T};
    max_interactions::Int = 10000,
    methods::Vector{Symbol} = [:mutual_information, :dcor, :hsic],
    threshold_quantile::T = T(0.99),
    parallel_chunks::Int = 10
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Initialize score matrix
    interaction_scores = Dict{Symbol, SparseMatrixCSC{T, Int}}()
    
    # Compute interaction scores using multiple methods
    for method in methods
        println("Computing $method scores...")
        scores = compute_interaction_scores(
            genotypes, phenotypes, method, parallel_chunks
        )
        interaction_scores[method] = scores
    end
    
    # Ensemble scoring - combine multiple criteria
    ensemble_scores = combine_interaction_scores(interaction_scores)
    
    # Select top interactions
    selected_interactions = select_top_interactions(
        ensemble_scores, max_interactions, threshold_quantile
    )
    
    return selected_interactions, ensemble_scores
end

"""
Compute interaction scores using specified method
"""
function compute_interaction_scores(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    method::Symbol,
    n_chunks::Int
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Chunk processing for memory efficiency
    chunk_size = cld(n_snps, n_chunks)
    
    # Pre-allocate sparse matrix for scores
    I_indices = Int[]
    J_indices = Int[]
    V_values = T[]
    
    # Process chunks in parallel
    @sync for chunk in 1:n_chunks
        @async begin
            start_idx = (chunk - 1) * chunk_size + 1
            end_idx = min(chunk * chunk_size, n_snps)
            
            # Compute scores for this chunk
            chunk_scores = compute_chunk_scores(
                genotypes, phenotypes, method,
                start_idx, end_idx
            )
            
            # Add to sparse matrix
            for ((i, j), score) in chunk_scores
                if abs(score) > 1e-6
                    push!(I_indices, i)
                    push!(J_indices, j)
                    push!(V_values, score)
                end
            end
        end
    end
    
    # Create sparse matrix
    scores = sparse(I_indices, J_indices, V_values, n_snps, n_snps)
    
    return scores
end

"""
Compute interaction scores for a chunk of SNPs
"""
function compute_chunk_scores(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    method::Symbol,
    start_idx::Int,
    end_idx::Int
) where T
    scores = Dict{Tuple{Int, Int}, T}()
    
    if method == :mutual_information
        scores = mutual_information_scores(
            genotypes, phenotypes, start_idx, end_idx
        )
    elseif method == :dcor
        scores = distance_correlation_scores(
            genotypes, phenotypes, start_idx, end_idx
        )
    elseif method == :hsic
        scores = hilbert_schmidt_scores(
            genotypes, phenotypes, start_idx, end_idx
        )
    end
    
    return scores
end

"""
Mutual Information based interaction scoring
"""
function mutual_information_scores(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    start_idx::Int,
    end_idx::Int
) where T
    scores = Dict{Tuple{Int, Int}, T}()
    n_individuals = size(genotypes, 1)
    
    # Transfer relevant data to CPU for MI computation
    geno_chunk = Array(genotypes[:, start_idx:end_idx])
    pheno = Array(phenotypes)
    
    # Discretize phenotypes for MI
    n_bins = ceil(Int, sqrt(n_individuals))
    pheno_discrete = discretize(pheno, n_bins)
    
    # Compute MI for each pair
    for i in 1:(end_idx - start_idx + 1)
        for j in start_idx:size(genotypes, 2)
            if start_idx + i - 1 < j
                # Create interaction variable
                interaction = geno_chunk[:, i] .* Array(genotypes[:, j])
                interaction_discrete = discretize(interaction, n_bins)
                
                # Compute MI
                mi = mutual_information(interaction_discrete, pheno_discrete)
                
                if mi > 0.01  # Threshold
                    scores[(start_idx + i - 1, j)] = T(mi)
                end
            end
        end
    end
    
    return scores
end

"""
Distance correlation based scoring
"""
function distance_correlation_scores(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    start_idx::Int,
    end_idx::Int
) where T
    scores = Dict{Tuple{Int, Int}, T}()
    
    # GPU kernel for distance correlation
    @cuda threads=256 blocks=cld((end_idx-start_idx+1)*size(genotypes,2), 256) dcor_kernel!(
        scores, genotypes, phenotypes, start_idx, end_idx
    )
    
    return scores
end

"""
Hilbert-Schmidt Independence Criterion scoring
"""
function hilbert_schmidt_scores(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    start_idx::Int,
    end_idx::Int
) where T
    scores = Dict{Tuple{Int, Int}, T}()
    n_individuals = size(genotypes, 1)
    
    # Compute kernel matrices
    K_y = gaussian_kernel(phenotypes)
    
    for i in start_idx:end_idx
        for j in (i+1):size(genotypes, 2)
            # Interaction kernel
            interaction = genotypes[:, i] .* genotypes[:, j]
            K_x = gaussian_kernel(interaction)
            
            # HSIC statistic
            hsic = compute_hsic(K_x, K_y)
            
            if hsic > 0.01
                scores[(i, j)] = T(hsic)
            end
        end
    end
    
    return scores
end

"""
Elastic Net regularized epistasis detection
"""
function elastic_net_epistasis(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    interactions::Vector{Tuple{Int32, Int32}};
    λ1::T = T(0.01),
    λ2::T = T(0.001),
    max_iter::Int = 1000,
    tol::T = T(1e-6)
) where T
    n_individuals = size(genotypes, 1)
    n_interactions = length(interactions)
    
    # Build interaction design matrix
    X = build_interaction_matrix(genotypes, interactions)
    
    # Initialize coefficients
    β = CUDA.zeros(T, n_interactions)
    
    # Proximal gradient descent with acceleration
    β = accelerated_proximal_gradient(
        X, phenotypes, β, λ1, λ2, max_iter, tol
    )
    
    # Extract non-zero coefficients
    non_zero = findall(x -> abs(x) > 1e-8, Array(β))
    sparse_interactions = interactions[non_zero]
    sparse_coefficients = Array(β)[non_zero]
    
    return SparseEpistaticModel(
        sparse_interactions,
        sparse_coefficients,
        λ1, λ2,
        T(length(non_zero) / n_interactions),
        :elastic_net
    )
end

"""
Group LASSO for epistasis (groups by SNP)
"""
function group_lasso_epistasis(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    interactions::Vector{Tuple{Int32, Int32}};
    λ::T = T(0.01),
    groups::Union{Nothing, Vector{Vector{Int}}} = nothing,
    max_iter::Int = 1000
) where T
    n_individuals = size(genotypes, 1)
    n_interactions = length(interactions)
    
    # Define groups if not provided
    if groups === nothing
        groups = define_snp_groups(interactions)
    end
    
    # Build design matrix
    X = build_interaction_matrix(genotypes, interactions)
    
    # Group LASSO optimization
    β = CUDA.zeros(T, n_interactions)
    
    for iter in 1:max_iter
        β_old = copy(β)
        
        # Update each group
        for group in groups
            β_group = β[group]
            X_group = X[:, group]
            
            # Group soft thresholding
            r = phenotypes - X * β + X_group * β_group
            s = X_group' * r
            
            norm_s = norm(s)
            if norm_s > λ * sqrt(length(group))
                β[group] = (1 - λ * sqrt(length(group)) / norm_s) * s
            else
                β[group] .= 0
            end
        end
        
        # Check convergence
        if norm(β - β_old) < 1e-6
            break
        end
    end
    
    # Extract results
    non_zero_groups = [g for g in groups if any(abs.(β[g]) .> 1e-8)]
    selected_interactions = Tuple{Int32, Int32}[]
    selected_coefficients = T[]
    
    for group in non_zero_groups
        for idx in group
            if abs(β[idx]) > 1e-8
                push!(selected_interactions, interactions[idx])
                push!(selected_coefficients, β[idx])
            end
        end
    end
    
    return SparseEpistaticModel(
        selected_interactions,
        selected_coefficients,
        λ, T(0),
        T(length(selected_interactions) / n_interactions),
        :group_lasso
    )
end

"""
Adaptive sparse learning with importance weighting
"""
function adaptive_sparse_learning(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T};
    initial_screen_size::Int = 50000,
    final_model_size::Int = 5000,
    n_stages::Int = 3
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Stage 1: Initial screening
    println("Stage 1: Initial screening...")
    initial_scores = fast_correlation_screening(
        genotypes, phenotypes, initial_screen_size
    )
    
    candidate_interactions = initial_scores.interactions
    current_size = length(candidate_interactions)
    
    # Multi-stage refinement
    for stage in 2:n_stages
        target_size = round(Int, current_size * (final_model_size / initial_screen_size)^(1/(n_stages-1)))
        println("Stage $stage: Refining to $target_size interactions...")
        
        # Build current model
        X = build_interaction_matrix(genotypes, candidate_interactions)
        
        # Compute importance weights
        weights = compute_importance_weights(X, phenotypes)
        
        # Select top interactions by weight
        top_indices = partialsortperm(weights, 1:target_size, rev=true)
        candidate_interactions = candidate_interactions[top_indices]
        current_size = target_size
    end
    
    # Final model fitting with selected interactions
    final_model = elastic_net_epistasis(
        genotypes, phenotypes, candidate_interactions,
        λ1 = T(0.001), λ2 = T(0.0001)
    )
    
    return final_model
end

# === Helper Functions ===

"""
Build interaction design matrix efficiently
"""
function build_interaction_matrix(
    genotypes::CuArray{T, 2},
    interactions::Vector{Tuple{Int32, Int32}}
) where T
    n_individuals = size(genotypes, 1)
    n_interactions = length(interactions)
    
    X = CUDA.zeros(T, n_individuals, n_interactions)
    
    # GPU kernel for efficient computation
    @cuda threads=256 blocks=cld(n_individuals * n_interactions, 256) build_interaction_kernel!(
        X, genotypes, interactions, n_individuals, n_interactions
    )
    
    return X
end

@kernel function build_interaction_kernel!(X, genotypes, interactions, n_ind, n_int)
    idx = @index(Global)
    
    if idx <= n_ind * n_int
        i = (idx - 1) % n_ind + 1
        j = (idx - 1) ÷ n_ind + 1
        
        snp1, snp2 = interactions[j]
        @inbounds X[i, j] = genotypes[i, snp1] * genotypes[i, snp2]
    end
end

"""
Accelerated proximal gradient descent
"""
function accelerated_proximal_gradient(
    X::CuArray{T, 2},
    y::CuVector{T},
    β₀::CuVector{T},
    λ1::T,
    λ2::T,
    max_iter::Int,
    tol::T
) where T
    n, p = size(X)
    
    # Initialize
    β = copy(β₀)
    β_prev = copy(β)
    v = copy(β)
    
    # Lipschitz constant
    L = opnorm(X' * X) / n + λ2
    
    # FISTA acceleration
    t = T(1)
    
    for iter in 1:max_iter
        # Gradient step
        grad = X' * (X * v - y) / n + λ2 * v
        z = v - grad / L
        
        # Soft thresholding (proximal operator for L1)
        β = soft_threshold(z, λ1 / L)
        
        # Acceleration
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2
        v = β + ((t - 1) / t_new) * (β - β_prev)
        
        # Check convergence
        if norm(β - β_prev) < tol
            break
        end
        
        β_prev = copy(β)
        t = t_new
    end
    
    return β
end

"""
Soft thresholding operator
"""
function soft_threshold(x::CuVector{T}, λ::T) where T
    return sign.(x) .* max.(abs.(x) .- λ, 0)
end

"""
Compute importance weights for interactions
"""
function compute_importance_weights(
    X::CuArray{T, 2},
    y::CuVector{T}
) where T
    n, p = size(X)
    
    # Use randomized SVD for efficiency
    k = min(100, p ÷ 10)
    U, S, V = randomized_svd(X, k)
    
    # Compute leverage scores
    leverage = sum(V.^2, dims=2)
    
    # Correlation with outcome
    correlations = abs.(X' * y) / n
    
    # Combined importance
    weights = sqrt.(leverage) .* correlations
    
    return vec(weights)
end

"""
Fast correlation screening for initial selection
"""
function fast_correlation_screening(
    genotypes::CuArray{T, 2},
    phenotypes::CuVector{T},
    top_k::Int
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Single SNP correlations
    single_correlations = abs.(genotypes' * phenotypes) / n_individuals
    
    # Select top SNPs
    top_snps = partialsortperm(vec(Array(single_correlations)), 1:min(1000, n_snps), rev=true)
    
    # Screen pairwise interactions among top SNPs
    interactions = Tuple{Int32, Int32}[]
    scores = T[]
    
    for i in 1:length(top_snps)
        for j in (i+1):length(top_snps)
            snp1, snp2 = top_snps[i], top_snps[j]
            
            # Compute interaction correlation
            interaction = genotypes[:, snp1] .* genotypes[:, snp2]
            corr = abs(dot(interaction, phenotypes)) / (norm(interaction) * norm(phenotypes))
            
            push!(interactions, (Int32(snp1), Int32(snp2)))
            push!(scores, corr)
        end
    end
    
    # Select top k
    top_indices = partialsortperm(scores, 1:min(top_k, length(scores)), rev=true)
    
    return (interactions = interactions[top_indices], scores = scores[top_indices])
end

"""
Define SNP groups for group LASSO
"""
function define_snp_groups(interactions::Vector{Tuple{Int32, Int32}})
    # Group by first SNP in interaction
    groups = Dict{Int32, Vector{Int}}()
    
    for (idx, (snp1, snp2)) in enumerate(interactions)
        if !haskey(groups, snp1)
            groups[snp1] = Int[]
        end
        push!(groups[snp1], idx)
    end
    
    return collect(values(groups))
end

"""
Gaussian kernel computation
"""
function gaussian_kernel(x::CuVector{T}; σ::T = T(1.0)) where T
    n = length(x)
    K = CUDA.zeros(T, n, n)
    
    @cuda threads=16 blocks=cld(n*n, 256) gaussian_kernel_compute!(K, x, σ, n)
    
    return K
end

@kernel function gaussian_kernel_compute!(K, x, σ, n)
    idx = @index(Global)
    
    if idx <= n * n
        i = (idx - 1) % n + 1
        j = (idx - 1) ÷ n + 1
        
        @inbounds K[i, j] = exp(-(x[i] - x[j])^2 / (2 * σ^2))
    end
end

"""
Compute HSIC statistic
"""
function compute_hsic(K_x::CuArray{T, 2}, K_y::CuArray{T, 2}) where T
    n = size(K_x, 1)
    
    # Center kernel matrices
    H = I - ones(n, n) / n
    K_x_centered = H * K_x * H
    K_y_centered = H * K_y * H
    
    # HSIC = tr(K_x * K_y) / n^2
    hsic = tr(K_x_centered * K_y_centered) / n^2
    
    return hsic
end

"""
Discretize continuous variable for MI computation
"""
function discretize(x::Vector{T}, n_bins::Int) where T
    edges = quantile(x, range(0, 1, length=n_bins+1))
    discrete = zeros(Int, length(x))
    
    for i in 1:length(x)
        for j in 1:n_bins
            if x[i] <= edges[j+1]
                discrete[i] = j
                break
            end
        end
    end
    
    return discrete
end

"""
Compute mutual information between discrete variables
"""
function mutual_information(x::Vector{Int}, y::Vector{Int})
    n = length(x)
    
    # Joint probability
    max_x = maximum(x)
    max_y = maximum(y)
    joint = zeros(max_x, max_y)
    
    for i in 1:n
        joint[x[i], y[i]] += 1/n
    end
    
    # Marginal probabilities
    px = sum(joint, dims=2)
    py = sum(joint, dims=1)
    
    # MI calculation
    mi = 0.0
    for i in 1:max_x
        for j in 1:max_y
            if joint[i,j] > 0
                mi += joint[i,j] * log(joint[i,j] / (px[i] * py[j]))
            end
        end
    end
    
    return mi
end

"""
Randomized SVD for large matrices
"""
function randomized_svd(A::CuArray{T, 2}, k::Int; oversampling::Int = 10) where T
    m, n = size(A)
    l = k + oversampling
    
    # Random projection
    Ω = CUDA.randn(T, n, l)
    Y = A * Ω
    
    # QR decomposition
    Q, _ = qr(Y)
    Q = Matrix(Q)[:, 1:l]
    
    # Project A
    B = Q' * A
    
    # SVD of smaller matrix
    U_small, S, V = svd(B)
    U = Q * U_small
    
    return U[:, 1:k], S[1:k], V[:, 1:k]
end

end # module SparseEpistasis

# ===== src/distributed_computing.jl =====
"""
Distributed computing support for massive datasets
Handles 100K+ SNPs with 10K+ individuals across multiple nodes
"""

module DistributedComputing

using Distributed
using DistributedArrays
using SharedArrays
using MPI
using CUDA
using JLD2
using BSON

export DistributedGenotypeMatrix, DistributedGBLUP,
       distribute_genotypes, distributed_grm_computation,
       distributed_epistasis_detection

"""
Distributed genotype matrix across multiple workers
"""
struct DistributedGenotypeMatrix{T<:AbstractFloat}
    data::DArray{T, 2}
    local_chunks::Dict{Int, Tuple{Int, Int}}  # Worker ID -> (row_range, col_range)
    n_individuals::Int
    n_snps::Int
    n_workers::Int
    chunk_size::Tuple{Int, Int}
end

"""
Distributed GBLUP model structure
"""
struct DistributedGBLUP{T<:AbstractFloat}
    G_chunks::DArray{T, 2}
    G_aa_chunks::Union{Nothing, DArray{T, 2}}
    variance_components::VarianceComponents{T}
    n_workers::Int
end

"""
Initialize distributed computing environment
"""
function initialize_distributed_env(;
    n_workers::Int = Sys.CPU_THREADS,
    use_gpu::Bool = CUDA.functional(),
    memory_per_worker::Int = 8  # GB
)
    # Add workers if needed
    if nworkers() < n_workers
        addprocs(n_workers - nworkers())
    end
    
    # Load required packages on all workers
    @everywhere begin
        using CUDA
        using LinearAlgebra
        using SparseArrays
        using Statistics
    end
    
    # Initialize MPI if available
    if @isdefined MPI
        MPI.Init()
    end
    
    # Check GPU availability on each worker
    gpu_workers = @distributed (vcat) for w in workers()
        CUDA.functional() ? w : nothing
    end
    gpu_workers = filter(!isnothing, gpu_workers)
    
    println("Initialized distributed environment:")
    println("  Workers: $(nworkers())")
    println("  GPU-enabled workers: $(length(gpu_workers))")
    
    return gpu_workers
end

"""
Distribute genotype matrix across workers
"""
function distribute_genotypes(
    genotypes::Matrix{T};
    chunk_strategy::Symbol = :balanced,
    overlap::Int = 0  # For boundary computations
) where T
    n_individuals, n_snps = size(genotypes)
    n_workers = nworkers()
    
    # Determine chunking strategy
    if chunk_strategy == :balanced
        # Balance computation and memory
        chunk_rows = ceil(Int, n_individuals / sqrt(n_workers))
        chunk_cols = ceil(Int, n_snps / sqrt(n_workers))
    elseif chunk_strategy == :row_wise
        # Distribute by individuals
        chunk_rows = ceil(Int, n_individuals / n_workers)
        chunk_cols = n_snps
    elseif chunk_strategy == :col_wise
        # Distribute by SNPs
        chunk_rows = n_individuals
        chunk_cols = ceil(Int, n_snps / n_workers)
    end
    
    # Create distributed array
    dist_geno = DArray((n_individuals, n_snps), workers(), [chunk_rows, chunk_cols]) do I
        # I is the index range for this chunk
        row_range = I[1]
        col_range = I[2]
        
        # Add overlap if specified
        row_start = max(1, first(row_range) - overlap)
        row_end = min(n_individuals, last(row_range) + overlap)
        col_start = max(1, first(col_range) - overlap)
        col_end = min(n_snps, last(col_range) + overlap)
        
        # Copy data to worker
        chunk_data = genotypes[row_start:row_end, col_start:col_end]
        
        # Transfer to GPU if available
        if CUDA.functional()
            return CuArray(chunk_data)
        else
            return chunk_data
        end
    end
    
    # Record chunk assignments
    local_chunks = Dict{Int, Tuple{Int, Int}}()
    for (idx, pid) in enumerate(dist_geno.pids)
        local_chunks[pid] = dist_geno.indices[idx]
    end
    
    return DistributedGenotypeMatrix(
        dist_geno,
        local_chunks,
        n_individuals,
        n_snps,
        n_workers,
        (chunk_rows, chunk_cols)
    )
end

"""
Distributed computation of genomic relationship matrix
"""
function distributed_grm_computation(
    dist_genotypes::DistributedGenotypeMatrix{T};
    method::Symbol = :vanraden,
    use_compression::Bool = true
) where T
    n_individuals = dist_genotypes.n_individuals
    
    # Step 1: Compute allele frequencies across all chunks
    println("Computing global allele frequencies...")
    allele_freq = distributed_allele_frequencies(dist_genotypes)
    
    # Step 2: Center genotypes on each worker
    println("Centering genotypes...")
    @sync for (worker, chunk_range) in dist_genotypes.local_chunks
        @spawnat worker begin
            local_data = localpart(dist_genotypes.data)
            center_genotypes_chunk!(local_data, allele_freq[chunk_range[2]])
        end
    end
    
    # Step 3: Compute GRM blocks in parallel
    println("Computing GRM blocks...")
    G_blocks = distributed_grm_blocks(dist_genotypes, allele_freq)
    
    # Step 4: Assemble full GRM (or keep distributed)
    if n_individuals <= 10000
        # Gather to master for smaller matrices
        G = assemble_grm_from_blocks(G_blocks, n_individuals)
        return G
    else
        # Keep distributed for large matrices
        return G_blocks
    end
end

"""
Compute allele frequencies in distributed fashion
"""
function distributed_allele_frequencies(dist_geno::DistributedGenotypeMatrix{T}) where T
    n_snps = dist_geno.n_snps
    
    # Compute local sums and counts
    local_stats = @distributed (vcat) for (worker, chunk_range) in dist_geno.local_chunks
        @spawnat worker begin
            local_data = localpart(dist_geno.data)
            col_range = chunk_range[2]
            
            sums = sum(local_data, dims=1)
            counts = sum(.!ismissing.(local_data), dims=1)
            
            (col_range, sums, counts)
        end
    end
    
    # Aggregate to global frequencies
    allele_freq = zeros(T, n_snps)
    total_counts = zeros(Int, n_snps)
    
    for (col_range, sums, counts) in local_stats
        allele_freq[col_range] .+= vec(sums)
        total_counts[col_range] .+= vec(counts)
    end
    
    # Normalize by total count (accounting for ploidy)
    allele_freq ./= (2 .* total_counts)
    
    return allele_freq
end

"""
Compute GRM blocks in distributed fashion
"""
function distributed_grm_blocks(
    dist_geno::DistributedGenotypeMatrix{T},
    allele_freq::Vector{T}
) where T
    n_individuals = dist_geno.n_individuals
    n_workers = dist_geno.n_workers
    
    # Initialize distributed GRM storage
    block_size = ceil(Int, n_individuals / sqrt(n_workers))
    G_blocks = DArray((n_individuals, n_individuals), workers(), [block_size, block_size]) do I
        zeros(T, length(I[1]), length(I[2]))
    end
    
    # Compute each block
    @sync for (bi, bj) in Iterators.product(1:size(G_blocks.indices, 1), 1:size(G_blocks.indices, 2))
        if bi <= bj  # Only upper triangle
            @spawnat G_blocks.pids[bi, bj] begin
                compute_grm_block!(
                    localpart(G_blocks),
                    dist_geno,
                    G_blocks.indices[bi, bj],
                    allele_freq
                )
            end
        end
    end
    
    # Symmetrize
    symmetrize_distributed_matrix!(G_blocks)
    
    return G_blocks
end

"""
Compute a single GRM block
"""
function compute_grm_block!(
    block::Array{T, 2},
    dist_geno::DistributedGenotypeMatrix{T},
    block_indices::Tuple{UnitRange{Int}, UnitRange{Int}},
    allele_freq::Vector{T}
) where T
    row_range, col_range = block_indices
    
    # Get relevant genotype chunks
    W_i = get_centered_chunk(dist_geno, row_range, :)
    W_j = get_centered_chunk(dist_geno, col_range, :)
    
    # Compute block: G_ij = W_i * W_j' / scale
    scale = compute_grm_scale(allele_freq)
    
    if CUDA.functional()
        # GPU computation
        W_i_gpu = CuArray(W_i)
        W_j_gpu = CuArray(W_j)
        block_gpu = W_i_gpu * W_j_gpu' / scale
        copyto!(block, block_gpu)
    else
        # CPU computation with BLAS
        BLAS.gemm!('N', 'T', T(1/scale), W_i, W_j, T(0), block)
    end
end

"""
Distributed epistatic GRM computation
"""
function distributed_epistatic_grm(
    dist_geno::DistributedGenotypeMatrix{T};
    method::Symbol = :symmetric_polynomial,
    max_memory_gb::Float64 = 8.0
) where T
    n_individuals = dist_geno.n_individuals
    n_snps = dist_geno.n_snps
    
    println("Computing distributed epistatic GRM...")
    println("  Method: $method")
    println("  Memory limit per worker: $max_memory_gb GB")
    
    if method == :symmetric_polynomial
        # Use symmetric polynomial approach for efficiency
        G_aa = distributed_symmetric_epistasis(dist_geno, max_memory_gb)
    elseif method == :block_wise
        # Block-wise computation with compression
        G_aa = distributed_blockwise_epistasis(dist_geno, max_memory_gb)
    else
        error("Unknown distributed epistasis method: $method")
    end
    
    return G_aa
end

"""
Distributed symmetric polynomial epistasis computation
"""
function distributed_symmetric_epistasis(
    dist_geno::DistributedGenotypeMatrix{T},
    max_memory_gb::Float64
) where T
    # Compute elementary symmetric polynomials in distributed fashion
    e1_dist = distributed_polynomial(dist_geno, 1)  # Sum
    e2_dist = distributed_polynomial(dist_geno, 2)  # Sum of products
    
    # Compute epistatic GRM using Newton's identities
    n_individuals = dist_geno.n_individuals
    G_aa = DArray((n_individuals, n_individuals), workers()) do I
        compute_epistatic_block_symmetric(
            dist_geno, e1_dist, e2_dist, I
        )
    end
    
    return G_aa
end

"""
Block-wise epistasis computation with compression
"""
function distributed_blockwise_epistasis(
    dist_geno::DistributedGenotypeMatrix{T},
    max_memory_gb::Float64
) where T
    n_individuals = dist_geno.n_individuals
    n_snps = dist_geno.n_snps
    
    # Estimate block size based on memory constraints
    bytes_per_element = sizeof(T)
    elements_per_gb = 1e9 / bytes_per_element
    max_elements = max_memory_gb * elements_per_gb
    
    # Block size for quadratic scaling
    block_size = floor(Int, sqrt(max_elements / n_individuals))
    n_blocks = ceil(Int, n_snps / block_size)
    
    println("  Block size: $block_size SNPs")
    println("  Number of blocks: $n_blocks")
    
    # Initialize result
    G_aa = SharedArray{T}(n_individuals, n_individuals)
    
    # Process blocks in parallel
    @sync @distributed for block_pair in vec([(i,j) for i in 1:n_blocks for j in i:n_blocks])
        block_i, block_j = block_pair
        
        # Compute contribution from this block pair
        contribution = compute_epistasis_block_contribution(
            dist_geno, block_i, block_j, block_size
        )
        
        # Add to result (thread-safe)
        @sync @distributed for idx in 1:n_individuals^2
            i = (idx - 1) ÷ n_individuals + 1
            j = (idx - 1) % n_individuals + 1
            G_aa[i,j] += contribution[i,j]
        end
    end
    
    # Normalize
    n_pairs = n_snps * (n_snps - 1) / 2
    G_aa ./= n_pairs
    
    return G_aa
end

"""
Distributed sparse epistasis detection
"""
function distributed_sparse_epistasis(
    dist_geno::DistributedGenotypeMatrix{T},
    phenotypes::Vector{T};
    target_sparsity::Float64 = 0.01,
    screening_method::Symbol = :sure_independence
) where T
    n_snps = dist_geno.n_snps
    max_interactions = ceil(Int, n_snps^2 * target_sparsity)
    
    println("Distributed sparse epistasis detection...")
    println("  Target sparsity: $target_sparsity")
    println("  Maximum interactions: $max_interactions")
    
    # Step 1: Marginal screening on each worker
    marginal_scores = distributed_marginal_screening(
        dist_geno, phenotypes, screening_method
    )
    
    # Step 2: Select candidate SNPs
    n_candidates = min(5000, ceil(Int, sqrt(max_interactions) * 10))
    candidate_snps = select_top_snps(marginal_scores, n_candidates)
    
    # Step 3: Distributed interaction testing
    interactions = distributed_interaction_testing(
        dist_geno, phenotypes, candidate_snps, max_interactions
    )
    
    return interactions
end

"""
Distributed marginal screening
"""
function distributed_marginal_screening(
    dist_geno::DistributedGenotypeMatrix{T},
    phenotypes::Vector{T},
    method::Symbol
) where T
    # Broadcast phenotypes to all workers
    pheno_shared = SharedArray(phenotypes)
    
    # Compute marginal statistics on each worker
    marginal_stats = @distributed (vcat) for (worker, chunk_range) in dist_geno.local_chunks
        @spawnat worker begin
            local_data = localpart(dist_geno.data)
            col_range = chunk_range[2]
            
            if method == :sure_independence
                scores = sure_independence_screening(local_data, pheno_shared)
            elseif method == :correlation
                scores = correlation_screening(local_data, pheno_shared)
            end
            
            (col_range, scores)
        end
    end
    
    # Aggregate results
    all_scores = zeros(T, dist_geno.n_snps)
    for (col_range, scores) in marginal_stats
        all_scores[col_range] = scores
    end
    
    return all_scores
end

"""
SURE Independence Screening
"""
function sure_independence_screening(
    genotypes::Union{Array{T,2}, CuArray{T,2}},
    phenotypes::SharedArray{T}
) where T
    n_individuals, n_snps = size(genotypes)
    scores = zeros(T, n_snps)
    
    # Convert to appropriate array type
    y = isa(genotypes, CuArray) ? CuArray(phenotypes) : Array(phenotypes)
    
    # Compute marginal utilities
    for j in 1:n_snps
        x = genotypes[:, j]
        
        # Standardize
        x_std = (x .- mean(x)) ./ std(x)
        y_std = (y .- mean(y)) ./ std(y)
        
        # Marginal utility
        scores[j] = abs(dot(x_std, y_std)) / n_individuals
    end
    
    return scores
end

# === Utility Functions ===

"""
Center genotype chunk in-place
"""
function center_genotypes_chunk!(
    chunk::Union{Array{T,2}, CuArray{T,2}},
    allele_freq::Vector{T}
) where T
    n_individuals, n_snps = size(chunk)
    
    if isa(chunk, CuArray)
        # GPU kernel
        @cuda threads=256 blocks=cld(n_individuals * n_snps, 256) center_kernel!(
            chunk, CuArray(allele_freq), n_individuals, n_snps
        )
    else
        # CPU version
        for j in 1:n_snps
            chunk[:, j] .-= 2 * allele_freq[j]
        end
    end
end

@kernel function center_kernel!(chunk, freq, n_ind, n_snps)
    idx = @index(Global)
    
    if idx <= n_ind * n_snps
        i = (idx - 1) % n_ind + 1
        j = (idx - 1) ÷ n_ind + 1
        
        @inbounds chunk[i, j] -= 2 * freq[j]
    end
end

"""
Get centered genotype chunk for specific indices
"""
function get_centered_chunk(
    dist_geno::DistributedGenotypeMatrix{T},
    row_range::UnitRange{Int},
    col_range::Union{Colon, UnitRange{Int}}
) where T
    # Find workers that have needed data
    relevant_workers = find_relevant_workers(dist_geno, row_range, col_range)
    
    # Gather chunks from workers
    chunks = []
    for worker in relevant_workers
        chunk = @spawnat worker begin
            local_data = localpart(dist_geno.data)
            # Extract relevant part
            # ... implementation details ...
        end
        push!(chunks, fetch(chunk))
    end
    
    # Combine chunks
    return vcat(chunks...)
end

"""
Symmetrize distributed matrix
"""
function symmetrize_distributed_matrix!(A::DArray{T, 2}) where T
    n = size(A, 1)
    
    @sync @distributed for idx in 1:n^2
        i = (idx - 1) ÷ n + 1
        j = (idx - 1) % n + 1
        
        if i < j
            A[j, i] = A[i, j]
        end
    end
end

"""
Save distributed results efficiently
"""
function save_distributed_results(
    filename::String,
    dist_data::Union{DArray, SharedArray};
    compression::Symbol = :gzip
)
    if isa(dist_data, DArray)
        # Gather to master and save
        data = convert(Array, dist_data)
    else
        data = Array(dist_data)
    end
    
    # Save with compression
    if compression == :gzip
        save(filename * ".jld2", "data", data, compress=true)
    elseif compression == :bson
        BSON.@save filename * ".bson" data
    else
        save(filename * ".jld2", "data", data)
    end
end

"""
Load and distribute saved data
"""
function load_distributed_data(
    filename::String;
    n_workers::Int = nworkers()
)
    # Load data
    if endswith(filename, ".jld2")
        data = load(filename, "data")
    elseif endswith(filename, ".bson")
        BSON.@load filename data
    else
        error("Unknown file format")
    end
    
    # Distribute
    return distribute(data, procs=workers()[1:n_workers])
end

end # module DistributedComputing

# ===== src/gpu_optimization.jl =====
"""
Advanced GPU optimization techniques for maximum performance
"""

module GPUOptimization

using CUDA
using KernelAbstractions
using CUDAKernels
using Adapt
using StaticArrays

export optimize_gpu_computation!, GPUConfig, 
       fused_grm_kernel!, mixed_precision_epistasis

"""
GPU configuration for optimal performance
"""
struct GPUConfig
    device::CuDevice
    max_threads::Int
    max_blocks::Int
    shared_memory::Int
    warp_size::Int
    compute_capability::VersionNumber
end

"""
Auto-tune GPU parameters for specific hardware
"""
function auto_tune_gpu(;verbose::Bool = true)
    device = CUDA.device()
    
    config = GPUConfig(
        device,
        attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
        attribute(device, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE),
        CUDA.capability(device)
    )
    
    if verbose
        println("GPU Configuration:")
        println("  Device: $(CUDA.name(device))")
        println("  Compute capability: $(config.compute_capability)")
        println("  Max threads/block: $(config.max_threads)")
        println("  Shared memory/block: $(config.shared_memory) bytes")
    end
    
    return config
end

"""
Optimized fused kernel for GRM computation
Combines centering and multiplication in single pass
"""
function fused_grm_kernel!(
    G::CuArray{T, 2},
    genotypes::CuArray{T, 2},
    allele_freq::CuVector{T};
    config::GPUConfig = auto_tune_gpu(verbose=false)
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Optimal block size based on hardware
    block_size = min(32, config.max_threads)  # 32x32 blocks typical for matrix ops
    grid_size = (cld(n_individuals, block_size), cld(n_individuals, block_size))
    
    # Launch fused kernel
    @cuda threads=(block_size, block_size) blocks=grid_size shmem=sizeof(T)*2*block_size*block_size fused_grm_compute!(
        G, genotypes, allele_freq, n_individuals, n_snps, Val(block_size)
    )
    
    return G
end

function fused_grm_compute!(
    G::CuDeviceArray{T, 2},
    genotypes::CuDeviceArray{T, 2},
    allele_freq::CuDeviceArray{T, 1},
    n_ind::Int,
    n_snps::Int,
    ::Val{BLOCK_SIZE}
) where {T, BLOCK_SIZE}
    # Thread and block indices
    tid_x = threadIdx().x
    tid_y = threadIdx().y
    bid_x = blockIdx().x
    bid_y = blockIdx().y
    
    # Global indices
    i = (bid_x - 1) * BLOCK_SIZE + tid_x
    j = (bid_y - 1) * BLOCK_SIZE + tid_y
    
    # Shared memory for tile computation
    tile_g1 = @cuDynamicSharedMem(T, (BLOCK_SIZE, BLOCK_SIZE))
    tile_g2 = @cuDynamicSharedMem(T, (BLOCK_SIZE, BLOCK_SIZE), sizeof(T) * BLOCK_SIZE * BLOCK_SIZE)
    
    # Initialize accumulator
    sum_ij = zero(T)
    
    # Tile-based computation
    n_tiles = cld(n_snps, BLOCK_SIZE)
    
    for tile in 1:n_tiles
        # Load tiles with bounds checking
        k = (tile - 1) * BLOCK_SIZE + tid_y
        
        if i <= n_ind && k <= n_snps
            @inbounds tile_g1[tid_x, tid_y] = genotypes[i, k] - 2 * allele_freq[k]
        else
            tile_g1[tid_x, tid_y] = zero(T)
        end
        
        if j <= n_ind && k <= n_snps
            @inbounds tile_g2[tid_y, tid_x] = genotypes[j, k] - 2 * allele_freq[k]
        else
            tile_g2[tid_y, tid_x] = zero(T)
        end
        
        sync_threads()
        
        # Compute partial dot product
        if i <= n_ind && j <= n_ind
            for k_local in 1:BLOCK_SIZE
                @inbounds sum_ij += tile_g1[tid_x, k_local] * tile_g2[tid_y, k_local]
            end
        end
        
        sync_threads()
    end
    
    # Write result with scaling
    if i <= n_ind && j <= n_ind
        scale = sum(k -> 2 * allele_freq[k] * (1 - allele_freq[k]), 1:n_snps)
        @inbounds G[i, j] = sum_ij / scale
    end
    
    return nothing
end

"""
Mixed precision computation for epistasis
Uses FP16 for computation, FP32 for accumulation
"""
function mixed_precision_epistasis(
    genotypes::CuArray{Float32, 2};
    use_tensor_cores::Bool = CUDA.capability(device()) >= v"7.0"
)
    n_individuals, n_snps = size(genotypes)
    
    # Convert to FP16 for computation
    genotypes_fp16 = CuArray{Float16}(genotypes)
    
    if use_tensor_cores
        # Use tensor cores for massive speedup
        G_aa = tensor_core_epistasis(genotypes_fp16)
    else
        # Standard mixed precision
        G_aa = standard_mixed_precision(genotypes_fp16)
    end
    
    return G_aa
end

"""
Tensor core accelerated epistasis computation
"""
function tensor_core_epistasis(genotypes_fp16::CuArray{Float16, 2})
    n_individuals, n_snps = size(genotypes_fp16)
    
    # Prepare for WMMA (Warp Matrix Multiply Accumulate)
    # Tensor cores work on 16x16x16 tiles
    WMMA_M = WMMA_N = WMMA_K = 16
    
    # Pad matrices to multiple of 16
    n_ind_padded = cld(n_individuals, WMMA_M) * WMMA_M
    n_snps_padded = cld(n_snps, WMMA_K) * WMMA_K
    
    # Padded matrices
    W_padded = CUDA.zeros(Float16, n_ind_padded, n_snps_padded)
    W_padded[1:n_individuals, 1:n_snps] = genotypes_fp16
    
    # Result matrix (FP32 for accuracy)
    G_aa = CUDA.zeros(Float32, n_ind_padded, n_ind_padded)
    
    # Launch tensor core kernel
    threads = (32, 4)  # Warp-level operation
    blocks = (cld(n_ind_padded, WMMA_M), cld(n_ind_padded, WMMA_N))
    
    @cuda threads=threads blocks=blocks tensor_epistasis_kernel!(
        G_aa, W_padded, n_ind_padded, n_snps_padded
    )
    
    # Extract actual result
    return G_aa[1:n_individuals, 1:n_individuals]
end

"""
Stream-based computation for overlapping compute and memory transfer
"""
function streaming_grm_computation(
    genotypes::Array{T, 2};
    n_streams::Int = 4,
    chunk_size::Int = 1000
) where T
    n_individuals, n_snps = size(genotypes)
    
    # Create CUDA streams
    streams = [CuStream() for _ in 1:n_streams]
    
    # Allocate device memory for each stream
    d_genotypes = [CUDA.zeros(T, n_individuals, chunk_size) for _ in 1:n_streams]
    d_results = [CUDA.zeros(T, n_individuals, n_individuals) for _ in 1:n_streams]
    
    # Final result
    G = CUDA.zeros(T, n_individuals, n_individuals)
    
    # Process chunks with overlapping
    chunk_starts = 1:chunk_size:n_snps
    
    for (idx, start_idx) in enumerate(chunk_starts)
        stream_idx = (idx - 1) % n_streams + 1
        stream = streams[stream_idx]
        
        end_idx = min(start_idx + chunk_size - 1, n_snps)
        chunk = genotypes[:, start_idx:end_idx]
        
        # Async copy to device
        copyto!(d_genotypes[stream_idx], chunk, stream=stream)
        
        # Compute on stream
        CUDA.@sync stream=stream begin
            compute_grm_chunk!(
                d_results[stream_idx],
                d_genotypes[stream_idx],
                end_idx - start_idx + 1
            )
        end
        
        # Accumulate result
        CUDA.@sync stream=stream begin
            G .+= d_results[stream_idx]
        end
    end
    
    # Wait for all streams
    for stream in streams
        synchronize(stream)
    end
    
    # Normalize
    G ./= n_snps
    
    return G
end

"""
Optimized memory access patterns for coalesced reads
"""
function optimize_memory_access!(
    kernel_func::Function,
    data::CuArray{T, N};
    access_pattern::Symbol = :coalesced
) where {T, N}
    if access_pattern == :coalesced
        # Ensure coalesced memory access
        # Transpose if necessary for column-major access
        if N == 2 && size(data, 1) < size(data, 2)
            data = transpose(data)
        end
    elseif access_pattern == :texture
        # Use texture memory for spatially local access
        # (Requires specific kernel design)
    end
    
    return data
end

"""
Dynamic parallelism for adaptive computation
"""
function adaptive_epistasis_kernel!(
    interactions::CuArray{Tuple{Int32, Int32}, 1},
    scores::CuArray{T, 1},
    genotypes::CuArray{T, 2},
    threshold::T
) where T
    # Parent kernel launches child kernels dynamically
    n_snps = size(genotypes, 2)
    
    @cuda threads=256 blocks=cld(n_snps, 256) parent_kernel!(
        interactions, scores, genotypes, threshold, n_snps
    )
end

function parent_kernel!(
    interactions, scores, genotypes, threshold, n_snps
)
    snp_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if snp_i <= n_snps
        # Check if this SNP is promising
        marginal_score = compute_marginal_score(genotypes, snp_i)
        
        if marginal_score > threshold
            # Launch child kernel for detailed analysis
            @cuda dynamic=true threads=64 child_kernel!(
                interactions, scores, genotypes, snp_i, n_snps
            )
        end
    end
end

"""
Persistent kernel for continuous processing
"""
function persistent_grm_kernel!(
    G::CuArray{T, 2},
    work_queue::CuArray{Tuple{Int, Int}, 1},
    genotypes::CuArray{T, 2}
) where T
    # Persistent threads continuously process work
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n_threads = blockDim().x * gridDim().x
    
    while true
        # Atomically get next work item
        work_idx = atomic_add!(work_counter, 1)
        
        if work_idx > length(work_queue)
            break
        end
        
        i, j = work_queue[work_idx]
        
        # Compute GRM element
        compute_grm_element!(G, genotypes, i, j)
    end
end

"""
Warp-level primitives for efficient reduction
"""
function warp_reduce_sum(val::T) where T
    # Butterfly reduction within warp
    offset = 16
    while offset > 0
        val += shfl_down_sync(0xffffffff, val, offset)
        offset ÷= 2
    end
    return val
end

"""
Cooperative groups for flexible synchronization
"""
function cooperative_epistasis!(
    G_aa::CuArray{T, 2},
    genotypes::CuArray{T, 2}
) where T
    # Use cooperative groups for block-wide operations
    # Requires CUDA.jl support for cooperative groups
    
    # Block-wide synchronization and reduction
    # ... implementation ...
end

end # module GPUOptimization

# ===== src/multivariate_extension.jl =====
"""
Multivariate trait analysis with epistasis
"""

module MultivariateAnalysis

using LinearAlgebra
using Statistics
using Distributions
using CUDA

export MultivariatePhenotypes, MultivariateGBLUP,
       multivariate_epistasis_gblup, genetic_correlation_analysis

"""
Multivariate phenotype structure
"""
struct MultivariatePhenotypes{T<:AbstractFloat}
    traits::Matrix{T}  # individuals × traits
    trait_names::Vector{Symbol}
    missing_mask::BitMatrix
    correlation_matrix::Matrix{T}
end

"""
Multivariate GBLUP model with epistasis
"""
struct MultivariateGBLUP{T<:AbstractFloat}
    G::CuArray{T, 2}
    G_aa::Union{Nothing, CuArray{T, 2}}
    genetic_covariances::Dict{Symbol, Matrix{T}}
    residual_covariances::Matrix{T}
    heritabilities::Matrix{T}  # Trait × Component
end

"""
Fit multivariate GBLUP with epistasis
"""
function multivariate_epistasis_gblup(
    genotypes::GenotypeMatrix{T},
    phenotypes::MultivariatePhenotypes{T};
    include_epistasis::Bool = true,
    genetic_correlation_constraint::Union{Nothing, Symbol} = nothing
) where T
    n_individuals = genotypes.n_individuals
    n_traits = size(phenotypes.traits, 2)
    
    # Compute relationship matrices
    G = compute_grm!(genotypes)
    G_aa = include_epistasis ? compute_epistatic_grm!(genotypes) : nothing
    
    # Initialize covariance matrices
    genetic_cov = Dict{Symbol, Matrix{T}}()
    genetic_cov[:additive] = initialize_genetic_covariance(phenotypes, :additive)
    if include_epistasis
        genetic_cov[:epistatic] = initialize_genetic_covariance(phenotypes, :epistatic)
    end
    residual_cov = initialize_residual_covariance(phenotypes)
    
    # Fit multivariate model
    model = fit_multivariate_reml(
        phenotypes.traits, G, G_aa,
        genetic_cov, residual_cov,
        constraint = genetic_correlation_constraint
    )
    
    return model
end

"""
Fit multivariate REML
"""
function fit_multivariate_reml(
    Y::Matrix{T},  # Phenotype matrix
    G::CuArray{T, 2},
    G_aa::Union{Nothing, CuArray{T, 2}},
    genetic_cov_init::Dict{Symbol, Matrix{T}},
    residual_cov_init::Matrix{T};
    constraint::Union{Nothing, Symbol} = nothing,
    max_iter::Int = 100
) where T
    n_individuals, n_traits = size(Y)
    
    # Current estimates
    genetic_cov = deepcopy(genetic_cov_init)
    residual_cov = copy(residual_cov_init)
    
    # Convert to GPU
    Y_gpu = CuArray(Y)
    
    converged = false
    iter = 0
    log_lik_prev = -Inf
    
    while !converged && iter < max_iter
        iter += 1
        
        # E-step: Compute expected values
        U_add, U_epi = compute_genetic_values_multivariate(
            Y_gpu, G, G_aa, genetic_cov, residual_cov
        )
        
        # M-step: Update covariance matrices
        genetic_cov[:additive] = update_genetic_covariance(
            U_add, G, constraint
        )
        
        if G_aa !== nothing
            genetic_cov[:epistatic] = update_genetic_covariance(
                U_epi, G_aa, constraint
            )
        end
        
        # Update residual covariance
        residuals = Y_gpu - U_add
        if G_aa !== nothing
            residuals -= U_epi
        end
        residual_cov = cov(Array(residuals))
        
        # Compute log-likelihood
        log_lik = multivariate_reml_loglik(
            Y_gpu, G, G_aa, genetic_cov, residual_cov
        )
        
        # Check convergence
        if abs(log_lik - log_lik_prev) < 1e-6
            converged = true
        end
        log_lik_prev = log_lik
        
        if iter % 10 == 0
            println("Iteration $iter: Log-likelihood = $log_lik")
        end
    end
    
    # Compute heritabilities
    heritabilities = compute_multivariate_heritabilities(
        genetic_cov, residual_cov
    )
    
    return MultivariateGBLUP(
        G, G_aa, genetic_cov, residual_cov, heritabilities
    )
end

"""
Genetic correlation analysis between traits
"""
function genetic_correlation_analysis(
    model::MultivariateGBLUP{T};
    components::Vector{Symbol} = [:additive, :epistatic],
    compute_se::Bool = true
) where T
    n_traits = size(model.residual_covariances, 1)
    
    correlations = Dict{Symbol, Matrix{T}}()
    standard_errors = Dict{Symbol, Matrix{T}}()
    
    for component in components
        if haskey(model.genetic_covariances, component)
            cov_matrix = model.genetic_covariances[component]
            
            # Compute genetic correlations
            corr_matrix = cov2cor(cov_matrix)
            correlations[component] = corr_matrix
            
            # Compute standard errors if requested
            if compute_se
                se_matrix = compute_genetic_correlation_se(
                    cov_matrix, model, component
                )
                standard_errors[component] = se_matrix
            end
        end
    end
    
    return correlations, standard_errors
end

"""
Compute genetic values for multivariate model
"""
function compute_genetic_values_multivariate(
    Y::CuArray{T, 2},
    G::CuArray{T, 2},
    G_aa::Union{Nothing, CuArray{T, 2}},
    genetic_cov::Dict{Symbol, Matrix{T}},
    residual_cov::Matrix{T}
) where T
    n_individuals, n_traits = size(Y)
    
    # Build kronecker structured covariance
    V = kron(residual_cov, I(n_individuals))
    V += kron(genetic_cov[:additive], G)
    
    if G_aa !== nothing && haskey(genetic_cov, :epistatic)
        V += kron(genetic_cov[:epistatic], G_aa)
    end
    
    # Solve for genetic values
    V_inv = inv(V)
    
    # Additive genetic values
    K_add = kron(genetic_cov[:additive], G)
    U_add_vec = K_add * V_inv * vec(Y)
    U_add = reshape(U_add_vec, n_individuals, n_traits)
    
    # Epistatic genetic values
    if G_aa !== nothing
        K_epi = kron(genetic_cov[:epistatic], G_aa)
        U_epi_vec = K_epi * V_inv * vec(Y)
        U_epi = reshape(U_epi_vec, n_individuals, n_traits)
    else
        U_epi = nothing
    end
    
    return U_add, U_epi
end

"""
Update genetic covariance with constraints
"""
function update_genetic_covariance(
    U::CuArray{T, 2},
    K::CuArray{T, 2},
    constraint::Union{Nothing, Symbol}
) where T
    n_individuals, n_traits = size(U)
    
    # Compute raw covariance
    S = U' * inv(K) * U / n_individuals
    
    # Apply constraints
    if constraint === nothing
        return Array(S)
    elseif constraint == :diagonal
        # Independent traits
        return Diagonal(diag(Array(S)))
    elseif constraint == :factor
        # Factor analytic structure
        return factor_analysis_constraint(Array(S))
    elseif constraint == :positive_definite
        # Ensure positive definiteness
        return nearest_positive_definite(Array(S))
    end
end

"""
Compute multivariate heritabilities
"""
function compute_multivariate_heritabilities(
    genetic_cov::Dict{Symbol, Matrix{T}},
    residual_cov::Matrix{T}
) where T
    n_traits = size(residual_cov, 1)
    
    # Total phenotypic covariance
    phenotypic_cov = residual_cov
    for (_, cov) in genetic_cov
        phenotypic_cov += cov
    end
    
    # Heritabilities for each trait and component
    heritabilities = zeros(T, n_traits, length(genetic_cov) + 1)
    
    for (idx, (component, cov)) in enumerate(genetic_cov)
        for trait in 1:n_traits
            heritabilities[trait, idx] = cov[trait, trait] / phenotypic_cov[trait, trait]
        end
    end
    
    # Total heritabilities
    total_genetic_var = zeros(n_traits)
    for (_, cov) in genetic_cov
        total_genetic_var .+= diag(cov)
    end
    
    heritabilities[:, end] = total_genetic_var ./ diag(phenotypic_cov)
    
    return heritabilities
end

"""
Test pleiotropic epistatic effects
"""
function test_pleiotropic_epistasis(
    model::MultivariateGBLUP{T};
    trait_pairs::Vector{Tuple{Int, Int}} = all_trait_pairs(model),
    method::Symbol = :likelihood_ratio
) where T
    results = Dict{Tuple{Int, Int}, NamedTuple}()
    
    for (trait1, trait2) in trait_pairs
        if method == :likelihood_ratio
            result = likelihood_ratio_test_pleiotropy(
                model, trait1, trait2
            )
        elseif method == :score_test
            result = score_test_pleiotropy(
                model, trait1, trait2
            )
        end
        
        results[(trait1, trait2)] = result
    end
    
    return results
end

# === Utility functions ===

"""
Initialize genetic covariance matrix
"""
function initialize_genetic_covariance(
    phenotypes::MultivariatePhenotypes{T},
    component::Symbol
) where T
    n_traits = size(phenotypes.traits, 2)
    
    # Use phenotypic correlations as starting point
    phenotypic_cov = cov(phenotypes.traits, dims=1)
    
    # Scale based on expected contribution
    if component == :additive
        scale = 0.4  # Assume 40% additive
    elseif component == :epistatic
        scale = 0.1  # Assume 10% epistatic
    else
        scale = 0.05
    end
    
    return phenotypic_cov * scale
end

"""
Nearest positive definite matrix
"""
function nearest_positive_definite(A::Matrix{T}) where T
    # Eigendecomposition
    eigen_decomp = eigen(Symmetric(A))
    
    # Set negative eigenvalues to small positive value
    eigen_decomp.values[eigen_decomp.values .< 1e-8] .= 1e-8
    
    # Reconstruct
    return eigen_decomp.vectors * Diagonal(eigen_decomp.values) * eigen_decomp.vectors'
end

"""
Factor analytic constraint for genetic covariance
"""
function factor_analysis_constraint(S::Matrix{T}; n_factors::Int = 2) where T
    n_traits = size(S, 1)
    
    # Simplified factor analysis
    eigen_decomp = eigen(Symmetric(S), sortby = x -> -x)
    
    # Keep top factors
    loadings = eigen_decomp.vectors[:, 1:n_factors]
    factors = Diagonal(eigen_decomp.values[1:n_factors])
    
    # Specific variances
    specific = Diagonal(diag(S) - sum(loadings.^2, dims=2))
    
    # Reconstructed covariance
    return loadings * factors * loadings' + specific
end

end # module MultivariateAnalysis

# ===== src/breeding_optimization.jl =====
"""
Breeding program optimization using epistatic information
"""

module BreedingOptimization

using JuMP
using Gurobi
using Combinatorics
using LinearAlgebra
using Random

export OptimalMatingPlan, optimize_breeding_program,
       epistatic_mate_allocation, long_term_genetic_gain

"""
Optimal mating plan considering epistasis
"""
struct OptimalMatingPlan{T<:AbstractFloat}
    matings::Vector{Tuple{Int, Int}}  # Parent pairs
    expected_genetic_gain::T
    expected_genetic_variance::T
    inbreeding_rate::T
    epistatic_value::T
end

"""
Optimize breeding program with epistatic effects
"""
function optimize_breeding_program(
    population::PopulationData{T},
    model::OrthogonalGBLUP{T};
    n_matings::Int = 100,
    offspring_per_mating::Int = 20,
    selection_intensity::Float64 = 0.2,
    optimization_criterion::Symbol = :genetic_gain,
    constraints::Dict{Symbol, Any} = Dict()
) where T
    n_candidates = population.genotypes.n_individuals
    
    # Get breeding values and epistatic effects
    gebv_add = get_additive_breeding_values(model, population)
    gebv_epi = get_epistatic_values(model, population)
    
    # Solve optimization problem
    if optimization_criterion == :genetic_gain
        plan = maximize_genetic_gain(
            gebv_add, gebv_epi, model.G, model.G_aa,
            n_matings, constraints
        )
    elseif optimization_criterion == :genetic_variance
        plan = maximize_genetic_variance(
            gebv_add, gebv_epi, model.G, model.G_aa,
            n_matings, constraints
        )
    elseif optimization_criterion == :epistatic_value
        plan = maximize_epistatic_combinations(
            population, model, n_matings, constraints
        )
    end
    
    return plan
end

"""
Maximize genetic gain with constraints
"""
function maximize_genetic_gain(
    gebv_add::Vector{T},
    gebv_epi::Vector{T},
    G::CuArray{T, 2},
    G_aa::CuArray{T, 2},
    n_matings::Int,
    constraints::Dict{Symbol, Any}
) where T
    n_candidates = length(gebv_add)
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    
    # Decision variables: mating allocation matrix
    @variable(model, x[1:n_candidates, 1:n_candidates], Bin)
    
    # Objective: maximize expected genetic value
    expected_value = zeros(AffExpr, n_candidates, n_candidates)
    for i in 1:n_candidates
        for j in i:n_candidates
            # Expected value of offspring
            ev = 0.5 * (gebv_add[i] + gebv_add[j])
            
            # Add epistatic expectation
            if G_aa !== nothing
                epistatic_expect = compute_epistatic_expectation(
                    i, j, gebv_epi, G_aa
                )
                ev += epistatic_expect
            end
            
            expected_value[i,j] = expected_value[j,i] = ev
        end
    end
    
    @objective(model, Max, sum(expected_value .* x))
    
    # Constraints
    
    # Number of matings
    @constraint(model, sum(x) == 2 * n_matings)  # Symmetric matrix
    
    # Mating restrictions
    if haskey(constraints, :max_use_per_parent)
        max_use = constraints[:max_use_per_parent]
        @constraint(model, [i=1:n_candidates], sum(x[i,:]) <= max_use)
    end
    
    # Inbreeding constraint
    if haskey(constraints, :max_inbreeding_rate)
        max_f = constraints[:max_inbreeding_rate]
        G_cpu = Array(G)
        
        # Average relationship constraint
        avg_relationship = sum((G_cpu[i,j] * x[i,j] for i in 1:n_candidates, j in 1:n_candidates))
        @constraint(model, avg_relationship / sum(x) <= 1 + 2*max_f)
    end
    
    # Solve
    optimize!(model)
    
    # Extract solution
    x_sol = value.(x)
    matings = Tuple{Int, Int}[]
    
    for i in 1:n_candidates
        for j in i:n_candidates
            if x_sol[i,j] > 0.5
                push!(matings, (i, j))
            end
        end
    end
    
    # Compute plan statistics
    plan = compute_plan_statistics(matings, gebv_add, gebv_epi, G, G_aa)
    
    return plan
end

"""
Epistatic mate allocation algorithm
"""
function epistatic_mate_allocation(
    population::PopulationData{T},
    model::OrthogonalGBLUP{T};
    n_matings::Int = 100,
    method::Symbol = :complementarity
) where T
    n_candidates = population.genotypes.n_individuals
    
    if method == :complementarity
        # Find complementary epistatic combinations
        matings = find_complementary_pairs(population, model, n_matings)
    elseif method == :positive_assortative
        # Positive assortative mating for epistasis
        matings = positive_assortative_epistasis(population, model, n_matings)
    elseif method == :diversity_epistasis
        # Balance diversity and epistatic value
        matings = diversity_aware_epistasis(population, model, n_matings)
    end
    
    return matings
end

"""
Find complementary epistatic pairs
"""
function find_complementary_pairs(
    population::PopulationData{T},
    model::OrthogonalGBLUP{T},
    n_matings::Int
) where T
    n_candidates = population.genotypes.n_individuals
    genotypes = Array(population.genotypes.data)
    
    # Identify beneficial epistatic interactions
    interactions = identify_beneficial_interactions(model)
    
    # Score all possible matings
    mating_scores = zeros(T, n_candidates, n_candidates)
    
    for i in 1:n_candidates
        for j in i:n_candidates
            score = compute_complementarity_score(
                genotypes[i, :], genotypes[j, :], interactions
            )
            mating_scores[i, j] = mating_scores[j, i] = score
        end
    end
    
    # Select top matings with diversity constraint
    selected_matings = select_diverse_matings(mating_scores, n_matings)
    
    return selected_matings
end

"""
Long-term genetic gain optimization
"""
function long_term_genetic_gain(
    initial_population::PopulationData{T},
    model::OrthogonalGBLUP{T};
    n_generations::Int = 10,
    n_matings_per_gen::Int = 50,
    selection_intensity::Float64 = 0.1,
    update_model_freq::Int = 2
) where T
    # Dynamic programming approach
    populations = [initial_population]
    models = [model]
    genetic_gains = T[]
    
    for gen in 1:n_generations
        current_pop = populations[end]
        current_model = models[end]
        
        # Optimize mating for multiple generations ahead
        if gen % update_model_freq == 0
            # Re-estimate model
            new_model, _ = orthogonal_epistasis_gblup(current_pop)
            push!(models, new_model)
            current_model = new_model
        end
        
        # Plan matings considering future generations
        mating_plan = multi_generation_optimization(
            current_pop, current_model,
            n_generations - gen + 1,  # Remaining generations
            n_matings_per_gen
        )
        
        # Generate offspring
        offspring_pop = execute_mating_plan(current_pop, mating_plan)
        
        # Selection
        selected_pop = truncation_selection(
            offspring_pop, selection_intensity
        )
        
        push!(populations, selected_pop)
        
        # Track genetic gain
        gain = mean(selected_pop.phenotypes.values) - mean(current_pop.phenotypes.values)
        push!(genetic_gains, gain)
    end
    
    return populations, models, genetic_gains
end

"""
Multi-generation optimization using dynamic programming
"""
function multi_generation_optimization(
    population::PopulationData{T},
    model::OrthogonalGBLUP{T},
    n_future_generations::Int,
    n_matings::Int
) where T
    # Simplified approach: weight short-term and long-term gains
    
    # Short-term gain (current generation)
    short_term_plan = optimize_breeding_program(
        population, model,
        n_matings = n_matings,
        optimization_criterion = :genetic_gain
    )
    
    # Long-term considerations
    if n_future_generations > 1
        # Maintain genetic variance
        variance_plan = optimize_breeding_program(
            population, model,
            n_matings = n_matings,
            optimization_criterion = :genetic_variance
        )
        
        # Weighted combination
        α = 0.7  # Weight for short-term gain
        combined_plan = combine_mating_plans(
            short_term_plan, variance_plan, α
        )
        
        return combined_plan
    else
        return short_term_plan
    end
end

"""
Compute expected epistatic value for offspring
"""
function compute_epistatic_expectation(
    parent1::Int,
    parent2::Int,
    gebv_epi::Vector{T},
    G_aa::CuArray{T, 2}
) where T
    # Expected epistatic value depends on:
    # 1. Parent epistatic values
    # 2. Epistatic relationship between parents
    
    G_aa_cpu = Array(G_aa)
    
    # Simple approximation
    parent_avg = 0.5 * (gebv_epi[parent1] + gebv_epi[parent2])
    relationship_factor = G_aa_cpu[parent1, parent2]
    
    # Expected value with some variance reduction
    expected = parent_avg * (1 + 0.1 * relationship_factor)
    
    return expected
end

"""
Identify beneficial epistatic interactions from model
"""
function identify_beneficial_interactions(model::OrthogonalGBLUP{T}) where T
    # Extract significant epistatic effects
    # This would require storing interaction effects in the model
    
    # Placeholder: return top interactions
    interactions = Vector{Tuple{Int, Int, T}}()
    
    # In practice, would extract from model fitting results
    # For now, return empty
    
    return interactions
end

"""
Compute complementarity score for mating
"""
function compute_complementarity_score(
    genotype1::Vector{T},
    genotype2::Vector{T},
    interactions::Vector{Tuple{Int, Int, T}}
) where T
    score = zero(T)
    
    for (snp1, snp2, effect) in interactions
        # Favorable allele combinations
        allele_sum1 = genotype1[snp1] + genotype2[snp1]
        allele_sum2 = genotype1[snp2] + genotype2[snp2]
        
        # Probability of favorable combination in offspring
        prob = (allele_sum1 / 4) * (allele_sum2 / 4)
        
        score += prob * effect
    end
    
    return score
end

"""
Select diverse matings from score matrix
"""
function select_diverse_matings(
    scores::Matrix{T},
    n_select::Int;
    diversity_weight::T = T(0.1)
) where T
    n_candidates = size(scores, 1)
    selected = Tuple{Int, Int}[]
    used_parents = Set{Int}()
    
    # Greedy selection with diversity
    while length(selected) < n_select
        best_score = -Inf
        best_pair = (0, 0)
        
        for i in 1:n_candidates
            for j in i:n_candidates
                # Skip if already selected
                if (i, j) in selected || (j, i) in selected
                    continue
                end
                
                # Diversity penalty
                diversity_penalty = 0
                if i in used_parents
                    diversity_penalty += diversity_weight
                end
                if j in used_parents
                    diversity_penalty += diversity_weight
                end
                
                adjusted_score = scores[i, j] - diversity_penalty
                
                if adjusted_score > best_score
                    best_score = adjusted_score
                    best_pair = (i, j)
                end
            end
        end
        
        if best_pair[1] > 0
            push!(selected, best_pair)
            push!(used_parents, best_pair[1])
            push!(used_parents, best_pair[2])
        else
            break
        end
    end
    
    return selected
end

"""
Execute mating plan to generate offspring
"""
function execute_mating_plan(
    population::PopulationData{T},
    plan::OptimalMatingPlan{T};
    offspring_per_mating::Int = 20
) where T
    # Implementation would generate offspring based on plan
    # Using functions from main module
    
    # Placeholder
    return population
end

"""
Compute statistics for mating plan
"""
function compute_plan_statistics(
    matings::Vector{Tuple{Int, Int}},
    gebv_add::Vector{T},
    gebv_epi::Vector{T},
    G::CuArray{T, 2},
    G_aa::Union{Nothing, CuArray{T, 2}}
) where T
    n_matings = length(matings)
    
    # Expected genetic gain
    expected_gain = zero(T)
    for (i, j) in matings
        expected_gain += 0.5 * (gebv_add[i] + gebv_add[j])
    end
    expected_gain /= n_matings
    
    # Expected genetic variance (simplified)
    parent_variance = var(gebv_add)
    expected_variance = 0.5 * parent_variance  # Rough approximation
    
    # Inbreeding rate
    G_cpu = Array(G)
    avg_relationship = zero(T)
    for (i, j) in matings
        avg_relationship += G_cpu[i, j]
    end
    avg_relationship /= n_matings
    inbreeding_rate = (avg_relationship - 1) / 2
    
    # Epistatic value
    epistatic_value = zero(T)
    if G_aa !== nothing
        for (i, j) in matings
            epistatic_value += compute_epistatic_expectation(
                i, j, gebv_epi, G_aa
            )
        end
        epistatic_value /= n_matings
    end
    
    return OptimalMatingPlan(
        matings,
        expected_gain,
        expected_variance,
        inbreeding_rate,
        epistatic_value
    )
end

end # module BreedingOptimization

# ===== src/visualization.jl =====
"""
Advanced visualization for epistatic genomic analysis
"""

module Visualization

using Plots
using StatsPlots
using PlotlyJS
using Colors
using NetworkLayout
using Graphs
using DataFrames
using Makie
using GLMakie

export plot_epistasis_network, plot_grm_heatmap,
       plot_variance_decomposition, interactive_manhattan_plot,
       plot_prediction_accuracy_surface

"""
Plot epistatic interaction network
"""
function plot_epistasis_network(
    interactions::Vector{Tuple{Int32, Int32}},
    scores::Vector{T};
    n_top::Int = 100,
    layout_algorithm::Symbol = :spring,
    node_colors::Union{Nothing, Vector} = nothing,
    save_path::Union{Nothing, String} = nothing
) where T
    # Select top interactions
    top_indices = partialsortperm(scores, 1:min(n_top, length(scores)), rev=true)
    top_interactions = interactions[top_indices]
    top_scores = scores[top_indices]
    
    # Build graph
    unique_snps = unique(vcat([i[1] for i in top_interactions], 
                             [i[2] for i in top_interactions]))
    snp_to_node = Dict(snp => i for (i, snp) in enumerate(unique_snps))
    
    g = SimpleGraph(length(unique_snps))
    edge_weights = Float64[]
    
    for (idx, (snp1, snp2)) in enumerate(top_interactions)
        add_edge!(g, snp_to_node[snp1], snp_to_node[snp2])
        push!(edge_weights, Float64(top_scores[idx]))
    end
    
    # Layout
    if layout_algorithm == :spring
        pos = spring_layout(g, weights=edge_weights)
    elseif layout_algorithm == :spectral
        pos = spectral_layout(g)
    else
        pos = circular_layout(g)
    end
    
    # Create plot
    fig = Figure(resolution = (1200, 800))
    ax = Axis(fig[1, 1], 
              title = "Epistatic Interaction Network",
              aspect = DataAspect())
    
    # Plot edges
    for (idx, e) in enumerate(edges(g))
        src_pos = pos[src(e)]
        dst_pos = pos[dst(e)]
        weight = edge_weights[idx]
        
        lines!(ax, [src_pos[1], dst_pos[1]], [src_pos[2], dst_pos[2]],
               linewidth = weight * 5,
               color = (:blue, weight / maximum(edge_weights)),
               transparency = true)
    end
    
    # Plot nodes
    node_sizes = [degree(g, v) * 10 + 20 for v in vertices(g)]
    
    if node_colors === nothing
        node_colors = node_sizes
    end
    
    scatter!(ax, [p[1] for p in pos], [p[2] for p in pos],
             markersize = node_sizes,
             color = node_colors,
             colormap = :viridis,
             strokewidth = 2,
             strokecolor = :black)
    
    # Add labels for high-degree nodes
    for (idx, v) in enumerate(vertices(g))
        if degree(g, v) > 5
            text!(ax, string(unique_snps[v]),
                  position = (pos[v][1], pos[v][2]),
                  textsize = 12,
                  align = (:center, :center))
        end
    end
    
    if save_path !== nothing
        save(save_path, fig)
    end
    
    return fig
end

"""
Interactive heatmap for genomic relationship matrices
"""
function plot_grm_heatmap(
    G::Matrix{T};
    title::String = "Genomic Relationship Matrix",
    cluster::Bool = true,
    save_path::Union{Nothing, String} = nothing
) where T
    n = size(G, 1)
    
    # Cluster if requested
    if cluster
        # Hierarchical clustering
        D = 1 .- G  # Convert to distance
        D[diagind(D)] .= 0
        hc = hclust(D, :average)
        order = hc.order
        G_clustered = G[order, order]
    else
        G_clustered = G
    end
    
    # Create interactive plot with PlotlyJS
    trace = PlotlyJS.heatmap(
        z = G_clustered,
        colorscale = "RdBu",
        zmid = 0,
        text = round.(G_clustered, digits=3),
        hovertemplate = "Individual %{y} vs %{x}<br>Relationship: %{text}<extra></extra>"
    )
    
    layout = PlotlyJS.Layout(
        title = title,
        xaxis = attr(title = "Individual", showgrid = false),
        yaxis = attr(title = "Individual", showgrid = false),
        width = 800,
        height = 800
    )
    
    p = PlotlyJS.plot(trace, layout)
    
    if save_path !== nothing
        PlotlyJS.savefig(p, save_path)
    end
    
    return p
end

"""
Variance decomposition visualization
"""
function plot_variance_decomposition(
    variance_components::VarianceComponents{T};
    save_path::Union{Nothing, String} = nothing
) where T
    # Prepare data
    components = ["Additive", "Epistatic", "Residual"]
    values = [variance_components.σ²_a,
              variance_components.σ²_aa,
              variance_components.σ²_e]
    percentages = values / variance_components.σ²_p * 100
    
    # Create figure with subplots
    fig = Figure(resolution = (1200, 600))
    
    # Pie chart
    ax1 = Axis(fig[1, 1], aspect = DataAspect())
    pie!(ax1, percentages,
         labels = [string(c, "\n", round(p, digits=1), "%") 
                  for (c, p) in zip(components, percentages)],
         colors = [:blue, :orange, :gray])
    
    # Bar chart with heritabilities
    ax2 = Axis(fig[1, 2],
               xlabel = "Heritability Type",
               ylabel = "Value",
               title = "Heritability Estimates")
    
    herit_types = ["h² (narrow)", "H² (broad)", "Epistatic\ncontribution"]
    herit_values = [variance_components.h²,
                    variance_components.H²,
                    variance_components.H² - variance_components.h²]
    
    barplot!(ax2, 1:3, herit_values,
             color = [:blue, :green, :orange],
             bar_labels = [string(round(v, digits=3)) for v in herit_values])
    
    ax2.xticks = (1:3, herit_types)
    ylims!(ax2, 0, 1)
    
    if save_path !== nothing
        save(save_path, fig)
    end
    
    return fig
end

"""
Interactive Manhattan plot for epistatic interactions
"""
function interactive_manhattan_plot(
    interactions::Vector{Tuple{Int32, Int32}},
    scores::Vector{T},
    chromosome_info::Dict{Int, Tuple{Int, Int}};  # chrom -> (start_snp, end_snp)
    threshold::Union{Nothing, T} = nothing,
    save_path::Union{Nothing, String} = nothing
) where T
    # Prepare data
    n_interactions = length(interactions)
    x_positions = zeros(n_interactions)
    y_values = -log10.(scores)
    colors = String[]
    hover_text = String[]
    
    # Assign positions and colors
    chrom_offset = 0
    chrom_colors = ["#1f77b4", "#ff7f0e"]
    
    for (chrom, (start_snp, end_snp)) in sort(chromosome_info)
        for (idx, (snp1, snp2)) in enumerate(interactions)
            if start_snp <= snp1 <= end_snp || start_snp <= snp2 <= end_snp
                # Position at midpoint
                pos1 = snp1 - start_snp + chrom_offset
                pos2 = snp2 - start_snp + chrom_offset
                x_positions[idx] = (pos1 + pos2) / 2
                
                push!(colors, chrom_colors[chrom % 2 + 1])
                push!(hover_text, "SNP$snp1 × SNP$snp2\nScore: $(round(scores[idx], digits=4))")
            end
        end
        chrom_offset += end_snp - start_snp + 1
    end
    
    # Create interactive plot
    trace = PlotlyJS.scatter(
        x = x_positions,
        y = y_values,
        mode = "markers",
        marker = attr(
            size = 5,
            color = colors,
            opacity = 0.7
        ),
        text = hover_text,
        hoverinfo = "text",
        name = "Interactions"
    )
    
    # Add threshold line if provided
    shapes = []
    if threshold !== nothing
        push!(shapes, PlotlyJS.line(
            x0 = 0, x1 = maximum(x_positions),
            y0 = -log10(threshold), y1 = -log10(threshold),
            line = attr(color = "red", dash = "dash")
        ))
    end
    
    layout = PlotlyJS.Layout(
        title = "Epistatic Interaction Manhattan Plot",
        xaxis = attr(title = "Genomic Position"),
        yaxis = attr(title = "-log₁₀(p-value)"),
        shapes = shapes,
        hovermode = "closest"
    )
    
    p = PlotlyJS.plot(trace, layout)
    
    if save_path !== nothing
        PlotlyJS.savefig(p, save_path)
    end
    
    return p
end

"""
3D surface plot for prediction accuracy
"""
function plot_prediction_accuracy_surface(
    results::DataFrame;
    x_var::Symbol = :n_snps,
    y_var::Symbol = :n_individuals,
    z_var::Symbol = :accuracy,
    method_column::Symbol = :method,
    save_path::Union{Nothing, String} = nothing
)
    # Create separate surface for each method
    methods = unique(results[!, method_column])
    
    fig = Figure(resolution = (1400, 600))
    
    for (idx, method) in enumerate(methods)
        method_data = filter(row -> row[method_column] == method, results)
        
        # Create grid
        x_unique = sort(unique(method_data[!, x_var]))
        y_unique = sort(unique(method_data[!, y_var]))
        
        Z = zeros(length(y_unique), length(x_unique))
        
        for (i, y) in enumerate(y_unique)
            for (j, x) in enumerate(x_unique)
                matching = filter(row -> row[x_var] == x && row[y_var] == y, method_data)
                if nrow(matching) > 0
                    Z[i, j] = mean(matching[!, z_var])
                end
            end
        end
        
        ax = Axis3(fig[1, idx],
                   xlabel = string(x_var),
                   ylabel = string(y_var),
                   zlabel = string(z_var),
                   title = string(method))
        
        surface!(ax, x_unique, y_unique, Z,
                 colormap = :viridis,
                 transparency = true)
    end
    
    if save_path !== nothing
        save(save_path, fig)
    end
    
    return fig
end

"""
Animated visualization of genetic gain over generations
"""
function animate_genetic_progress(
    populations::Vector{PopulationData{T}},
    models::Vector{OrthogonalGBLUP{T}};
    trait_name::String = "Yield",
    save_path::Union{Nothing, String} = nothing
) where T
    n_generations = length(populations)
    
    # Prepare data
    mean_phenotypes = [mean(pop.phenotypes.values) for pop in populations]
    var_phenotypes = [var(pop.phenotypes.values) for pop in populations]
    h2_narrow = [m.variance.h² for m in models]
    h2_broad = [m.variance.H² for m in models]
    
    # Create animation
    anim = @animate for gen in 1:n_generations
        layout = @layout [a b; c d]
        
        # Phenotype distribution
        p1 = histogram(populations[gen].phenotypes.values,
                      bins = 30,
                      title = "Generation $gen",
                      xlabel = trait_name,
                      ylabel = "Frequency",
                      legend = false)
        
        # Genetic trend
        p2 = plot(1:gen, mean_phenotypes[1:gen],
                 marker = :circle,
                 title = "Genetic Trend",
                 xlabel = "Generation",
                 ylabel = "Mean $trait_name",
                 legend = false)
        
        # Variance components
        p3 = plot(1:gen, [h2_narrow[1:gen] h2_broad[1:gen]],
                 marker = :circle,
                 title = "Heritability",
                 xlabel = "Generation",
                 ylabel = "Heritability",
                 label = ["h² (narrow)" "H² (broad)"])
        
        # Genetic variance
        p4 = plot(1:gen, var_phenotypes[1:gen],
                 marker = :circle,
                 title = "Phenotypic Variance",
                 xlabel = "Generation",
                 ylabel = "Variance",
                 legend = false)
        
        plot(p1, p2, p3, p4, layout = layout, size = (1000, 800))
    end
    
    if save_path !== nothing
        gif(anim, save_path, fps = 2)
    end
    
    return anim
end

end # module Visualization

# ===== test/comprehensive_tests.jl =====
"""
Comprehensive test suite for DynamicEpistasisGBLUP package
"""

module TestSuite

using Test
using BenchmarkTools
using Random
using LinearAlgebra
using Statistics

# Include all modules to test
using ..DynamicEpistasisGBLUP
using ..SparseEpistasis
using ..DistributedComputing
using ..GPUOptimization
using ..MultivariateAnalysis
using ..BreedingOptimization
using ..Visualization

# Test data generation
function generate_test_data(n_ind::Int = 100, n_snps::Int = 1000)
    Random.seed!(42)
    
    # Simulate small dataset
    pop = simulate_population(
        n_individuals = n_ind,
        n_snps = n_snps,
        n_qtl_additive = 10,
        n_qtl_epistatic = 10,
        h2_narrow = 0.3,
        h2_broad = 0.4
    )
    
    return pop
end

@testset "DynamicEpistasisGBLUP.jl" begin
    
    @testset "Core Functionality" begin
        pop = generate_test_data(50, 500)
        
        @testset "GRM Computation" begin
            G = compute_grm!(pop.genotypes)
            @test size(G) == (50, 50)
            @test issymmetric(Array(G))
            @test all(diag(Array(G)) .>= 0)
        end
        
        @testset "Epistatic GRM" begin
            G_aa = compute_epistatic_grm!(pop.genotypes)
            @test size(G_aa) == (50, 50)
            @test issymmetric(Array(G_aa))
        end
        
        @testset "Orthogonal GBLUP" begin
            model, gebv = orthogonal_epistasis_gblup(pop)
            @test length(gebv) == 50
            @test model.variance.h² > 0
            @test model.variance.H² >= model.variance.h²
        end
    end
    
    @testset "Sparse Epistasis" begin
        pop = generate_test_data(100, 1000)
        
        @testset "Sparse Detection" begin
            interactions, scores = detect_sparse_interactions(
                pop.genotypes.data,
                CuArray(pop.phenotypes.values),
                max_interactions = 100
            )
            
            @test length(interactions) <= 100
            @test all(length.(interactions) .== 2)
        end
        
        @testset "Elastic Net" begin
            # Select some interactions
            interactions = [(Int32(i), Int32(j)) for i in 1:10 for j in (i+1):20]
            
            model = elastic_net_epistasis(
                pop.genotypes.data,
                CuArray(pop.phenotypes.values),
                interactions,
                λ1 = 0.01,
                λ2 = 0.001
            )
            
            @test model.sparsity_level >= 0
            @test model.sparsity_level <= 1
        end
    end
    
    @testset "GPU Optimization" begin
        if CUDA.functional()
            @testset "Auto-tuning" begin
                config = auto_tune_gpu(verbose=false)
                @test config.max_threads > 0
                @test config.compute_capability > v"0.0"
            end
            
            @testset "Fused Kernels" begin
                pop = generate_test_data(100, 1000)
                G = CUDA.zeros(Float32, 100, 100)
                
                fused_grm_kernel!(
                    G,
                    pop.genotypes.data,
                    pop.genotypes.allele_freq
                )
                
                @test all(isfinite.(Array(G)))
            end
        else
            @test_skip "GPU not available"
        end
    end
    
    @testset "Performance Benchmarks" begin
        println("\n=== Performance Benchmarks ===")
        
        # GRM computation scaling
        for n in [100, 500, 1000]
            pop = generate_test_data(n, 5000)
            
            time_grm = @belapsed compute_grm!($pop.genotypes)
            println("GRM computation ($n × $n): $(round(time_grm, digits=3)) seconds")
            
            if n <= 500  # Skip large epistatic GRM
                time_epi = @belapsed compute_epistatic_grm!($pop.genotypes)
                println("Epistatic GRM ($n × $n): $(round(time_epi, digits=3)) seconds")
            end
        end
    end
    
    @testset "Cross-validation" begin
        pop = generate_test_data(200, 2000)
        
        cv_results = cross_validation(
            pop,
            n_folds = 3,
            include_epistasis = true
        )
        
        @test nrow(cv_results) == 3
        @test all(cv_results.accuracy .>= 0)
        @test all(cv_results.accuracy .<= 1)
    end
    
    @testset "Multivariate Analysis" begin
        # Generate multivariate phenotypes
        n_ind = 100
        n_traits = 3
        
        traits = randn(n_ind, n_traits)
        traits[:, 2] += 0.5 * traits[:, 1]  # Correlated traits
        
        mv_pheno = MultivariatePhenotypes(
            traits,
            [:trait1, :trait2, :trait3],
            falses(n_ind, n_traits),
            cor(traits)
        )
        
        pop = generate_test_data(n_ind, 1000)
        
        @testset "Multivariate GBLUP" begin
            mv_model = multivariate_epistasis_gblup(
                pop.genotypes,
                mv_pheno,
                include_epistasis = false  # Faster for testing
            )
            
            @test size(mv_model.heritabilities) == (3, 2)
            @test all(mv_model.heritabilities .>= 0)
        end
    end
    
    @testset "Integration Tests" begin
        # Full pipeline test
        pop = generate_test_data(100, 1000)
        
        # Run selection
        populations, models = simulate_selection(
            pop,
            n_generations = 3,
            selection_intensity = 0.5,
            update_model_frequency = 1
        )
        
        @test length(populations) == 4  # Initial + 3 generations
        @test length(models) == 4
        
        # Check genetic gain
        initial_mean = mean(populations[1].phenotypes.values)
        final_mean = mean(populations[end].phenotypes.values)
        @test final_mean > initial_mean
    end
    
end

# Run tests
@testset "Package Tests" begin
    TestSuite.runtests()
end

end # module TestSuite

# ===== Project.toml =====
"""
[Project.toml content for the package]

name = "DynamicEpistasisGBLUP"
uuid = "12345678-1234-5678-1234-567812345678"
authors = ["Advanced AI Implementation"]
version = "1.0.0"

[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
DistributedArrays = "aaf54ef3-cdf8-58ed-94cc-d5827