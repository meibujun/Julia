function run_pca(dataset::IntegratedDataset; components::Int=3)
    components > 0 || throw(ArgumentError("Number of components must be positive"))
    n_samples = length(dataset.samples)
    n_samples > 1 || throw(ArgumentError("PCA requires at least two samples"))
    mat = dataset.matrix
    centered = mat .- mean(mat, dims=2)
    data = transpose(centered)
    svd_res = svd(data; full=false)
    k = min(components, length(svd_res.S))
    scores = svd_res.U[:, 1:k] * Diagonal(svd_res.S[1:k])
    loadings = svd_res.V[:, 1:k]
    denom = max(n_samples - 1, 1)
    explained_variance = (svd_res.S .^ 2) / denom
    explained_ratio = explained_variance ./ sum(explained_variance)
    return PCAResult(scores, loadings, explained_variance[1:k], explained_ratio[1:k])
end

function _initialise_centroids(data::Matrix{Float64}, k::Int, seed::Int)
    Random.seed!(seed)
    indices = randperm(size(data, 2))[1:k]
    return data[:, indices]
end

function _assign_clusters(data::Matrix{Float64}, centroids::Matrix{Float64})
    assignments = Vector{Int}(undef, size(data, 2))
    for j in 1:size(data, 2)
        column = view(data, :, j)
        best_idx = 1
        best_dist = Inf
        for c in 1:size(centroids, 2)
            dist = sum((column .- view(centroids, :, c)).^2)
            if dist < best_dist
                best_dist = dist
                best_idx = c
            end
        end
        assignments[j] = best_idx
    end
    return assignments
end

function _update_centroids(data::Matrix{Float64}, assignments::Vector{Int}, k::Int)
    dim = size(data, 1)
    centroids = zeros(Float64, dim, k)
    counts = zeros(Int, k)
    for j in 1:length(assignments)
        cluster = assignments[j]
        centroids[:, cluster] .+= view(data, :, j)
        counts[cluster] += 1
    end
    for c in 1:k
        if counts[c] > 0
            centroids[:, c] ./= counts[c]
        end
    end
    return centroids, counts
end

function _compute_inertia(data::Matrix{Float64}, assignments::Vector{Int}, centroids::Matrix{Float64})
    total = 0.0
    for j in 1:size(data, 2)
        cluster = assignments[j]
        diff = view(data, :, j) .- view(centroids, :, cluster)
        total += sum(diff .^ 2)
    end
    return total
end

function run_kmeans(dataset::IntegratedDataset; k::Int=3, maxiter::Int=300, tol::Float64=1e-4, seed::Int=42)
    k > 0 || throw(ArgumentError("Number of clusters must be positive"))
    data = dataset.matrix
    size(data, 2) >= k || throw(ArgumentError("Number of clusters cannot exceed sample count"))
    centroids = _initialise_centroids(data, k, seed)
    assignments = _assign_clusters(data, centroids)
    previous_inertia = Inf
    for iter in 1:maxiter
        centroids, counts = _update_centroids(data, assignments, k)
        for c in 1:k
            if counts[c] == 0
                centroids[:, c] = data[:, rand(1:size(data, 2))]
            end
        end
        assignments = _assign_clusters(data, centroids)
        inertia = _compute_inertia(data, assignments, centroids)
        if abs(previous_inertia - inertia) < tol
            previous_inertia = inertia
            break
        end
        previous_inertia = inertia
    end
    return ClusteringResult(assignments, centroids, _compute_inertia(data, assignments, centroids))
end

function run_analysis(dataset::IntegratedDataset, config::AnalysisConfig)
    pca_result = nothing
    if config.run_pca
        pca_result = run_pca(dataset; components=config.pca_components)
    end
    clustering_result = nothing
    if config.clustering === :kmeans
        clustering_result = run_kmeans(dataset; k=config.cluster_count, seed=config.random_seed)
    elseif config.clustering === nothing || config.clustering === :none
        clustering_result = nothing
    else
        throw(ArgumentError("Unsupported clustering strategy: $(config.clustering)"))
    end
    return AnalysisResult(pca=pca_result, clustering=clustering_result)
end
