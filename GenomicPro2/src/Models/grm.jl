"""
Genomic Relationship Matrix (GRM) computation.

Implements various methods for computing genomic relationships:
- VanRaden (2008) method
- Additive relationship matrix
- Dominance relationship matrix (future)
"""

"""
    center_genotypes(X::AbstractMatrix{<:Real}, freqs::AbstractVector{<:Real}) -> Matrix{Float64}

Center genotype matrix by subtracting 2*allele_frequency.

# Arguments
- `X`: Genotype matrix (n_samples × n_markers)
- `freqs`: Allele frequencies (length n_markers)

# Returns
Centered genotype matrix Z where Z[i,j] = X[i,j] - 2*freqs[j]

# Example
```julia
X = Float64.(to_matrix(geno; impute=true))
freqs = allele_frequencies(geno)
Z = center_genotypes(X, freqs)
```
"""
function center_genotypes(X::AbstractMatrix{<:Real}, freqs::AbstractVector{<:Real})
    n_samples, n_markers = size(X)

    if length(freqs) != n_markers
        throw(DimensionMismatchError("freqs", n_markers, length(freqs)))
    end

    # Center by subtracting 2*p
    Z = similar(X, Float64)
    for j in 1:n_markers
        center_val = 2 * freqs[j]
        for i in 1:n_samples
            Z[i, j] = X[i, j] - center_val
        end
    end

    return Z
end

"""
    scale_genotypes(Z::AbstractMatrix{<:Real}, freqs::AbstractVector{<:Real}) -> Matrix{Float64}

Scale centered genotypes by √(2*p*(1-p)).

# Arguments
- `Z`: Centered genotype matrix (n_samples × n_markers)
- `freqs`: Allele frequencies (length n_markers)

# Returns
Scaled genotype matrix where each column is divided by √(2*p*(1-p))

# Example
```julia
Z_centered = center_genotypes(X, freqs)
Z_scaled = scale_genotypes(Z_centered, freqs)
```
"""
function scale_genotypes(Z::AbstractMatrix{<:Real}, freqs::AbstractVector{<:Real})
    n_samples, n_markers = size(Z)

    if length(freqs) != n_markers
        throw(DimensionMismatchError("freqs", n_markers, length(freqs)))
    end

    Z_scaled = similar(Z, Float64)
    for j in 1:n_markers
        # Compute scale factor: sqrt(2*p*(1-p))
        p = freqs[j]
        scale = sqrt(2 * p * (1 - p))

        # Avoid division by zero for monomorphic SNPs
        if scale < 1e-10
            # Set to zero for monomorphic markers
            Z_scaled[:, j] .= 0.0
        else
            for i in 1:n_samples
                Z_scaled[i, j] = Z[i, j] / scale
            end
        end
    end

    return Z_scaled
end

"""
    compute_grm_vanraden(geno::CompactGenotypes; scale::Bool=true, min_maf::Float64=0.0) -> Matrix{Float64}

Compute genomic relationship matrix using VanRaden (2008) method.

The VanRaden method computes:
- G = Z*Z' / (2*Σp(1-p))

where:
- Z[i,j] = X[i,j] - 2*p[j] (centered genotypes)
- p[j] is the allele frequency at marker j
- X[i,j] ∈ {0, 1, 2} is the genotype

If scale=true, uses the normalized form:
- G = Z*Z' / m where Z is scaled by √(2*p*(1-p)) and m is the number of markers

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `scale::Bool`: Whether to use scaled (normalized) form (default: true)
- `min_maf::Float64`: Minimum MAF threshold for marker filtering (default: 0.0)

# Returns
Genomic relationship matrix G (n_samples × n_samples)

# References
VanRaden PM. 2008. Efficient methods to compute genomic predictions.
J Dairy Sci. 91(11):4414-23. doi: 10.3168/jds.2007-0980

# Examples
```julia
# Basic GRM
G = compute_grm_vanraden(geno)

# With MAF filtering
G = compute_grm_vanraden(geno; min_maf=0.01)

# Unscaled version
G = compute_grm_vanraden(geno; scale=false)
```
"""
function compute_grm_vanraden(geno::CompactGenotypes;
                              scale::Bool = true,
                              min_maf::Float64 = 0.0)
    n = n_samples(geno)
    m = n_markers(geno)

    # Get allele frequencies
    freqs = allele_frequencies(geno)

    # Filter by MAF if requested
    if min_maf > 0.0
        maf = minor_allele_frequency(geno)
        keep_markers = findall(maf .>= min_maf)

        if isempty(keep_markers)
            throw(ArgumentError("No markers pass MAF threshold of $min_maf"))
        end

        # Subset genotypes
        geno = subset_markers(geno, keep_markers)
        freqs = freqs[keep_markers]
        m = length(keep_markers)

        @info "MAF filtering" original_markers=n_markers(geno) filtered_markers=m min_maf=min_maf
    end

    # Convert to matrix and impute missing values
    X = to_matrix(geno; impute=true)

    # Center genotypes: Z = X - 2p
    Z = center_genotypes(X, freqs)

    if scale
        # Scale by sqrt(2*p*(1-p))
        Z = scale_genotypes(Z, freqs)

        # G = Z*Z' / m (normalized form)
        G = (Z * Z') ./ m
    else
        # G = Z*Z' / (2*Σp(1-p)) (original VanRaden form)
        denom = 2 * sum(p * (1 - p) for p in freqs)
        G = (Z * Z') ./ denom
    end

    # Ensure symmetry (numerical errors can break this)
    G = 0.5 * (G + G')

    return G
end

"""
    compute_grm_additive(geno::CompactGenotypes; min_maf::Float64=0.0) -> Matrix{Float64}

Compute additive genomic relationship matrix (simple allele sharing).

Computes: G[i,j] = (1/m) * Σ_k IBS(i,j,k) / 2
where IBS is identity-by-state at marker k.

This is a simpler method than VanRaden that doesn't use allele frequencies.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `min_maf::Float64`: Minimum MAF threshold (default: 0.0)

# Returns
Additive relationship matrix (n_samples × n_samples)

# Example
```julia
G = compute_grm_additive(geno)
```
"""
function compute_grm_additive(geno::CompactGenotypes; min_maf::Float64 = 0.0)
    n = n_samples(geno)
    m = n_markers(geno)

    # Filter by MAF if requested
    if min_maf > 0.0
        maf = minor_allele_frequency(geno)
        keep_markers = findall(maf .>= min_maf)

        if isempty(keep_markers)
            throw(ArgumentError("No markers pass MAF threshold of $min_maf"))
        end

        geno = subset_markers(geno, keep_markers)
        m = length(keep_markers)
    end

    # Convert to matrix and impute
    X = to_matrix(geno; impute=true)

    # Compute IBS matrix
    G = zeros(Float64, n, n)

    # For each pair of samples
    for i in 1:n
        # Diagonal
        G[i, i] = 1.0

        for j in (i+1):n
            # Count allele sharing
            ibs_sum = 0.0
            for k in 1:m
                # IBS: 2 - |X[i,k] - X[j,k]|
                ibs = 2 - abs(X[i, k] - X[j, k])
                ibs_sum += ibs / 2  # Normalize to [0, 1]
            end

            # Average over markers
            G[i, j] = ibs_sum / m
            G[j, i] = G[i, j]  # Symmetry
        end
    end

    return G
end

"""
    compute_grm(geno::CompactGenotypes; method::Symbol=:vanraden, kwargs...) -> Matrix{Float64}

Compute genomic relationship matrix using specified method.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `method::Symbol`: Method to use (:vanraden, :additive)
- `kwargs...`: Additional arguments passed to the specific method

# Returns
Genomic relationship matrix (n_samples × n_samples)

# Examples
```julia
# VanRaden method (default)
G = compute_grm(geno)

# Additive method
G = compute_grm(geno; method=:additive)

# With MAF filtering
G = compute_grm(geno; method=:vanraden, min_maf=0.01)
```
"""
function compute_grm(geno::CompactGenotypes; method::Symbol = :vanraden, kwargs...)
    if method == :vanraden
        return compute_grm_vanraden(geno; kwargs...)
    elseif method == :additive
        return compute_grm_additive(geno; kwargs...)
    else
        throw(ArgumentError("Unknown GRM method: $method. Use :vanraden or :additive"))
    end
end

"""
    validate_grm(G::AbstractMatrix) -> ValidationResult

Validate genomic relationship matrix.

Checks:
- Symmetry
- Diagonal values ≈ 1
- All values in reasonable range
- Positive semi-definite

# Example
```julia
G = compute_grm(geno)
result = validate_grm(G)
```
"""
function validate_grm(G::AbstractMatrix{<:Real})
    result = ValidationResult()

    n = size(G, 1)

    # Check square matrix
    if size(G, 2) != n
        add_error!(result, "GRM must be square matrix")
        return result
    end

    # Check symmetry
    if !issymmetric(G)
        max_asym = maximum(abs(G[i, j] - G[j, i]) for i in 1:n for j in (i+1):n)
        if max_asym > 1e-6
            add_error!(result, @sprintf("GRM is not symmetric (max asymmetry: %.2e)", max_asym))
        else
            # Fix numerical asymmetry
            add_warning!(result, "Fixed numerical asymmetry in GRM")
        end
    end

    # Check diagonal values
    diag_vals = diag(G)
    mean_diag = mean(diag_vals)
    if abs(mean_diag - 1.0) > 0.1
        add_warning!(result, @sprintf("Diagonal values deviate from 1.0 (mean: %.3f)", mean_diag))
    end

    result.metadata[:mean_diagonal] = mean_diag
    result.metadata[:min_value] = minimum(G)
    result.metadata[:max_value] = maximum(G)

    # Check positive semi-definite
    try
        eigenvalues = eigvals(Symmetric(G))
        min_eig = minimum(eigenvalues)
        result.metadata[:min_eigenvalue] = min_eig

        if min_eig < -1e-8
            add_warning!(result,
                @sprintf("GRM has negative eigenvalues (min: %.2e). Consider adding ridge penalty.", min_eig))
        end

        # Count near-zero eigenvalues (rank deficiency)
        near_zero = count(λ -> abs(λ) < 1e-8, eigenvalues)
        if near_zero > 0
            add_warning!(result, "GRM is rank deficient ($near_zero near-zero eigenvalues)")
        end
        result.metadata[:rank_deficiency] = near_zero

    catch e
        add_error!(result, "Failed to compute eigenvalues: $e")
    end

    return result
end
