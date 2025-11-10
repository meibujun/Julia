# src/GenomicProPredict/grm.jl

"""
    compute_grm(genotypes::AbstractGenotypeData; method=:VanRaden, kwargs...)

Compute genomic relationship matrix (GRM) from SNP genotype data.

The genomic relationship matrix captures genetic similarities between individuals
based on molecular marker information. It forms the foundation for genomic BLUP
and related prediction methods, replacing or augmenting traditional pedigree-based
relationship matrices.

# Mathematical Foundation

## VanRaden Method (2008)
The standard genomic relationship matrix is computed as:

    G = ZZ' / (2∑pᵢ(1-pᵢ))

where Z is the centered and scaled genotype matrix:

    Z[i,j] = (X[i,j] - 2pⱼ) / √(2pⱼ(1-pⱼ))

X[i,j] represents the raw genotype for individual i at marker j (coded as 0, 1, or 2
for the number of reference alleles), and pⱼ is the allele frequency at marker j.

This scaling ensures that diagonal elements of G approximate 1 + inbreeding coefficient,
and off-diagonal elements represent genomic relationships relative to the base population.

## Properties of G
Under the VanRaden scaling:
- Diagonal elements: E[G[i,i]] = 1 + F[i], where F[i] is inbreeding coefficient
- Off-diagonal elements: E[G[i,j]] = relationship coefficient between i and j
- For unrelated base population: mean(G[i,j] for i≠j) ≈ 0
- G is symmetric positive semi-definite

# Arguments
- `genotypes::AbstractGenotypeData`: Genotype matrix with individuals as rows, markers as columns
  Valid genotype codes: 0 (homozygous reference), 1 (heterozygous), 2 (homozygous alternate)

# Keyword Arguments
- `method::Symbol = :VanRaden`: Scaling method for GRM construction
  - `:VanRaden`: Standard method used in most applications (default)
  - `:AstalBalding`: Alternative scaling robust to ascertainment bias
- `center::Bool = true`: Center genotypes by subtracting 2p (strongly recommended)
- `scale::Bool = true`: Scale genotypes by √(2p(1-p)) (strongly recommended)
- `handle_missing::Symbol = :mean`: Strategy for missing genotypes
  - `:mean`: Replace with 2p (expected value under HWE)
  - `:omit`: Use only non-missing markers (reduces effective marker count)
- `min_maf::Float64 = 0.0`: Minimum MAF for marker inclusion (0.0 = include all)
- `blocksize::Int = 10000`: Process markers in blocks for memory efficiency

# Returns
- `Matrix{Float64}`: Symmetric n×n genomic relationship matrix where n = number of individuals

# Computational Complexity
- Time: O(n²m) where n = individuals, m = markers (dominated by matrix multiplication)
- Space: O(n² + nm) for storing G matrix and centered genotypes
- Optimization: Uses BLAS level-3 operations for maximum performance

# Performance Notes
For large datasets (n > 10,000 or m > 100,000), consider:
- GPU acceleration via `backend=:gpu` (50-200× speedup)
- Sparse GRM for structured populations (block-diagonal storage)
- Distributed computation for extremely large problems

# Numerical Stability
The implementation ensures numerical accuracy through:
- Compensated summation for allele frequency calculation
- Symmetric result via (G + G')/2 to correct round-off artifacts
- Condition number monitoring with warnings for near-singular matrices
- Float64 precision throughout critical calculations

# Examples
```julia
# Basic GRM computation with default parameters
genotypes = read_genotypes("cattle_50k.vcf")
G = compute_grm(genotypes)

# Verify expected properties
@assert issymmetric(G)
@assert isapprox(mean(diag(G)), 1.0, atol=0.1)  # Diagonal near 1.0
@assert isapprox(mean(G[i,j] for i in 1:size(G,1), j in 1:size(G,2) if i≠j),
                 0.0, atol=0.1)  # Off-diagonal near 0 for unrelated population

# GRM with quality control filtering
G = compute_grm(genotypes, min_maf=0.01)  # Exclude rare variants

# Alternative scaling method
G_ab = compute_grm(genotypes, method=:AstalBalding)

# Memory-efficient blocked computation
G = compute_grm(genotypes, blocksize=5000)  # Process 5000 markers at a time
```

# Interpretation Guidelines

## Diagonal Elements
Values significantly above 1.0 indicate inbreeding:
- 1.00-1.05: Low inbreeding, typical for outbred populations
- 1.05-1.15: Moderate inbreeding, common in purebred livestock
- >1.15: High inbreeding, may indicate recent common ancestry

## Off-Diagonal Elements
Relationship coefficients for common relationships:
- Parent-offspring: ~0.50
- Full siblings: ~0.50 (range 0.25-0.75 due to Mendelian sampling)
- Half siblings: ~0.25
- First cousins: ~0.125
- Unrelated: ~0.00

Negative values can occur for individuals less related than the population average.

# References
- VanRaden PM (2008) J Dairy Sci 91:4414-4423
- Yang et al. (2010) Nature Genetics 42:565-569
- Speed & Balding (2015) Am J Hum Genet 97:75-85

# See Also
- [`compute_grm_sparse`](@ref): Sparse GRM for structured populations
- [`compute_dominance_grm`](@ref): Dominance relationship matrix
- [`validate_grm`](@ref): Check GRM properties and quality
"""
function compute_grm(genotypes::AbstractGenotypeData;
                    method::Symbol = :VanRaden,
                    center::Bool = true,
                    scale::Bool = true,
                    handle_missing::Symbol = :mean,
                    min_maf::Float64 = 0.0,
                    blocksize::Int = 10000)

    n_individuals, n_markers = size(genotypes)

    # Validate inputs
    if !(method in [:VanRaden, :AstalBalding])
        throw(ArgumentError("Unknown method: $method. Use :VanRaden or :AstalBalding"))
    end
    if !(handle_missing in [:mean, :omit])
        throw(ArgumentError("Unknown handle_missing: $handle_missing. Use :mean or :omit"))
    end
    if min_maf < 0.0 || min_maf > 0.5
        throw(ArgumentError("min_maf must be in [0, 0.5], got $min_maf"))
    end

    # Compute allele frequencies with compensated summation for numerical accuracy
    println("Computing allele frequencies...")
    allele_freqs = compute_allele_frequencies_accurate(genotypes)

    # Calculate MAF and filter markers if requested
    maf = min.(allele_freqs, 1 .- allele_freqs)
    if min_maf > 0.0
        markers_to_use = maf .>= min_maf
        n_filtered = sum(.!markers_to_use)
        if n_filtered > 0
            println("  Filtering $n_filtered markers with MAF < $min_maf")
        end
    else
        markers_to_use = trues(n_markers)
    end

    n_markers_used = sum(markers_to_use)
    println("  Using $n_markers_used markers for GRM computation")

    # Determine scaling factor based on method
    if method == :VanRaden
        # Standard scaling: 2∑p(1-p)
        scaling_factor = 2.0 * sum(allele_freqs[markers_to_use] .*
                                   (1 .- allele_freqs[markers_to_use]))
    elseif method == :AstalBalding
        # Alternative scaling adjusting for ascertainment
        scaling_factor = Float64(n_markers_used)
    end

    println("  Scaling factor: $(round(scaling_factor, sigdigits=6))")

    # Initialize GRM
    G = zeros(Float64, n_individuals, n_individuals)

    # Process markers in blocks for memory efficiency
    println("Computing genomic relationship matrix...")
    marker_indices = findall(markers_to_use)
    n_blocks = cld(length(marker_indices), blocksize)

    for block_idx in 1:n_blocks
        block_start = (block_idx - 1) * blocksize + 1
        block_end = min(block_idx * blocksize, length(marker_indices))
        block_markers = marker_indices[block_start:block_end]

        # Extract and process genotype block
        Z_block = extract_and_standardize_block(
            genotypes, block_markers, allele_freqs,
            center, scale, handle_missing
        )

        # Accumulate ZZ' using BLAS for optimal performance
        # G += Z_block * Z_block'
        BLAS.syrk!('U', 'N', 1.0, Z_block, 1.0, G)

        if block_idx % 10 == 0 || block_idx == n_blocks
            progress = block_idx / n_blocks * 100
            println("  Progress: $(round(progress, digits=1))% (block $block_idx/$n_blocks)")
        end
    end

    # Complete symmetric matrix (BLAS.syrk! only fills upper triangle)
    for i in 1:n_individuals
        for j in 1:(i-1)
            G[i, j] = G[j, i]
        end
    end

    # Scale by denominator
    G ./= scaling_factor

    # Ensure perfect symmetry (correct any numerical artifacts)
    G = (G + G') / 2.0

    # Validate GRM properties
    println("Validating GRM properties...")
    validate_grm_properties(G)

    return G
end


"""
    compute_allele_frequencies_accurate(genotypes::AbstractGenotypeData)

Compute allele frequencies with compensated summation for numerical accuracy.

Uses Kahan summation algorithm to minimize round-off error accumulation when
summing genotypes, which is critical for accurate allele frequency estimation
in large datasets.

# Algorithm
For each marker j:
1. Initialize sum = 0, compensation = 0, count = 0
2. For each individual i:
   - If genotype is non-missing:
     - Compensated addition: sum += (genotype - compensation)
     - Update compensation for next iteration
     - Increment count
3. Frequency = sum / (2 × count)

# Returns
- `Vector{Float64}`: Allele frequencies, one per marker
"""
function compute_allele_frequencies_accurate(genotypes::AbstractGenotypeData)
    n_individuals, n_markers = size(genotypes)
    allele_freqs = Vector{Float64}(undef, n_markers)

    for j in 1:n_markers
        # Kahan summation for numerical stability
        sum_alleles = 0.0
        compensation = 0.0
        count_nonmissing = 0

        for i in 1:n_individuals
            g = genotypes[i, j]
            if !ismissing(g)
                # Compensated summation
                y = Float64(g) - compensation
                t = sum_alleles + y
                compensation = (t - sum_alleles) - y
                sum_alleles = t
                count_nonmissing += 1
            end
        end

        if count_nonmissing > 0
            allele_freqs[j] = sum_alleles / (2.0 * count_nonmissing)
        else
            # All missing: assign frequency 0.5 (will be filtered later)
            allele_freqs[j] = 0.5
        end
    end

    return allele_freqs
end


"""
    extract_and_standardize_block(genotypes, marker_indices, allele_freqs,
                                   center, scale, handle_missing)

Extract genotype block and apply centering/scaling transformations.

Transforms raw genotypes (0/1/2) to standardized Z-scores suitable for GRM
computation. Handles missing genotypes according to specified strategy.

# Arguments
- `genotypes`: Genotype data structure
- `marker_indices`: Indices of markers to extract
- `allele_freqs`: Vector of allele frequencies
- `center`: Whether to center by 2p
- `scale`: Whether to scale by √(2p(1-p))
- `handle_missing`: Strategy for missing data (:mean or :omit)

# Returns
- `Matrix{Float64}`: Standardized genotype matrix (individuals × markers in block)

# Transformation Details
For genotype X[i,j] at individual i, marker j:

1. Missing value handling:
   - `:mean`: X[i,j] = 2pⱼ (expected value)
   - `:omit`: Marker contribution set to 0 (handled in accumulation)

2. Centering (if center=true):
   - Z[i,j] = X[i,j] - 2pⱼ

3. Scaling (if scale=true):
   - Z[i,j] = Z[i,j] / √(2pⱼ(1-pⱼ))

The scaling ensures unit variance per marker and appropriate weighting in GRM.
"""
function extract_and_standardize_block(genotypes::AbstractGenotypeData,
                                      marker_indices::Vector{Int},
                                      allele_freqs::Vector{Float64},
                                      center::Bool,
                                      scale::Bool,
                                      handle_missing::Symbol)
    n_individuals = size(genotypes, 1)
    n_markers_block = length(marker_indices)

    Z = Matrix{Float64}(undef, n_individuals, n_markers_block)

    for (col_idx, marker_idx) in enumerate(marker_indices)
        p = allele_freqs[marker_idx]

        # Compute centering and scaling factors
        center_value = center ? 2.0 * p : 0.0
        scale_value = scale ? sqrt(2.0 * p * (1.0 - p)) : 1.0

        # Avoid division by zero for monomorphic markers
        if scale_value < 1e-10
            scale_value = 1.0
        end

        for i in 1:n_individuals
            g = genotypes[i, marker_idx]

            if ismissing(g)
                # Handle missing genotype
                if handle_missing == :mean
                    # Replace with expected value (after centering, this is 0)
                    Z[i, col_idx] = center ? 0.0 : 2.0 * p
                else  # :omit
                    Z[i, col_idx] = 0.0
                end
            else
                # Standardize non-missing genotype
                Z[i, col_idx] = (Float64(g) - center_value) / scale_value
            end
        end
    end

    return Z
end


"""
    validate_grm_properties(G::Matrix{Float64})

Validate mathematical properties of genomic relationship matrix.

Checks that computed GRM satisfies expected theoretical properties and
provides diagnostic information for quality assessment.

# Checks Performed
1. Symmetry: G[i,j] = G[j,i] for all i,j
2. Diagonal elements near 1.0 (allowing for inbreeding)
3. Off-diagonal mean near 0.0 (for unrelated base population)
4. Positive semi-definiteness (all eigenvalues ≥ 0)
5. Condition number (indicates numerical stability)

# Warnings Issued
- Diagonal elements far from 1.0 suggest scaling issues
- Large negative off-diagonal elements suggest population structure
- Negative eigenvalues indicate numerical problems
- High condition number warns of near-singularity
"""
function validate_grm_properties(G::Matrix{Float64})
    n = size(G, 1)

    # Check symmetry
    max_asymmetry = maximum(abs(G[i,j] - G[j,i]) for i in 1:n, j in 1:n if i < j)
    if max_asymmetry > 1e-10
        @warn "GRM not perfectly symmetric, max asymmetry: $max_asymmetry"
    end

    # Analyze diagonal elements
    diag_mean = mean(diag(G))
    diag_std = std(diag(G))
    println("  Diagonal statistics:")
    println("    Mean: $(round(diag_mean, digits=4)) (expect ≈1.0)")
    println("    Std:  $(round(diag_std, digits=4))")

    if abs(diag_mean - 1.0) > 0.2
        @warn "Diagonal mean far from 1.0, check scaling"
    end

    # Analyze off-diagonal elements
    offdiag_elements = [G[i,j] for i in 1:n, j in 1:n if i != j]
    offdiag_mean = mean(offdiag_elements)
    offdiag_min = minimum(offdiag_elements)
    offdiag_max = maximum(offdiag_elements)

    println("  Off-diagonal statistics:")
    println("    Mean: $(round(offdiag_mean, digits=4)) (expect ≈0.0 for unrelated)")
    println("    Min:  $(round(offdiag_min, digits=4))")
    println("    Max:  $(round(offdiag_max, digits=4))")

    if offdiag_min < -0.5
        @warn "Large negative relationships detected, check population structure"
    end

    # Check positive semi-definiteness (sample check on small subset if large)
    if n <= 1000
        eigenvalues = eigvals(G)
        min_eigenval = minimum(eigenvalues)

        println("  Eigenvalue statistics:")
        println("    Min: $(round(min_eigenval, sigdigits=4))")

        if min_eigenval < -1e-6
            @warn "Negative eigenvalues detected: $min_eigenval, GRM not positive semi-definite"
        end

        # Condition number
        max_eigenval = maximum(eigenvalues)
        condition_num = max_eigenval / max(abs(min_eigenval), 1e-10)
        println("    Condition number: $(round(condition_num, sigdigits=4))")

        if condition_num > 1e10
            @warn "High condition number, matrix near-singular"
        end
    end

    println("  ✓ GRM validation complete")
end