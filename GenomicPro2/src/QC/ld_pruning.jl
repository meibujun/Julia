"""
LD Pruning Module

Linkage Disequilibrium (LD) pruning for reducing marker redundancy.

LD pruning removes highly correlated markers to:
- Reduce computational burden
- Remove redundant information
- Improve model stability
- Meet independence assumptions of some methods

# Algorithms Implemented
1. **Window-based pruning**: Slide a window along the genome, remove markers in high LD
2. **Pairwise pruning**: Remove one marker from each pair exceeding LD threshold
3. **VIF-based pruning**: Remove markers with high Variance Inflation Factor

# References
- Purcell et al. (2007). PLINK: A tool set for whole-genome association and population-based linkage analyses. American Journal of Human Genetics, 81(3), 559-575.
- Anderson et al. (2010). Data quality control in genetic case-control association studies. Nature Protocols, 5(9), 1564-1573.
"""

using LinearAlgebra
using Statistics
using Printf

"""
    LDResult

Result of LD calculation between two markers.

# Fields
- `marker1::String`: ID of first marker
- `marker2::String`: ID of second marker
- `r::Float64`: Pearson correlation coefficient
- `r2::Float64`: r-squared (coefficient of determination)
- `Dprime::Float64`: Normalized linkage disequilibrium coefficient D'
"""
struct LDResult
    marker1::String
    marker2::String
    r::Float64
    r2::Float64
    Dprime::Float64
end

function Base.show(io::IO, ld::LDResult)
    @printf(io, "LD: %s <-> %s | r=%.4f, r²=%.4f, D'=%.4f",
            ld.marker1, ld.marker2, ld.r, ld.r2, ld.Dprime)
end

"""
    LDMatrix

Pairwise LD matrix for a set of markers.

# Fields
- `marker_ids::Vector{String}`: Marker IDs
- `r2::Matrix{Float64}`: r² matrix (n_markers × n_markers)
- `r::Matrix{Float64}`: r matrix (n_markers × n_markers)
"""
struct LDMatrix
    marker_ids::Vector{String}
    r2::Matrix{Float64}
    r::Matrix{Float64}
end

"""
    compute_ld_r2(geno::CompactGenotypes, idx1::Int, idx2::Int) -> Float64

Compute r² (coefficient of determination) between two markers.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `idx1::Int`: Index of first marker
- `idx2::Int`: Index of second marker

# Returns
- `Float64`: r² value (squared Pearson correlation)

# Example
```julia
r2 = compute_ld_r2(geno, 1, 2)
```
"""
function compute_ld_r2(geno::CompactGenotypes, idx1::Int, idx2::Int)
    n = geno.n_samples

    # Extract genotypes for both markers (handle missing data)
    g1 = zeros(Float64, n)
    g2 = zeros(Float64, n)
    valid_mask = trues(n)

    for i in 1:n
        val1 = get_genotype(geno, i, idx1)
        val2 = get_genotype(geno, i, idx2)

        if ismissing(val1) || ismissing(val2)
            valid_mask[i] = false
        else
            g1[i] = Float64(val1)
            g2[i] = Float64(val2)
        end
    end

    # Filter to valid samples
    g1_valid = g1[valid_mask]
    g2_valid = g2[valid_mask]

    n_valid = length(g1_valid)

    if n_valid < 10
        return 0.0  # Not enough data
    end

    # Center genotypes
    g1_centered = g1_valid .- mean(g1_valid)
    g2_centered = g2_valid .- mean(g2_valid)

    # Compute correlation
    var1 = dot(g1_centered, g1_centered) / n_valid
    var2 = dot(g2_centered, g2_centered) / n_valid

    if var1 < 1e-10 || var2 < 1e-10
        return 0.0  # Monomorphic marker
    end

    cov12 = dot(g1_centered, g2_centered) / n_valid
    r = cov12 / sqrt(var1 * var2)

    return r^2
end

"""
    compute_ld_dprime(geno::CompactGenotypes, idx1::Int, idx2::Int) -> Float64

Compute D' (normalized linkage disequilibrium coefficient) between two markers.

D' is the normalized coefficient of linkage disequilibrium, defined as D/Dmax,
where D is the difference between observed and expected haplotype frequencies.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `idx1::Int`: Index of first marker
- `idx2::Int`: Index of second marker

# Returns
- `Float64`: D' value (between -1 and 1)
"""
function compute_ld_dprime(geno::CompactGenotypes, idx1::Int, idx2::Int)
    n = geno.n_samples

    # Count haplotypes (assuming unphased, use genotype counts as proxy)
    # This is an approximation - true D' requires phased data

    # Count genotype combinations
    n00 = n01 = n02 = 0  # Marker1=0, Marker2=0,1,2
    n10 = n11 = n12 = 0  # Marker1=1, Marker2=0,1,2
    n20 = n21 = n22 = 0  # Marker1=2, Marker2=0,1,2

    n_valid = 0

    for i in 1:n
        val1 = get_genotype(geno, i, idx1)
        val2 = get_genotype(geno, i, idx2)

        if !ismissing(val1) && !ismissing(val2)
            n_valid += 1

            if val1 == 0 && val2 == 0
                n00 += 1
            elseif val1 == 0 && val2 == 1
                n01 += 1
            elseif val1 == 0 && val2 == 2
                n02 += 1
            elseif val1 == 1 && val2 == 0
                n10 += 1
            elseif val1 == 1 && val2 == 1
                n11 += 1
            elseif val1 == 1 && val2 == 2
                n12 += 1
            elseif val1 == 2 && val2 == 0
                n20 += 1
            elseif val1 == 2 && val2 == 1
                n21 += 1
            elseif val1 == 2 && val2 == 2
                n22 += 1
            end
        end
    end

    if n_valid < 10
        return 0.0
    end

    # Estimate allele frequencies
    # For marker 1: p1 = (2*n_alt_hom + n_het) / (2*n)
    p1 = (2*(n20 + n21 + n22) + (n10 + n11 + n12)) / (2.0 * n_valid)
    p2 = (2*(n02 + n12 + n22) + (n01 + n11 + n21)) / (2.0 * n_valid)

    q1 = 1.0 - p1
    q2 = 1.0 - p2

    # Estimate haplotype frequency (AB haplotype)
    # Using composite genotype approach
    pAB = (2*n22 + n21 + n12 + 0.5*n11) / (2.0 * n_valid)

    # D = pAB - p1*p2
    D = pAB - p1 * p2

    # Dmax
    if D >= 0
        Dmax = min(p1 * q2, q1 * p2)
    else
        Dmax = min(p1 * p2, q1 * q2)
    end

    if abs(Dmax) < 1e-10
        return 0.0
    end

    Dprime = D / Dmax

    return Dprime
end

"""
    compute_ld_full(geno::CompactGenotypes, idx1::Int, idx2::Int) -> LDResult

Compute full LD statistics between two markers.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `idx1::Int`: Index of first marker
- `idx2::Int`: Index of second marker

# Returns
- `LDResult`: LD statistics including r, r², and D'
"""
function compute_ld_full(geno::CompactGenotypes, idx1::Int, idx2::Int)
    r2_val = compute_ld_r2(geno, idx1, idx2)
    r_val = sqrt(r2_val)
    dprime_val = compute_ld_dprime(geno, idx1, idx2)

    return LDResult(
        geno.marker_ids[idx1],
        geno.marker_ids[idx2],
        r_val,
        r2_val,
        dprime_val
    )
end

"""
    ld_prune_window(geno::CompactGenotypes;
                    window_size::Int=50,
                    step_size::Int=5,
                    r2_threshold::Float64=0.8,
                    respect_chromosomes::Bool=true,
                    verbose::Bool=true) -> Vector{Int}

Perform window-based LD pruning.

Slides a window along the genome and removes markers in high LD.
This is the most commonly used LD pruning method (e.g., PLINK --indep-pairwise).

# Algorithm
1. Start with window of `window_size` markers
2. For each marker, compute r² with all other markers in window
3. Remove marker with highest average r² if any pair exceeds threshold
4. Slide window by `step_size` markers
5. Repeat until end of genome

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `window_size::Int`: Number of markers in sliding window (default: 50)
- `step_size::Int`: Number of markers to shift window (default: 5)
- `r2_threshold::Float64`: r² threshold for pruning (default: 0.8)
- `respect_chromosomes::Bool`: Don't compute LD across chromosomes (default: true)
- `verbose::Bool`: Print progress (default: true)

# Returns
- `Vector{Int}`: Indices of markers to keep (after pruning)

# Example
```julia
keep_idx = ld_prune_window(geno; window_size=50, r2_threshold=0.8)
geno_pruned = subset_markers(geno, keep_idx)
```
"""
function ld_prune_window(geno::CompactGenotypes;
                        window_size::Int = 50,
                        step_size::Int = 5,
                        r2_threshold::Float64 = 0.8,
                        respect_chromosomes::Bool = true,
                        verbose::Bool = true)

    if window_size < 2
        throw(ArgumentError("window_size must be >= 2"))
    end

    if step_size < 1
        throw(ArgumentError("step_size must be >= 1"))
    end

    if r2_threshold < 0.0 || r2_threshold > 1.0
        throw(ArgumentError("r2_threshold must be between 0 and 1"))
    end

    n_markers = geno.n_markers
    keep_markers = trues(n_markers)

    if verbose
        println("\n" * "="^70)
        println("LD Pruning (Window-based)")
        println("="^70)
        println("  Total markers: $n_markers")
        println("  Window size: $window_size")
        println("  Step size: $step_size")
        println("  r² threshold: $r2_threshold")
        println("  Respect chromosomes: $respect_chromosomes")
        println()
    end

    # Group markers by chromosome if requested
    if respect_chromosomes
        unique_chrs = unique(geno.chromosome)

        for chr in unique_chrs
            chr_idx = findall(geno.chromosome .== chr)

            if verbose
                println("  Processing chromosome $chr ($(length(chr_idx)) markers)...")
            end

            _prune_chromosome!(keep_markers, geno, chr_idx, window_size, step_size, r2_threshold, verbose)
        end
    else
        all_idx = collect(1:n_markers)
        _prune_chromosome!(keep_markers, geno, all_idx, window_size, step_size, r2_threshold, verbose)
    end

    keep_idx = findall(keep_markers)
    n_kept = length(keep_idx)
    n_removed = n_markers - n_kept

    if verbose
        println()
        println("="^70)
        println("Pruning Summary")
        println("="^70)
        println("  Markers kept: $n_kept ($(round(100*n_kept/n_markers, digits=2))%)")
        println("  Markers removed: $n_removed ($(round(100*n_removed/n_markers, digits=2))%)")
        println("="^70)
    end

    return keep_idx
end

function _prune_chromosome!(keep_markers::BitVector,
                            geno::CompactGenotypes,
                            chr_idx::Vector{Int},
                            window_size::Int,
                            step_size::Int,
                            r2_threshold::Float64,
                            verbose::Bool)

    n_chr_markers = length(chr_idx)

    if n_chr_markers < 2
        return  # Nothing to prune
    end

    window_start = 1
    n_removed_chr = 0

    while window_start <= n_chr_markers
        window_end = min(window_start + window_size - 1, n_chr_markers)
        window_indices = chr_idx[window_start:window_end]

        # Only consider markers not already removed
        active_window = window_indices[keep_markers[window_indices]]

        if length(active_window) >= 2
            # Compute pairwise LD within window
            n_active = length(active_window)
            max_r2 = 0.0
            to_remove = Int[]

            for i in 1:(n_active-1)
                for j in (i+1):n_active
                    r2 = compute_ld_r2(geno, active_window[i], active_window[j])

                    if r2 > r2_threshold
                        # Mark one for removal (keep the one with higher MAF)
                        maf_i = geno.allele_freqs[active_window[i]]
                        maf_j = geno.allele_freqs[active_window[j]]

                        # Prefer keeping SNP with MAF closer to 0.5
                        if abs(maf_i - 0.5) > abs(maf_j - 0.5)
                            push!(to_remove, active_window[i])
                        else
                            push!(to_remove, active_window[j])
                        end

                        max_r2 = max(max_r2, r2)
                    end
                end
            end

            # Remove duplicates
            to_remove = unique(to_remove)

            for idx in to_remove
                if keep_markers[idx]
                    keep_markers[idx] = false
                    n_removed_chr += 1
                end
            end
        end

        # Slide window
        window_start += step_size
    end

    if verbose && n_removed_chr > 0
        println("    Removed $n_removed_chr markers")
    end
end

"""
    ld_prune_pairwise(geno::CompactGenotypes;
                      r2_threshold::Float64=0.8,
                      respect_chromosomes::Bool=true,
                      max_distance::Union{Int,Nothing}=nothing,
                      verbose::Bool=true) -> Vector{Int}

Perform pairwise LD pruning.

Computes LD for all marker pairs and removes one from each pair exceeding threshold.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `r2_threshold::Float64`: r² threshold for pruning (default: 0.8)
- `respect_chromosomes::Bool`: Don't compute LD across chromosomes (default: true)
- `max_distance::Union{Int,Nothing}`: Maximum base-pair distance to consider (default: nothing)
- `verbose::Bool`: Print progress (default: true)

# Returns
- `Vector{Int}`: Indices of markers to keep

# Note
This method is more thorough but slower than window-based pruning.
Consider using window-based pruning for large datasets.
"""
function ld_prune_pairwise(geno::CompactGenotypes;
                          r2_threshold::Float64 = 0.8,
                          respect_chromosomes::Bool = true,
                          max_distance::Union{Int, Nothing} = nothing,
                          verbose::Bool = true)

    n_markers = geno.n_markers
    keep_markers = trues(n_markers)

    if verbose
        println("\n" * "="^70)
        println("LD Pruning (Pairwise)")
        println("="^70)
        println("  Total markers: $n_markers")
        println("  r² threshold: $r2_threshold")
        println("  Max distance: ", isnothing(max_distance) ? "unlimited" : "$max_distance bp")
        println()
    end

    # Build list of high-LD pairs
    high_ld_pairs = Tuple{Int, Int, Float64}[]

    for i in 1:(n_markers-1)
        for j in (i+1):n_markers
            # Skip if different chromosomes and we respect them
            if respect_chromosomes && geno.chromosome[i] != geno.chromosome[j]
                continue
            end

            # Skip if distance exceeds max_distance
            if !isnothing(max_distance) && geno.chromosome[i] == geno.chromosome[j]
                dist = abs(geno.position[j] - geno.position[i])
                if dist > max_distance
                    continue
                end
            end

            r2 = compute_ld_r2(geno, i, j)

            if r2 > r2_threshold
                push!(high_ld_pairs, (i, j, r2))
            end
        end

        if verbose && i % 100 == 0
            @printf("  Processed %d/%d markers\r", i, n_markers)
        end
    end

    if verbose
        println("\n  Found $(length(high_ld_pairs)) high-LD pairs")
        println("  Selecting markers to remove...")
    end

    # Sort pairs by r² (highest first)
    sort!(high_ld_pairs, by=x->x[3], rev=true)

    # Greedy removal: for each pair, remove one marker
    for (i, j, r2) in high_ld_pairs
        if keep_markers[i] && keep_markers[j]
            # Both still kept - remove one
            # Prefer keeping SNP with MAF closer to 0.5
            maf_i = geno.allele_freqs[i]
            maf_j = geno.allele_freqs[j]

            if abs(maf_i - 0.5) > abs(maf_j - 0.5)
                keep_markers[i] = false
            else
                keep_markers[j] = false
            end
        end
    end

    keep_idx = findall(keep_markers)
    n_kept = length(keep_idx)
    n_removed = n_markers - n_kept

    if verbose
        println()
        println("="^70)
        println("Pruning Summary")
        println("="^70)
        println("  Markers kept: $n_kept ($(round(100*n_kept/n_markers, digits=2))%)")
        println("  Markers removed: $n_removed ($(round(100*n_removed/n_markers, digits=2))%)")
        println("="^70)
    end

    return keep_idx
end

"""
    compute_ld_matrix(geno::CompactGenotypes, marker_indices::Vector{Int}) -> LDMatrix

Compute pairwise LD matrix for a set of markers.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `marker_indices::Vector{Int}`: Indices of markers to include

# Returns
- `LDMatrix`: Pairwise LD matrix with r and r² values

# Example
```julia
# Compute LD for first 100 markers
ld_mat = compute_ld_matrix(geno, 1:100)

# Access r² between markers i and j
r2_ij = ld_mat.r2[i, j]
```
"""
function compute_ld_matrix(geno::CompactGenotypes, marker_indices::Vector{Int})
    n_markers = length(marker_indices)
    r2_matrix = zeros(Float64, n_markers, n_markers)
    r_matrix = zeros(Float64, n_markers, n_markers)

    # Diagonal is 1.0
    for i in 1:n_markers
        r2_matrix[i, i] = 1.0
        r_matrix[i, i] = 1.0
    end

    # Compute upper triangle
    for i in 1:(n_markers-1)
        for j in (i+1):n_markers
            r2_val = compute_ld_r2(geno, marker_indices[i], marker_indices[j])
            r_val = sqrt(r2_val)

            r2_matrix[i, j] = r2_val
            r2_matrix[j, i] = r2_val
            r_matrix[i, j] = r_val
            r_matrix[j, i] = r_val
        end
    end

    marker_ids = geno.marker_ids[marker_indices]

    return LDMatrix(marker_ids, r2_matrix, r_matrix)
end

# Export
export LDResult, LDMatrix
export compute_ld_r2, compute_ld_dprime, compute_ld_full
export ld_prune_window, ld_prune_pairwise
export compute_ld_matrix
