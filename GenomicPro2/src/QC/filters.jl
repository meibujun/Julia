"""
Quality control filters for genomic data.
"""

"""
    QCFilters

Container for quality control filter parameters.

# Fields
- `min_maf::Float64`: Minimum minor allele frequency (default: 0.01)
- `max_missing_per_marker::Float64`: Maximum missing rate per marker (default: 0.1)
- `max_missing_per_sample::Float64`: Maximum missing rate per sample (default: 0.1)
- `min_call_rate::Float64`: Minimum call rate (default: 0.9)
- `hwe_pvalue::Float64`: Hardy-Weinberg equilibrium p-value threshold (default: 1e-6)
- `min_samples::Int`: Minimum number of samples after filtering (default: 10)
- `min_markers::Int`: Minimum number of markers after filtering (default: 100)

# Example
```julia
filters = QCFilters(
    min_maf = 0.05,
    max_missing_per_marker = 0.05,
    hwe_pvalue = 1e-8
)
```
"""
struct QCFilters
    min_maf::Float64
    max_missing_per_marker::Float64
    max_missing_per_sample::Float64
    min_call_rate::Float64
    hwe_pvalue::Float64
    min_samples::Int
    min_markers::Int

    function QCFilters(;
        min_maf::Float64 = 0.01,
        max_missing_per_marker::Float64 = 0.1,
        max_missing_per_sample::Float64 = 0.1,
        min_call_rate::Float64 = 0.9,
        hwe_pvalue::Float64 = 1e-6,
        min_samples::Int = 10,
        min_markers::Int = 100
    )
        # Validation
        if !(0 <= min_maf <= 0.5)
            throw(ArgumentError("min_maf must be in [0, 0.5]"))
        end
        if !(0 <= max_missing_per_marker <= 1)
            throw(ArgumentError("max_missing_per_marker must be in [0, 1]"))
        end
        if !(0 <= max_missing_per_sample <= 1)
            throw(ArgumentError("max_missing_per_sample must be in [0, 1]"))
        end
        if !(0 <= min_call_rate <= 1)
            throw(ArgumentError("min_call_rate must be in [0, 1]"))
        end
        if hwe_pvalue < 0
            throw(ArgumentError("hwe_pvalue must be >= 0"))
        end

        new(min_maf, max_missing_per_marker, max_missing_per_sample,
            min_call_rate, hwe_pvalue, min_samples, min_markers)
    end
end

"""
    filter_maf(geno::CompactGenotypes, min_maf::Float64) -> Vector{Int}

Identify markers passing minor allele frequency threshold.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `min_maf::Float64`: Minimum MAF (default: 0.01)

# Returns
Vector of marker indices passing filter

# Example
```julia
keep_markers = filter_maf(geno, 0.05)
geno_filtered = subset_markers(geno, keep_markers)
```
"""
function filter_maf(geno::CompactGenotypes, min_maf::Float64 = 0.01)
    maf = minor_allele_frequency(geno)
    keep_idx = findall(maf .>= min_maf)

    @info "MAF filter" total_markers=n_markers(geno) passing=length(keep_idx) failing=n_markers(geno)-length(keep_idx) threshold=min_maf

    return keep_idx
end

"""
    filter_missing_markers(geno::CompactGenotypes, max_missing::Float64) -> Vector{Int}

Identify markers passing missing rate threshold.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `max_missing::Float64`: Maximum missing rate (default: 0.1)

# Returns
Vector of marker indices passing filter

# Example
```julia
keep_markers = filter_missing_markers(geno, 0.05)
```
"""
function filter_missing_markers(geno::CompactGenotypes, max_missing::Float64 = 0.1)
    marker_missing = missing_rate(geno; dim=2)
    keep_idx = findall(marker_missing .<= max_missing)

    @info "Missing rate filter (markers)" total=n_markers(geno) passing=length(keep_idx) failing=n_markers(geno)-length(keep_idx) threshold=max_missing

    return keep_idx
end

"""
    filter_missing_samples(geno::CompactGenotypes, max_missing::Float64) -> Vector{Int}

Identify samples passing missing rate threshold.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `max_missing::Float64`: Maximum missing rate (default: 0.1)

# Returns
Vector of sample indices passing filter

# Example
```julia
keep_samples = filter_missing_samples(geno, 0.05)
geno_filtered = subset_samples(geno, keep_samples)
```
"""
function filter_missing_samples(geno::CompactGenotypes, max_missing::Float64 = 0.1)
    sample_missing = missing_rate(geno; dim=1)
    keep_idx = findall(sample_missing .<= max_missing)

    @info "Missing rate filter (samples)" total=n_samples(geno) passing=length(keep_idx) failing=n_samples(geno)-length(keep_idx) threshold=max_missing

    return keep_idx
end

"""
    filter_hwe(geno::CompactGenotypes, pvalue_threshold::Float64) -> Vector{Int}

Identify markers passing Hardy-Weinberg equilibrium test.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `pvalue_threshold::Float64`: P-value threshold (default: 1e-6)

# Returns
Vector of marker indices passing filter

# Example
```julia
keep_markers = filter_hwe(geno, 1e-8)
```
"""
function filter_hwe(geno::CompactGenotypes, pvalue_threshold::Float64 = 1e-6)
    # Compute HWE p-values for all markers
    pvalues = Float64[]

    for j in 1:n_markers(geno)
        # Count genotypes for this marker
        n0 = 0  # AA (0)
        n1 = 0  # Aa (1)
        n2 = 0  # aa (2)
        n_valid = 0

        for i in 1:n_samples(geno)
            if !ismissing(geno, i, j)
                val = geno[i, j]
                if val == 0
                    n0 += 1
                elseif val == 1
                    n1 += 1
                elseif val == 2
                    n2 += 1
                end
                n_valid += 1
            end
        end

        # Compute HWE p-value
        if n_valid > 0
            pval = hardy_weinberg_test(n0, n1, n2)
            push!(pvalues, pval)
        else
            push!(pvalues, 0.0)  # No data, fail
        end
    end

    keep_idx = findall(pvalues .> pvalue_threshold)

    @info "HWE filter" total=n_markers(geno) passing=length(keep_idx) failing=n_markers(geno)-length(keep_idx) threshold=pvalue_threshold

    return keep_idx
end

"""
    quality_control(geno::CompactGenotypes; kwargs...) -> CompactGenotypes

Apply comprehensive quality control filters to genotype data.

# Arguments
- `geno::CompactGenotypes`: Input genotype data
- `min_maf::Float64`: Minimum MAF (default: 0.01)
- `max_missing_per_marker::Float64`: Maximum missing per marker (default: 0.1)
- `max_missing_per_sample::Float64`: Maximum missing per sample (default: 0.1)
- `hwe_pvalue::Float64`: HWE p-value threshold (default: 1e-6)
- `apply_hwe::Bool`: Whether to apply HWE filter (default: true)
- `verbose::Bool`: Print detailed information (default: true)

# Returns
Filtered CompactGenotypes

# Example
```julia
# Standard QC
geno_qc = quality_control(geno)

# Strict QC
geno_qc = quality_control(geno;
    min_maf = 0.05,
    max_missing_per_marker = 0.02,
    max_missing_per_sample = 0.05,
    hwe_pvalue = 1e-8
)

# Skip HWE filter
geno_qc = quality_control(geno; apply_hwe = false)
```
"""
function quality_control(
    geno::CompactGenotypes;
    min_maf::Float64 = 0.01,
    max_missing_per_marker::Float64 = 0.1,
    max_missing_per_sample::Float64 = 0.1,
    hwe_pvalue::Float64 = 1e-6,
    apply_hwe::Bool = true,
    verbose::Bool = true
)
    if verbose
        println("\n" * "="^70)
        println("Quality Control Pipeline")
        println("="^70)
        println("\nInput data:")
        println("  Samples: $(n_samples(geno))")
        println("  Markers: $(n_markers(geno))")
        println("  Overall missing rate: $(@sprintf("%.2f%%", missing_rate(geno) * 100))")
    end

    # Step 1: Filter samples by missing rate
    if verbose
        println("\n[1/4] Filtering samples by missing rate (threshold: $(max_missing_per_sample*100)%)...")
    end
    keep_samples = filter_missing_samples(geno, max_missing_per_sample)

    if length(keep_samples) < 10
        error("Too few samples remain after missing rate filter: $(length(keep_samples))")
    end

    geno = subset_samples(geno, keep_samples)

    # Step 2: Filter markers by missing rate
    if verbose
        println("\n[2/4] Filtering markers by missing rate (threshold: $(max_missing_per_marker*100)%)...")
    end
    keep_markers = filter_missing_markers(geno, max_missing_per_marker)

    if length(keep_markers) < 100
        error("Too few markers remain after missing rate filter: $(length(keep_markers))")
    end

    geno = subset_markers(geno, keep_markers)

    # Step 3: Filter by MAF
    if verbose
        println("\n[3/4] Filtering markers by MAF (threshold: $min_maf)...")
    end
    keep_markers = filter_maf(geno, min_maf)

    if length(keep_markers) < 100
        error("Too few markers remain after MAF filter: $(length(keep_markers))")
    end

    geno = subset_markers(geno, keep_markers)

    # Step 4: Filter by HWE (optional)
    if apply_hwe
        if verbose
            println("\n[4/4] Filtering markers by HWE (p-value threshold: $hwe_pvalue)...")
        end
        keep_markers = filter_hwe(geno, hwe_pvalue)

        if length(keep_markers) < 100
            @warn "Too few markers remain after HWE filter: $(length(keep_markers)). Skipping HWE filter."
        else
            geno = subset_markers(geno, keep_markers)
        end
    else
        if verbose
            println("\n[4/4] Skipping HWE filter (apply_hwe = false)")
        end
    end

    # Final summary
    if verbose
        println("\n" * "="^70)
        println("Quality Control Summary")
        println("="^70)
        println("  Final samples: $(n_samples(geno))")
        println("  Final markers: $(n_markers(geno))")
        println("  Final missing rate: $(@sprintf("%.2f%%", missing_rate(geno) * 100))")
        println("="^70)
    end

    return geno
end

"""
    identify_duplicates(geno::CompactGenotypes; threshold::Float64 = 0.95) -> Vector{Tuple{Int,Int,Float64}}

Identify potential duplicate samples based on genotype correlation.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `threshold::Float64`: Correlation threshold for duplicates (default: 0.95)

# Returns
Vector of tuples (sample1_idx, sample2_idx, correlation)

# Example
```julia
duplicates = identify_duplicates(geno, threshold=0.98)
for (i, j, cor) in duplicates
    println("Samples \$(sample_ids(geno)[i]) and \$(sample_ids(geno)[j]): r = \$cor")
end
```
"""
function identify_duplicates(geno::CompactGenotypes; threshold::Float64 = 0.95)
    duplicates = Tuple{Int,Int,Float64}[]

    n = n_samples(geno)
    X = to_matrix(geno; impute=true)

    for i in 1:n
        for j in (i+1):n
            # Compute correlation between samples
            cor_val = cor(X[i, :], X[j, :])

            if cor_val >= threshold
                push!(duplicates, (i, j, cor_val))
            end
        end
    end

    if !isempty(duplicates)
        @warn "Found $(length(duplicates)) potential duplicate sample pairs"
    end

    return duplicates
end

"""
    compute_sample_correlation(geno::CompactGenotypes) -> Matrix{Float64}

Compute pairwise correlation matrix between all samples.

Useful for identifying related individuals or duplicates.

# Arguments
- `geno::CompactGenotypes`: Genotype data

# Returns
Correlation matrix (n_samples × n_samples)

# Example
```julia
cor_mat = compute_sample_correlation(geno)
heatmap(cor_mat)  # Visualize with Plots.jl
```
"""
function compute_sample_correlation(geno::CompactGenotypes)
    X = to_matrix(geno; impute=true)

    n = n_samples(geno)
    cor_mat = zeros(Float64, n, n)

    for i in 1:n
        cor_mat[i, i] = 1.0
        for j in (i+1):n
            cor_val = cor(X[i, :], X[j, :])
            cor_mat[i, j] = cor_val
            cor_mat[j, i] = cor_val
        end
    end

    return cor_mat
end
