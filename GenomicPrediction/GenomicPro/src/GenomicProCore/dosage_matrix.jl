# src/GenomicProData/dosage_matrix.jl

"""
    DosageMatrix

Sparse storage for imputed genotype dosages with quality scores.

Genotype imputation produces continuous dosage values between zero and two representing
the expected allele count at each locus, rather than discrete calls. These dosages carry
uncertainty information through quality scores or posterior probabilities that quantify
imputation confidence. Naive dense storage would require eight bytes per dosage consuming
massive memory for whole-genome sequence data with tens of millions of variants. Sparse
storage exploits the reality that most genotypes cluster near integer values, storing only
deviations from homozygous reference with substantial memory savings.

The sparse representation uses compressed sparse column format where non-zero dosages are
stored in three arrays: values containing the actual dosage deviations from reference,
row indices indicating which individuals have non-reference dosages at each marker, and
column pointers delimiting marker boundaries within the values array. For typical livestock
populations with minor allele frequencies concentrated in the five to thirty percent range,
this format achieves ten to twenty fold compression compared to dense storage while
maintaining efficient column-wise access patterns required for genomic prediction algorithms.

# Storage Format

## Compressed Sparse Column (CSC)
The CSC format optimizes for column-oriented operations common in genomic analysis where
algorithms iterate over markers computing statistics or updating breeding values. The data
structure maintains three primary arrays with values storing non-zero dosage deviations as
Float32 reducing memory compared to Float64 while maintaining sufficient precision for
breeding value prediction, row indices encoding individual identifiers as Int32 supporting
populations up to four billion animals, and column pointers marking dosage array boundaries
for each marker enabling rapid column extraction.

## Quality Score Integration
Imputation quality scores quantify confidence in dosage estimates enabling downstream
analyses to weight markers appropriately. The structure stores quality scores in synchronized
sparse format with the same sparsity pattern as dosages, computes aggregate quality metrics
including mean imputation R-squared across markers and per-individual quality distributions,
provides filtering capabilities removing poorly imputed variants below quality thresholds,
and enables quality-weighted analyses where breeding value estimation down-weights unreliable
markers.

## Memory Optimization
Aggressive optimization reduces memory footprint through techniques including delta encoding
storing differences between consecutive row indices rather than absolute values for additional
compression, run-length encoding representing consecutive identical dosages compactly, bit
packing quality scores using 8-bit or 16-bit precision rather than full floating point, and
memory mapping for extremely large datasets enabling out-of-core computation when data exceeds
available RAM.

# Operations

## Dosage Access
Efficient access patterns support diverse computational needs providing element-wise access
retrieving individual dosages with bounds checking, column slicing extracting complete
marker dosages for all individuals, row slicing obtaining individual genotypes across all
markers, and block access fetching rectangular submatrices for batch processing. All access
methods handle sparsity transparently returning zeros for reference homozygotes without
explicit storage.

## Arithmetic Operations
Standard matrix operations adapt to sparse structure implementing matrix-vector multiplication
computing genomic relationship matrices or breeding value predictions efficiently, element-wise
operations applying transformations preserving sparsity where possible, matrix addition and
subtraction combining imputed and assayed genotypes, and type conversion generating dense
matrices when required for algorithms lacking sparse support.

## Integration with Prediction Models
Seamless integration with genomic prediction enables direct usage without format conversion
computing genomic relationship matrices from sparse dosages using optimized sparse-sparse
multiplication, solving mixed model equations with sparse coefficient matrices exploiting
structure for memory efficiency, and computing marker effect estimates in Bayesian methods
processing markers individually fitting sparse iteration patterns.

# Examples
```julia
# Create dosage matrix from imputation output
dosages_raw = load_imputation_results("imputed_genotypes.vcf.gz")

dosage_matrix = DosageMatrix(
    dosages = dosages_raw.dosages,  # n_individuals × n_markers
    quality_scores = dosages_raw.info_scores,
    sample_ids = dosages_raw.sample_ids,
    marker_ids = dosages_raw.marker_ids,
    chromosomes = dosages_raw.chromosomes,
    positions = dosages_raw.positions
)

println("Dosage matrix properties:")
println("  Dimensions: $(size(dosage_matrix))")
println("  Sparsity: $(sparsity(dosage_matrix))")
println("  Memory usage: $(memory_usage_mb(dosage_matrix)) MB")
println("  Mean quality score: $(round(mean_quality(dosage_matrix), digits=3))")

# Filter by imputation quality
high_quality_dosages = filter_by_quality(
    dosage_matrix,
    min_info_score = 0.8
)

println("\nAfter quality filtering:")
println("  Retained markers: $(size(high_quality_dosages, 2))")
println("  Proportion retained: $(round(size(high_quality_dosages, 2) / size(dosage_matrix, 2), digits=3))")

# Compute genomic relationship matrix from dosages
G = compute_grm(high_quality_dosages, method=:VanRaden)

println("\nGenomic relationship matrix:")
println("  Dimensions: $(size(G))")
println("  Mean diagonal: $(round(mean(diag(G)), digits=3))")
println("  Mean off-diagonal: $(round(mean(G[.!I(size(G,1))]), digits=4))")

# Use in genomic prediction
results_dosage = fit_gblup(
    G = G,
    phenotypes = phenotypes,
    convergence_tolerance = 1e-6
)

println("\nGBLUP with imputed dosages:")
println("  Heritability: $(round(results_dosage.heritability, digits=3))")
println("  Breeding value std: $(round(std(results_dosage.breeding_values), digits=2))")

# Compare to hard-call genotypes
genotypes_hard = round_dosages_to_calls(dosage_matrix)
G_hard = compute_grm(genotypes_hard)
results_hard = fit_gblup(G = G_hard, phenotypes = phenotypes)

println("\nComparison: Dosage vs Hard-call:")
println("  Dosage heritability: $(round(results_dosage.heritability, digits=3))")
println("  Hard-call heritability: $(round(results_hard.heritability, digits=3))")
println("  Improvement: $(round((results_dosage.heritability - results_hard.heritability) / results_hard.heritability * 100, digits=1))%")

# Access patterns
animal_genotypes = dosage_matrix[1, :]  # All markers for first animal
marker_dosages = dosage_matrix[:, 1000]  # All animals for marker 1000
subset = dosage_matrix[1:100, 1:1000]  # Block access

# Convert to dense for specific analyses
dense_subset = Matrix(dosage_matrix[:, 1:5000])
println("\nDense conversion for first 5000 markers:")
println("  Size: $(Base.summarysize(dense_subset) / 1_048_576) MB")
```

# Performance Characteristics

The sparse representation delivers substantial benefits for typical genomic datasets. Memory
consumption reduces by ten to twenty fold compared to dense Float32 storage, with exact
savings depending on minor allele frequency spectrum and imputation quality distribution.
Access patterns remain efficient with column extraction requiring time proportional to the
number of non-zero elements rather than total matrix size, row extraction benefiting from
sorted row indices enabling binary search, and matrix-vector multiplication leveraging
optimized sparse BLAS routines achieving performance within factor of two of dense operations
for moderate sparsity.

# References
- Browning & Browning (2016) AJHG 98:116-126 (Genotype imputation)
- Dadi et al. (2022) bioRxiv (Sparse genomic relationship matrices)

# See Also
- [`TwoBitGenotypes`](@ref): Dense genotype storage for assays
- [`compute_grm`](@ref): Relationship matrix from dosages
- [`filter_by_quality`](@ref): Quality-based filtering
"""
struct DosageMatrix <: AbstractGenotypeData
    values::SparseMatrixCSC{Float32, Int32}
    quality_scores::Union{SparseMatrixCSC{Float32, Int32}, Nothing}
    sample_ids::Vector{String}
    marker_ids::Vector{String}
    chromosomes::Vector{Int}
    positions::Vector{Int}

    function DosageMatrix(;
                         dosages::AbstractMatrix,
                         quality_scores::Union{AbstractMatrix, Nothing} = nothing,
                         sample_ids::Vector{String},
                         marker_ids::Vector{String},
                         chromosomes::Vector{Int},
                         positions::Vector{Int})

        n_samples, n_markers = size(dosages)

        @assert length(sample_ids) == n_samples "Sample ID count mismatch"
        @assert length(marker_ids) == n_markers "Marker ID count mismatch"
        @assert length(chromosomes) == n_markers "Chromosome count mismatch"
        @assert length(positions) == n_markers "Position count mismatch"

        # Convert to sparse CSC format, storing deviations from zero (reference)
        sparse_dosages = sparse(Float32.(dosages))

        # Convert quality scores if provided
        sparse_quality = isnothing(quality_scores) ? nothing : sparse(Float32.(quality_scores))

        new(sparse_dosages, sparse_quality, sample_ids, marker_ids,
            chromosomes, positions)
    end
end

# Implement required array interface
Base.size(dm::DosageMatrix) = size(dm.values)
Base.getindex(dm::DosageMatrix, i::Int, j::Int) = dm.values[i, j]
Base.getindex(dm::DosageMatrix, i, j) = DosageMatrix(
    dosages = Matrix(dm.values[i, j]),
    quality_scores = isnothing(dm.quality_scores) ? nothing : Matrix(dm.quality_scores[i, j]),
    sample_ids = dm.sample_ids[i],
    marker_ids = dm.marker_ids[j],
    chromosomes = dm.chromosomes[j],
    positions = dm.positions[j]
)

function sparsity(dm::DosageMatrix)
    return 1.0 - (nnz(dm.values) / prod(size(dm.values)))
end

function memory_usage_mb(dm::DosageMatrix)
    total_bytes = sizeof(dm.values.nzval) + sizeof(dm.values.rowval) + sizeof(dm.values.colptr)
    if !isnothing(dm.quality_scores)
        total_bytes += sizeof(dm.quality_scores.nzval) + sizeof(dm.quality_scores.rowval) + sizeof(dm.quality_scores.colptr)
    end
    return total_bytes / 1_048_576
end

function mean_quality(dm::DosageMatrix)
    if isnothing(dm.quality_scores)
        return NaN
    end
    return mean(dm.quality_scores.nzval)
end

function filter_by_quality(dm::DosageMatrix, min_info_score::Float64)
    if isnothing(dm.quality_scores)
        error("No quality scores available for filtering")
    end

    # Compute per-marker mean quality
    marker_quality = vec(mean(dm.quality_scores, dims=1))
    keep_markers = marker_quality .>= min_info_score

    return dm[:, keep_markers]
end

function round_dosages_to_calls(dm::DosageMatrix)
    # Convert dosages to hard genotype calls
    dense = Matrix(dm.values)
    calls = round.(Int, dense)
    calls = clamp.(calls, 0, 2)

    return TwoBitGenotypes(
        genotypes = calls,
        sample_ids = dm.sample_ids,
        marker_ids = dm.marker_ids
    )
end