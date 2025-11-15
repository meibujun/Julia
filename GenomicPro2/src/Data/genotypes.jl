"""
Genotype data structures and operations.

Implements memory-efficient genotype storage using 2-bit encoding.
"""

# ============================================================================
# CompactGenotypes - 2-bit Encoding
# ============================================================================

"""
    CompactGenotypes{T<:Integer} <: AbstractGenotypeData{T}

Memory-efficient genotype storage using 2-bit encoding.

# Encoding Scheme

Each genotype is encoded in 2 bits:
- `00` (0): Homozygous reference (0 alt alleles)
- `01` (1): Heterozygous (1 alt allele)
- `10` (2): Homozygous alternate (2 alt alleles)
- `11` (3): Missing (reserved, tracked separately)

# Memory Savings

For n samples × m markers:
- Standard Float64: n × m × 8 bytes
- Compact 2-bit: n × m / 4 bytes
- Savings: **96.875%**

Example: 10,000 samples × 100,000 SNPs
- Standard: 7.45 GB
- Compact: 244 MB
- **97% memory reduction!**

# Fields

- `data::Vector{UInt8}`: Packed 2-bit encoded genotypes (4 per byte)
- `n_samples::Int`: Number of samples
- `n_markers::Int`: Number of markers
- `sample_ids::Vector{String}`: Sample identifiers
- `marker_ids::Vector{String}`: Marker identifiers
- `missing_mask::BitMatrix`: Boolean mask for missing values
- `chromosome::Vector{String}`: Chromosome for each marker
- `position::Vector{Int}`: Position for each marker
- `ref_allele::Vector{String}`: Reference allele
- `alt_allele::Vector{String}`: Alternate allele
- `allele_freqs::Vector{Float64}`: Cached allele frequencies

# Examples

```julia
# Create from matrix
data = rand(0:2, 1000, 5000)
geno = CompactGenotypes(data, sample_ids, marker_ids)

# Memory usage
mem = memory_usage(geno)
println("Memory saved: \$(mem.savings * 100)%")

# Access genotypes
g = geno[1, 100]  # Get genotype for sample 1, marker 100

# Compute allele frequencies
freqs = allele_frequencies(geno)  # Cached for performance

# Validate
result = validate(geno)
```

# See Also

- [`encode_genotypes`](@ref): Convert matrix to 2-bit encoding
- [`decode_genotypes`](@ref): Convert 2-bit encoding back to matrix
- [`to_matrix`](@ref): Extract as standard matrix
"""
struct CompactGenotypes{T<:Integer} <: AbstractGenotypeData{T}
    # Core data (2-bit encoded)
    data::Vector{UInt8}

    # Dimensions
    n_samples::Int
    n_markers::Int

    # Identifiers
    sample_ids::Vector{String}
    marker_ids::Vector{String}

    # Missing value tracking
    missing_mask::BitMatrix

    # Marker metadata
    chromosome::Vector{String}
    position::Vector{Int}
    ref_allele::Vector{String}
    alt_allele::Vector{String}

    # Cached statistics (computed lazily)
    allele_freqs::Vector{Float64}

    """
        CompactGenotypes(data, sample_ids, marker_ids; kwargs...)

    Construct CompactGenotypes from a genotype matrix.

    # Arguments

    - `data::AbstractMatrix`: Genotype matrix (n_samples × n_markers)
    - `sample_ids::Vector{String}`: Sample identifiers
    - `marker_ids::Vector{String}`: Marker identifiers

    # Keyword Arguments

    - `chromosome::Vector{String}`: Chromosome for each marker
    - `position::Vector{Int}`: Position for each marker
    - `ref_allele::Vector{String}`: Reference alleles
    - `alt_allele::Vector{String}`: Alternate alleles

    # Returns

    CompactGenotypes object with 2-bit encoded data.
    """
    function CompactGenotypes(
        data::AbstractMatrix{T},
        sample_ids::Vector{String},
        marker_ids::Vector{String};
        chromosome::Union{Vector{String}, Nothing} = nothing,
        position::Union{Vector{Int}, Nothing} = nothing,
        ref_allele::Union{Vector{String}, Nothing} = nothing,
        alt_allele::Union{Vector{String}, Nothing} = nothing
    ) where T<:Integer

        n_samples, n_markers = size(data)

        # Validate inputs
        if length(sample_ids) != n_samples
            throw(DimensionMismatchError((n_samples,), (length(sample_ids),)))
        end

        if length(marker_ids) != n_markers
            throw(DimensionMismatchError((n_markers,), (length(marker_ids),)))
        end

        # Validate genotype values
        for val in data
            if !ismissing(val) && !(val in [0, 1, 2])
                throw(DataValidationError(
                    "Invalid genotype value",
                    :genotype,
                    val
                ))
            end
        end

        # Encode to 2-bit format
        encoded, missing_mask = encode_genotypes(data)

        # Compute allele frequencies (Kahan summation for numerical stability)
        freqs = compute_allele_frequencies_kahan(data, missing_mask)

        # Default metadata
        chrom = isnothing(chromosome) ? fill("0", n_markers) : chromosome
        pos = isnothing(position) ? collect(1:n_markers) : position
        ref = isnothing(ref_allele) ? fill("A", n_markers) : ref_allele
        alt = isnothing(alt_allele) ? fill("T", n_markers) : alt_allele

        new{T}(
            encoded,
            n_samples,
            n_markers,
            sample_ids,
            marker_ids,
            missing_mask,
            chrom,
            pos,
            ref,
            alt,
            freqs
        )
    end
end

# ============================================================================
# Encoding/Decoding Functions
# ============================================================================

"""
    encode_genotypes(data::AbstractMatrix) -> (Vector{UInt8}, BitMatrix)

Encode genotype matrix to 2-bit format.

Each UInt8 stores 4 genotypes (2 bits each).

# Arguments

- `data`: Genotype matrix (values: 0, 1, 2, or missing)

# Returns

- `encoded::Vector{UInt8}`: Packed 2-bit encoded data
- `missing_mask::BitMatrix`: Boolean mask for missing values

# Example

```julia
data = [0 1 2 missing; 1 2 0 1]
encoded, mask = encode_genotypes(data)
```
"""
function encode_genotypes(data::AbstractMatrix)
    n_samples, n_markers = size(data)

    # Each UInt8 stores 4 genotypes (2 bits each)
    n_bytes = cld(n_samples * n_markers, 4)
    encoded = zeros(UInt8, n_bytes)
    missing_mask = falses(n_samples, n_markers)

    byte_idx = 1
    bit_offset = 0

    # Encode column-major (Julia's default)
    for j in 1:n_markers
        for i in 1:n_samples
            val = data[i, j]

            # Determine encoding
            if ismissing(val)
                missing_mask[i, j] = true
                code = 0b00  # Store as 0, track in mask
            else
                code = UInt8(val) & 0b11  # Ensure 2-bit value
            end

            # Pack into byte
            encoded[byte_idx] |= (code << bit_offset)

            # Update position
            bit_offset += 2
            if bit_offset == 8
                bit_offset = 0
                byte_idx += 1
            end
        end
    end

    return encoded, missing_mask
end

"""
    decode_genotypes(cg::CompactGenotypes) -> Matrix

Decode 2-bit encoded data back to standard matrix.

Missing values are represented as `missing`.

# Example

```julia
decoded = decode_genotypes(geno)
```
"""
function decode_genotypes(cg::CompactGenotypes{T}) where T
    data = Matrix{Union{T, Missing}}(undef, cg.n_samples, cg.n_markers)

    byte_idx = 1
    bit_offset = 0

    for j in 1:cg.n_markers
        for i in 1:cg.n_samples
            # Extract 2 bits
            code = (cg.data[byte_idx] >> bit_offset) & 0b11

            # Decode
            if cg.missing_mask[i, j]
                data[i, j] = missing
            else
                data[i, j] = T(code)
            end

            # Update position
            bit_offset += 2
            if bit_offset == 8
                bit_offset = 0
                byte_idx += 1
            end
        end
    end

    return data
end

# ============================================================================
# Allele Frequency Computation (Kahan Summation)
# ============================================================================

"""
    compute_allele_frequencies_kahan(data, missing_mask) -> Vector{Float64}

Compute allele frequencies using Kahan summation for numerical stability.

Kahan summation compensates for floating-point rounding errors, providing
more accurate results especially for large datasets.

# Arguments

- `data::AbstractMatrix`: Genotype matrix
- `missing_mask::BitMatrix`: Missing value mask

# Returns

Vector of allele frequencies (proportion of alternate alleles)

# Algorithm

For each marker j:
1. Count valid (non-missing) genotypes
2. Sum genotype values using Kahan summation
3. Frequency = sum / (2 × n_valid)

# Reference

Kahan, W. (1965). "Further remarks on reducing truncation errors".
Communications of the ACM 8 (1): 40.
"""
function compute_allele_frequencies_kahan(
    data::AbstractMatrix,
    missing_mask::BitMatrix
)
    n_samples, n_markers = size(data)
    freqs = zeros(Float64, n_markers)

    for j in 1:n_markers
        # Kahan summation variables
        sum_val = 0.0
        compensation = 0.0
        n_valid = 0

        for i in 1:n_samples
            if !missing_mask[i, j]
                val = Float64(data[i, j])

                # Kahan summation
                y = val - compensation
                t = sum_val + y
                compensation = (t - sum_val) - y
                sum_val = t

                n_valid += 1
            end
        end

        # Frequency = sum / (2n) because diploid
        freqs[j] = n_valid > 0 ? sum_val / (2 * n_valid) : 0.0
    end

    return freqs
end

# ============================================================================
# Interface Implementation
# ============================================================================

Core.n_samples(cg::CompactGenotypes) = cg.n_samples
Core.n_markers(cg::CompactGenotypes) = cg.n_markers
Core.sample_ids(cg::CompactGenotypes) = cg.sample_ids
Core.marker_ids(cg::CompactGenotypes) = cg.marker_ids
Core.allele_frequencies(cg::CompactGenotypes) = cg.allele_freqs

Base.size(cg::CompactGenotypes) = (cg.n_samples, cg.n_markers)

"""
    getindex(cg::CompactGenotypes, i::Int, j::Int)

Access single genotype value.

Returns genotype value (0, 1, 2) or `missing`.
"""
function Base.getindex(cg::CompactGenotypes{T}, i::Int, j::Int) where T
    @boundscheck checkbounds(cg, i, j)

    # Calculate linear index in column-major order
    linear_idx = (j - 1) * cg.n_samples + i

    # Calculate byte and bit position
    byte_idx = div(linear_idx - 1, 4) + 1
    bit_offset = 2 * mod(linear_idx - 1, 4)

    # Extract 2 bits
    code = (cg.data[byte_idx] >> bit_offset) & 0b11

    # Check if missing
    return cg.missing_mask[i, j] ? missing : T(code)
end

"""
    missing_rate(cg::CompactGenotypes; dim=0)

Compute missing data rate.
"""
function Core.missing_rate(cg::CompactGenotypes; dim::Int=0)
    if dim == 0
        # Overall missing rate
        return sum(cg.missing_mask) / length(cg.missing_mask)
    elseif dim == 1
        # Per sample
        return vec(sum(cg.missing_mask, dims=2)) ./ cg.n_markers
    elseif dim == 2
        # Per marker
        return vec(sum(cg.missing_mask, dims=1)) ./ cg.n_samples
    else
        throw(ArgumentError("dim must be 0, 1, or 2"))
    end
end

# ============================================================================
# Validation
# ============================================================================

"""
    validate(cg::CompactGenotypes) -> ValidationResult

Validate genotype data for consistency and quality.

Checks:
- Sample/marker counts > 0
- ID uniqueness
- Missing rate
- Allele frequency range

# Example

```julia
result = validate(geno)
if !result.valid
    @error "Validation failed" result
end
```
"""
function Core.validate(cg::CompactGenotypes)
    result = ValidationResult()

    # Check dimensions
    if cg.n_samples <= 0
        add_error!(result, "Number of samples must be > 0")
    end

    if cg.n_markers <= 0
        add_error!(result, "Number of markers must be > 0")
    end

    # Check ID uniqueness
    id_result = validate_sample_ids(cg.sample_ids)
    if !id_result.valid
        merge!([result, id_result])
    end

    if length(unique(cg.marker_ids)) != cg.n_markers
        add_error!(result, "Marker IDs are not unique")
    end

    # Check missing rate
    overall_missing = missing_rate(cg; dim=0)
    result.metadata[:overall_missing_rate] = overall_missing

    if overall_missing > 0.5
        add_warning!(result,
            @sprintf("High missing rate: %.1f%%", overall_missing * 100))
    end

    # Check allele frequencies
    if any(f -> !(0 <= f <= 1), cg.allele_freqs)
        add_error!(result, "Invalid allele frequencies detected")
    end

    # Store metadata
    result.metadata[:n_samples] = cg.n_samples
    result.metadata[:n_markers] = cg.n_markers
    result.metadata[:memory_bytes] = sizeof(cg.data) + sizeof(cg.missing_mask)

    return result
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    to_matrix(cg::CompactGenotypes; impute=false) -> Matrix{Float64}

Convert to standard matrix, optionally imputing missing values.

# Arguments

- `cg`: CompactGenotypes object
- `impute`: If true, impute missing values with mean (2p)

# Returns

Matrix{Float64} of genotypes
"""
function to_matrix(cg::CompactGenotypes; impute::Bool=false)
    data = decode_genotypes(cg)

    if impute
        for j in 1:cg.n_markers
            impute_val = 2 * cg.allele_freqs[j]
            for i in 1:cg.n_samples
                if cg.missing_mask[i, j]
                    data[i, j] = impute_val
                end
            end
        end
    end

    return Float64.(data)
end

"""
    memory_usage(cg::CompactGenotypes) -> NamedTuple

Calculate memory usage statistics.

# Returns

NamedTuple with fields:
- `data`: Bytes used by encoded data
- `missing_mask`: Bytes used by missing mask
- `metadata`: Bytes used by metadata
- `total`: Total bytes
- `savings`: Proportion saved vs Float64 matrix

# Example

```julia
mem = memory_usage(geno)
println("Total: \$(mem.total / 1e6) MB")
println("Savings: \$(mem.savings * 100)%")
```
"""
function memory_usage(cg::CompactGenotypes)
    data_bytes = sizeof(cg.data)
    mask_bytes = sizeof(cg.missing_mask)

    metadata_bytes = sum([
        sizeof(cg.sample_ids),
        sizeof(cg.marker_ids),
        sizeof(cg.chromosome),
        sizeof(cg.position),
        sizeof(cg.ref_allele),
        sizeof(cg.alt_allele),
        sizeof(cg.allele_freqs)
    ])

    total_bytes = data_bytes + mask_bytes + metadata_bytes

    # Naive Float64 storage
    naive_bytes = cg.n_samples * cg.n_markers * sizeof(Float64)

    savings = 1 - (total_bytes / naive_bytes)

    return (
        data = data_bytes,
        missing_mask = mask_bytes,
        metadata = metadata_bytes,
        total = total_bytes,
        naive = naive_bytes,
        savings = savings
    )
end
