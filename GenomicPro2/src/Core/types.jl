"""
Core type definitions for GenomicPro2.

Defines the abstract type hierarchy that all concrete types must implement.
"""

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractGenomicData{T}

Root abstract type for all genomic data types.

All genomic data types (genotypes, phenotypes, pedigrees) inherit from this type
and must implement the core interface functions.

# Interface Requirements

Subtypes must implement:
- `n_samples(::AbstractGenomicData)` - return number of samples
- `sample_ids(::AbstractGenomicData)` - return vector of sample IDs
- `validate(::AbstractGenomicData)` - validate data integrity

# Type Parameter

- `T`: The element type of the data
"""
abstract type AbstractGenomicData{T} end

"""
    AbstractGenotypeData{T} <: AbstractGenomicData{T}

Abstract type for genotype data.

# Additional Interface Requirements

In addition to `AbstractGenomicData` requirements, subtypes must implement:
- `n_markers(::AbstractGenotypeData)` - return number of genetic markers
- `marker_ids(::AbstractGenotypeData)` - return vector of marker IDs
- `allele_frequencies(::AbstractGenotypeData)` - return allele frequencies
- `Base.getindex(::AbstractGenotypeData, i, j)` - access genotype at position

# Supported Implementations

- `CompactGenotypes`: Memory-efficient 2-bit encoding
- `SparseGenotypes`: Sparse matrix representation
- `MappedGenotypes`: Memory-mapped file access
"""
abstract type AbstractGenotypeData{T} <: AbstractGenomicData{T} end

"""
    AbstractPhenotypeData{T} <: AbstractGenomicData{T}

Abstract type for phenotype data.

# Additional Interface Requirements

- `trait_names(::AbstractPhenotypeData)` - return vector of trait names
- `get_trait(::AbstractPhenotypeData, trait)` - access specific trait
"""
abstract type AbstractPhenotypeData{T} <: AbstractGenomicData{T} end

"""
    AbstractPedigreeData{T} <: AbstractGenomicData{T}

Abstract type for pedigree data.

# Additional Interface Requirements

- `get_sire(::AbstractPedigreeData, id)` - get sire of individual
- `get_dam(::AbstractPedigreeData, id)` - get dam of individual
- `compute_A_inverse(::AbstractPedigreeData)` - compute inverse of numerator relationship matrix
"""
abstract type AbstractPedigreeData{T} <: AbstractGenomicData{T} end

# ============================================================================
# Value Objects
# ============================================================================

"""
    GenotypeValue

Immutable value object representing a single genotype call.

Valid values are:
- `0`: Homozygous reference (0 alt alleles)
- `1`: Heterozygous (1 alt allele)
- `2`: Homozygous alternate (2 alt alleles)
- `missing`: Missing genotype

# Examples

```julia
g = GenotypeValue(1)  # Heterozygous
g.value  # returns 1

# Invalid value throws error
GenotypeValue(3)  # DomainError
```
"""
struct GenotypeValue
    value::Union{UInt8, Missing}

    function GenotypeValue(val::Union{Integer, Missing})
        if ismissing(val)
            return new(missing)
        end

        if !(val in [0, 1, 2])
            throw(DomainError(val, "Genotype value must be 0, 1, or 2"))
        end

        return new(UInt8(val))
    end
end

Base.:(==)(a::GenotypeValue, b::GenotypeValue) = a.value == b.value
Base.ismissing(g::GenotypeValue) = ismissing(g.value)

"""
    AlleleFrequency

Immutable value object representing an allele frequency.

Allele frequency must be in the range [0, 1].

# Examples

```julia
af = AlleleFrequency(0.3)
is_rare(af)  # false
is_rare(af, 0.4)  # true (rare if < 0.4)
```
"""
struct AlleleFrequency
    value::Float64

    function AlleleFrequency(freq::Real)
        if !(0 <= freq <= 1)
            throw(DomainError(freq, "Allele frequency must be in [0, 1]"))
        end
        return new(Float64(freq))
    end
end

"""
    is_rare(af::AlleleFrequency, threshold=0.01)

Check if allele frequency is considered rare (< threshold or > 1-threshold).
"""
function is_rare(af::AlleleFrequency, threshold::Float64=0.01)::Bool
    return af.value < threshold || af.value > (1 - threshold)
end

Base.:(==)(a::AlleleFrequency, b::AlleleFrequency) = a.value == b.value
Base.isless(a::AlleleFrequency, b::AlleleFrequency) = a.value < b.value
