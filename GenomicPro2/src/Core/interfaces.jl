"""
Interface functions that all genomic data types must implement.

These are the core operations that define the contract for genomic data types.
"""

# ============================================================================
# Common Interface (All genomic data types)
# ============================================================================

"""
    n_samples(data::AbstractGenomicData) -> Int

Return the number of samples in the dataset.

# Examples

```julia
geno = CompactGenotypes(...)
n = n_samples(geno)  # 5000
```
"""
function n_samples end

"""
    sample_ids(data::AbstractGenomicData) -> Vector{String}

Return the vector of sample identifiers.

Sample IDs should be unique within a dataset.

# Examples

```julia
ids = sample_ids(geno)
ids[1]  # "sample_001"
```
"""
function sample_ids end

# ============================================================================
# Genotype-Specific Interface
# ============================================================================

"""
    n_markers(geno::AbstractGenotypeData) -> Int

Return the number of genetic markers (SNPs).

# Examples

```julia
m = n_markers(geno)  # 50000
```
"""
function n_markers end

"""
    marker_ids(geno::AbstractGenotypeData) -> Vector{String}

Return the vector of marker identifiers.

# Examples

```julia
markers = marker_ids(geno)
markers[1]  # "rs123456"
```
"""
function marker_ids end

"""
    allele_frequencies(geno::AbstractGenotypeData) -> Vector{Float64}

Return the allele frequencies for all markers.

Frequencies are computed as the proportion of alternate alleles,
typically cached for performance.

# Examples

```julia
freqs = allele_frequencies(geno)
freqs[1]  # 0.35
```
"""
function allele_frequencies end

"""
    missing_rate(geno::AbstractGenotypeData; dim=0) -> Union{Float64, Vector{Float64}}

Compute missing data rate.

# Arguments

- `geno`: Genotype data
- `dim`: Dimension along which to compute missing rate
  - `0`: Overall missing rate (default)
  - `1`: Missing rate per sample
  - `2`: Missing rate per marker

# Returns

- If `dim=0`: Single Float64 value (overall rate)
- If `dim=1`: Vector of per-sample missing rates
- If `dim=2`: Vector of per-marker missing rates

# Examples

```julia
overall = missing_rate(geno)  # 0.02
per_sample = missing_rate(geno; dim=1)
per_marker = missing_rate(geno; dim=2)
```
"""
function missing_rate end

# ============================================================================
# Phenotype-Specific Interface
# ============================================================================

"""
    trait_names(pheno::AbstractPhenotypeData) -> Vector{Symbol}

Return the names of all traits in the phenotype data.
"""
function trait_names end

"""
    get_trait(pheno::AbstractPhenotypeData, trait::Symbol) -> Vector

Get the values for a specific trait.
"""
function get_trait end

# ============================================================================
# Pedigree-Specific Interface
# ============================================================================

"""
    get_sire(ped::AbstractPedigreeData, id::String) -> Union{String, Missing}

Get the sire (father) of an individual.

Returns `missing` if sire is unknown.
"""
function get_sire end

"""
    get_dam(ped::AbstractPedigreeData, id::String) -> Union{String, Missing}

Get the dam (mother) of an individual.

Returns `missing` if dam is unknown.
"""
function get_dam end

"""
    compute_A_inverse(ped::AbstractPedigreeData) -> SparseMatrixCSC

Compute the inverse of the numerator relationship matrix (A⁻¹).

Uses Henderson's method for efficient computation.
"""
function compute_A_inverse end

# ============================================================================
# Array Interface for Genotypes
# ============================================================================

"""
Genotype data types should implement the array interface for convenient access.

# Required Methods

- `Base.size(::AbstractGenotypeData)` -> (n_samples, n_markers)
- `Base.getindex(::AbstractGenotypeData, i, j)` -> genotype value
- `Base.getindex(::AbstractGenotypeData, i, :)` -> sample genotypes
- `Base.getindex(::AbstractGenotypeData, :, j)` -> marker genotypes
"""

# These are implemented by subtypes
# Base.size(::AbstractGenotypeData)
# Base.getindex(::AbstractGenotypeData, ...)
