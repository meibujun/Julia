# src/GenomicProCore/types.jl

"""
    AbstractGenomicData

Root abstract type for all genomic data structures in GenomicPro.jl.

This type hierarchy provides a unified interface for working with diverse genomic
data types including genotypes, phenotypes, pedigrees, and multi-omics data. All
concrete types should implement appropriate methods for data access, subsetting,
and validation.

# Type Hierarchy
- `AbstractGenomicData` (root)
  - `AbstractGenotypeData`: SNP genotypes, sequence data
  - `AbstractPhenotypeData`: Phenotypic measurements and traits
  - `AbstractPedigreeData`: Family relationships and ancestry
  - `AbstractOmicsData`: Multi-omics measurements (RNA-seq, metabolomics, etc.)
  - `AbstractRelationshipMatrix`: Kinship and relationship matrices

# Interface Requirements
Subtypes should implement:
- `Base.size(data)`: Return dimensions as tuple
- `Base.getindex(data, i...)`: Element or subset access
- `get_sample_ids(data)`: Return vector of sample identifiers
- `validate(data)`: Check data integrity and consistency

# Examples
```julia
# Example usage with concrete type
genotypes = read_genotypes("data.vcf")  # Returns AbstractGenotypeData
n_samples, n_markers = size(genotypes)
sample_ids = get_sample_ids(genotypes)
```

# See Also
- [`AbstractGenotypeData`](@ref): Genotype-specific interface
- [`AbstractPhenotypeData`](@ref): Phenotype-specific interface
"""
abstract type AbstractGenomicData end


"""
    AbstractGenotypeData <: AbstractGenomicData

Abstract type for genotype data including SNP arrays, sequence data, and dosages.

Genotype data represents genetic variants across individuals. Common encodings include:
- Biallelic SNPs: 0/1/2 representing reference homozygote/heterozygote/alternate homozygote
- Dosage data: Continuous [0,2] representing expected allele counts from imputation
- Sequence data: Raw nucleotide sequences with quality scores

# Interface Requirements
In addition to `AbstractGenomicData` interface, subtypes should implement:
- `get_allele_frequencies(data)`: Compute allele frequencies per marker
- `get_genotype(data, sample_idx, marker_idx)`: Access individual genotype
- `center_and_scale!(data)`: Standardize genotypes for relationship matrix computation

# Storage Considerations
Efficient storage is critical for genomic-scale data. Common strategies include:
- Two-bit encoding: 2 bits per genotype reducing memory 4-fold
- Sparse matrices: For dosage data with many zeros
- Memory-mapped files: For datasets exceeding RAM capacity

# Examples
```julia
# Load genotypes and compute basic statistics
genotypes = read_genotypes("cattle.vcf")
n_individuals, n_snps = size(genotypes)
allele_freqs = get_allele_frequencies(genotypes)
maf = min.(allele_freqs, 1 .- allele_freqs)
```

# See Also
- [`TwoBitGenotypes`](@ref): Memory-efficient SNP storage
- [`DosageMatrix`](@ref): Imputed genotype probabilities
"""
abstract type AbstractGenotypeData <: AbstractGenomicData end


"""
    AbstractPhenotypeData <: AbstractGenomicData

Abstract type for phenotypic trait measurements and associated metadata.

Phenotype data includes quantitative traits (continuous measurements like body weight),
qualitative traits (categorical like disease status), and associated covariates
(environmental factors, management groups, etc.).

# Interface Requirements
In addition to `AbstractGenomicData` interface, subtypes should implement:
- `get_traits(data)`: Return vector of trait names
- `get_phenotype(data, sample_id, trait)`: Access phenotype value
- `get_covariates(data)`: Return DataFrame of covariate values
- `has_missing(data, trait)`: Check for missing phenotype values

# Data Quality Considerations
Phenotype data quality significantly impacts prediction accuracy:
- Outlier detection: Identify extreme values requiring review
- Missing data patterns: Distinguish missing completely at random (MCAR) from
  missing not at random (MNAR) which may require different handling
- Covariate relationships: Model fixed effects like age, sex, management group

# Examples
```julia
# Load phenotypes and examine distribution
phenotypes = read_phenotypes("traits.csv", trait_names=["milk_yield", "fertility"])
trait_names = get_traits(phenotypes)
milk = get_phenotype(phenotypes, :, "milk_yield")

# Summary statistics
using Statistics
println("Mean milk yield: ", mean(skipmissing(milk)))
println("Missing rate: ", count(ismissing, milk) / length(milk))
```

# See Also
- [`PhenotypeData`](@ref): Standard phenotype implementation
- [`validate_phenotypes`](@ref): Data quality checking
"""
abstract type AbstractPhenotypeData <: AbstractGenomicData end


"""
    AbstractPedigreeData <: AbstractGenomicData

Abstract type for pedigree information representing family relationships.

Pedigree data encodes parent-offspring relationships enabling computation of
expected genetic relationships (numerator relationship matrix) and integration
with genomic information in single-step methods.

# Interface Requirements
In addition to `AbstractGenomicData` interface, subtypes should implement:
- `get_parents(data, individual_id)`: Return (sire, dam) tuple
- `compute_inbreeding(data)`: Calculate inbreeding coefficients
- `compute_numerator_relationship(data)`: Build additive relationship matrix A
- `validate_pedigree(data)`: Check for impossible relationships and cycles

# Pedigree Structure
Well-structured pedigrees should:
- Use consistent individual identifiers across generations
- Define unknown parents as missing or special codes (0, "Unknown", etc.)
- Avoid cycles (individual as own ancestor)
- Include generation numbers for temporal analysis

# Computational Considerations
Relationship matrix computation scales as O(n³) for direct inversion or O(n²)
for recursive algorithms. For large pedigrees (>50,000 individuals), consider:
- Sparse storage exploiting relationship sparsity
- Pruning distant relationships with negligible contribution
- Decomposition methods avoiding full matrix inversion

# Examples
```julia
# Load pedigree and compute relationships
pedigree = read_pedigree("family.ped")
A = compute_numerator_relationship(pedigree)

# Check pedigree quality
inbreeding = compute_inbreeding(pedigree)
highly_inbred = findall(inbreeding .> 0.25)
```

# See Also
- [`PedigreeData`](@ref): Standard pedigree implementation
- [`compute_inverse_numerator_relationship`](@ref): Efficient A⁻¹ computation
"""
abstract type AbstractPedigreeData <: AbstractGenomicData end


"""
    AbstractRelationshipMatrix <: AbstractGenomicData

Abstract type for genetic relationship matrices capturing similarity between individuals.

Relationship matrices quantify genetic similarity and are fundamental to genomic
prediction. Types include:
- Additive relationship: Expected proportion of genome shared IBD (G matrix)
- Dominance relationship: Captures dominance effects
- Epistatic relationship: Models interaction effects
- Multi-kernel: Combinations of above for complex genetic architecture

# Mathematical Foundation
For additive relationships, the genomic relationship matrix G is constructed as:
    G = ZZ' / (2 ∑pᵢ(1-pᵢ))
where Z is the centered and scaled genotype matrix with:
    Z[i,j] = (X[i,j] - 2pⱼ) / √(2pⱼ(1-pⱼ))
X is raw genotypes (0/1/2) and pⱼ is allele frequency at marker j.

# Interface Requirements
Subtypes should implement:
- `get_relationship(data, id1, id2)`: Access relationship between two individuals
- `get_diagonal(data)`: Extract diagonal elements (self-relationships)
- Matrix operations: `*`, `\`, `inv` for integration with linear solvers

# Storage Strategies
Relationship matrices are typically n×n symmetric positive semi-definite:
- Dense storage: For populations <10,000 individuals
- Sparse storage: For structured populations with clear subgroups
- Low-rank approximation: Using eigendecomposition for large n
- Block-diagonal: For multi-breed or multi-population datasets

# Examples
```julia
# Compute genomic relationship matrix
genotypes = read_genotypes("markers.bed")
G = compute_grm(genotypes, method=:VanRaden)

# Examine relationship distribution
using Statistics
diag_mean = mean(diag(G))  # Should be ≈1.0 + average inbreeding
offdiag = [G[i,j] for i in 1:size(G,1), j in 1:size(G,2) if i != j]
offdiag_mean = mean(offdiag)  # Should be ≈0.0 for unrelated base population
```

# References
- VanRaden PM (2008) J Dairy Sci 91:4414-4423
- Yang et al. (2010) Nature Genetics 42:565-569

# See Also
- [`GenomicRelationshipMatrix`](@ref): Standard GRM implementation
- [`compute_grm`](@ref): GRM computation with multiple methods
"""
abstract type AbstractRelationshipMatrix <: AbstractGenomicData end


"""
    AbstractQCFilter

Abstract type for quality control filters applied to genomic data.

Quality control is essential for removing low-quality samples and markers that
introduce noise and bias into genomic predictions. Filters should be composable,
allowing construction of multi-stage QC pipelines.

# Filter Design Principles
1. Single Responsibility: Each filter addresses one quality metric
2. Composability: Filters chain together via function composition
3. Reporting: Each filter documents what was removed and why
4. Reversibility: Maintain original data for alternative QC strategies

# Common Quality Metrics
- Missing rate: Proportion of missing genotypes per sample/marker
- Minor allele frequency: Rare variants may be uninformative or errors
- Hardy-Weinberg equilibrium: Deviation suggests genotyping errors
- Call rate: Minimum successful genotyping proportion
- Mendelian consistency: Parent-offspring genotype concordance

# Interface Requirements
Subtypes should implement:
- `apply_filter(filter, data)`: Apply filter returning filtered data and report
- `get_threshold(filter)`: Return filter threshold parameter(s)
- `get_filtered_indices(filter, data)`: Return indices that fail QC

# Examples
```julia
# Create QC pipeline
qc_pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.10, marker_threshold=0.10),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])

# Apply to data
genotypes_clean, qc_report = apply_qc(genotypes, qc_pipeline)
println(qc_report)
```

# See Also
- [`QCPipeline`](@ref): Compose multiple filters
- [`QCReport`](@ref): Comprehensive quality control summary
"""
abstract type AbstractQCFilter end


"""
    AbstractImputationMethod

Abstract type for genotype imputation algorithms.

Imputation infers missing genotypes leveraging linkage disequilibrium patterns
and population structure. Methods range from simple (mean imputation) to
sophisticated (deep learning models).

# Imputation Strategies
1. Population-based: Use allele frequencies (mean/mode imputation)
2. Family-based: Leverage pedigree for Mendelian-consistent filling
3. LD-based: Exploit haplotype structure (Beagle, IMPUTE2)
4. Deep learning: Learn complex patterns (VAE, autoencoder)

# Interface Requirements
Subtypes should implement:
- `impute!(method, data)`: Impute missing genotypes in-place
- `impute(method, data)`: Impute returning new data (original unchanged)
- `assess_accuracy(method, data, truth)`: Evaluate imputation quality
- `get_confidence_scores(method, data)`: Per-genotype imputation confidence

# Accuracy Considerations
Imputation accuracy depends on:
- Reference panel size and diversity: Larger, more diverse panels improve accuracy
- Marker density: Denser panels capture LD structure better
- Relatedness: Related individuals provide stronger information
- MAF: Rare variants are harder to impute accurately

# Examples
```julia
# Simple mean imputation
genotypes_imputed = impute(MeanImputation(), genotypes_with_missing)

# Deep learning imputation with pre-trained model
vae_model = load_pretrained_vae("cattle_50k_model.jld2")
genotypes_imputed = impute(VAEImputation(vae_model), genotypes_with_missing)
confidence = get_confidence_scores(vae_model, genotypes_imputed)
```

# References
- Browning & Browning (2007) Am J Hum Genet 81:1084-1097 (Beagle)
- Howie et al. (2009) PLoS Genet 5:e1000529 (IMPUTE2)
- Ros-Freixedes et al. (2022) Genet Sel Evol 54:58 (Deep learning)

# See Also
- [`VAEImputation`](@ref): Variational autoencoder imputation
- [`BeagleImputation`](@ref): HMM-based imputation
"""
abstract type AbstractImputationMethod end


"""
    AbstractValidator

Abstract type for data validation routines checking data integrity.

Validators perform automated checks identifying data quality issues requiring
manual review or correction. Unlike filters which automatically remove problematic
data, validators report issues for user decision-making.

# Validation Categories
1. Logical consistency: Impossible values or relationships
2. Statistical anomalies: Extreme outliers, unexpected distributions
3. Cross-reference integrity: Mismatches between related datasets
4. Biological plausibility: Violations of genetic principles

# Interface Requirements
Subtypes should implement:
- `validate(validator, data)`: Perform validation returning detailed report
- `is_valid(validator, data)`: Return boolean indicating overall validity
- `get_issues(validator, data)`: Return vector of specific issues found

# Examples
```julia
# Validate Mendelian consistency
pedigree = read_pedigree("family.ped")
genotypes = read_genotypes("offspring.vcf")
validator = MendelianConsistencyValidator(pedigree)
report = validate(validator, genotypes)

# Review inconsistencies
if !is_valid(validator, genotypes)
    issues = get_issues(validator, genotypes)
    for issue in issues
        println("Sample $(issue.sample_id), Marker $(issue.marker_id): $(issue.description)")
    end
end
```

# See Also
- [`MendelianConsistencyValidator`](@ref): Parent-offspring concordance
- [`PopulationStructureValidator`](@ref): Detect stratification
"""
abstract type AbstractValidator end