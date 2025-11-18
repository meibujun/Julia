"""
    GenomicPro2

High-performance genomic prediction and analysis toolkit.

GenomicPro2 is a complete rewrite of GenomicPro with focus on:
- Memory efficiency (2-bit genotype encoding)
- Performance (GPU acceleration, parallel computing)
- Scalability (support for millions of SNPs)
- Modern software architecture (hexagonal architecture, DDD)

# Quick Start

```julia
using GenomicPro2

# Load data
geno = read_genotypes("data.vcf.gz")
pheno = read_phenotypes("phenotypes.csv")

# Quality control
geno_qc, pheno_qc = quality_control(geno, pheno)

# Compute GRM
G = compute_grm(geno_qc)

# Train model
model = GBLUPModel()
fit!(model, geno_qc, pheno_qc; G=G)

# Predict
predictions = predict(model, geno_qc)
```

# Modules

- `Core`: Core types, interfaces, and exceptions
- `Data`: Data structures (genotypes, phenotypes, pedigrees)
- `IO`: File I/O (VCF, PLINK, HDF5, etc.)
- `LinearAlgebra`: GRM computation, linear solvers
- `Models`: Prediction models (GBLUP, BayesR, Deep Learning)
- `Utils`: Utilities (logging, configuration, etc.)

# Version

This is version 2.0.0, a complete redesign from version 1.0.

# License

MIT License
"""
module GenomicPro2

using LinearAlgebra
using SparseArrays
using Statistics
using Printf

# Include core modules
include("Core/Core.jl")
using .Core

# Include data modules
include("Data/Data.jl")
using .Data

# Include I/O modules
include("IO/IO.jl")
using .IO

# Include Models modules
include("Models/Models.jl")
using .Models

# Include QC modules
include("QC/QC.jl")
using .QC

# Include Utils modules
include("Utils/Utils.jl")
using .Utils

# Include PopulationStructure modules
include("PopulationStructure/PopulationStructure.jl")
using .PopulationStructure

# Include Visualization modules
include("Visualization/Visualization.jl")
using .Visualization

# Export core types
export AbstractGenomicData, AbstractGenotypeData, AbstractPhenotypeData
export ValidationResult

# Export data types
export CompactGenotypes, PhenotypeData

# Export core functions
export n_samples, n_markers, sample_ids, marker_ids
export validate, allele_frequencies, missing_rate

# Export data manipulation functions
export subset_samples, subset_markers, subset
export minor_allele_frequency, to_matrix, memory_usage

# Export I/O functions
export read_plink, write_plink
export read_phenotypes, write_phenotypes
export read_vcf, write_vcf
export merge_genotype_phenotype
export VCFHeader

# Export Models
export GBLUPModel, GBLUPResult
export fit!, predict
export compute_grm, compute_grm_vanraden, compute_grm_additive
export validate_grm

# Export Parallel GRM
export compute_grm_parallel, compute_grm_vanraden_parallel, compute_grm_additive_parallel
export benchmark_threading

# Export Cross-validation
export CVResult
export kfold_cv, loo_cv, random_cv
export create_folds

# Export BayesR
export BayesRModel, BayesRResult

# Export BayesCπ
export BayesCπModel, BayesCπResult

# Export RKHS
export RKHSModel, RKHSResult
export compute_kernel, center_kernel_matrix!

# Export QC
export quality_control, qc_report, QCReport, QCFilters
export filter_maf, filter_missing_markers, filter_missing_samples, filter_hwe
export hardy_weinberg_test, call_rate, heterozygosity_rate
export expected_heterozygosity, inbreeding_coefficient
export identify_duplicates, compute_sample_correlation

# Export LD Pruning
export LDResult, LDMatrix
export compute_ld_r2, compute_ld_dprime, compute_ld_full
export ld_prune_window, ld_prune_pairwise
export compute_ld_matrix

# Export Utils
export GenotypeDataSummary, PhenotypeDataSummary
export summarize, compare_datasets, detect_outliers, marker_quality_summary

# Export Population Structure
export PCAResult
export pca, detect_outliers_pca, cluster_samples

# Export Visualization
export GWASResult
export manhattan_plot_data, qq_plot_data, pca_plot_data, ld_heatmap_data
export save_plot_data

end # module GenomicPro2
