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

# Export core types
export AbstractGenomicData, AbstractGenotypeData, AbstractPhenotypeData
export ValidationResult

# Export data types
export CompactGenotypes

# Export functions
export n_samples, n_markers, sample_ids, marker_ids
export validate

end # module GenomicPro2
