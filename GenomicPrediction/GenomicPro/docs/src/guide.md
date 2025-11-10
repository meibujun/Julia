# User Guide

Welcome to the `GenomicPro.jl` User Guide. This document provides a comprehensive overview of the package's functionalities, from data loading and quality control to advanced modeling with multi-omics data.

## 1. Installation

To install `GenomicPro.jl`, open the Julia REPL and run:

```julia
using Pkg
Pkg.add("GenomicPro")
```

## 2. Core Concepts and Data Structures

`GenomicPro.jl` is built around a set of core data structures that represent the different types of data used in genomic prediction.

### 2.1 Genotype Data

-   **`TwoBitGenotypes`**: A memory-efficient data structure for storing biallelic SNP genotypes. It uses a two-bit encoding scheme to reduce memory usage by 75% compared to standard arrays.
-   **`DosageMatrix`**: A sparse matrix for storing imputed genotype dosages.

### 2.2 Phenotype Data

-   **`PhenotypeData`**: A container for phenotypic trait data, built on top of `DataFrames.jl`.

### 2.3 Pedigree Data

-   **`PedigreeData`**: A container for pedigree information, also built on `DataFrames.jl`.

### 2.4 Multi-Omics Data

-   **`ExpressionData`**: A container for gene expression data.
-   **`MultiOmicsData`**: A flexible container for holding multiple omics data types in a dictionary.

## 3. Basic GBLUP Workflow

This section walks you through a standard GBLUP analysis.

### 3.1 Loading Data

```julia
using GenomicPro

# Load genotype data from a VCF file
geno = read_genotypes("path/to/genotypes.vcf")

# Load phenotype data from a CSV file
pheno = read_phenotypes("path/to/phenotypes.csv", :ID, [:Trait1, :Trait2])
```

### 3.2 Quality Control

```julia
# Create a QC pipeline
pipeline = QCPipeline([
    MissingRateFilter(sample_threshold=0.1, marker_threshold=0.1),
    MAFFilter(min_maf=0.01),
    HWEFilter(pvalue_threshold=1e-6)
])

# Apply the pipeline to the genotype data
geno_qc, qc_reports = apply_qc(geno, pipeline)
```

### 3.3 GBLUP Analysis

```julia
# Compute the Genomic Relationship Matrix (GRM)
G = compute_grm(geno_qc)

# Extract the phenotype vector
y = pheno.table[:, :Trait1]

# Estimate variance components
vc = estimate_variance_components(G, y)
λ = vc.residual_variance / vc.genetic_variance

# Solve for breeding values
results = solve_gblup(G, y, λ)

println("Estimated Breeding Values: ", results.breeding_values)
```

## 4. Multi-Omics Integration Workflow

This section demonstrates how to integrate genomic and transcriptomic data for improved prediction accuracy.

### 4.1 Loading Multi-Omics Data

```julia
# Load SNP and gene expression data
snp_data = read_genotypes("path/to/snps.vcf")
expression_data = ExpressionData(...) # Assuming a constructor or reader function

# Create a MultiOmicsData container
multi_omics_data = MultiOmicsData(Dict(
    :genotypes => snp_data,
    :expression => expression_data
))
```

### 4.2 Defining a Multi-Omics Model

```julia
using Lux

latent_dim = 64
n_attention_heads = 4

# Define encoders for each modality
snp_encoder = build_snp_encoder(size(snp_data, 2), latent_dim)
rnaseq_encoder = build_rnaseq_vae(size(expression_data.table, 2) - 2, latent_dim)

# Define the fusion layer and predictor
fusion_layer = CrossAttentionFusion(latent_dim, n_heads=n_attention_heads)
predictor = Chain(
    Dense(latent_dim * 2, 128, relu),
    Dense(128, 1)
)

# Combine into a single model
multi_omics_model = (
    snp_encoder = snp_encoder,
    expression_encoder = rnaseq_encoder,
    fusion_layer = fusion_layer,
    predictor = predictor
)
```

### 4.3 Training the Multi-Omics Model

```julia
# Split data into training and validation sets
# ... (code for splitting data)

# Train the model
ps, st = train_multiomics_model(multi_omics_model, train_data, y_train,
                                val_data, y_val,
                                n_epochs=50, batch_size=32, early_stopping_patience=5)
```
