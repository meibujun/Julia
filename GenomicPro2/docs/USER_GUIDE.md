# GenomicPro2 User Guide

Complete guide to using GenomicPro2 for genomic prediction and analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Loading](#data-loading)
3. [Quality Control](#quality-control)
4. [LD Pruning](#ld-pruning)
5. [Genomic Prediction Models](#genomic-prediction-models)
6. [Cross-Validation](#cross-validation)
7. [Data Summary and Exploration](#data-summary-and-exploration)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/GenomicPro2.jl")
```

### Quick Start

```julia
using GenomicPro2

# Load data
geno = read_plink("mydata")
pheno = read_phenotypes("phenotypes.csv")

# Quality control
geno_qc = quality_control(geno; min_maf=0.01)

# Train model
G = compute_grm(geno_qc)
model = GBLUPModel()
fit!(model, geno_qc, pheno; G=G)

# Predict
gebv = predict(model, geno_qc)
```

---

## Data Loading

### PLINK Format

The most common format for genomic data.

```julia
# Read PLINK files (.bed/.bim/.fam)
geno = read_plink("path/to/data")

# With options
geno = read_plink("data";
    max_markers = 10000,  # Limit markers
    verbose = true
)

# Write PLINK files
write_plink("output", geno)
```

**File Structure:**
- `.bed`: Binary genotype data (SNP-major format)
- `.bim`: Marker information (chr, ID, position, alleles)
- `.fam`: Sample information (family, individual IDs)

### VCF Format

Standard format for variant call data.

```julia
# Read VCF
geno = read_vcf("data.vcf.gz";
    regions = ["1", "2"],        # Specific chromosomes
    min_qual = 30.0,             # Quality threshold
    pass_only = true,            # Only PASS variants
    samples = ["S1", "S2"]       # Specific samples
)

# Write VCF
write_vcf("output.vcf", geno;
    compress = true,             # Gzip compression
    reference = "GRCh38"         # Reference genome
)
```

**VCF vs PLINK:**
| Feature | VCF | PLINK |
|---------|-----|-------|
| File size | Large | Small (5-10x smaller) |
| Read speed | Slower | Faster (2-5x) |
| Standard format | ✓ | ✓ |
| Quality scores | ✓ | ✗ |
| Multi-allelic | ✓ | Limited |
| **Best for** | Data exchange | Analysis |

### Phenotype Data

```julia
# Read CSV phenotypes
pheno = read_phenotypes("pheno.csv";
    id_col = "ID",              # Sample ID column
    trait_cols = ["Yield", "Height"],  # Trait columns
    covariate_cols = ["Age", "Sex"]    # Covariate columns
)

# Write phenotypes
write_phenotypes("output.csv", pheno)
```

---

## Quality Control

### Basic QC Pipeline

```julia
# Comprehensive QC
geno_qc = quality_control(geno;
    min_maf = 0.01,                    # Minimum minor allele frequency
    max_missing_per_marker = 0.1,      # Maximum 10% missing per SNP
    max_missing_per_sample = 0.1,      # Maximum 10% missing per sample
    hwe_pvalue = 1e-6,                 # Hardy-Weinberg equilibrium
    apply_hwe = true,
    verbose = true
)
```

### QC Report

```julia
# Generate detailed QC report
report = qc_report(geno)
println(report)
```

**QC Report Includes:**
- Overall statistics (samples, markers, missing rate)
- MAF distribution
- Missing rate distribution
- HWE test results
- Heterozygosity analysis
- Duplicate samples

### Manual Filtering

```julia
# MAF filtering
geno_maf = filter_maf(geno; min_maf=0.05, max_maf=0.5)

# Missing rate filtering
geno_miss = filter_missing_markers(geno; max_missing=0.05)

# HWE filtering
geno_hwe = filter_hwe(geno; pvalue_threshold=1e-5)
```

### Duplicate Detection

```julia
# Find duplicate samples
duplicates = identify_duplicates(geno; threshold=0.95)

for (id1, id2, correlation) in duplicates
    println("$id1 <-> $id2: correlation = $correlation")
end
```

---

## LD Pruning

### Window-Based Pruning (Recommended)

```julia
# Standard LD pruning
keep_idx = ld_prune_window(geno;
    window_size = 50,        # SNPs in window
    step_size = 10,          # Window shift
    r2_threshold = 0.8,      # LD threshold
    respect_chromosomes = true
)

geno_pruned = subset_markers(geno, keep_idx)
```

**Threshold Guidelines:**
- `r² > 0.99`: Very strict (removes duplicates only)
- `r² > 0.90`: Strict (recommended for imputed data)
- `r² > 0.80`: **Standard (recommended for most analyses)**
- `r² > 0.50`: Moderate pruning
- `r² > 0.20`: Aggressive pruning

### Pairwise Pruning

More thorough but slower:

```julia
keep_idx = ld_prune_pairwise(geno;
    r2_threshold = 0.8,
    max_distance = 500000    # 500kb window
)
```

### LD Statistics

```julia
# Compute r² between two SNPs
r2 = compute_ld_r2(geno, 1, 2)

# Full LD statistics
ld = compute_ld_full(geno, 1, 2)
println("r = $(ld.r), r² = $(ld.r2), D' = $(ld.Dprime)")

# LD matrix for region
ld_mat = compute_ld_matrix(geno, 1:100)
```

---

## Genomic Prediction Models

### GBLUP

Genomic Best Linear Unbiased Prediction.

```julia
# Compute GRM
G = compute_grm(geno;
    method = :vanraden,     # VanRaden (2008) method
    scale = true,
    min_maf = 0.01
)

# Fit GBLUP model
model = GBLUPModel(
    method = :cholesky,           # :cholesky or :pcg
    estimate_variances = true,    # EM-REML
    max_iter = 100
)

fit!(model, geno, pheno;
    G = G,
    trait_index = 1
)

# Results
println("Heritability: $(model.result.h2)")
println("Genetic variance: $(model.result.var_u)")
println("Residual variance: $(model.result.var_e)")

# Predict
gebv = predict(model, geno)
```

**When to Use GBLUP:**
- ✓ Highly polygenic traits
- ✓ Large sample sizes
- ✓ Fast computation needed
- ✓ Routine genomic evaluation

### BayesR

Bayesian variable selection with mixture priors.

```julia
model = BayesRModel(
    n_iter = 50000,                # MCMC iterations
    burn_in = 20000,               # Burn-in
    thin = 10,                     # Thinning
    update_pi = true,              # Estimate mixture proportions
    seed = 123                     # Reproducibility
)

fit!(model, geno, pheno)

# Results
result = model.result
println("Heritability: $(result.heritability)")
println("Non-zero SNPs: $(sum(result.marker_pip .> 0.5))")

# Identify top QTLs
top_qtls = sortperm(result.marker_pip, rev=true)[1:20]
println("Top 20 SNPs: $(geno.marker_ids[top_qtls])")

# Effect sizes with uncertainty
for i in top_qtls[1:10]
    @printf("%s: β = %.4f ± %.4f (PIP = %.4f)\n",
            geno.marker_ids[i],
            result.marker_effects[i],
            result.marker_effects_sd[i],
            result.marker_pip[i])
end
```

**When to Use BayesR:**
- ✓ Sparse genetic architecture (few large-effect QTLs)
- ✓ Variable selection needed
- ✓ QTL mapping applications
- ✓ Need effect size estimates with uncertainty

**BayesR Tuning:**
- Short MCMC for testing: 5,000 iterations
- Standard analysis: 50,000-100,000 iterations
- Production: 100,000-200,000 iterations
- Monitor convergence by running multiple chains

### Model Comparison

```julia
# Fit both models
model_gblup = GBLUPModel()
fit!(model_gblup, geno, pheno; G=G)

model_bayesr = BayesRModel(n_iter=10000, burn_in=5000)
fit!(model_bayesr, geno, pheno)

# Compare predictions
gebv_gblup = predict(model_gblup, geno)
gebv_bayesr = model_bayesr.result.gebv_train

println("GBLUP   accuracy: $(cor(gebv_gblup, true_breeding_values))")
println("BayesR  accuracy: $(cor(gebv_bayesr, true_breeding_values))")
println("Agreement: $(cor(gebv_gblup, gebv_bayesr))")
```

---

## Cross-Validation

### K-Fold Cross-Validation

```julia
# 5-fold CV
cv_result = kfold_cv(
    () -> GBLUPModel(),    # Model constructor
    geno, pheno;
    k = 5,
    grm_options = (method = :vanraden, min_maf = 0.01),
    seed = 123,
    verbose = true
)

# Results
println("CV Metrics:")
println("  Correlation: $(cv_result.metrics.correlation) ± $(cv_result.metrics.std_fold_correlation)")
println("  R²: $(cv_result.metrics.r_squared)")
println("  MSE: $(cv_result.metrics.mse)")
println("  Bias: $(cv_result.metrics.bias)")

# Per-fold results
for (i, fold) in enumerate(cv_result.fold_results)
    println("Fold $i: r = $(fold.correlation)")
end
```

### Leave-One-Out CV

For small datasets:

```julia
cv_result = loo_cv(
    () -> GBLUPModel(),
    geno, pheno;
    grm_options = (method = :vanraden,)
)
```

### Random Sub-sampling

```julia
cv_result = random_cv(
    () -> GBLUPModel(),
    geno, pheno;
    n_folds = 10,
    train_fraction = 0.8,
    seed = 123
)
```

---

## Data Summary and Exploration

### Genotype Summary

```julia
# Comprehensive summary
summary = summarize(geno)
println(summary)
```

**Summary Includes:**
- Dimensions (samples, markers, chromosomes)
- Missing data statistics
- MAF distribution
- Genotype distribution (0/1/2 counts)
- Memory usage

### Compare Datasets

```julia
# Before and after QC
compare_datasets(geno_raw, geno_qc;
    name1 = "Raw Data",
    name2 = "After QC"
)
```

### Outlier Detection

```julia
# Detect outlier samples
outliers = detect_outliers(geno;
    method = :iqr,           # :iqr or :sd
    threshold = 3.0
)

println("Outlier samples: $(geno.sample_ids[outliers])")

# Remove outliers
geno_clean = subset_samples(geno, setdiff(1:geno.n_samples, outliers))
```

### Marker Quality Summary

```julia
# Per-marker quality metrics
marker_stats = marker_quality_summary(geno)

# Filter low-quality markers
high_quality = findall(
    (marker_stats.maf .>= 0.05) .&
    (marker_stats.missing_rate .<= 0.05) .&
    (marker_stats.hwe_pvalue .> 1e-6)
)

geno_hq = subset_markers(geno, high_quality)
```

---

## Performance Optimization

### Multi-Threading

Enable threading when running Julia:

```bash
julia --threads=8 --project=. script.jl
```

Use parallel GRM computation:

```julia
# Automatic threading
G = compute_grm_parallel(geno; use_threads=true)

# Benchmark threading performance
benchmark_threading(geno; method=:vanraden)
```

### Memory Optimization

```julia
# Check memory usage
mem = memory_usage(geno)
println("Memory: $(mem.total / 1e6) MB")
println("Savings: $(mem.savings * 100)%")

# Use LD pruning to reduce dataset
geno_pruned = subset_markers(geno, ld_prune_window(geno; r2_threshold=0.8))

# Convert formats for faster I/O
write_plink("data", geno)  # PLINK is 5-10x smaller than VCF
```

### Large Datasets

For very large datasets (>100k samples or >1M SNPs):

```julia
# 1. Read by chromosome
geno_chr1 = read_vcf("data.vcf.gz"; regions=["1"])

# 2. Apply QC during loading
geno = read_vcf("data.vcf.gz";
    min_qual=30.0,
    pass_only=true,
    biallelic_only=true
)

# 3. Use PCG solver for large GRM
model = GBLUPModel(method=:pcg)  # Faster for n > 5000

# 4. Use parallel GRM
G = compute_grm_parallel(geno; use_threads=true)
```

---

## Best Practices

### Complete Workflow

```julia
using GenomicPro2

# 1. Load data
geno = read_vcf("data.vcf.gz"; regions=["1"])
pheno = read_phenotypes("phenotypes.csv")

# 2. Data exploration
summary_raw = summarize(geno)
println(summary_raw)

# 3. Quality control
geno_qc = quality_control(geno;
    min_maf = 0.01,
    max_missing_per_marker = 0.1,
    hwe_pvalue = 1e-6
)

report = qc_report(geno_qc)
println(report)

# 4. LD pruning
keep_idx = ld_prune_window(geno_qc; r2_threshold=0.8)
geno_pruned = subset_markers(geno_qc, keep_idx)

# 5. Genomic prediction
G = compute_grm_parallel(geno_pruned; method=:vanraden)
model = GBLUPModel(estimate_variances=true)
fit!(model, geno_pruned, pheno; G=G)

# 6. Cross-validation
cv_result = kfold_cv(
    () -> GBLUPModel(),
    geno_pruned, pheno;
    k=5,
    grm_options=(method=:vanraden,)
)

println("CV Accuracy: $(cv_result.metrics.correlation)")

# 7. Predict on new data
gebv = predict(model, geno_test)

# 8. Save results
write_plink("results", geno_pruned)
write_phenotypes("gebv.csv", pheno)
```

### Recommended Settings

**Quality Control:**
- MAF ≥ 0.01 (standard) or 0.05 (conservative)
- Missing rate ≤ 10% per marker/sample
- HWE p-value > 1e-6

**LD Pruning:**
- Window size: 50 SNPs
- Step size: 10 SNPs
- r² threshold: 0.8 (standard) or 0.5 (moderate)

**Cross-Validation:**
- 5-fold CV (standard)
- 10-fold CV (more stable but slower)
- Multiple replicates for robust estimates

**GBLUP:**
- Cholesky solver: n < 10,000
- PCG solver: n > 10,000
- Estimate variances: Yes (for heritability)

**BayesR:**
- Minimum 50,000 iterations
- Burn-in: 20,000-40,000
- Thinning: 5-10
- Run multiple chains to check convergence

---

## Troubleshooting

### Common Issues

**Issue: "No common samples between genotype and phenotype"**

Solution:
```julia
# Check sample IDs
println("Genotype samples: ", geno.sample_ids[1:5])
println("Phenotype samples: ", pheno.sample_ids[1:5])

# Find overlap
common = intersect(geno.sample_ids, pheno.sample_ids)
println("Common samples: $(length(common))")
```

**Issue: "GRM is not positive definite"**

Solutions:
1. Add ridge penalty:
```julia
model = GBLUPModel(ridge=0.01)
```

2. Filter low-MAF markers:
```julia
geno_filtered = filter_maf(geno; min_maf=0.05)
```

3. Remove duplicate samples

**Issue: Slow performance**

Solutions:
1. Enable multi-threading
2. Use LD pruning
3. Convert to PLINK format
4. Use PCG solver for large datasets

**Issue: High memory usage**

Solutions:
1. LD pruning
2. Load by chromosome
3. Filter markers during loading

---

## Additional Resources

- **API Documentation**: See docstrings (`?function_name`)
- **Examples**: `GenomicPro2/examples/`
- **GitHub Issues**: Report bugs and request features
- **Performance**: Run `julia test/benchmark.jl`

---

**Version**: 2.0.0
**Last Updated**: 2024-11-17
