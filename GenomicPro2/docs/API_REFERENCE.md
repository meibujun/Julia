# GenomicPro2 API Reference

Complete API reference for GenomicPro2 v2.0.0

---

## Table of Contents

1. [Core Module](#core-module)
2. [Data Module](#data-module)
3. [IO Module](#io-module)
4. [Models Module](#models-module)
5. [GWAS Module](#gwas-module)
6. [QC Module](#qc-module)
7. [Population Structure Module](#population-structure-module)
8. [Visualization Module](#visualization-module)
9. [GPU Module](#gpu-module)
10. [Config Module](#config-module)
11. [Logging Module](#logging-module)
12. [Web API Module](#web-api-module)

---

## Core Module

### Types

#### `AbstractGenomicModel`

Base type for all genomic prediction models.

```julia
abstract type AbstractGenomicModel end
```

**Required Methods:**
- `fit!(model, genotypes, phenotypes; kwargs...)`
- `predict(model, genotypes; kwargs...)`
- `model_name(model)`
- `model_type(model)`

**Optional Methods:**
- `is_fitted(model)`
- `score(model, genotypes, phenotypes; metric=:correlation)`
- `get_hyperparameters(model)`
- `heritability(model)`

#### `ModelType`

Enum for model categories.

```julia
@enum ModelType begin
    LINEAR_MODEL
    BAYESIAN_MODEL
    KERNEL_MODEL
    DEEP_LEARNING
    ENSEMBLE_MODEL
end
```

---

## Data Module

### CompactGenotypes

Memory-efficient 2-bit genotype storage.

#### Constructor

```julia
CompactGenotypes(n_samples::Int, n_snps::Int)
```

**Parameters:**
- `n_samples`: Number of samples
- `n_snps`: Number of SNPs

**Memory Usage:** ~2 bits per genotype (96.8% reduction vs. Float64)

#### Methods

```julia
getindex(cg::CompactGenotypes, i::Int, j::Int) -> UInt8
setindex!(cg::CompactGenotypes, value::UInt8, i::Int, j::Int)
size(cg::CompactGenotypes) -> Tuple{Int, Int}
```

---

## IO Module

### PLINK Format

#### `read_plink`

Read PLINK binary format files (.bed, .bim, .fam).

```julia
read_plink(prefix::String; verbose=true) -> CompactGenotypes
```

**Parameters:**
- `prefix`: File prefix (e.g., "data/genotypes" for genotypes.bed/bim/fam)
- `verbose`: Show progress information

**Returns:** `CompactGenotypes` object

**Example:**
```julia
genotypes = read_plink("data/genotypes")
println(size(genotypes))  # (n_samples, n_snps)
```

#### `write_plink`

Write genotypes to PLINK binary format.

```julia
write_plink(prefix::String, genotypes::CompactGenotypes,
            sample_ids::Vector{String}, snp_ids::Vector{String})
```

### VCF Format

#### `read_vcf`

Read VCF (Variant Call Format) files.

```julia
read_vcf(filename::String; samples=nothing, regions=nothing) -> CompactGenotypes
```

**Parameters:**
- `filename`: VCF file path (.vcf or .vcf.gz)
- `samples`: Subset of samples to read (optional)
- `regions`: Genomic regions to read (optional)

### Phenotype Files

#### `read_phenotypes`

Read phenotype data from CSV.

```julia
read_phenotypes(filename::String; id_col=1, pheno_col=2) -> PhenotypeData
```

**Parameters:**
- `filename`: CSV file path
- `id_col`: Column index for sample IDs
- `pheno_col`: Column index for phenotype values

**Example:**
```julia
phenotypes = read_phenotypes("data/phenotypes.csv")
```

---

## Models Module

### GBLUP Model

Genomic Best Linear Unbiased Prediction.

#### `GBLUPModel`

```julia
mutable struct GBLUPModel <: AbstractGenomicModel
    grm::Union{Matrix{Float64}, Nothing}
    heritability::Float64
    genetic_variance::Float64
    environmental_variance::Float64
    gebv::Vector{Float64}
end
```

#### `fit!`

Fit GBLUP model.

```julia
fit!(model::GBLUPModel, genotypes, phenotypes;
     grm=nothing, method=:reml, verbose=true)
```

**Parameters:**
- `grm`: Pre-computed GRM (optional, will compute if not provided)
- `method`: `:reml` or `:ml` for variance estimation
- `verbose`: Print progress

**Example:**
```julia
model = GBLUPModel()
fit!(model, genotypes, phenotypes)
println("Heritability: ", model.heritability)
```

#### `predict`

Predict breeding values.

```julia
predict(model::GBLUPModel, genotypes) -> Vector{Float64}
```

### BayesR Model

Bayesian mixture model with SNP-specific variances.

```julia
struct BayesRModel <: AbstractGenomicModel
    n_components::Int
    variance_proportions::Vector{Float64}
    niter::Int
    burnin::Int
end
```

**Default Variance Proportions:** `[0.0, 0.0001, 0.001, 0.01]`

#### Constructor

```julia
BayesRModel(; n_components=4, niter=50000, burnin=10000)
```

### BayesCπ Model

Bayesian variable selection with automatic π estimation.

#### `fit_bayescpi`

```julia
fit_bayescpi(genotypes, phenotypes;
             niter=50000, burnin=10000,
             estimate_pi=true, pi_prior=0.995) -> BayesCpiResults
```

**Parameters:**
- `niter`: Total MCMC iterations
- `burnin`: Burn-in period
- `estimate_pi`: Estimate π automatically
- `pi_prior`: Prior for π (proportion of zero-effect SNPs)

**Returns:**
```julia
struct BayesCpiResults
    beta::Vector{Float64}             # SNP effects
    pi_estimated::Float64              # Estimated π
    h2_estimated::Float64              # Heritability
    inclusion_prob::Vector{Float64}    # SNP inclusion probabilities
    genetic_variance::Float64
    environmental_variance::Float64
end
```

### RKHS Model

Reproducing Kernel Hilbert Space regression.

#### Kernel Functions

```julia
abstract type KernelFunction end

struct GaussianKernel <: KernelFunction
    bandwidth::Float64
end

struct PolynomialKernel <: KernelFunction
    degree::Int
    scale::Float64
    offset::Float64
end

struct ExponentialKernel <: KernelFunction
    bandwidth::Float64
end
```

#### `fit_rkhs`

```julia
fit_rkhs(genotypes, phenotypes;
         kernel=GaussianKernel(),
         auto_bandwidth=true,
         lambda=1e-5) -> RKHSResults
```

**Parameters:**
- `kernel`: Kernel function
- `auto_bandwidth`: Automatically select bandwidth
- `lambda`: Regularization parameter

### Deep GBLUP Model

Deep neural network for genomic prediction.

```julia
mutable struct DeepGBLUP <: AbstractGenomicModel
    input_dim::Int
    hidden_layers::Vector{Int}
    activation::Symbol
    dropout_rate::Float64
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
end
```

#### Constructor

```julia
DeepGBLUP(input_dim::Int;
          hidden_layers=[256, 128, 64],
          activation=:relu,
          dropout_rate=0.2)
```

#### `train_deepgblup!`

```julia
train_deepgblup!(model::DeepGBLUP, genotypes, phenotypes;
                 epochs=100, batch_size=32, learning_rate=0.001,
                 early_stopping=true, patience=10) -> DeepGBLUPResults
```

### Ensemble Model

Combine multiple models with weighted averaging.

```julia
model = EnsembleModel([
    GBLUPModel(),
    BayesRModel(),
    RKHSModel()
])

fit!(model, genotypes, phenotypes, optimize_weights=true)
predictions = predict(model, genotypes)
```

### Model Comparison

```julia
models = [GBLUPModel(), BayesRModel(), RKHSModel()]
results = compare_models(models, genotypes, phenotypes,
                        cv_folds=5, metrics=[:correlation, :mse])
```

---

## GWAS Module

### GWAS Models

#### `LinearModelGWAS`

Simple linear regression GWAS.

```julia
struct LinearModelGWAS <: AbstractGWASModel
    adjust_population_structure::Bool
    n_pcs::Int
end
```

**Example:**
```julia
model = LinearModelGWAS(adjust_population_structure=true, n_pcs=10)
results = perform_gwas(genotypes, phenotypes, model)
```

#### `MixedModelGWAS`

Mixed linear model with random effects.

```julia
struct MixedModelGWAS <: AbstractGWASModel
    grm::Union{Matrix{Float64}, Nothing}
    reml::Bool
end
```

### `perform_gwas`

Run genome-wide association analysis.

```julia
perform_gwas(genotypes, phenotypes, model::AbstractGWASModel;
             parallel=true, verbose=true) -> GWASResult
```

**Returns:**
```julia
struct GWASResult
    snp_ids::Vector{String}
    chromosomes::Vector{Int}
    positions::Vector{Int}
    pvalues::Vector{Float64}
    effect_sizes::Vector{Float64}
    std_errors::Vector{Float64}
    genomic_control_lambda::Float64
    heritability::Union{Float64, Nothing}
end
```

### Multiple Testing Correction

```julia
adjust_pvalues(pvalues::Vector{Float64}; method=:bonferroni) -> Vector{Float64}
```

**Methods:**
- `:bonferroni` - Bonferroni correction
- `:fdr` - Benjamini-Hochberg FDR
- `:sidak` - Šidák correction

**Example:**
```julia
adjusted = adjust_pvalues(results.pvalues, method=:fdr)
significant = findall(adjusted .< 0.05)
```

### Genomic Inflation

```julia
lambda = calculate_genomic_control_lambda(pvalues)
```

---

## QC Module

### Quality Control Filters

```julia
struct QCFilters
    maf_threshold::Float64              # Minor allele frequency
    missing_rate_threshold::Float64     # Maximum missing rate
    hwe_pvalue::Float64                 # Hardy-Weinberg p-value
    ld_window_size::Int                 # LD pruning window
    ld_threshold::Float64               # LD r² threshold
end
```

#### Constructor

```julia
QCFilters(; maf_threshold=0.05,
           missing_rate_threshold=0.1,
           hwe_pvalue=1e-6,
           ld_window_size=50,
           ld_threshold=0.2)
```

### `quality_control`

Apply quality control filters.

```julia
quality_control(genotypes, phenotypes, filters::QCFilters)
    -> (filtered_genotypes, filtered_phenotypes, QCReport)
```

**Example:**
```julia
filters = QCFilters(maf_threshold=0.05, missing_rate_threshold=0.1)
geno_qc, pheno_qc, report = quality_control(genotypes, phenotypes, filters)

println(report)
# QC Report:
#   SNPs before: 500000
#   SNPs after: 425000
#   Samples before: 1000
#   Samples after: 985
```

### LD Pruning

```julia
pruned_snps = ld_pruning(genotypes; window_size=50, threshold=0.2)
```

---

## Population Structure Module

### PCA Analysis

```julia
perform_pca(genotypes; n_components=20, method=:svd) -> PCAResults
```

**Returns:**
```julia
struct PCAResults
    scores::Matrix{Float64}              # PC scores (samples × PCs)
    loadings::Matrix{Float64}            # PC loadings (SNPs × PCs)
    explained_variance::Vector{Float64}   # Variance per PC
    cumulative_variance::Vector{Float64}  # Cumulative variance
end
```

**Example:**
```julia
pca = perform_pca(genotypes, n_components=10)
println("First PC explains: ", pca.explained_variance[1] * 100, "%")
```

### ADMIXTURE Analysis

```julia
perform_admixture(genotypes; K=3, niter=1000, tol=1e-4) -> AdmixtureResults
```

**Parameters:**
- `K`: Number of ancestral populations
- `niter`: Maximum iterations
- `tol`: Convergence tolerance

**Returns:**
```julia
struct AdmixtureResults
    Q::Matrix{Float64}          # Admixture proportions (samples × K)
    F::Matrix{Float64}          # Allele frequencies (SNPs × K)
    log_likelihood::Float64
    converged::Bool
end
```

### FST Calculation

```julia
calculate_fst(genotypes, population1_indices, population2_indices) -> Float64
```

---

## Visualization Module

### Manhattan Plot

```julia
prepare_manhattan_plot(gwas::GWASResult;
                      significant_threshold=5e-8,
                      suggestive_threshold=1e-5) -> ManhattanPlotData
```

### QQ Plot

```julia
prepare_qq_plot(pvalues::Vector{Float64};
               confidence_interval=0.95) -> QQPlotData
```

### PCA Plot

```julia
prepare_pca_plot(pca::PCAResults; pc_x=1, pc_y=2,
                colors=nothing, labels=nothing) -> PCAPlotData
```

### Export Plot Data

```julia
export_plot_data(filename::String, plot_data)
```

Exports to JSON format for web visualization.

---

## GPU Module

### GPU Availability

```julia
has_cuda() -> Bool
```

Check if CUDA is available.

### GPU Information

```julia
gpu_info() -> Dict{String, Any}
```

Get GPU device information.

### GPU-Accelerated GRM

```julia
compute_grm_gpu(genotypes; batch_size=1000) -> Matrix{Float64}
```

**Speedup:** 10-50x faster than CPU for large datasets.

**Example:**
```julia
if has_cuda()
    grm = compute_grm_gpu(genotypes)
else
    grm = compute_grm(genotypes)
end
```

### GPU-Accelerated GWAS

```julia
gwas_gpu(genotypes, phenotypes; model=:linear) -> GWASResult
```

---

## Config Module

### Configuration Structure

```julia
struct GenomicProConfig
    compute::ComputeConfig
    memory::MemoryConfig
    io::IOConfig
    logging::LogConfig
    api::APIConfig
    analysis::AnalysisConfig
end
```

### Load Configuration

```julia
config = load_config("GenomicPro2.toml")
```

### Print Configuration

```julia
print_config(config)
```

### Environment Variables

Override configuration with environment variables:
- `GENOMICPRO_THREADS` - Number of threads
- `GENOMICPRO_GPU` - Enable GPU (true/false)
- `GENOMICPRO_LOG_LEVEL` - Log level (DEBUG/INFO/WARN/ERROR)

---

## Logging Module

### Setup Logging

```julia
setup_logging(; level="INFO",
               log_file="genomicpro2.log",
               console=true,
               performance=false)
```

### Log Macros

```julia
@debug "Debug message" param=value
@info "Info message" n_samples=1000
@warn "Warning message"
@error "Error occurred" exception=e
```

### Performance Logging

```julia
@log_performance "GRM Computation" begin
    grm = compute_grm(genotypes)
end
```

### Performance Timer

```julia
timer = PerformanceTimer("My Analysis")
start!(timer)
# ... do work ...
stop!(timer)
log_performance(timer)
```

---

## Web API Module

### Start Server

```julia
start_server(; host="127.0.0.1", port=8080, verbose=true)
```

### API Endpoints

#### Data Management
- `POST /api/data/upload` - Upload dataset
- `GET /api/data/list` - List datasets
- `DELETE /api/data/:id` - Delete dataset

#### Analysis
- `POST /api/analysis/gwas` - Run GWAS
- `POST /api/analysis/gblup` - Run GBLUP
- `POST /api/analysis/pca` - Run PCA

#### Jobs
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/:id` - Get job status
- `DELETE /api/jobs/:id` - Cancel job

#### Visualization
- `GET /api/viz/manhattan?job_id=...` - Manhattan plot data
- `GET /api/viz/qq?job_id=...` - QQ plot data

#### Health
- `GET /api/health` - Server health check

### Example API Usage

```bash
# Upload data
curl -X POST http://localhost:8080/api/data/upload \
  -H "Content-Type: application/json" \
  -d '{"name": "my_dataset", "type": "genotype"}'

# Run GWAS
curl -X POST http://localhost:8080/api/analysis/gwas \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "abc123", "model": "mixed"}'

# Check job status
curl http://localhost:8080/api/jobs/xyz789
```

---

## Cross-Validation

### K-Fold Cross-Validation

```julia
kfold_cv(genotypes, phenotypes; k=5, model=GBLUPModel(),
         metric=:correlation) -> CVResult
```

### Leave-One-Out Cross-Validation

```julia
loo_cv(genotypes, phenotypes; model=GBLUPModel()) -> CVResult
```

### Random Cross-Validation

```julia
random_cv(genotypes, phenotypes; n_reps=10, train_fraction=0.8,
          model=GBLUPModel()) -> CVResult
```

**Results:**
```julia
struct CVResult
    accuracies::Vector{Float64}
    correlations::Vector{Float64}
    mse::Vector{Float64}
    mean_accuracy::Float64
    mean_correlation::Float64
    mean_mse::Float64
end
```

---

## Utility Functions

### GRM Computation

```julia
compute_grm(genotypes; method=:vanraden) -> Matrix{Float64}
```

**Methods:**
- `:vanraden` - VanRaden method (default)
- `:additive` - Additive relationship matrix

### Parallel GRM

```julia
compute_grm_parallel(genotypes; method=:vanraden,
                     n_threads=Threads.nthreads()) -> Matrix{Float64}
```

### Center and Scale

```julia
center_genotypes(genotypes) -> Matrix{Float64}
scale_genotypes(genotypes) -> Matrix{Float64}
```

---

## Complete Workflow Example

```julia
using GenomicPro2

# 1. Load configuration
config = load_config("GenomicPro2.toml")
setup_logging_from_config(config)

# 2. Load data
genotypes = read_plink("data/genotypes")
phenotypes = read_phenotypes("data/phenotypes.csv")

# 3. Quality control
filters = QCFilters(maf_threshold=0.05, missing_rate_threshold=0.1)
geno_qc, pheno_qc, qc_report = quality_control(genotypes, phenotypes, filters)

# 4. GWAS analysis
model = MixedModelGWAS()
gwas_results = perform_gwas(geno_qc, pheno_qc, model)

# 5. Multiple testing correction
adjusted_pvalues = adjust_pvalues(gwas_results.pvalues, method=:fdr)

# 6. Genomic prediction
gblup = GBLUPModel()
fit!(gblup, geno_qc, pheno_qc)
predictions = predict(gblup, geno_qc)

# 7. Cross-validation
cv_results = kfold_cv(geno_qc, pheno_qc, k=5, model=GBLUPModel())
println("Mean accuracy: ", cv_results.mean_correlation)

# 8. Visualization
manhattan_data = prepare_manhattan_plot(gwas_results)
export_plot_data("manhattan.json", manhattan_data)
```

---

## Version Information

GenomicPro2 v2.0.0

**Minimum Julia Version:** 1.10+

**Dependencies:**
- LinearAlgebra.jl
- Statistics.jl
- Random.jl
- CUDA.jl (optional, for GPU support)
- HTTP.jl (for Web API)
- JSON3.jl (for JSON serialization)
- TOML.jl (for configuration)
- ArgParse.jl (for CLI)

---

## License

MIT License - See LICENSE file for details

---

## Citation

```bibtex
@software{genomicpro2_2024,
  title = {GenomicPro2: A High-Performance Genomic Prediction Toolkit},
  author = {GenomicPro2 Development Team},
  year = {2024},
  url = {https://github.com/meibujun/Julia}
}
```

---

**Last Updated:** 2024-11-18
