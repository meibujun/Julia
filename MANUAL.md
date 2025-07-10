# DynamicEpistasisGBLUP.jl - User Manual

## Table of Contents

1.  [Introduction](#1-introduction)
    *   [Purpose](#purpose)
    *   [Key Features](#key-features)
    *   [Scientific Background](#scientific-background)
    *   [Intended Audience](#intended-audience)
    *   [License](#license)
    *   [Citing the Package](#citing-the-package)
2.  [Installation](#2-installation)
    *   [Prerequisites](#prerequisites)
    *   [Installing Julia](#installing-julia)
    *   [Installing DynamicEpistasisGBLUP.jl](#installing-dynamicEpistasisGBLUPjl)
    *   [GPU Setup](#gpu-setup)
    *   [Verifying Installation](#verifying-installation)
3.  [Core Concepts](#3-core-concepts)
    *   [Genomic Prediction Overview](#genomic-prediction-overview)
    *   [GBLUP (Genomic Best Linear Unbiased Prediction)](#gblup)
    *   [Epistasis in Quantitative Genetics](#epistasis-in-quantitative-genetics)
    *   [Orthogonal Decomposition of Genetic Effects](#orthogonal-decomposition-of-genetic-effects)
    *   [Dynamic Modeling Across Generations](#dynamic-modeling-across-generations)
    *   [Key Data Structures](#key-data-structures)
4.  [Tutorials and Examples](#4-tutorials-and-examples)
    *   [4.1. Simulating a Base Population](#41-simulating-a-base-population)
    *   [4.2. Computing Relationship Matrices (GRMs)](#42-computing-relationship-matrices-grms)
    *   [4.3. Fitting GBLUP Models](#43-fitting-gblup-models)
    *   [4.4. Genomic Prediction for New Individuals](#44-genomic-prediction-for-new-individuals)
    *   [4.5. Cross-Validation](#45-cross-validation)
    *   [4.6. Multi-Generation Selection Simulation](#46-multi-generation-selection-simulation)
5.  [Advanced Modules and Features](#5-advanced-modules-and-features)
    *   [5.1. Walsh-Hadamard Transform (`WalshHadamard.jl`)](#51-walsh-hadamard-transform-walshhadamardjl)
    *   [5.2. NOIA Framework (`NOIAFramework.jl`)](#52-noia-framework-noiaframeworkjl)
    *   [5.3. Sparse Epistasis Detection (`SparseEpistasis.jl`)](#53-sparse-epistasis-detection-sparseepistasisjl)
    *   [5.4. GPU Optimizations (`gpu_optimization.jl`)](#54-gpu-optimizations-gpu_optimizationjl)
    *   [5.5. Distributed Computing (`distributed_computing.jl`)](#55-distributed-computing-distributed_computingjl)
    *   [5.6. Multivariate Analysis (`multivariate_extension.jl`)](#56-multivariate-analysis-multivariate_extensionjl)
    *   [5.7. Breeding Program Optimization (`breeding_optimization.jl`)](#57-breeding-program-optimization-breeding_optimizationjl)
6.  [API Reference](#6-api-reference)
7.  [Program Structure](#7-program-structure)
8.  [Troubleshooting / FAQ](#8-troubleshooting--faq)
9.  [Contributing](#9-contributing)
10. [References](#10-references)

---
*(Sections below will be filled in subsequent steps)*
---

## 1. Introduction

### Purpose
`DynamicEpistasisGBLUP.jl` is a high-performance Julia package designed for advanced genomic prediction in the field of quantitative genetics, with a primary focus on livestock breeding. Its core purpose is to implement and make accessible state-of-the-art statistical models that incorporate dynamic orthogonal epistatic effects within the Genomic Best Linear Unbiased Prediction (GBLUP) framework. This allows for more accurate breeding value estimation by accounting for non-additive genetic variance and the changing genetic architecture of populations under selection.

### Key Features
*   **Orthogonal Epistasis GBLUP:** Implements GBLUP models considering additive and additive-by-additive epistatic effects with an orthogonal decomposition to ensure unbiased variance component estimation.
*   **Dynamic Modeling:** Supports the dynamic update of allele frequencies and relationship matrices, crucial for multi-generation selection scenarios where genetic architectures evolve.
*   **GPU Acceleration:** Leverages CUDA.jl and KernelAbstractions.jl for significant performance gains in computationally intensive tasks like GRM calculation and model fitting, making large-scale analyses feasible.
*   **Population Simulation:** Includes a comprehensive module for simulating populations with complex genetic architectures (customizable QTLs, heritabilities, epistatic patterns) to test and validate models.
*   **Advanced Methodologies (Partially or Fully Implemented):**
    *   Walsh-Hadamard Transform for specific epistasis detection approaches.
    *   Natural and Orthogonal Interactions (NOIA) framework principles for effect parameterization.
    *   Symmetric polynomial algorithms for efficient epistatic GRM computation.
    *   Sparse epistasis detection techniques (e.g., screening, Elastic Net).
    *   Frameworks for GPU optimization, distributed computing, multivariate analysis, and breeding program optimization (many of these are currently stubs or conceptual outlines).
*   **Modular Design:** Built with a modular structure in Julia for clarity, extensibility, and maintainability.

### Scientific Background
The methodologies implemented in this package are rooted in quantitative genetics theory, building upon the foundational GBLUP method. The core innovation, "Dynamic Orthogonal Epistasis," is based on the principles described in the research document `Notes.docx` (provided with the project), which details how to properly partition and account for epistatic variance in evolving populations. This involves allele-frequency-dependent genotype coding and dynamic recalculation of model parameters to maintain orthogonality and prediction accuracy over time.

### Intended Audience
This package is intended for:
*   Quantitative geneticists and researchers in animal/plant breeding.
*   Statisticians and bioinformaticians working on genomic prediction models.
*   Students learning about advanced genomic selection techniques.
*   Professionals in the livestock industry looking to apply cutting-edge prediction methods.

A good understanding of linear mixed models, quantitative genetics, and basic Julia programming is recommended for effective use and extension of this package.

### License
This package is distributed under the MIT License. (A `LICENSE` file should ideally be present in the repository root).

### Citing the Package
(Placeholder - Details to be added when the package is published or associated with a publication.)
Users of this software are encouraged to cite the relevant research papers and the software itself.

## 2. Installation

### Prerequisites
*   **Julia:** Version 1.9 or higher is required. Download from [julialang.org](https://julialang.org/downloads/).
*   **CUDA Toolkit (for GPU support):** If you intend to use GPU acceleration, an NVIDIA GPU compatible with CUDA is required. You will also need the NVIDIA CUDA Toolkit installed on your system. The version should be compatible with the version of `CUDA.jl` used by the package (check `Project.toml`). Visit the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download.
*   **Git:** For cloning the development version of the package.

### Installing Julia
Follow the instructions on the [Julia language website](https://julialang.org/downloads/platform/) for your operating system. Ensure that the `julia` executable is added to your system's PATH.

### Installing DynamicEpistasisGBLUP.jl
Currently, `DynamicEpistasisGBLUP.jl` is under development and not yet registered in the general Julia registry. To install it:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>  # Replace <repository_url> with the actual URL
    cd DynamicEpistasisGBLUP.jl
    ```
2.  **Install using Julia's Pkg manager:**
    Open Julia REPL in the cloned directory and run:
    ```julia
    import Pkg
    Pkg.activate(".")  # Activates the project environment
    Pkg.instantiate()  # Installs all dependencies from Project.toml
    ```

Alternatively, if you have the package files locally, you can `dev` the package:
```julia
import Pkg
Pkg.dev("path/to/DynamicEpistasisGBLUP.jl")
```

### GPU Setup
For GPU acceleration, `CUDA.jl` needs to correctly identify your NVIDIA GPU and CUDA toolkit.

1.  **Install `CUDA.jl`:** If not already listed as a direct dependency that `Pkg.instantiate()` handles, you might need to add it (though it's listed in this project's `Project.toml`):
    ```julia
    import Pkg
    Pkg.add("CUDA")
    ```
2.  **Test CUDA.jl:** In a Julia REPL, run:
    ```julia
    using CUDA
    CUDA.functional()
    # Should return true if a compatible GPU and CUDA toolkit are found.
    CUDA.devices()
    # Should list your available NVIDIA GPU(s).
    ```
    If `CUDA.functional()` is `false`, consult the [CUDA.jl documentation](https://cuda.juliagpu.org/stable/installation/overview/) for troubleshooting steps. This often involves setting environment variables like `CUDA_HOME` or ensuring driver compatibility.

    **Important for GPU performance:**
    ```julia
    CUDA.allowscalar(false)
    ```
    This is generally recommended for `CUDA.jl` to prevent accidental scalar operations on the GPU which are very slow. The package's kernels are designed for vectorized operations.

### Verifying Installation
After installation, you can try to load the package and run a basic function (if a simple exported test function exists) or the demo:

```julia
using DynamicEpistasisGBLUP

# Check if CUDA is functional (optional, but good for GPU features)
using CUDA
if CUDA.functional()
    println("CUDA is functional with device: ", CUDA.name(CUDA.device()))
    CUDA.allowscalar(false)
else
    println("CUDA not functional. GPU features will be unavailable.")
end

# Try running the built-in demo (if available and configured for quick run)
# This demo uses GPU functionalities if available.
println("Attempting to run a small demo...")
try
    populations_history, models_history, results_df = run_demo()
    println("Demo completed successfully.")
catch e
    println("Error running demo: ", e)
    println("This might be due to missing full implementations for some advanced features called in the demo,")
    println("or issues with the GPU environment if CUDA is expected.")
end
```
Successful execution of the demo (even with warnings about stubbed features) indicates that the core package and its main dependencies are loaded correctly. Refer to the `TESTING_PLAN.md` for more comprehensive testing once the environment is fully set up.

## 3. Core Concepts

This section introduces fundamental concepts essential for understanding and effectively utilizing the `DynamicEpistasisGBLUP.jl` package.

### Genomic Prediction Overview
Genomic Prediction (GP), also known as Genomic Selection (GS), aims to predict the genetic merit (e.g., breeding value for a trait) of individuals using genome-wide marker data (like SNPs). By capturing the effects of numerous small genetic variations across the genome, GP allows for more accurate selection decisions, especially for complex traits and at early stages in an individual's life, thereby accelerating genetic gain in breeding programs.

### GBLUP (Genomic Best Linear Unbiased Prediction)
GBLUP is a widely used method for genomic prediction. It assumes that the trait is influenced by many genes, each with a small effect, and these effects are normally distributed. Instead of estimating individual marker effects, GBLUP estimates the total genetic value of an individual using a Genomic Relationship Matrix (GRM). The GRM quantifies the genomic similarity between pairs of individuals based on their marker genotypes. The model is typically a linear mixed model:

`y = Xβ + Zg + e`

where:
- `y` is the vector of phenotypes.
- `β` is a vector of fixed effects (e.g., mean, herd-year-season).
- `X` is an incidence matrix for fixed effects.
- `g` is a vector of random genomic breeding values, assumed `g ~ N(0, Gσ²_g)`, where `G` is the GRM and `σ²_g` is the additive genetic variance.
- `Z` is an incidence matrix linking `g` to `y` (often an identity matrix in animal models where `g` represents individual animal effects).
- `e` is a vector of random residual errors, assumed `e ~ N(0, Iσ²_e)`.

Variance components (σ²_g, σ²_e) are typically estimated using REML.

### Epistasis in Quantitative Genetics
Epistasis refers to non-additive interactions between alleles at different genetic loci. While additive effects (where the effect of an allele sums up independently) often explain the largest portion of genetic variance, epistatic interactions can significantly contribute to the variation of complex traits. Ignoring epistasis can lead to:
-   Underestimation of total genetic variance.
-   Biased estimates of additive effects if epistasis is confounded with them.
-   Reduced accuracy of genomic predictions, especially for traits with substantial interaction components.
-   Missed opportunities to leverage favorable gene combinations in selection or mating strategies.

Within its core GBLUP framework, `DynamicEpistasisGBLUP.jl` primarily focuses on modeling **additive-by-additive epistasis (A x A)** by incorporating specific epistatic GRMs. This is often considered the most significant type of pairwise interaction for quantitative traits. Other tools within the package (like WHT or NOIA utilities) can be used to explore or define different types or higher orders of epistasis more explicitly.

### Orthogonal Decomposition of Genetic Effects
A key challenge in modeling epistasis is that statistical estimates of additive and epistatic effects (and their variances) can become confounded, especially when allele frequencies deviate from 0.5 or when populations are not in idealized equilibrium. This confounding makes it difficult to interpret the true contribution of each genetic component.

The **orthogonal decomposition** approach, with principles aligned with frameworks like NOIA (Natural and Orthogonal Interactions), aims to define genetic effects (additive, dominance, epistatic) such that they are statistically independent (orthogonal) in the specific population being analyzed. This is often achieved by using allele frequency-dependent codings for genotypes when defining effects, or by constructing relationship matrices that ensure orthogonality between variance components.
Benefits of orthogonality include:
-   Unbiased and independent estimation of variance components (e.g., σ²_a for additive, σ²_aa for A x A epistasis).
-   More accurate partitioning of total genetic variance.
-   Stable interpretation of genetic architecture even when model complexity increases (e.g., adding epistatic terms doesn't distort additive variance estimates).

### Dynamic Modeling Across Generations
In populations undergoing selection, allele frequencies change over generations. This can alter the magnitude and variance of genetic effects, including epistatic interactions. A model parameterized for one generation might become suboptimal for predicting future generations.

**Dynamic modeling**, as implemented in this package, addresses this by:
-   Recalculating allele frequencies for the current generation based on its genotype data.
-   Updating genotype codings (if using explicit NOIA parameterization) and genomic relationship matrices (both additive `G` and epistatic `G_aa`) based on these current allele frequencies.
This ensures that the model adapts to the evolving genetic background of the population, aiming to maintain prediction accuracy and the validity of variance component estimates over time. This approach is particularly important for long-term selection programs, as highlighted in the `Notes.docx` paper.

### Key Data Structures
The package utilizes several core data structures defined in `src/types.jl` to manage genomic and phenotypic information efficiently:

*   **`GenotypeMatrix{T}`**: This mutable struct is central to storing genotype data.
    *   `data::CuArray{T, 2}`: A GPU array (Individuals × SNPs) holding the numerical representation of genotypes (e.g., 0, 1, 2 for allele counts, or standardized values for GRM computations).
    *   `missing_mask::CuSparseMatrixCSR{Bool, Int32}`: A sparse matrix on the GPU indicating missing genotypes (`true` for missing). This allows for efficient storage when missing data is sparse.
    *   `allele_freq::CuVector{T}`: A GPU vector storing the reference allele frequency (p) for each SNP. This is dynamically updated.
    *   `n_individuals::Int32`, `n_snps::Int32`: Dimensions of the genotype data.
    *   `ploidy::Int8`: Ploidy of the organism (typically 2 for diploid species).
    For full details, use `?DynamicEpistasisGBLUP.GenotypeMatrix` in the Julia REPL.


*   **`PhenotypeData{T}`**: Stores phenotypic information.
    *   `values::Vector{T}`: A vector of phenotypic observations for the trait(s) under analysis.
    *   `trait_names::Vector{Symbol}`: Names of the traits.
    *   `fixed_effects::Union{Nothing, DataFrame}`: Optional DataFrame for fixed effect covariates.
    *   `random_effects::Union{Nothing, DataFrame}`: Optional DataFrame for other known random effects.
    See `?DynamicEpistasisGBLUP.PhenotypeData`.

*   **`PopulationData{T}`**: A container that groups `GenotypeMatrix`, `PhenotypeData`, and other relevant information for a specific population or generation.
    *   `genotypes::GenotypeMatrix{T}`
    *   `phenotypes::PhenotypeData{T}`
    *   `pedigree::Union{Nothing, SparseMatrixCSC{T, Int}}`: Optional pedigree information.
    *   `generation::Int32`: Identifier for the generation.
    *   `metadata::Dict{Symbol, Any}`: For storing simulation parameters, true genetic values, etc.
    See `?DynamicEpistasisGBLUP.PopulationData`.

*   **`GeneticArchitecture{T}`**: Used in simulations to define the true genetic model of traits.
    *   `n_qtl_additive::Int32`, `additive_qtl_actual_indices::Vector{Int32}`, `additive_effects::Vector{T}`: Define number, genomic positions (column indices), and effects of additive QTLs.
    *   `n_epistatic_pairs::Int32`, `epistatic_pairs_actual_indices::Vector{Tuple{Int32, Int32}}`, `epistatic_effects::Vector{T}`: Define number, SNP pair indices, and effects of epistatic interactions.
    *   `h2_narrow_target::T`, `h2_broad_target::T`: Target heritabilities for the simulation setup.
    See `?DynamicEpistasisGBLUP.GeneticArchitecture`.

*   **`VarianceComponents{T}`**: A mutable struct to hold estimated variance components from REML analysis.
    *   `σ²_a::T` (additive), `σ²_aa::T` (epistatic), `σ²_e::T` (residual), `σ²_p::T` (phenotypic).
    *   `h²::T` (narrow-sense heritability), `H²::T` (broad-sense heritability).
    See `?DynamicEpistasisGBLUP.VarianceComponents`.

*   **`OrthogonalGBLUP{T}`**: Represents a fitted GBLUP model.
    *   `G::CuArray{T,2}`: The additive GRM used.
    *   `G_aa::Union{Nothing, CuArray{T,2}}`: The epistatic GRM (if included).
    *   `variance::VarianceComponents{T}`: The estimated variance components.
    *   `fixed_effects::Union{Nothing, Matrix{T}}`: Estimated fixed effect coefficients.
    *   `generation::Int32`: The generation context of the model.
    See `?DynamicEpistasisGBLUP.OrthogonalGBLUP`.

A thorough understanding of these concepts and data structures will facilitate the effective use and extension of `DynamicEpistasisGBLUP.jl`.

## 4. Tutorials and Examples

### 4.1. Simulating a Base Population

The foundation of many genetic analyses and model testing is a realistically simulated population. `DynamicEpistasisGBLUP.jl` provides robust tools for this in its `Simulation` module (accessed via functions exported from the main `DynamicEpistasisGBLUP` module).

**Key Functions:**

*   `simulate_population(...)`: The main function to generate a base population.
*   `initialize_genetic_architecture(...)`: Defines the number of QTLs, their effects (additive and epistatic), and target heritabilities. This is usually called internally by `simulate_population`.
*   `generate_base_genotypes(...)`: Creates genotype data based on allele frequencies and Hardy-Weinberg equilibrium. Called by `simulate_population`.
*   `calculate_true_genetic_values(...)`: Computes true genetic values based on the defined architecture. Called by `simulate_population`.
*   `generate_phenotypes(...)`: Generates phenotypes by adding environmental noise. Called by `simulate_population`.

**Example: Simulating a Small Sheep-like Population**

Let's simulate a small population with 200 individuals and 1,000 SNPs, aiming for a narrow-sense heritability (h²) of 0.3 and a broad-sense heritability (H²) of 0.4 (implying 10% variance from epistasis).

```julia
using DynamicEpistasisGBLUP
using Distributions # For Beta distribution for MAF
using CUDA # To check functionality, not strictly needed for this example if not using GPU later immediately

# Ensure CUDA is functional if you plan to use GPU features later
# if CUDA.functional()
#     CUDA.allowscalar(false)
# end

# Simulation parameters
n_ind = 200
n_snp = 1000
n_add_qtl = 20
n_epi_pairs = 10
h2_narrow = 0.3
h2_broad = 0.4

# Simulate the population
# Note: The `Float` type used internally (e.g., Float32 for GPU efficiency) is set in DynamicEpistasisGBLUP module.
base_population = simulate_population(
    n_individuals=n_ind,
    n_snps=n_snp,
    n_qtl_additive=n_add_qtl,
    n_qtl_epistatic_pairs=n_epi_pairs,
    h2_narrow=h2_narrow,
    h2_broad=h2_broad,
    maf_distribution=Beta(0.5, 0.5), # U-shaped MAF distribution (more rare alleles)
    seed=123 # For reproducibility
)

println("Simulation complete. Population data generated:")
println("Number of individuals: ", base_population.genotypes.n_individuals)
println("Number of SNPs: ", base_population.genotypes.n_snps)
println("Number of phenotypes: ", length(base_population.phenotypes.values))

# Inspecting the Genetic Architecture
arch = base_population.metadata[:architecture]
println("\nGenetic Architecture Details:")
println("  Number of Additive QTLs: ", arch.n_qtl_additive)
println("  Additive QTL Indices (first 5): ", arch.additive_qtl_actual_indices[1:min(5, end)])
println("  Additive Effects (first 5): ", round.(arch.additive_effects[1:min(5, end)], digits=3))
println("  Number of Epistatic Pairs: ", arch.n_epistatic_pairs)
println("  Epistatic Pairs (first 3): ", arch.epistatic_pairs_actual_indices[1:min(3, end)])
println("  Epistatic Effects (first 3): ", round.(arch.epistatic_effects[1:min(3, end)], digits=3))
println("  Target h² (narrow): ", arch.h2_narrow_target)
println("  Target H² (broad): ", arch.h2_broad_target)

# Inspecting True Genetic Values (example for first 5 individuals)
true_bvs = base_population.metadata[:true_breeding_values]
println("\nTrue Breeding Values (first 5 individuals):")
println(round.(true_bvs[1:min(5, end)], digits=3))

# Inspecting Phenotypes (example for first 5 individuals)
phenos = base_population.phenotypes.values
println("\nPhenotypes (first 5 individuals):")
println(round.(phenos[1:min(5, end)], digits=3))

# Accessing Genotype Data (example: first 3 individuals, first 5 SNPs)
# Note: base_population.genotypes.data is a CuArray if CUDA is functional.
# We need to bring it to CPU with Array() for printing.
println("\nGenotype Data (first 3 individuals, first 5 SNPs):")
if CUDA.functional() && isa(base_population.genotypes.data, CuArray)
    println(Array(base_population.genotypes.data[1:min(3,end), 1:min(5,end)]))
else
    # If not on GPU or data is already CPU Matrix (e.g. during CPU-only path development)
    println(base_population.genotypes.data[1:min(3,end), 1:min(5,end)])
end
println("Allele Frequencies (first 5 SNPs):")
if CUDA.functional() && isa(base_population.genotypes.allele_freq, CuArray)
    println(round.(Array(base_population.genotypes.allele_freq[1:min(5,end)]), digits=3))
else
    println(round.(base_population.genotypes.allele_freq[1:min(5,end)], digits=3))
end
```

**Explanation of the Example:**
1.  We import necessary packages.
2.  Define parameters for our simulation: number of individuals, SNPs, QTLs, and target heritabilities.
3.  `simulate_population(...)` is called. Internally, this:
    *   Calls `initialize_genetic_architecture` to decide which SNPs are QTLs and what their effects are. It attempts to scale these effects so that the resulting genetic variance components in the simulated population will roughly match the target `h2_narrow` and `h2_broad`.
    *   Calls `generate_base_genotypes` to create the `GenotypeMatrix`. Allele frequencies are drawn from the specified `Beta(0.5, 0.5)` distribution, which tends to produce more SNPs with lower minor allele frequencies. Genotypes are then assigned based on these frequencies assuming Hardy-Weinberg Equilibrium.
    *   Calls `calculate_true_genetic_values` using the defined architecture and the generated genotypes to compute the true additive, epistatic, and total genetic values for each individual. These true total genetic values are stored in `base_population.metadata[:true_breeding_values]`.
    *   Calls `generate_phenotypes` to add random environmental noise to the true total genetic values, such that the resulting phenotypic variance aligns with the target `h2_broad`.
4.  The returned `base_population` object (of type `PopulationData`) contains all this information.
5.  The example then shows how to access and print some of this information, such as the parameters of the genetic architecture, a sample of true breeding values, phenotypes, and genotype data.

This simulated `PopulationData` object can then be used as input for GRM computation, model fitting, and other analyses within the package.

### 4.2. Computing Relationship Matrices (GRMs)
*(This section will be detailed in a subsequent step, covering `grm_computation.jl` and `gpu_kernels.jl` for additive and epistatic GRMs, including CPU and GPU versions, and cross-population GRMs. For now, assume you have computed `Ga_train` (additive GRM) and `Gaa_train` (epistatic GRM) for your training population, and `Ga_val_train` / `Gaa_val_train` for relationships between validation and training sets.)*

### 4.3. Fitting GBLUP Models

Once you have your phenotype data (`y_train`) and the necessary Genomic Relationship Matrices (GRMs) for your training population (e.g., `Ga_train` for additive effects, `Gaa_train` for A×A epistatic effects), the next step is to fit the GBLUP model. This primarily involves estimating variance components using Restricted Maximum Likelihood (REML). The core functionalities for this are in `reml.jl` and `epistasis_core.jl` (though `epistasis_core.jl` is more about foundational operations for epistasis, while GRM construction itself is in `grm_computation.jl`).

**Key Functions:**

*   `estimate_variance_components_reml!(...)` (from `reml.jl`): The main function for AI-REML estimation of variance components.
*   `REMLParameters(...)` (from `types.jl`): Struct to control REML algorithm parameters.
*   `REMLResults(...)` (from `types.jl`): Struct to store results from REML.

**The Mixed Model Equation (MME) Context:**

The package solves variants of the MME. For a model with additive and A×A epistatic effects, it's conceptually:
`y = Xb + Zg_a + Wg_aa + e`
where:
*   `y`: phenotypes
*   `Xb`: fixed effects (e.g., overall mean)
*   `g_a ~ N(0, G_a * σ²_a)`: additive genetic effects, `G_a` is the additive GRM.
*   `g_aa ~ N(0, G_aa * σ²_aa)`: A×A epistatic genetic effects, `G_aa` is the A×A epistatic GRM.
*   `e ~ N(0, I * σ²_e)`: residuals.
The REML algorithm estimates `σ²_a`, `σ²_aa`, and `σ²_e`.

**Example: Fitting an Additive + Epistatic GBLUP Model**

Let's assume `base_population` from the simulation example (Section 4.1) is our training set. We first need to compute the GRMs. (We'll placeholder this step for now, as GRM computation is detailed in 4.2, but imagine `Ga` and `Gaa` are available).

```julia
using DynamicEpistasisGBLUP, LinearAlgebra, CUDA
# Assuming 'base_population' from the simulation step (section 4.1) is available.
# y_train = base_population.phenotypes.values
# Z_train_full_snp_matrix = base_population.genotypes.data # Individuals x SNPs (on GPU if functional)
# allele_freqs_train = base_population.genotypes.allele_freq # (on GPU if functional)

# --- Placeholder for GRM computation (details in Section 4.2) ---
# For this example, let's create dummy GRMs.
# In a real scenario, these would be computed using functions from grm_computation.jl
n_train = base_population.genotypes.n_individuals
# Ga_train = compute_additive_grm(Z_train_full_snp_matrix, allele_freqs_train) # Conceptual
# Gaa_train = compute_epistatic_grm(Z_train_full_snp_matrix, allele_freqs_train) # Conceptual

# Dummy GRMs for demonstration if actual computation is too slow for manual example:
# Ensure they are on the correct device (CPU or GPU) and Float type.
# Default data type in the package is Float32 for GPU ops.
dtype = DynamicEpistasisGBLUP.FLOAT_TYPE
current_device = DynamicEpistasisGBLUP.get_device()

# Create symmetric positive semi-definite dummy GRMs
_Ga_cpu = Symmetric(rand(dtype, n_train, n_train) * rand(dtype, n_train, n_train)')
_Ga_cpu ./= tr(_Ga_cpu) / n_train # Basic scaling
Ga_train = current_device(_Ga_cpu)

_Gaa_cpu = Symmetric(rand(dtype, n_train, n_train) * rand(dtype, n_train, n_train)')
_Gaa_cpu ./= tr(_Gaa_cpu) / n_train
Gaa_train = current_device(_Gaa_cpu)

y_train = current_device(convert(Vector{dtype}, base_population.phenotypes.values))
X_train = current_device(ones(dtype, n_train, 1)) # Intercept as fixed effect
# --- End Placeholder for GRM computation ---

# Define REML parameters
reml_params = REMLParameters(
    max_iter=100,
    tol=1e-6,
    verbose=true,
    min_variance_value=1e-8, # To keep variance estimates positive
    use_gpu = CUDA.functional() && DynamicEpistasisGBLUP.USE_GPU[] # Use GPU if available and enabled
)

# Initial variance component guesses (Additive, Epistatic, Residual)
# Order must match the order of GRMs provided to the function + residual at the end.
total_pheno_var_approx = var(base_population.phenotypes.values) # Use CPU version for var
initial_va = dtype(0.4 * total_pheno_var_approx)
initial_vaa = dtype(0.2 * total_pheno_var_approx)
initial_ve = dtype(0.4 * total_pheno_var_approx)
initial_variances = [initial_va, initial_vaa, initial_ve]

# List of GRMs for random effects (excluding identity for residuals, handled internally)
# These should be on the GPU if use_gpu is true.
active_GRMs = [Ga_train, Gaa_train]

println("Starting REML estimation...")
# Call the REML estimation function
# Ensure all inputs are on the correct device (CPU/GPU) and of the correct Float type.
reml_results = estimate_variance_components_reml!(
    y_train,
    active_GRMs,
    X_train,
    initial_variances, # Initial guesses for VCs (additive, epistatic, residual)
    reml_params
)

# Display results
if reml_results.converged
    println("\nREML converged in ", reml_results.iterations, " iterations.")
    est_va, est_vaa, est_ve = reml_results.var_components
    println("  Estimated Additive Variance (σ²_a): ", round(est_va, digits=4))
    println("  Estimated Epistatic Variance (σ²_aa): ", round(est_vaa, digits=4))
    println("  Estimated Residual Variance (σ²_e): ", round(est_ve, digits=4))

    total_genetic_var_est = est_va + est_vaa
    total_pheno_var_est = total_genetic_var_est + est_ve
    h2_est = est_va / total_pheno_var_est
    H2_est = total_genetic_var_est / total_pheno_var_est
    println("  Estimated Narrow-sense Heritability (h²): ", round(h2_est, digits=3))
    println("  Estimated Broad-sense Heritability (H²): ", round(H2_est, digits=3))
    println("  REML Log-Likelihood: ", round(reml_results.log_likelihood, digits=2))

    # Store results in a VarianceComponents struct
    vc_estimates = VarianceComponents(
        σ²_a=est_va, σ²_aa=est_vaa, σ²_e=est_ve, σ²_p=total_pheno_var_est,
        h²=h2_est, H²=H2_est
    )
    println("\nVarianceComponents struct: ", vc_estimates)

else
    println("\nREML did not converge.")
    println("  Last variance estimates: ", reml_results.var_components)
    println("  Log-likelihood: ", reml_results.log_likelihood)
end

# The reml_results struct also contains P and V_inv matrices (reml_results.P, reml_results.V_inv)
# which are essential for calculating BLUPs. These will be on the GPU if use_gpu was true.
# Example: size(reml_results.P)
```

**Explanation:**

1.  **Prepare Inputs:**
    *   `y_train`: Vector of phenotypes.
    *   `active_GRMs`: A list/vector of GRM matrices (e.g., `[Ga_train, Gaa_train]`). These should correspond to the random genetic effects you want to model. The residual component (`I*σ²_e`) is handled internally by `estimate_variance_components_reml!`.
    *   `X_train`: Incidence matrix for fixed effects (a column of ones for an overall mean is common).
    *   `initial_variances`: A vector of starting values for the variance components. The order is critical: it must match the order of GRMs in `active_GRMs`, followed by the initial guess for the residual variance `σ²_e`.
    *   `reml_params`: A `REMLParameters` struct to control the algorithm's behavior (iterations, tolerance, verbosity, device).

2.  **Run REML:**
    *   `estimate_variance_components_reml!` is called. This function implements an Average Information (AI) REML algorithm.
    *   It iteratively updates variance components by:
        *   Constructing the total phenotypic covariance matrix `V = Σ(GRM_i * σ²_i) + I * σ²_e`.
        *   Calculating `V⁻¹` and the projection matrix `P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹`.
        *   Updating variance components using derivatives of the REML log-likelihood and the AI matrix.
        *   Employing techniques like backtracking line search to aid convergence.
    *   The `use_gpu` flag in `REMLParameters` determines if GPU acceleration (via `CUDA.jl`) is used for matrix operations, provided CUDA is functional. All input matrices (`y`, `GRMs`, `X`) should be on the chosen device.

3.  **Interpret Results:**
    *   The function returns a `REMLResults` struct.
    *   `var_components`: A vector of estimated variance components (`[σ²_a_est, σ²_aa_est, σ²_e_est]`).
    *   `log_likelihood`: The REML log-likelihood value at convergence.
    *   `converged`, `iterations`: Status and count of iterations.
    *   `P`, `V_inv`: Matrices useful for subsequent BLUP calculations.

**`epistasis_core.jl` Role:**
While `reml.jl` handles the variance component estimation, `epistasis_core.jl` contains more fundamental building blocks that *could* be used in forming specialized epistatic terms or models, particularly if one were not using a pre-computed epistatic GRM. For example:
*   `construct_interaction_terms_cpu/gpu(...)`: Can create explicit marker interaction terms (e.g., element-wise product of two marker vectors). This is more relevant for marker-effect models (like BayesEpistasis) than standard GBLUP with epistatic GRMs.
*   `encode_marker_value`, `decode_marker_value`: Utilities for specific genotype encodings, sometimes used in certain epistatic scan algorithms.

In the context of the GBLUP workflow described above, the primary contribution for epistasis comes from providing a correctly computed epistatic GRM (e.g., `Gaa_train` from `grm_computation.jl`) to the `estimate_variance_components_reml!` function. The REML algorithm then estimates its associated variance component.

Once variance components are estimated, the next step is to calculate Best Linear Unbiased Predictions (BLUPs) for genetic values, which is covered in Section 4.4.

### 4.4. Genomic Prediction for New Individuals

After fitting a GBLUP model and estimating variance components using the training data (as shown in Section 4.3), you can predict the genetic values (BLUPs - Best Linear Unbiased Predictors) for new individuals (validation or selection candidates) that have genotypes but no phenotypes. This is the core of genomic selection. The `prediction.jl` module handles these calculations.

**Key Functions:**

*   `genomic_prediction(...)` (from `prediction.jl`): Calculates BLUPs for a set of validation individuals.
*   `calculate_accuracy(...)` (from `prediction.jl` or a utility module): Computes prediction accuracy (e.g., correlation between predicted and true BVs if known from simulation).

**Conceptual Basis for Prediction:**

Given the estimated variance components (`σ²_a`, `σ²_aa`, `σ²_e`) from the training set, and the fixed effects estimates (`β_hat`), the BLUP for the total genetic value (`u_val`) of a validation individual is:

`u_val = Cov(u_val, u_train) * [Cov(u_train, u_train) + I*(σ²_e / (relevant scaling for G))]⁻¹ * (y_train - X_train * β_hat)`

Where:
*   `u_train`: Genetic values of training individuals.
*   `Cov(u_val, u_train)`: Covariance matrix (or vector if predicting one individual) between validation and training individuals. This is constructed from the relevant GRMs (e.g., `G_avt` for additive effects, `G_aavt` for epistatic effects between validation and training sets) scaled by their respective variance components.
    *   `Cov_total_val_train = G_avt * σ²_a + G_aavt * σ²_aa + ...`
*   `Cov(u_train, u_train)`: Total genetic covariance matrix for training individuals, i.e., `G_a * σ²_a + G_aa * σ²_aa + ...`.
*   The term `[Cov(u_train, u_train) + I*(σ²_e / relevant_scaling)]⁻¹ * (y_train - X_train * β_hat)` is equivalent to `Z' * V_inv * (y_train - X_train * β_hat)` from the MME solution, where `Z` is identity here. More simply, `P * y_train` (where `P` is from REML results) can be used if `y_train` is already adjusted for fixed effects, or `V_inv * (y_train - X_train * β_hat)` is used.

The `genomic_prediction` function simplifies this by taking the necessary GRMs and estimated variance components.

**Example: Predicting for a Validation Set**

Let's assume we have:
*   `base_population` (used as training set, from Section 4.1).
*   `reml_results` (from fitting the model on `base_population`, Section 4.3).
*   A new set of individuals, `validation_population`, simulated similarly to `base_population`, for whom we want to predict BVs.
*   GRMs:
    *   `Ga_train`, `Gaa_train` (training set GRMs, already computed).
    *   `Ga_val_train`, `Gaa_val_train` (cross-GRMs between validation and training individuals). These need to be computed.

```julia
using DynamicEpistasisGBLUP, LinearAlgebra, CUDA, Statistics
# Assume 'base_population' (training), 'reml_results' are available.
# est_va, est_vaa, est_ve = reml_results.var_components

# --- Simulate a Validation Population (similar to Section 4.1) ---
n_val = 100
validation_population = simulate_population(
    n_individuals=n_val,
    n_snps=base_population.genotypes.n_snps, # Same SNP set
    n_qtl_additive=base_population.metadata[:architecture].n_qtl_additive,
    n_qtl_epistatic_pairs=base_population.metadata[:architecture].n_epistatic_pairs,
    h2_narrow=base_population.metadata[:architecture].h2_narrow_target,
    h2_broad=base_population.metadata[:architecture].h2_broad_target,
    # Use the same architecture for consistency in true BVs
    architecture_template=base_population.metadata[:architecture],
    seed=456
)
# --- End Validation Population Simulation ---


# --- Placeholder for GRM computation (details in Section 4.2) ---
# We need:
# Ga_train, Gaa_train (already used for REML, assume available on correct device)
# Ga_val_train: Additive GRM between validation (rows) and training (cols)
# Gaa_val_train: Epistatic GRM between validation (rows) and training (cols)

dtype = DynamicEpistasisGBLUP.FLOAT_TYPE
current_device = DynamicEpistasisGBLUP.get_device()
n_train = base_population.genotypes.n_individuals

# Dummy cross-GRMs for demonstration
_Gavt_cpu = rand(dtype, n_val, n_train)
# Basic scaling might not be appropriate for cross-GRMs without proper centering.
# For real cross-GRMs, use compute_grm_cross and compute_epistatic_grm_cross.
Ga_val_train = current_device(_Gavt_cpu)

_Gaavt_cpu = rand(dtype, n_val, n_train)
Gaa_val_train = current_device(_Gaavt_cpu)

# Training GRMs (re-using from REML example, or re-define if not in scope)
# For simplicity, let's assume Ga_train and Gaa_train from Section 4.3 are in scope and on device.
# If not, they would be:
# _Ga_cpu = Symmetric(rand(dtype, n_train, n_train) * rand(dtype, n_train, n_train)'); _Ga_cpu ./= tr(_Ga_cpu) / n_train
# Ga_train = current_device(_Ga_cpu)
# _Gaa_cpu = Symmetric(rand(dtype, n_train, n_train) * rand(dtype, n_train, n_train)'); _Gaa_cpu ./= tr(_Gaa_cpu) / n_train
# Gaa_train = current_device(_Gaa_cpu)
# --- End Placeholder for GRM computation ---


# Phenotypes and fixed effects matrix for training set (ensure on correct device)
y_train = current_device(convert(Vector{dtype}, base_population.phenotypes.values))
X_train = current_device(ones(dtype, n_train, 1)) # Intercept for training

# Fixed effects for validation set (usually just an intercept if no other fixed effects)
X_val = current_device(ones(dtype, n_val, 1))

# Estimated variance components from REML (ensure they are correct type)
est_variances = convert(Vector{dtype}, reml_results.var_components) # va, vaa, ve

# Package GRMs for prediction function
# Training GRMs (diagonal blocks for constructing V)
train_GRMs = [Ga_train, Gaa_train] # Additive, Epistatic
# Cross GRMs (off-diagonal blocks for Cov(val, train))
cross_GRMs = [Ga_val_train, Gaa_val_train] # Additive_val_train, Epistatic_val_train

println("Starting genomic prediction for validation set...")
predicted_bvs_val = genomic_prediction(
    y_train,
    X_train,
    X_val,
    train_GRMs,    # GRMs for training individuals [G_a_train, G_aa_train]
    cross_GRMs,    # GRMs between validation and training [G_a_val_train, G_aa_val_train]
    est_variances, # Estimated [sigma_a^2, sigma_aa^2, sigma_e^2]
    reml_results.V_inv, # Optional: V_inv from training REML results for efficiency
    reml_results.P      # Optional: P matrix from training REML for efficiency
)
# If V_inv and P are not provided, genomic_prediction will re-calculate them.

println("Predicted BVs for first 5 validation individuals:")
println(round.(Array(predicted_bvs_val[1:min(5, end)]), digits=3)) # Move to CPU with Array() for printing

# Calculate prediction accuracy if true BVs for validation set are known
true_bvs_val = validation_population.metadata[:true_breeding_values]
# Ensure true_bvs_val is on CPU and correct type for correlation
true_bvs_val_cpu = convert(Vector{dtype}, true_bvs_val)
predicted_bvs_val_cpu = Array(predicted_bvs_val) # Move predictions to CPU

accuracy = calculate_accuracy(predicted_bvs_val_cpu, true_bvs_val_cpu)
println("Prediction Accuracy (correlation): ", round(accuracy, digits=3))

# The `calculate_accuracy` function is simple:
# function calculate_accuracy(predicted_bvs::AbstractVector, true_bvs::AbstractVector)
#     return cor(predicted_bvs, true_bvs)
# end
```

**Explanation:**

1.  **Simulate/Prepare Validation Data:** A new set of individuals (`validation_population`) is created. Crucially, they share the same SNP map as the training data, allowing for the computation of cross-GRMs.
2.  **Compute Cross-GRMs:**
    *   `Ga_val_train`: Additive relationships between validation individuals and training individuals.
    *   `Gaa_val_train`: Epistatic relationships between validation and training individuals.
    *   These are computed using functions like `compute_grm_cross_population_cpu/gpu` and `compute_epistatic_grm_cross_cpu/gpu` (from `grm_computation.jl`), which take genotype matrices from both populations and the reference allele frequencies from the training (or base) population. *(Actual implementation details of these GRM functions are in Section 4.2)*.
3.  **Call `genomic_prediction`:**
    *   `y_train`, `X_train`: Phenotypes and fixed effects incidence matrix for the training set.
    *   `X_val`: Fixed effects incidence matrix for the validation set.
    *   `train_GRMs`: A list of GRMs for the training set (e.g., `[Ga_train, Gaa_train]`).
    *   `cross_GRMs`: A list of GRMs connecting validation to training individuals (e.g., `[Ga_val_train, Gaa_val_train]`). The order must correspond to `train_GRMs`.
    *   `est_variances`: The vector of estimated variance components (`[σ²_a, σ²_aa, σ²_e]`) from `reml_results`.
    *   `V_inv`, `P` (optional): Passing these matrices from the training `reml_results` can speed up calculations as they are reused.
4.  **Output:** The function returns a vector of predicted total genetic values (BLUPs) for the validation individuals.
5.  **Accuracy:** If true BVs are known (common in simulations), `calculate_accuracy` computes the Pearson correlation between predicted and true BVs.

### 4.5. Cross-Validation

Cross-validation (CV) is a standard technique to assess the predictive performance of a model and to avoid overfitting. In genomic prediction, it typically involves dividing the dataset with known phenotypes into multiple folds (subsets). The model is trained on some folds and validated on the remaining fold(s). This process is repeated until each fold has served as a validation set.

**`DynamicEpistasisGBLUP.jl` does not provide a high-level, automated cross-validation runner function out-of-the-box.** However, the tools described (`estimate_variance_components_reml!`, `genomic_prediction`, GRM computation functions) provide all the necessary building blocks to implement a custom CV scheme.

**Manual k-Fold Cross-Validation Workflow:**

1.  **Partition Data:** Divide your full dataset (individuals with genotypes and phenotypes) into `k` folds.
2.  **Loop `k` times:** In each iteration `i`:
    *   **Define Training and Validation Sets:** Use fold `i` as the validation set and the remaining `k-1` folds as the training set.
    *   **Compute GRMs:**
        *   `Ga_train`, `Gaa_train` (and other epistatic GRMs if used) for the current training set.
        *   `Ga_val_train`, `Gaa_val_train` (etc.) for relationships between the current validation set and the current training set.
        *   *Crucially, allele frequencies used for standardizing genotypes to compute these GRMs should be estimated **only from the current training set** to avoid information leakage from the validation set.*
    *   **Estimate Variance Components:** Run `estimate_variance_components_reml!` using the current training set's phenotypes (`y_train_fold_i`) and GRMs (`Ga_train`, `Gaa_train`). Get `reml_results_fold_i`.
    *   **Predict BVs:** Use `genomic_prediction` to predict BVs for the current validation set, using `y_train_fold_i`, the GRMs, and the estimated variance components from `reml_results_fold_i`.
    *   **Store Predictions:** Collect the predicted BVs for the validation fold.
3.  **Aggregate Results:** After all `k` iterations, you will have predicted BVs for all individuals (each predicted when it was in a validation fold).
4.  **Calculate Overall Accuracy:** Compute the correlation between the aggregated predicted BVs and the true phenotypes (or true BVs if known).

**Example Snippet (Conceptual for one fold of CV):**

```julia
# Assume 'full_population_data' is a PopulationData object with all individuals
# Assume 'fold_indices' is a list of lists, where each inner list contains indices for a fold.
k_folds = 5 # Example: 5-fold CV
all_indices = 1:full_population_data.genotypes.n_individuals

for i = 1:k_folds
    val_idx = fold_indices[i]
    train_idx = setdiff(all_indices, val_idx)

    # --- Subset data for current fold ---
    # genotypes_train = full_population_data.genotypes.data[train_idx, :] (and on device)
    # genotypes_val = full_population_data.genotypes.data[val_idx, :] (and on device)
    # phenotypes_train = full_population_data.phenotypes.values[train_idx] (and on device)
    # true_bvs_val_fold = full_population_data.metadata[:true_breeding_values][val_idx]

    # --- IMPORTANT: Estimate allele frequencies ONLY from training set of this fold ---
    # allele_freq_train_fold = calculate_allele_frequencies(genotypes_train) # Conceptual

    # --- Compute GRMs for this fold ---
    # Ga_train_fold = compute_additive_grm(genotypes_train, allele_freq_train_fold)
    # Gaa_train_fold = compute_epistatic_grm(genotypes_train, allele_freq_train_fold)
    # Ga_val_train_fold = compute_grm_cross(genotypes_val, genotypes_train, allele_freq_train_fold)
    # Gaa_val_train_fold = compute_epistatic_grm_cross(genotypes_val, genotypes_train, allele_freq_train_fold)
    # (Ensure all GRMs are on the correct device and float type)

    # --- Fit REML for this fold ---
    # y_train_fold_gpu = current_device(phenotypes_train)
    # X_train_fold_gpu = current_device(ones(dtype, length(train_idx), 1))
    # initial_variances_fold = ... # Potentially re-estimate or use overall initial guesses
    # active_GRMs_fold = [Ga_train_fold, Gaa_train_fold]

    # reml_results_fold = estimate_variance_components_reml!(
    #     y_train_fold_gpu, active_GRMs_fold, X_train_fold_gpu, initial_variances_fold, reml_params
    # )
    # est_variances_fold = convert(Vector{dtype}, reml_results_fold.var_components)

    # --- Predict for validation part of this fold ---
    # X_val_fold_gpu = current_device(ones(dtype, length(val_idx), 1))
    # cross_GRMs_fold = [Ga_val_train_fold, Gaa_val_train_fold]
    # predicted_bvs_val_fold = genomic_prediction(
    #     y_train_fold_gpu, X_train_fold_gpu, X_val_fold_gpu,
    #     active_GRMs_fold, cross_GRMs_fold, est_variances_fold,
    #     reml_results_fold.V_inv, reml_results_fold.P
    # )

    # Store predicted_bvs_val_fold and true_bvs_val_fold for overall accuracy calculation later
    # ...
    println("Completed CV fold $i / $k_folds")
end

# After loop, aggregate all predicted BVs and all true BVs and calculate overall correlation.
```

**Considerations for Cross-Validation:**

*   **Computational Cost:** CV can be very computationally expensive, as it involves repeated GRM computations and REML estimations.
*   **Allele Frequencies:** Strict adherence to using only training data for allele frequency estimation at each fold is crucial for unbiased accuracy assessment.
*   **Family Structure:** If strong family structures exist (e.g., dairy cattle data), random assignment to folds might lead to overly optimistic accuracies. More structured CV schemes (e.g., leave-one-family-out, or based on birth year) might be necessary.
*   **Dynamic Models:** For dynamic models across generations, CV would typically be performed within each generation, or by predicting future generations from past ones.

While `DynamicEpistasisGBLUP.jl` provides the engine, the user needs to design and implement the specific CV strategy suited to their data and research questions.

### 4.6. Multi-Generation Selection Simulation
*(This section will be detailed in a subsequent step, covering how to simulate selection across multiple generations, update allele frequencies dynamically, re-evaluate models, and track genetic gain. It will tie together simulation, GRM computation, REML, and prediction in a loop.)*

## 5. Advanced Modules and Features

This part of the manual delves into specialized modules within `DynamicEpistasisGBLUP.jl` that offer advanced functionalities or implement specific algorithms relevant to quantitative genetics and genomic prediction. Many of these modules provide tools that can be used to extend the core GBLUP framework or to perform more detailed genetic analyses.

### 5.1. Walsh-Hadamard Transform (`WalshHadamard.jl`)

The Walsh-Hadamard Transform (WHT) is a mathematical tool that can be applied in quantitative genetics, particularly for analyzing epistatic interactions. It provides a way to transform genotypic values into an orthogonal set of genetic effects (e.g., additive, dominance, and various orders of epistatic interactions). This is especially useful when working with biallelic markers and assuming a specific coding for genotypes (like -1, 0, 1 or -1, 1).

**Key Concepts:**

*   **Orthogonal Basis:** The WHT decomposes genetic values onto an orthogonal basis, where each basis vector corresponds to a specific type of genetic effect (mean, additive effect of SNP1, additive effect of SNP2, interaction SNP1×SNP2, etc.).
*   **Efficiency:** For `k` markers, the Fast Walsh-Hadamard Transform (FWHT) can compute all `2^k` possible interaction effects efficiently, typically in `O(N * 2^N * log(2^N))` time, where `N` is the number of loci involved in an interaction block. This makes it feasible for analyzing interactions among small sets of loci.
*   **Application in Epistasis Scans:** The FWHT can be used to estimate all possible interaction effects among a subset of markers. This can help identify significant epistatic interactions contributing to trait variation.

**Functionality in `WalshHadamard.jl`:**

The `WalshHadamard.jl` module in this package provides implementations of the FWHT.

*   **`fwht_natural_order_gpu!(data::CuArray{T}) where T`**: Performs an in-place Fast Walsh-Hadamard Transform on a GPU array. The "natural order" (also known as Hadamard order) means the input data should correspond to genotypic values arranged in a specific sequence.
*   **`fwht_natural_order_cpu!(data::AbstractVector{T}) where T`**: CPU version of the FWHT.
*   **`normalized_fwht_cpu/gpu!(...)`**: Versions that normalize the transform.
*   **`get_interactions_from_fwht(...)`**: A utility to extract specific interaction terms from the transformed vector.

**Genotype Coding for WHT:**
For the WHT to yield meaningful genetic effects (like additive, dominance, epistasis), the input genotypic values must be coded appropriately. A common coding for a biallelic SNP (alleles A and a) is:
*   `aa`: -1 (or 0, or 1, depending on the specific parameterization model)
*   `Aa`: 0 (or 1, or 0)
*   `AA`: 1 (or 2, or -1)

If you have `k` SNPs, you'd form `2^k` multi-locus genotype combinations. For each combination, you'd have an average phenotypic value. The WHT takes this vector of `2^k` phenotypic values and transforms it into `2^k` orthogonal components representing the overall mean, main effects of each SNP, and all possible two-way, three-way, ..., k-way interactions.

**Example Usage (Conceptual):**

Imagine you have 2 SNPs (A/a, B/b) and you have measured the average phenotype for each of the 4 (since 2^2=4 for two loci if we consider gametes, or 3^2=9 if we consider diploid genotypes) possible two-locus genotype combinations.
Let's simplify and assume we have one individual per genotypic class for 2 SNPs, coded as -1 (homozygote 1), 0 (heterozygote), 1 (homozygote 2).
The input vector for `fwht` would be the phenotypic values for these combinations, ordered correctly (e.g., lexicographically by genotype codes).

```julia
using DynamicEpistasisGBLUP, CUDA

# Assume we have phenotypic values for 2 loci, coded -1, 1 for alleles
# Genotypes: (-1,-1), (-1,1), (1,-1), (1,1) for locus1, locus2
# These correspond to specific 2-locus diploid genotypes.
# For example, if SNP1 has alleles A1, A2 and SNP2 has B1, B2:
# A1A1B1B1 -> code (-1,-1) -> phenotype_val1
# A1A1B1B2 -> not directly in this simple 2^k WHT, which usually assumes haploid or homozygous lines.
# More commonly, one might use it on deviations from mean for specific genotype classes.

# Let's use a more standard example: 3 SNPs, so 2^3 = 8 combinations.
# Suppose these are average phenotypes for 8 homozygous lines (e.g., from a MAGIC population or RILs)
# or effects derived from a regression model for each multi-locus genotype.
phenotypic_values_for_genotypes = Float32[10.2, 11.0, 9.8, 10.5, 12.1, 11.5, 10.9, 11.2] # Length 2^k

# Ensure data is on GPU if using GPU version
if CUDA.functional() && DynamicEpistasisGBLUP.USE_GPU[]
    data_gpu = CuArray(phenotypic_values_for_genotypes)
    DynamicEpistasisGBLUP.WalshHadamard.fwht_natural_order_gpu!(data_gpu)
    transformed_effects_gpu = data_gpu
    transformed_effects_cpu = Array(transformed_effects_gpu) # Move to CPU for inspection
else
    data_cpu = copy(phenotypic_values_for_genotypes) # FWHT is in-place
    DynamicEpistasisGBLUP.WalshHadamard.fwht_natural_order_cpu!(data_cpu)
    transformed_effects_cpu = data_cpu
end

println("Transformed effects (WHT coefficients):")
println(round.(transformed_effects_cpu, digits=3))

# Interpretation of transformed_effects_cpu (length 8 for 3 SNPs):
# Index 0 (after transform): Overall mean (or sum, depending on normalization)
# Index 1: Main effect of SNP 3 (if SNPs ordered 1,2,3 and FWHT processes from right)
# Index 2: Main effect of SNP 2
# Index 3: Interaction SNP2 x SNP3
# Index 4: Main effect of SNP 1
# Index 5: Interaction SNP1 x SNP3
# Index 6: Interaction SNP1 x SNP2
# Index 7: Interaction SNP1 x SNP2 x SNP3
# (The exact mapping of indices to effects depends on the specific FWHT algorithm's conventions
# and the ordering of input genotype combinations. Refer to literature like Cockerham or Walsh.)

# The module might also provide helper functions to map these indices to named effects.
# For instance, get_interactions_from_fwht(transformed_effects_cpu, num_loci=3)
# would ideally return a dictionary or named tuple.
```

**Practical Considerations:**

*   **Curse of Dimensionality:** The WHT is powerful for small `k`. For larger numbers of SNPs (e.g., genome-wide), applying FWHT to all `2^k` combinations is computationally infeasible. It's typically used for:
    *   Analyzing interactions among a pre-selected small set of candidate SNPs (e.g., significant QTLs).
    *   Theoretical derivations of genetic variance components.
*   **Genotype Data Preparation:** The main challenge is preparing the input vector of mean phenotypic values for all `2^k` (or `3^k` for diploid unphased) genotypic combinations. This often requires large sample sizes to accurately estimate these means or fitting a regression model that includes all interaction terms.
*   **Linkage Disequilibrium (LD):** Strong LD between markers can complicate the interpretation of interaction effects derived from WHT, as main effects and interaction effects can become statistically confounded.
*   **Orthogonality:** The WHT coefficients are orthogonal under linkage equilibrium and when using specific genotype codings (e.g., -1, 1 for two alleles at a locus, and assuming Hardy-Weinberg for interpretation of variance components).

**Relevance to Dynamic Orthogonal Epistasis GBLUP:**

While the main GBLUP framework in this package uses GRMs to capture epistatic variance implicitly, the WHT tools in `WalshHadamard.jl` can be valuable for:
1.  **Understanding Genetic Architecture:** For specific subsets of important markers (e.g., major QTLs identified), WHT can dissect the nature of their interactions.
2.  **Parameterizing Models:** In more complex models that aim to explicitly fit specific high-order interactions (beyond the pairwise A×A in the default epistatic GRM), WHT can provide a way to define these effects orthogonally.
3.  **Validating Orthogonal Decompositions:** The principles of WHT align with the goal of orthogonal decomposition of genetic effects. It can serve as a reference or tool for developing and verifying custom orthogonal coding schemes.

The functions in `WalshHadamard.jl` are generally low-level. Users would typically build higher-level analysis scripts around them to prepare data, run the transform, and interpret the results in a specific genetic context.

### 5.2. NOIA Framework (`NOIAFramework.jl`)

The Natural and Orthogonal Interactions (NOIA) framework, proposed by Alvarez-Castro and Carlborg (and building on earlier work by Cockerham and Kempthorne), provides a systematic way to define, estimate, and interpret genetic effects (additive, dominance, epistasis) in populations, particularly when allele frequencies are arbitrary (not necessarily 0.5) and linkage disequilibrium may exist. A key feature of NOIA is its emphasis on **orthogonality**, ensuring that estimates of different types of genetic effects (and their corresponding variance components) are uncorrelated.

**Key Concepts of NOIA:**

1.  **Reference Point:** Genetic effects are defined relative to a specific reference point, which is typically the current population's mean genotypic value. This makes the effects population-specific.
2.  **Allele Frequencies:** Genotype codings and effect definitions explicitly incorporate allele frequencies (`p` and `q` for two alleles at a locus). This is crucial for achieving orthogonality when `p ≠ q`.
3.  **Orthogonal Scaling Factors:** Genotypic values are scaled by factors derived from allele frequencies to ensure that the statistical model's design matrix for different genetic effects (e.g., additive vs. dominance at a single locus, or additive vs. additive-additive interaction between two loci) has orthogonal columns.
4.  **Decomposition of Genetic Variance:** NOIA allows for the total genetic variance to be partitioned into orthogonal components: additive variance (σ²_A), dominance variance (σ²_D), additive-by-additive epistatic variance (σ²_AA), additive-by-dominance (σ²_AD), etc.
5.  **Statistical vs. Physiological Epistasis:** NOIA primarily deals with statistical epistasis (detectable deviations from additivity in a statistical model). Orthogonal decomposition helps in clearly separating these statistical effects.

**Functionality in `NOIAFramework.jl`:**

The `NOIAFramework.jl` module aims to provide tools and functions consistent with NOIA principles. This includes:

*   **Genotype Coding Functions:**
    *   `noia_orthogonal_codes_biallelic(p::Real)`: Calculates the orthogonal genotype codes for a biallelic locus given the allele frequency `p` of one allele. For genotypes `A1A1`, `A1A2`, `A2A2`, it returns codes for the additive effect (`x_a`) and dominance deviation (`x_d`).
        *   `A1A1`: `x_a = 2q`, `x_d = -2q^2`
        *   `A1A2`: `x_a = q-p`, `x_d = 2pq`
        *   `A2A2`: `x_a = -2p`, `x_d = -2p^2`
        (where `q = 1-p`)
    *   Functions to apply these codings to a genotype matrix to create design matrices for additive and dominance effects.

*   **Interaction Term Construction:**
    *   `compute_pairwise_effects_kernel_noia_gpu!(...)` (from `gpu_kernels.jl`, but conceptually part of NOIA): A GPU kernel that could be used to construct interaction terms based on NOIA principles (e.g., product of orthogonal additive codes for two loci for A×A interaction).
    *   Host functions in `NOIAFramework.jl` would orchestrate the creation of design matrices for epistatic effects (A×A, A×D, D×D) using these orthogonal codings. For example, the A×A interaction term for locus `i` and locus `j` would be the product of their respective orthogonal additive codes: `(x_a_i) * (x_a_j)`.

*   **Variance Component Analysis (Conceptual):**
    *   While REML (`reml.jl`) estimates variance components given GRMs, if one were to fit a linear model with NOIA-coded effects directly (e.g., `y = μ + X_a*β_a + X_d*β_d + X_aa*β_aa + ... + e`), the sum of squares from such a model could be used to estimate the orthogonal variance components.
    *   The GRMs used in the GBLUP part of this package (e.g., `compute_additive_grm`, `compute_epistatic_grm`) are constructed in a way that aims to be consistent with this orthogonal partitioning, especially when using VanRaden's Method 1 for additive GRM (which centers using `2p_i`) and similar allele-frequency-aware methods for epistatic GRMs.

**Example: Orthogonal Coding for a Single Locus**

```julia
using DynamicEpistasisGBLUP

# Allele frequency of A1
p_A1 = 0.7
q_A2 = 1 - p_A1

# Get NOIA orthogonal codes for additive (alpha) and dominance (delta) effects
codes = DynamicEpistasisGBLUP.NOIAFramework.noia_orthogonal_codes_biallelic(p_A1)

println("NOIA Orthogonal Codes for p_A1 = $p_A1:")
println("  Genotype A1A1: Additive code = $(codes.A1A1_a), Dominance code = $(codes.A1A1_d)")
println("  Genotype A1A2: Additive code = $(codes.A1A2_a), Dominance code = $(codes.A1A2_d)")
println("  Genotype A2A2: Additive code = $(codes.A2A2_a), Dominance code = $(codes.A2A2_d)")

# Expected values:
# A1A1: x_a = 2*q_A2, x_d = -2*q_A2^2
# A1A2: x_a = q_A2-p_A1, x_d = 2*p_A1*q_A2
# A2A2: x_a = -2*p_A1, x_d = -2*p_A1^2

# Verify orthogonality (weighted by genotype frequencies assuming HWE)
# Frequencies: f(A1A1)=p^2, f(A1A2)=2pq, f(A2A2)=q^2
# Weighted sum of additive codes should be 0:
# p_A1^2 * (2*q_A2) + 2*p_A1*q_A2 * (q_A2-p_A1) + q_A2^2 * (-2*p_A1) ≈ 0
sum_a_weighted = p_A1^2 * codes.A1A1_a + 2*p_A1*q_A2 * codes.A1A2_a + q_A2^2 * codes.A2A2_a
println("Weighted sum of additive codes: ", round(sum_a_weighted, digits=10)) # Should be near 0

# Weighted sum of dominance codes should be 0:
sum_d_weighted = p_A1^2 * codes.A1A1_d + 2*p_A1*q_A2 * codes.A1A2_d + q_A2^2 * codes.A2A2_d
println("Weighted sum of dominance codes: ", round(sum_d_weighted, digits=10)) # Should be near 0

# Weighted sum of product of additive and dominance codes should be 0:
sum_ad_weighted = p_A1^2 * codes.A1A1_a * codes.A1A1_d +
                  2*p_A1*q_A2 * codes.A1A2_a * codes.A1A2_d +
                  q_A2^2 * codes.A2A2_a * codes.A2A2_d
println("Weighted sum of (additive * dominance) codes: ", round(sum_ad_weighted, digits=10)) # Should be near 0
```

**Relevance to `DynamicEpistasisGBLUP.jl`:**

The NOIA framework provides the theoretical underpinning for the "Dynamic Orthogonal Epistasis" concept mentioned in the project's goals and `Notes.docx`.
1.  **Dynamic Allele Frequencies:** As selection proceeds over generations, allele frequencies change. The NOIA framework dictates that genetic effects and their partitioning are dependent on these frequencies. Thus, for a dynamic model, genotype codings (implicitly in GRMs or explicitly in effect estimation models) must be updated with current generation-specific allele frequencies to maintain orthogonality.
2.  **GRM Construction:** The methods used to construct additive and epistatic GRMs in `grm_computation.jl` are designed to be consistent with NOIA principles. For example:
    *   The additive GRM `G_a` is typically constructed from a marker matrix `M` centered using `2p_i` for each marker `i`. This centering ensures that the resulting breeding values are orthogonal to mean effects, aligning with NOIA's additive definition.
    *   Epistatic GRMs (e.g., `G_aa`) are often constructed using Hadamard products of such centered (and possibly scaled) marker matrices (e.g., `(M_centered) # (M_centered)`). This implicitly uses products of NOIA-like additive codes, aiming for an orthogonal A×A component.
3.  **Interpretation of Variance Components:** When variance components (σ²_a, σ²_aa, etc.) are estimated using REML with these NOIA-consistent GRMs, they can be interpreted as orthogonal components of the total genetic variance, providing a clearer picture of the genetic architecture.
4.  **Extensibility:** If the package were to be extended to explicitly fit higher-order epistatic effects (e.g., A×D×A) or to perform detailed QTL mapping with epistasis, the functions and principles within `NOIAFramework.jl` would be directly used to construct the necessary design matrices and interpret the effects.

**Current Status and Use:**

*   The `noia_orthogonal_codes_biallelic` function provides a direct implementation of NOIA coding.
*   Other functions in `NOIAFramework.jl` might be more conceptual or serve as building blocks for future extensions.
*   The primary application of NOIA principles in the current GBLUP-focused package is through the construction of allele-frequency-dependent GRMs that lead to an orthogonal partitioning of variance. The "dynamic" aspect means that these GRMs (and the underlying allele frequencies) are updated across generations or analysis cohorts.

The `NOIAFramework.jl` module, therefore, acts as both a provider of specific utility functions and a guiding theoretical framework for the methods implemented throughout the `DynamicEpistasisGBLUP.jl` package.

### 5.3. Sparse Epistasis Detection (`SparseEpistasis.jl`)

While GBLUP models with epistatic GRMs (like `G_aa`) account for overall epistatic variance, they don't explicitly identify which specific pairs (or higher-order combinations) of SNPs are responsible for the epistatic effects. **Sparse epistasis detection** methods aim to find a relatively small number of specific interacting SNP pairs that significantly contribute to trait variation from the vast number of potential interactions (e.g., `m(m-1)/2` pairs for `m` markers).

This is important for:
*   Understanding the genetic architecture of complex traits in more detail.
*   Identifying candidate gene interactions for biological validation.
*   Potentially developing more parsimonious prediction models if only a few interactions are key.

The `SparseEpistasis.jl` module is intended to house algorithms for such detection. Due to the computational challenge (the "curse of dimensionality"), these methods often involve screening steps, regularization techniques, or specialized statistical tests.

**Key Approaches and Potential Functionality:**

1.  **Screening Methods:**
    *   **Two-Stage Approaches:**
        *   Stage 1: Screen individual markers for main effects.
        *   Stage 2: Test interactions only between markers that showed significant main effects or passed some other filtering criteria. This drastically reduces the search space but might miss purely epistatic interactions (where individual loci have no main effects).
    *   **Information Theoretic Approaches:** Methods like Multifactor Dimensionality Reduction (MDR) or those based on mutual information or entropy to identify sets of interacting loci.
    *   **Distance Correlation (dCor):** A measure of dependence between random vectors. It can be used to screen for pairs of SNPs whose joint distribution significantly deviates from independence with respect to the phenotype, potentially indicating interaction.
        *   `compute_distance_correlation_vectors(...)` and associated GPU kernels (`distance_matrix_kernel!`, `double_center_distance_matrix_kernel!`, `dcor_pairs_kernel!`) are designed for this.
        *   **Note:** The current implementation of dCor-related functions are marked as **stubs or highly experimental** and may not be fully functional or validated. Calculating dCor efficiently and correctly for all pairs is complex.

2.  **Regularized Regression Methods:**
    *   **LASSO (Least Absolute Shrinkage and Selection Operator) / Elastic Net:** These methods can be applied to a model including all main effects and all pairwise interaction terms. The regularization penalizes the number of non-zero effect estimates, leading to a sparse solution where only the most influential main effects and interactions are retained.
        *   `fit_sparse_epistasis_model(...)` (conceptual function): This would involve constructing a very wide design matrix (individuals × (markers + marker pairs)) and then applying a penalized regression solver.
        *   Specialized algorithms (e.g., for "group LASSO" where interactions are grouped) might be needed.
    *   **Bayesian Sparse Regression (e.g., BayesCπ, BayesR with interactions):** These methods use prior distributions that favor sparsity, effectively shrinking most interaction effects to zero.

3.  **Iterative Search Algorithms:**
    *   Algorithms that iteratively build up a model of interacting SNPs, adding or removing terms based on some statistical criterion.

**Functionality in `SparseEpistasis.jl` (Current and Planned):**

*   **Distance Correlation (dCor) Based Screening (Experimental/Stubbed):**
    *   `distance_matrix_kernel!(output, X, Y)`: GPU kernel to compute pairwise Euclidean distances between columns of `X` and `Y` (or `X` and `X` if `Y` is the same).
    *   `double_center_distance_matrix_kernel!(A)`: GPU kernel to double-center a distance matrix (U-centering part of dCor calculation). This is currently a **stub** and needs proper implementation.
    *   `dcor_pairs_kernel!(...)`: GPU kernel intended to compute dCor for pairs of SNPs against the phenotype. Also largely a **stub**.
    *   `compute_distance_correlation_vectors(genotypes_gpu, phenotypes_gpu, ...)`: Host function to orchestrate the dCor calculation for SNP pairs. Relies on the above kernels and is thus **experimental/incomplete**.
    *   `distance_correlation_scores_for_chunk(...)`: A helper for processing chunks of SNP pairs.

    **How dCor for epistasis works (conceptually):**
    For each pair of SNPs (SNP_i, SNP_j), create a joint genotype variable (e.g., 9 states for unphased diploid SNPs). Calculate dCor between this joint variable and the phenotype vector. High dCor suggests a non-linear relationship, potentially epistasis. This is computationally intensive if done for all pairs.

*   **Other Planned/Conceptual Functions:**
    *   `screen_interactions_by_effect_size(...)`: A simpler screening method based on fitting single-locus and two-locus models and filtering by effect size or p-value (computationally very demanding for all pairs).
    *   Wrappers or interfaces to external sparse regression packages (e.g., from `MLJ.jl` or specialized Bayesian packages) if a full design matrix of interactions can be formed.

**Example Usage (Conceptual for dCor - current implementation is stubbed):**

```julia
using DynamicEpistasisGBLUP, CUDA

# Assume:
# Z_gpu: Genotype matrix (individuals x SNPs) on GPU, coded e.g. 0,1,2
# y_gpu: Phenotype vector (individuals x 1) on GPU

# --- This is highly conceptual due to current stub status ---
# if CUDA.functional() && DynamicEpistasisGBLUP.USE_GPU[] && false # Disabled as it's a stub
#   println("Attempting dCor based screening (EXPERIMENTAL)...")
#   try
#     # The actual function would need more parameters (e.g., which pairs to test, or batching)
#     # It would return a list of SNP pairs and their dCor scores with the phenotype.
#     top_interacting_pairs_dcor = DynamicEpistasisGBLUP.SparseEpistasis.compute_distance_correlation_vectors(
#         Z_gpu,
#         y_gpu,
#         num_pairs_to_screen = 1000 # Example: screen a subset or top pairs by some other metric
#     )
#
#     if !isempty(top_interacting_pairs_dcor)
#       println("Top candidate interacting pairs (dCor):")
#       for item in top_interacting_pairs_dcor[1:min(5,end)]
#         println("  Pair: $(item.pair_indices), dCor Score: $(round(item.score, digits=4))")
#       end
#     else
#       println("No dCor results (likely due to stubbed implementation or no pairs screened).")
#     end
#
#   catch e
#     println("Error during conceptual dCor screening: ", e)
#     println("This is expected as the dCor feature is largely a stub.")
#   end
# else
#   println("Skipping dCor screening example (GPU not available or feature disabled).")
# end
println("NOTE: Distance Correlation functionalities in SparseEpistasis.jl are currently stubs/experimental.")
println("The example above is purely conceptual for future implementation.")
```

**Challenges and Considerations:**

*   **Computational Burden:** The number of possible pairwise interactions (`m*(m-1)/2`) is huge for typical SNP datasets. Genome-wide exhaustive searches are often infeasible without massive computation or very efficient screening.
*   **Multiple Testing Problem:** When testing millions of pairs, correcting for multiple testing is critical to control false positives. Standard Bonferroni correction is often too conservative. FDR (False Discovery Rate) control is more common.
*   **Statistical Power:** Detecting interactions often requires larger sample sizes than detecting main effects, as interaction effects are typically smaller or distributed across more degrees of freedom.
*   **Interpretation:** Even if statistically significant interactions are found, interpreting their biological meaning can be challenging.
*   **Implementation Status:** As noted, many functions in `SparseEpistasis.jl`, especially those related to dCor and advanced GPU kernels, are currently **stubs or placeholders**. Full, robust implementation of these methods is a significant undertaking.

**Future Directions for `SparseEpistasis.jl`:**

*   Complete and validate the dCor calculation pipeline.
*   Implement efficient screening algorithms (e.g., based on simpler linear model tests for pairs).
*   Integrate or develop efficient solvers for sparse regression models (LASSO/Elastic Net for interaction terms).
*   Provide tools for significance assessment and multiple testing correction for identified interactions.

For users interested in sparse epistasis detection with the current version of the package, it's important to be aware of the experimental nature of this module. Custom scripts leveraging more mature external tools for sparse regression might be a more practical approach until these features are fully developed within `DynamicEpistasisGBLUP.jl`.

### 5.4. GPU Optimizations (`gpu_optimization.jl`)

The `DynamicEpistasisGBLUP.jl` package heavily relies on GPU acceleration via `CUDA.jl` and `KernelAbstractions.jl` to handle the computationally intensive tasks common in genomic prediction, especially when dealing with large datasets and complex epistatic models. The `gpu_optimization.jl` module, along with specialized kernels in `gpu_kernels.jl`, is dedicated to implementing and exploring advanced GPU programming techniques to maximize performance.

**Core GPU Usage in the Package:**

*   **GRM Computation:** Calculating additive (`G_a`) and epistatic (`G_aa`, `G_aaa`, etc.) relationship matrices involves large matrix multiplications and element-wise operations on genotype data. These are prime candidates for GPU acceleration. Kernels like `grm_kernel!`, `epistatic_grm_chunk_kernel!`, `higher_order_epistatic_kernel!` are designed for this.
*   **REML Algorithm:** The AI-REML algorithm involves repeated inversions and multiplications of large covariance matrices (`V`, `P`). These linear algebra operations are significantly faster on GPUs using `CUDA.jl`'s built-in functionalities or custom kernels.
*   **Specialized Algorithms:** Modules like `WalshHadamard.jl` and parts of `SparseEpistasis.jl` (e.g., distance correlation attempts) also have GPU-accelerated versions of their core computations.

**Advanced GPU Optimization Techniques Explored (some are stubs/experimental):**

The `gpu_optimization.jl` module and related kernels aim to go beyond basic GPU porting by leveraging more advanced CUDA features:

1.  **Tensor Cores (`tensor_epistasis_kernel_placeholder!`)**
    *   **Concept:** Modern NVIDIA GPUs (Volta architecture and newer) feature Tensor Cores, specialized hardware units designed to accelerate mixed-precision matrix multiply-accumulate operations (typically FP16 multiplication with FP32 accumulation). These can provide a significant speedup for specific types of computations.
    *   **Application:** For epistatic GRM calculations that involve products of marker scores, if marker data can be appropriately represented or transformed into lower precision (e.g., FP16 or even INT8/INT4 with suitable scaling), Tensor Cores could be leveraged. This is particularly relevant for very large epistatic GRMs.
    *   **Status:** The `tensor_epistasis_kernel_placeholder!` in `gpu_kernels.jl` is a **placeholder/stub**. Actual Tensor Core programming requires using low-level CUDA libraries like WMMA (Warp Matrix Multiply-Accumulate) intrinsics or PTX assembly, which are complex to implement directly in Julia or via `KernelAbstractions.jl` without specific library support that might still be evolving.
    *   **Challenges:** Data type conversion, maintaining numerical stability with lower precision, and structuring computations to fit the specific matrix dimensions Tensor Cores operate on (e.g., 16x16x16).

2.  **Dynamic Parallelism (`parent_kernel_dynamic!`, `child_kernel_dynamic!`)**
    *   **Concept:** CUDA Dynamic Parallelism allows a kernel running on the GPU to launch other kernels on the same GPU. This can be useful for algorithms with nested parallelism, adaptive workloads, or irregular computation patterns where the amount of work for a subproblem is not known until runtime on the GPU.
    *   **Application:** In epistasis detection, if a "parent" kernel identifies a potentially interesting region or subset of interactions, it could launch "child" kernels to perform more detailed computations only for those specific regions/interactions. This could avoid unnecessary global synchronization or communication with the CPU.
    *   **Status:**
        *   `parent_kernel_dynamic!` and `child_kernel_dynamic!` in `gpu_kernels.jl` are implemented as **conceptual stubs/examples**.
        *   `adaptive_epistasis_kernel_launcher!` in `gpu_optimization.jl` is a host-side launcher for such a dynamic kernel.
        *   True CUDA dynamic parallelism from within `KernelAbstractions.jl` kernels can be tricky or might have limitations compared to native CUDA C++. The current stubs simulate the idea but may not fully leverage hardware dynamic parallelism without more direct CUDA API calls.
    *   **Challenges:** Increased complexity in kernel design, managing resource allocation for child kernels, potential for launch overhead if child kernels are too small.

3.  **Shared Memory Optimization:**
    *   **Concept:** GPUs have small, fast on-chip shared memory accessible by threads within a block. Kernels can achieve significant speedups by staging frequently accessed data into shared memory, reducing reliance on slower global memory.
    *   **Application:** Many kernels in `gpu_kernels.jl` (e.g., for GRM computation, matrix multiplication components in REML if custom-written) implicitly or explicitly use shared memory for tasks like block-wise matrix multiplication, reductions, or caching parts of genotype vectors.
    *   **Status:** This is a standard GPU optimization technique and is generally applied where beneficial. `KernelAbstractions.jl` often facilitates this through its `@localmem` macro or by how it maps loops to thread blocks.

4.  **Stream Concurrency:**
    *   **Concept:** CUDA streams allow for overlapping data transfers (CPU-GPU, GPU-CPU) with kernel execution, and executing multiple kernels concurrently if they are independent and hardware resources allow.
    *   **Application:** For large datasets, one might process data in chunks. While one chunk is being processed by a kernel, the next chunk can be transferred from CPU to GPU, and results from a previous chunk can be transferred back to CPU.
    *   **Status:** The overall package structure (e.g., processing GRMs in chunks with `epistatic_grm_chunk_kernel!`) is amenable to stream concurrency. Explicit stream management is typically handled at a higher level in the host code that launches these kernels.

**Key Functions and Variables in `gpu_optimization.jl`:**

*   `tensor_core_epistasis_config(...)`: Placeholder for configuring Tensor Core usage.
*   `adaptive_epistasis_kernel_launcher!(...)`: Host-side function to launch the conceptual dynamic parallelism kernel.
*   `DYNAMIC_INTERACTION_COUNTER`, `reset_dynamic_interaction_counter!()`: Atomics for managing counters in GPU kernels, often used in dynamic or adaptive scenarios.
*   Constants related to GPU architecture (e.g., `MAX_THREADS_PER_BLOCK`, `WARP_SIZE`) used to configure kernel launches optimally.

**Example: Conceptual Use of Dynamic Parallelism Counter**

```julia
using DynamicEpistasisGBLUP, CUDA, KernelAbstractions

# This is a simplified illustration of how a counter might be used
# The actual dynamic parallelism kernels are more complex.

if CUDA.functional() && DynamicEpistasisGBLUP.USE_GPU[]
    # Reset a GPU-side counter (if it were used by a real dynamic kernel)
    DynamicEpistasisGBLUP.reset_dynamic_interaction_counter!() # Sets counter to 0

    # Imagine a kernel that increments this counter under certain conditions
    @kernel function my_conditional_increment_kernel(counter_dev, data_dev)
        idx = @index(Global)
        if data_dev[idx] > 0.5 # Some condition
            CUDA.atomic_add!(pointer(counter_dev), Int32(1))
        end
    end

    data = rand(Float32, 100)
    data_d = CuArray(data)
    # Get the device pointer to the counter (assuming it's a CuArray of size 1)
    counter_d_ptr = DynamicEpistasisGBLUP.gpu_optimization.DYNAMIC_INTERACTION_COUNTER

    # Launch the kernel
    kernel = my_conditional_increment_kernel(DynamicEpistasisGBLUP.get_device(), 256)
    event = kernel(counter_d_ptr, data_d, ndrange=length(data_d))
    wait(event)

    # Read the counter value back (example, actual counter is on device)
    # In a real scenario, the counter's value might determine subsequent kernel launches.
    # For this example, we'd need to copy it back to see it.
    # current_count = DynamicEpistasisGBLUP.DYNAMIC_INTERACTION_COUNTER[1] # This is conceptual
    # A proper way to get the value would be:
    current_count_cpu = Array(DynamicEpistasisGBLUP.gpu_optimization.DYNAMIC_INTERACTION_COUNTER)[1]

    println("GPU counter (example): ", current_count_cpu)
    expected_count = count(x -> x > 0.5, data)
    println("Expected count based on CPU data: ", expected_count)
    # Note: The DYNAMIC_INTERACTION_COUNTER is a global-like mutable CuArray in the module,
    # which is generally okay for illustrative/experimental purposes but might need careful handling
    # in complex, multi-kernel scenarios to avoid race conditions if not used with atomics.
else
    println("Skipping GPU counter example (GPU not available or feature disabled).")
end
```

**Overall GPU Strategy:**

*   **KernelAbstractions.jl:** Abstract away some of the CUDA-specific boilerplate, allowing for more portable kernel code (though this package is heavily CUDA-focused).
*   **Data Types:** Default to `Float32` for most GPU computations to balance precision and performance (as `Float64` is significantly slower on most consumer GPUs).
*   **Memory Management:** Use `CuArray` for GPU data. Be mindful of CPU-GPU data transfers, as these can be bottlenecks. Minimize transfers by performing as much computation on the GPU as possible.
*   **Profiling:** For serious GPU optimization, using NVIDIA's Nsight Systems / Nsight Compute profilers is essential to identify bottlenecks in kernels or memory operations.

The `gpu_optimization.jl` module represents the commitment to pushing performance boundaries. However, users should be aware that some of the most advanced features (like Tensor Cores and true Dynamic Parallelism via Julia) are at the cutting edge and may be experimental or have evolving best practices for implementation.

### 5.5. Distributed Computing (`distributed_computing.jl`)

For extremely large datasets or computationally demanding models (e.g., very high-order epistasis, or genome-wide sparse epistasis scans), even a single powerful GPU might not be sufficient in terms of memory or processing power. Distributed computing offers a way to scale analyses across multiple machines (nodes) in a cluster, or even across multiple GPUs within a single powerful node, beyond what `CUDA.jl` handles for a single process.

The `distributed_computing.jl` module in `DynamicEpistasisGBLUP.jl` is intended to lay the groundwork for such distributed capabilities, primarily leveraging Julia's built-in `Distributed` module and potentially specialized packages like `DistributedArrays.jl` or MPI wrappers.

**Key Concepts for Distributed Genomic Analysis:**

1.  **Data Parallelism:**
    *   **Distributing Individuals:** Genotype and phenotype data for different individuals can be distributed across multiple worker processes. Each worker then computes partial results (e.g., a block of a GRM, or partial sums for REML statistics), which are later aggregated.
    *   **Distributing Markers:** For tasks like marker effect estimation or some screening algorithms, different sets of markers can be processed in parallel by different workers.

2.  **Model Parallelism:**
    *   For some complex models, different parts of the model itself might be handled by different workers. This is less common for GBLUP but could be relevant for very large neural networks or highly modular Bayesian models.

3.  **Communication:** Efficient communication between worker processes is crucial. This involves sending data (e.g., subsets of genotype matrices, partial sums) and synchronizing operations. Julia's `Distributed` module provides mechanisms like `@spawnat`, `fetch`, `put!`, `RemoteChannel`, and distributed arrays.

4.  **Task Distribution and Load Balancing:** Deciding how to split the work and ensuring all workers are utilized effectively.

**Potential Functionality in `distributed_computing.jl` (Largely Conceptual/Stubbed):**

The current `distributed_computing.jl` is mostly a **placeholder/stub** module, indicating future ambitions rather than implemented features. Here's what it might entail:

*   **Distributed GRM Computation (`distribute_grm_computation`):**
    *   **Strategy:** Divide the `N x M` genotype matrix `Z` by individuals (rows) or markers (columns).
        *   *Row-wise (individuals):* If `Z` is split into `k` blocks `Z_1, Z_2, ..., Z_k` (each `N/k x M`), each worker `i` can compute a diagonal block `Z_i * Z_i'` of the GRM `G`. Off-diagonal blocks `Z_i * Z_j'` would require sending `Z_j` to worker `i` (or vice-versa).
        *   *Column-wise (markers):* `G = sum_j (Z_col_j * Z_col_j') / num_markers_scaling`. Each worker can compute partial sums of these outer products for a subset of markers, which are then summed up globally. This is often more communication-efficient for GRM.
    *   **Implementation:** Would use `DistributedArrays` to represent `Z` across workers, and then parallel loops (`@distributed for`) or custom remote calls to compute partial GRMs, followed by an aggregation step (e.g., `reduce(+, ...)` for partial sums).
    *   The existing `distribute_grm_computation` function is a **stub**.

*   **Distributed REML (`distribute_reml_iterations`):**
    *   **Challenge:** REML involves dense matrix operations (inversion of `V`) which are hard to distribute efficiently without specialized distributed linear algebra libraries that can handle matrix inversions across nodes.
    *   **Potential Strategies:**
        *   If `V` is block-diagonal or has a specific sparse structure (not typical for standard GRMs), specialized distributed solvers could be used.
        *   Approximation methods like Average Information REML with Monte Carlo sampling for log-determinants or traces might be more amenable to distribution.
        *   Iterative methods to solve the MME (e.g., Preconditioned Conjugate Gradient) can be distributed, and variance components updated based on these solutions.
    *   The existing `distribute_reml_iterations` function is a **stub**.

*   **Distributed Sparse Epistasis Scans:**
    *   **Strategy:** The search space of SNP pairs can be easily divided. Each worker can screen a subset of all possible pairs for interactions (e.g., using dCor or simpler model fitting). Results (e.g., top `k` pairs per worker) are then aggregated.
    *   This is often called an "embarrassingly parallel" problem for the screening stage.

*   **Helper Functions:**
    *   `setup_distributed_env()`: To initialize Julia worker processes (e.g., `addprocs()`).
    *   Functions for scattering/gathering data to/from workers.

**Example (Conceptual - Distributing a Simple Task):**

This example doesn't use functions from `distributed_computing.jl` (as they are stubs) but shows the basic Julia `Distributed` pattern that would be foundational.

```julia
using Distributed

# Add worker processes (e.g., 4 workers on the local machine)
# In a cluster environment, this would be configured differently (e.g., using a cluster manager).
if nprocs() == 1 # Only add procs if we are the master and no workers exist yet
    addprocs(min(4, Sys.CPU_THREADS - 1)) # Add up to 4 workers, or fewer if not enough cores
end

# Ensure all workers have access to the necessary code/modules
@everywhere using DynamicEpistasisGBLUP # Or specific submodules

@everywhere function my_parallel_task(data_chunk)
    # Simulate some computation on a chunk of data
    # In reality, this could be a partial GRM calculation, a part of an epistasis scan, etc.
    println("Worker $(myid()) processing data chunk of size $(length(data_chunk))")
    # Ensure operations are compatible with where data_chunk resides (e.g. if it's a CuArray on a specific GPU)
    # For this CPU example, sum is fine.
    # If data_chunk was a CuArray, this worker would need a GPU.
    # This example assumes data_chunk is CPU data for simplicity of Distributed.
    return sum(data_chunk .* data_chunk)
end

# Create some data and split it for workers
full_data = rand(Float64, 10000)
num_workers = nworkers()
chunk_size = ceil(Int, length(full_data) / num_workers)
data_chunks = [full_data[i:min(i + chunk_size - 1, end)] for i in 1:chunk_size:length(full_data)];

# Distribute the task using pmap (parallel map)
# pmap handles sending chunks to workers and collecting results.
# For more complex scenarios, manual @spawnat and fetch might be needed.
println("Distributing tasks to $num_workers workers...")
partial_results = pmap(my_parallel_task, data_chunks)

# Aggregate results
total_sum_of_squares = sum(partial_results)
println("Total sum of squares (from distributed computation): ", total_sum_of_squares)
println("Total sum of squares (local computation for verification): ", sum(full_data .* full_data))

# removeprocs(workers()) # Clean up workers if added by script
```

**Current Status and Challenges:**

*   **Largely Unimplemented:** The `distributed_computing.jl` module is currently a **placeholder**. Implementing robust and efficient distributed versions of complex algorithms like GRM computation and REML is a significant software engineering effort.
*   **Communication Overhead:** The speedup from distributed computing can be limited by the time spent communicating data between nodes. Algorithms need to be designed to minimize this.
*   **Fault Tolerance:** In long-running distributed jobs, handling potential worker failures becomes important.
*   **GPU Management in Distributed Settings:** If each node has GPUs, managing which worker process uses which GPU, and distributing GPU data (`CuArray`s) using tools like `DistributedArrays.jl` with GPU support (e.g., `DaggerCUDA.jl` or similar patterns) adds another layer of complexity. Standard `DistributedArrays` might not directly support `CuArray` elements without custom serialization or specialized distributed GPU array types.

**Future Directions:**

*   Implement distributed GRM computation, likely starting with a column-wise (marker-based) distribution strategy.
*   Explore distributed solvers or iterative approaches for the MME part of REML.
*   Develop strategies for distributing large-scale epistasis screening tasks.
*   Integrate with frameworks like `Dagger.jl` which can simplify distributed task scheduling, including for GPUs.

For users needing to analyze datasets that exceed single-machine capabilities *now*, custom scripting using Julia's `Distributed` module for simpler forms of parallelism (like splitting individuals for independent predictions or cross-validation folds across machines) might be feasible. Full distributed model fitting is a future goal.

### 5.6. Multivariate Analysis (`multivariate_extension.jl`)

Often in breeding programs or genetic studies, multiple traits are measured on the same individuals, and these traits may be genetically correlated. Multivariate (or multi-trait) analysis considers these traits simultaneously, which can lead to:

*   **Increased Accuracy of Prediction:** If traits are genetically correlated, information from one trait can help improve the prediction accuracy for another, especially if one trait has higher heritability or is measured on more individuals.
*   **Understanding Genetic Correlations:** Estimation of genetic covariances between traits, which is crucial for understanding pleiotropy (genes affecting multiple traits) and for designing selection indexes that aim to improve multiple traits simultaneously.
*   **Selection on Correlated Traits:** Allows for optimal selection strategies when indirect selection is desired or when selecting against undesirable correlations.

The `multivariate_extension.jl` module is intended to provide the framework and functions for extending the GBLUP models in this package to handle multiple traits.

**Key Concepts in Multivariate GBLUP:**

1.  **Model Structure:**
    For `t` traits, the basic multivariate mixed model can be written as:
    `y = Xb + Zu + e`
    Where:
    *   `y = [y₁', y₂', ..., y_t']'` is a stacked vector of phenotypes for all traits.
    *   `X` is a block-diagonal incidence matrix for fixed effects `b` for each trait.
    *   `Z` is a block-diagonal incidence matrix linking individuals to their genetic effects `u`.
    *   `u = [u₁', u₂', ..., u_t']'` is a stacked vector of genetic effects (e.g., additive BVs) for all traits.
    *   The covariance structure for `u` is `Var(u) = K_g ⊗ G`, where:
        *   `G` is the genomic relationship matrix (e.g., `G_a` for additive effects).
        *   `K_g` is a `t x t` genetic covariance matrix between traits. For additive effects, `K_g = Σ_a` (additive genetic covariance matrix).
        *   `⊗` denotes the Kronecker product.
    *   The covariance structure for residuals `e` is `Var(e) = K_e ⊗ I`, where `K_e = Σ_e` is a `t x t` residual covariance matrix between traits.

2.  **Variance/Covariance Components:**
    The key parameters to estimate are the elements of the genetic covariance matrix `K_g` (e.g., `Σ_a` for additive effects, `Σ_aa` for A×A epistatic effects) and the residual covariance matrix `K_e`.
    *   `Σ_a = [ σ²_a(1)    σ_a(1,2)  ... ;`
              `  σ_a(2,1)   σ²_a(2)   ... ; ... ]`
    *   `Σ_e = [ σ²_e(1)    σ_e(1,2)  ... ;`
              `  σ_e(2,1)   σ²_e(2)   ... ; ... ]`

3.  **REML Estimation:**
    Multivariate REML (often AI-REML) is used to estimate these covariance matrices. The likelihood function becomes more complex, and the number of parameters to estimate increases significantly (`t(t+1)/2` parameters for each covariance matrix).

**Potential Functionality in `multivariate_extension.jl` (Largely Conceptual/Stubbed):**

The `multivariate_extension.jl` module is currently a **placeholder/stub**. Implementing full multivariate GBLUP is a significant extension. Here's what it would involve:

*   **Data Structures for Multivariate Data:**
    *   `MultivariatePhenotypes`: To store phenotype vectors for multiple traits, and potentially trait names.
    *   Structures to hold estimated covariance matrices (`Σ_a`, `Σ_e`, `Σ_aa`, etc.).

*   **Multivariate REML (`estimate_multitrait_variance_components_reml`):**
    *   This would be a major new function, adapting the single-trait REML logic.
    *   It would need to construct the large Kronecker product covariance matrices (`K_g ⊗ G`, `K_e ⊗ I`) or use specialized algorithms that avoid their explicit formation if possible (e.g., "iteration on data" or derivative-free methods for smaller numbers of traits).
    *   The Average Information matrix calculation becomes more complex.
    *   The function `estimate_multitrait_reml_parameters_gpu` in the provided stubs hints at this.

*   **Multivariate BLUP (`multitrait_genomic_prediction`):**
    *   Once covariance components are estimated, BLUPs for each trait are calculated simultaneously.
    *   `u_hat = (K_g ⊗ G) * Z' * V⁻¹ * (y - Xb_hat)` where `V = (K_g ⊗ G) + (K_e ⊗ I_N)` (for a simple model with one random genetic effect per trait).

*   **Epistasis in Multivariate Models:**
    *   Epistatic effects can also be modeled for multiple traits. This would involve estimating an epistatic covariance matrix between traits (e.g., `Σ_aa`).
    *   The model would include terms like `(Σ_aa ⊗ G_aa)`.

*   **Helper Functions:**
    *   To construct Kronecker products efficiently.
    *   To manage and initialize the larger number of variance/covariance parameters.
    *   To calculate genetic correlations from estimated covariance matrices: `r_g(1,2) = σ_a(1,2) / (σ_a(1) * σ_a(2))`.

**Example (Highly Conceptual - Illustrating Data and Parameter Structure):**

```julia
using DynamicEpistasisGBLUP # Assuming future multivariate capabilities

# --- Conceptual Data for 2 Traits ---
# N = number of individuals
# y1 = phenotypes for trait 1 (N x 1)
# y2 = phenotypes for trait 2 (N x 1)
# y_multitrait = [y1; y2] # Stacked vector (2N x 1)

# G_a: Additive GRM (N x N)

# --- Conceptual Covariance Matrices to be Estimated (2x2 for 2 traits) ---
# Additive Genetic Covariance Matrix (Sigma_a)
# Sa = [ var_a_trait1    cov_a_trait1_trait2 ;
#        cov_a_trait2_trait1 var_a_trait2    ]

# Residual Covariance Matrix (Sigma_e)
# Se = [ var_e_trait1    cov_e_trait1_trait2 ;
#        cov_e_trait2_trait1 var_e_trait2    ]

# --- Conceptual Multivariate REML Call ---
# multivariate_reml_results = estimate_multitrait_variance_components_reml(
#     y_multitrait,
#     [G_a], # List of GRMs (could include G_aa for epistatic model)
#     X_multitrait, # Incidence for fixed effects
#     initial_Sigma_a, # Initial guess for Sigma_a matrix
#     initial_Sigma_e, # Initial guess for Sigma_e matrix
#     # ... other parameters ...
# )

# --- Conceptual Prediction ---
# predicted_bvs_multitrait = multitrait_genomic_prediction(
#     y_multitrait_train,
#     X_multitrait_train, X_multitrait_val,
#     [G_a_train], [G_a_val_train], # Training and cross-GRMs
#     estimated_Sigma_a, estimated_Sigma_e
# )
# Result would be [u_hat_trait1_val; u_hat_trait2_val]

println("NOTE: Multivariate functionalities in multivariate_extension.jl are currently stubs.")
println("The example above is purely conceptual for future implementation.")
```

**Current Status and Challenges:**

*   **Placeholder Module:** `multivariate_extension.jl` is almost entirely a **stub** with placeholder function names like `estimate_multitrait_reml_parameters_gpu` and `construct_multivariate_grm_block_gpu`.
*   **Computational Complexity:** Multivariate models are significantly more computationally demanding than univariate models, both in terms of memory (Kronecker products can be huge) and floating-point operations.
*   **Parameter Estimation:** Convergence of REML can be more challenging with a larger number of covariance parameters. Good starting values are important.
*   **Numerical Stability:** Ensuring numerical stability in matrix inversions and likelihood calculations is critical.
*   **GPU Implementation:** Adapting kernels for multivariate operations, especially if avoiding explicit Kronecker products, requires careful design. For instance, `construct_multivariate_grm_block_gpu` hints at building blocks for `K ⊗ G` on the GPU.

**Future Directions:**

*   Develop data structures for handling multi-trait phenotypes and covariance matrices.
*   Implement a robust multivariate AI-REML algorithm, potentially starting with a two-trait model and then generalizing.
*   Optimize linear algebra, especially Kronecker products and solutions of large linear systems, for GPUs.
*   Extend prediction functions to the multivariate case.
*   Incorporate multivariate epistatic effects.

Multivariate analysis is a common and important extension in quantitative genetics. Its implementation would greatly enhance the capabilities of `DynamicEpistasisGBLUP.jl`, but it represents a substantial development effort.

### 5.7. Breeding Program Optimization (`breeding_optimization.jl`)

The ultimate goal of genomic prediction in animal and plant breeding is to make better selection and mating decisions to maximize genetic gain for desired traits, while often needing to manage genetic diversity or inbreeding. The `breeding_optimization.jl` module is envisioned as a high-level component for implementing algorithms and strategies related to optimizing breeding programs using outputs from the genomic prediction models.

This module is currently a **placeholder/stub** and represents a significant area for future development, potentially integrating concepts from operations research, optimization theory, and advanced breeding simulation.

**Key Areas in Breeding Program Optimization:**

1.  **Selection Strategies:**
    *   **Selection Index Theory:** Combining information from multiple traits (and their BLUPs/GEBVs) into a single index value for ranking and selecting individuals. This often involves weighting traits by their economic importance and considering genetic and phenotypic correlations.
    *   **Optimal Contribution Selection (OCS):** Methods that aim to maximize genetic gain while constraining the rate of inbreeding or maintaining genetic diversity. This involves finding the optimal genetic contribution (e.g., number of offspring) from each selection candidate to the next generation. This often involves solving optimization problems.
    *   **Dynamic Selection Rules:** Adapting selection criteria over generations as allele frequencies change or as new economic weights become relevant.

2.  **Mating Strategies:**
    *   **Minimizing Inbreeding:** Designing mating plans to avoid mating closely related individuals, using pedigree or genomic relationship information.
    *   **Maximizing Genetic Merit in Offspring:** Mating individuals with complementary high GEBVs.
    *   **Exploiting Non-Additive Effects:** If significant dominance or epistatic effects are identified and can be predicted, mating strategies might try to create favorable combinations of alleles in offspring (though this is complex).
    *   **Mate Allocation Algorithms:** Using optimization algorithms (e.g., linear programming, simulated annealing, genetic algorithms) to find the best set of matings given a pool of selected males and females, subject to various constraints (e.g., number of matings per individual, avoiding specific crosses).

3.  **Management of Genetic Diversity:**
    *   Monitoring inbreeding levels using pedigree (`A` matrix) or genomic (`G` matrix) information.
    *   Strategies to balance short-term genetic gain with long-term maintenance of diversity to ensure future selection potential and population viability. This often involves OCS or modifications to selection indexes.

4.  **Simulation and Evaluation of Breeding Programs:**
    *   Simulating entire breeding programs over multiple generations under different strategies to compare their long-term outcomes in terms of genetic gain, inbreeding, variance, etc. This would heavily leverage the `simulation.jl` module but add layers for selection, mating, and population advancement.

**Potential Functionality in `breeding_optimization.jl` (Conceptual/Future):**

*   **Selection Index Calculation (`calculate_selection_index`):**
    *   Input: GEBVs for multiple traits, economic weights, genetic/phenotypic covariance matrices (from multivariate analysis).
    *   Output: Index values for selection candidates.

*   **Optimal Contribution Selection Solvers (`solve_ocs`):**
    *   Input: GEBVs, relationship matrix (genomic or pedigree), constraints on inbreeding or contributions.
    *   Output: Optimal number of offspring or contribution proportions for each candidate.
    *   This would likely require integrating with Julia optimization packages like `JuMP.jl`.

*   **Mate Allocation Algorithms (`allocate_matings`):**
    *   Input: Lists of selected males and females, their GEBVs, relationship information, mating constraints.
    *   Output: A proposed mating list.
    *   Could implement various heuristic or exact optimization algorithms.
    *   The stubbed `optimize_breeding_program_gpu` hints at a high-level optimization function.

*   **Inbreeding Calculation (`calculate_inbreeding_coefficients`):**
    *   Functions to calculate inbreeding from pedigree or genomic data.

*   **Breeding Program Simulation Engine (`simulate_breeding_program`):**
    *   A high-level function to orchestrate multi-generational simulations incorporating genomic evaluation, selection, mating, and population dynamics.
    *   This would be a major component, building upon many other modules.

**Example (Highly Conceptual - Optimal Contribution Selection Idea):**

```julia
using DynamicEpistasisGBLUP, JuMP, Ipopt # JuMP for modeling, Ipopt as an example solver

# --- Conceptual Data ---
# gebvs: Vector of GEBVs for N candidates for a single trait or an index
# G_matrix: Genomic Relationship Matrix for the N candidates (N x N)
# max_total_contribution: e.g., total number of offspring for next generation
# max_inbreeding_rate: Constraint on average inbreeding of offspring

# --- Conceptual OCS Function (using JuMP) ---
# function solve_optimal_contribution(gebvs, G_matrix, max_total_contribution, target_inbreeding)
#     N = length(gebvs)
#     model = Model(Ipopt.Optimizer)
#
#     @variable(model, c[1:N] >= 0) # c_i is the contribution of candidate i
#
#     @objective(model, Max, sum(gebvs[i] * c[i] for i in 1:N)) # Maximize genetic gain
#
#     @constraint(model, sum(c[i] for i in 1:N) == 1.0) # Contributions sum to 1 (proportions)
#                                                      # Or == max_offspring if c is number of offspring
#
#     # Inbreeding constraint: c' * G_matrix * c / 2 <= target_inbreeding
#     # This is a quadratic constraint.
#     # @constraint(model, 0.5 * c' * G_matrix * c <= target_inbreeding) # JuMP syntax for quadratic
#
#     # Other constraints (e.g., max contribution per individual)
#     # @constraint(model, c[i] <= max_individual_contribution)
#
#     optimize!(model)
#     return JuMP.value.(c)
# end

# --- Call the conceptual function ---
# contributions = solve_optimal_contribution(candidate_gebvs, candidate_G, ...)
# selected_parents_and_contributions = findall(contributions .> 1e-6) # Get selected parents

println("NOTE: Breeding program optimization functionalities are currently stubs.")
println("The example above is purely conceptual for future implementation using optimization packages.")
```

**Current Status and Challenges:**

*   **Placeholder Module:** `breeding_optimization.jl` is almost entirely a **stub** with placeholder function names like `optimize_breeding_program_gpu` and `calculate_genetic_gain_gpu`.
*   **Interdisciplinary Nature:** Requires knowledge of quantitative genetics, optimization algorithms, and potentially simulation modeling.
*   **Complexity of Optimization Problems:** Many breeding optimization problems (especially mate allocation with complex objectives) are NP-hard, requiring heuristic approaches for large problem sizes.
*   **Defining Objectives:** Clearly defining the breeding goal (e.g., weighting of traits, acceptable inbreeding levels) is crucial and often specific to each breeding program.
*   **Software Integration:** May require interfacing with specialized optimization solvers or libraries.

**Future Directions:**

*   Implement basic selection index calculations.
*   Develop functions for calculating and monitoring inbreeding.
*   Integrate with `JuMP.jl` or other optimization packages to implement OCS.
*   Design and implement heuristic algorithms for mate allocation.
*   Build a flexible simulation engine for evaluating long-term breeding strategies.

Optimizing breeding programs is the practical application of the models and predictions generated by the rest of the package. Developing this module would provide end-to-end capabilities, from raw genotype/phenotype data to optimized selection decisions.

## 6. API Reference
*(This section would ideally be auto-generated using tools like Documenter.jl from the docstrings in the source code. It would list all exported modules, structs, and functions with their detailed documentation. For now, it's a placeholder. Users are encouraged to use Julia's built-in help system, e.g., `?DynamicEpistasisGBLUP.function_name`)*

## 7. Program Structure

This section provides an overview of the organization of the `DynamicEpistasisGBLUP.jl` package, detailing the purpose of key source files and directories. Understanding the structure can help users and developers navigate the codebase, contribute new features, or customize existing functionalities.

**Core Directories and Files:**

*   **`src/`**: Contains all the core source code for the package.
    *   **`DynamicEpistasisGBLUP.jl`**: The main module file. It `include`s all other source files and `export`s the public API of the package (functions, types, etc., intended for users). It also sets up global configurations like `FLOAT_TYPE` and `USE_GPU`.
    *   **`types.jl`**: Defines all custom data structures (structs) used throughout the package, such as `GenotypeMatrix`, `PopulationData`, `GeneticArchitecture`, `REMLParameters`, `REMLResults`, `VarianceComponents`, etc. Keeping types in a separate file helps with organization and managing dependencies between modules.
    *   **`simulation.jl`**: Contains functions for simulating genetic data, including defining genetic architectures, generating genotypes, calculating true genetic values, and simulating phenotypes (e.g., `simulate_population`, `initialize_genetic_architecture`).
    *   **`grm_computation.jl`**: Houses functions for computing Genomic Relationship Matrices (GRMs), both additive and epistatic (e.g., `compute_additive_grm_cpu`, `compute_epistatic_grm_gpu`, `compute_grm_cross_population_cpu`). It relies heavily on `gpu_kernels.jl` for GPU-accelerated versions.
    *   **`gpu_kernels.jl`**: Contains the actual CUDA kernels (written using `KernelAbstractions.jl`) that perform the low-level computations on the GPU for GRM calculations, matrix operations, and other specialized algorithms. This file is central to the package's GPU performance.
    *   **`epistasis_core.jl`**: Provides fundamental utilities related to epistasis, such as constructing interaction terms from marker data (e.g., `construct_interaction_terms_cpu/gpu`) and genotype encoding/decoding utilities.
    *   **`reml.jl`**: Implements the Restricted Maximum Likelihood (REML) algorithm (specifically AI-REML) for estimating variance components (e.g., `estimate_variance_components_reml!`).
    *   **`prediction.jl`**: Contains functions for genomic prediction, i.e., calculating Best Linear Unbiased Predictors (BLUPs) for new individuals given a fitted model (e.g., `genomic_prediction`, `calculate_accuracy`).
    *   **Advanced Modules (each in its own file):**
        *   `WalshHadamard.jl`: Functions for Fast Walsh-Hadamard Transforms.
        *   `NOIAFramework.jl`: Tools related to the Natural and Orthogonal Interactions framework.
        *   `SymmetricPolynomials.jl`: (Potentially) Algorithms for epistatic GRMs using symmetric polynomials (currently contains stubs like `compute_higher_order_epistatic_grm`).
        *   `AugmentedAIREML.jl`: (Potentially) Alternative or specialized REML algorithms (currently contains stubs like `augmented_aireml_grm_gpu`).
        *   `SparseEpistasis.jl`: Methods for detecting specific sparse epistatic interactions (largely experimental/stubs, e.g., dCor related functions).
        *   `gpu_optimization.jl`: Explores advanced GPU techniques like Tensor Cores and dynamic parallelism (largely experimental/stubs).
        *   `distributed_computing.jl`: Placeholders for distributing computations across multiple nodes/machines (stubs).
        *   `multivariate_extension.jl`: Placeholders for extending models to multiple traits (stubs).
        *   `breeding_optimization.jl`: Placeholders for optimizing breeding programs (stubs).
    *   **`utilities.jl`**: Contains miscellaneous helper functions used across different modules (e.g., functions for data conversion, device management (`get_device`), logging, etc.).
    *   **`visualization.jl`**: (Placeholder) Intended for functions related to plotting results, such as genetic gain, Manhattan plots for QTLs/interactions, etc. Likely to use `Plots.jl` or other Julia plotting packages.

*   **`test/`**: Contains test scripts for verifying the correctness of the package's functionalities.
    *   **`runtests.jl`**: The main test script, usually invoked by `Pkg.test("DynamicEpistasisGBLUP")`. It includes other test files.
    *   **`comprehensive_tests.jl`**: A more detailed test suite that aims to cover a wider range of functionalities, including simulations, GRM computations, REML, and prediction. This is where the `run_demo()` function is often defined and tested.
    *   **(Other specific test files)**: Ideally, each module in `src/` would have a corresponding test file in `test/` (e.g., `test_simulation.jl`, `test_grm.jl`).

*   **`Project.toml`**: Julia package manager file. Defines package metadata (name, UUID, version, authors), dependencies (other Julia packages required), and compatibility constraints.
*   **`Manifest.toml`**: Records the exact versions of all direct and indirect dependencies, ensuring reproducibility of the package environment. Automatically managed by `Pkg`.
*   **`README.md`**: The main introductory document for the package, usually displayed on the package repository's front page. Provides a general overview, installation instructions, and basic usage examples.
*   **`LICENSE`**: Contains the license under which the package is distributed (e.g., MIT License).
*   **`MANUAL.md`**: (This document) The detailed user manual.
*   **`TESTING_PLAN.md`**: Outlines the strategy for testing different components of the package, especially useful when direct execution of tests is limited by the environment.
*   **`Notes.docx` (or `Notes.md`)**: The scientific document or notes detailing the theoretical background and methodologies, particularly for the "Dynamic Orthogonal Epistasis" concept.

**Module Dependencies and Workflow:**

A typical analysis workflow involves several modules:
1.  **Data Input/Simulation:** Phenotypes and genotypes are loaded or generated using `simulation.jl` and represented by structures from `types.jl`.
2.  **GRM Computation:** `grm_computation.jl` (using kernels from `gpu_kernels.jl`) calculates additive and epistatic GRMs from genotype data.
3.  **Model Fitting:** `reml.jl` estimates variance components using the GRMs and phenotype data.
4.  **Prediction:** `prediction.jl` uses the estimated variance components and GRMs to predict breeding values for new individuals.
5.  **Advanced Analyses:** Specialized modules like `WalshHadamard.jl`, `SparseEpistasis.jl`, etc., can be used for more in-depth investigations, often taking genotype data and phenotypes as input.

The design emphasizes modularity, allowing for individual components to be developed, tested, and improved independently. The use of `KernelAbstractions.jl` aims to provide a degree of backend-agnosticism for GPU kernels, though `CUDA.jl` is the primary target.

## 8. Troubleshooting / FAQ

This section provides guidance on common issues, frequently asked questions, and tips for debugging when using `DynamicEpistasisGBLUP.jl`.

**Common Issues and Solutions:**

1.  **GPU Not Detected / CUDA Errors:**
    *   **Symptom:** Errors like `CUDA.jl functionality test failed`, `CuArray not defined`, or kernels failing with CUDA-specific error codes.
    *   **Troubleshooting:**
        *   Ensure you have an NVIDIA GPU and the correct version of the CUDA Toolkit installed system-wide. Refer to the NVIDIA CUDA documentation.
        *   Verify that your NVIDIA drivers are up-to-date and compatible with your CUDA Toolkit version.
        *   In Julia, run `using CUDA; CUDA.functional()`. If `false`, `CUDA.jl` cannot find or use your GPU. Consult the [CUDA.jl installation guide](https://cuda.juliagpu.org/stable/installation/overview/) for detailed troubleshooting (e.g., setting `CUDA_HOME` environment variable, driver issues).
        *   Check if other CUDA applications are running correctly on your system.
        *   If you have multiple GPUs, ensure `CUDA.jl` is targeting the correct one (see `CUDA.devices()`).

2.  **Slow Performance (Especially on GPU):**
    *   **Symptom:** GPU computations are slower than expected.
    *   **Troubleshooting:**
        *   **Scalar Operations:** Ensure `CUDA.allowscalar(false)` is set (usually done in `DynamicEpistasisGBLUP.jl`'s `__init__`). Accidental scalar indexing into `CuArray`s is extremely slow.
        *   **Data Transfers:** Minimize CPU-GPU data transfers. Operations like `Array(cu_array)` (GPU to CPU) or `CuArray(array)` (CPU to GPU) inside performance-critical loops are major bottlenecks. Perform as much of the computation pipeline on the GPU as possible.
        *   **Kernel Launch Configuration:** Kernel launch parameters (threads per block, grid size) can impact performance. The package attempts to use sensible defaults, but for specific hardware or problem sizes, tuning might be needed (advanced).
        *   **GPU Utilization:** Use tools like `nvidia-smi` (command line) or NVIDIA Nsight Compute/Systems to check if the GPU is actually being utilized during computation and to profile kernel performance.
        *   **Data Type:** The package typically uses `Float32` for GPU operations for performance. Using `Float64` (double precision) is significantly slower on most consumer GPUs.

3.  **REML Convergence Issues:**
    *   **Symptom:** The `estimate_variance_components_reml!` function does not converge (`reml_results.converged == false`) or takes an excessive number of iterations.
    *   **Troubleshooting:**
        *   **Initial Values:** Poor initial variance component guesses can hinder convergence. Try different starting values, perhaps based on prior knowledge or rough estimates from data variance.
        *   **Numerical Stability:** If GRMs are ill-conditioned (e.g., nearly singular due to highly related individuals or redundant markers), it can cause issues. Ensure GRMs are positive semi-definite. Adding a small diagonal "nugget" (e.g., `G + nugget*I`) to GRMs can sometimes help stabilize, though this also slightly biases estimates.
        *   **Small Variance Components:** If a true variance component is very close to zero, REML might struggle to estimate it precisely and may push it to the `min_variance_value` defined in `REMLParameters`.
        *   **Model Misspecification:** If the model is a poor fit for the data (e.g., ignoring major fixed effects), REML might have trouble.
        *   **Tolerance (`tol`):** A very strict tolerance in `REMLParameters` might be hard to achieve. Try relaxing it slightly if convergence is close but not quite met.
        *   **Data Quality:** Errors or extreme outliers in phenotype data can affect REML.

4.  **Errors Related to `KernelAbstractions.jl`:**
    *   **Symptom:** Errors originating from within `KernelAbstractions.jl` macros (`@kernel`, `@index`, etc.).
    *   **Troubleshooting:**
        *   These are often due to incorrect kernel code (e.g., type mismatches, out-of-bounds access within the kernel, unsupported operations on the GPU).
        *   Ensure all data passed to kernels is on the correct device (e.g., `CuArray`s for GPU kernels).
        *   Debugging GPU kernels can be challenging. Use `print` statements within kernels (they usually output to the Julia REPL when the kernel finishes or is synchronized) for basic debugging, or resort to NVIDIA's advanced debugging tools (Nsight VSE for Visual Studio, cuda-gdb).

5.  **Out-of-Memory Errors (GPU or CPU):**
    *   **Symptom:** `OutOfMemoryError()` from Julia, or CUDA-specific out-of-memory errors.
    *   **Troubleshooting:**
        *   **GPU Memory:** Large datasets (many individuals, many SNPs) result in large genotype matrices and GRMs. If these exceed GPU VRAM:
            *   Reduce batch sizes if processing in chunks.
            *   Consider using lower precision data types if appropriate (though the package mainly uses `Float32`).
            *   For GRM computation, ensure only necessary matrices are on the GPU at any given time.
            *   Some epistatic GRMs (especially higher-order) can be extremely large.
        *   **CPU Memory:** Even if GPU memory is sufficient, intermediate host-side operations or storing many large matrices can exhaust CPU RAM. Clear unused large variables (`var = nothing; GC.gc()`).

**Frequently Asked Questions (FAQ):**

*   **Q: Can I use this package without an NVIDIA GPU?**
    *   A: Many core computations are heavily optimized for NVIDIA GPUs via CUDA. While some functions have CPU fallbacks (e.g., `compute_additive_grm_cpu`), performance for large datasets will be significantly slower. The advanced GPU optimization features and many high-performance kernels are CUDA-specific. Full CPU-only support for all features is not the primary design goal.

*   **Q: How do I choose initial variance components for REML?**
    *   A: A common heuristic is to estimate total phenotypic variance (`var(y)`) and then partition it based on expected heritabilities (e.g., if h² ≈ 0.3, initial additive variance = `0.3 * var(y)`; if epistatic variance is expected to be 0.1, initial epistatic variance = `0.1 * var(y)`; residual variance = `(1 - 0.3 - 0.1) * var(y)`). If REML struggles, try different plausible values.

*   **Q: What genotype coding does the package expect?**
    *   A: For direct input to GRM functions, typically numerical codes like 0, 1, 2 (count of one allele) are expected. The GRM functions then internally center and scale these based on allele frequencies. For WHT or NOIA-specific functions, different codings (e.g., -1, 0, 1 or allele-frequency dependent codes) might be required as per their specific methodologies.

*   **Q: How are missing genotypes handled?**
    *   A: The `GenotypeMatrix` struct includes a `missing_mask`. GRM computation functions should ideally have strategies for handling missing values (e.g., mean imputation based on allele frequencies before centering). Check the documentation or source code of specific GRM functions for their exact handling. *(This area might need further development for robust handling across all functions).*

*   **Q: Why are many advanced modules (Sparse Epistasis, Distributed Computing, Multivariate) marked as stubs or experimental?**
    *   A: Implementing these advanced features robustly and efficiently is a very large undertaking. The stubs indicate the planned architecture and future ambitions of the package. Users should be cautious when using these experimental modules and check for warnings or notes in the documentation.

**Debugging Tips:**

*   **Start Small:** Test your analysis pipeline with a very small simulated dataset where you might know the expected outcomes or can manually verify intermediate steps.
*   **Verbose Output:** Enable `verbose=true` in `REMLParameters` or add print statements in your scripts to track progress and inspect intermediate values.
*   **Check Dimensions and Types:** Mismatched matrix dimensions or incorrect data types are common sources of errors, especially when interfacing with GPU kernels.
*   **Isolate the Problem:** If a complex workflow fails, try to run individual components separately to identify where the error occurs.
*   **Consult Julia and CUDA.jl Documentation:** For general Julia errors or CUDA-specific issues, the official documentation for these tools is invaluable.
*   **Use `try-catch` blocks:** To gracefully handle potential errors and print more informative messages.

If you encounter persistent issues that you suspect are bugs in the package, please consider reporting them on the package's issue tracker (if available), providing a minimal reproducible example.

## 9. Contributing

## 10. References
