# Testing Plan for DynamicEpistasisGBLUP.jl

This document outlines the testing strategy and requirements for the `DynamicEpistasisGBLUP.jl` package. Due to current environmental limitations preventing direct execution of the Julia test suite by the agent, this plan serves as a guide for future validation.

## 1. Overview of Existing Test Suite

The primary test suite is located in `test/comprehensive_tests.jl` and is executed via `test/runtests.jl`. It is structured using Julia's built-in `Test` package.

The suite currently includes `@testset` blocks for:
-   **Simulation Module:** Validates `simulate_population` and the structure of `GeneticArchitecture`, ensuring populations are generated with expected parameters and that genetic value calculations (post-refinement) are sensible.
-   **GRM Computation:** Tests `compute_grm!` and `compute_epistatic_grm!` for both GPU and CPU paths, checking for correct dimensions, symmetry, and non-negativity of diagonal elements.
-   **Core GBLUP Fitting:** Tests `orthogonal_epistasis_gblup` for additive-only and additive-plus-epistatic models (primarily on GPU due to current REML/MME structure), verifying model object creation, GEBV vector length, and plausible variance component estimates (e.g., positive variances, H² >= h²).
-   **Prediction and Cross-Validation:**
    -   Includes basic tests for `cross_validation` structure and output (DataFrame format, number of folds).
    -   `genomic_prediction` tests are currently minimal stubs due to the dependency on fully implemented and validated cross-GRM functions and a clear model state for prediction.
-   **Advanced Modules Sanity Checks:** Basic execution checks for functions within `WalshHadamard.jl`, `NOIAFramework.jl`, and `SparseEpistasis.jl` to ensure they run without immediate errors on small datasets (GPU only for most). These are not exhaustive functional tests.
-   **Integration: Simulation Pipeline:** A short run of `simulate_long_term_genetic_gain` to check basic pipeline integrity.
-   **Basic Benchmarks:** A call to `benchmark_grm_computation` for a quick performance sanity check (GPU only).

## 2. Command to Run Tests

Once a Julia environment (version 1.9+ as specified in `Project.toml`) with all package dependencies installed is available, the test suite can be run from the root directory of the package using:

```bash
julia --project=. test/runtests.jl
```

## 3. Key Areas Requiring Rigorous Future Testing

While the existing suite provides foundational checks, the following areas need more comprehensive and rigorous testing to meet "international top-level" standards:

### 3.1. REML Implementation (`reml.jl`, `augmented_aireml.jl`)
-   **Convergence Properties:** Test convergence across a wider range of genetic architectures, heritabilities (low, moderate, high), dataset sizes (N, M), and levels of epistasis.
-   **Accuracy of Variance Components:** Validate estimated variance components (σ²_a, σ²_aa, σ²_e) against datasets with known (simulated) true values. Assess bias and precision.
-   **Numerical Stability:**
    -   Test with nearly singular or ill-conditioned GRMs.
    -   Test behavior when variance components approach boundaries (e.g., zero).
    -   Verify robustness of matrix inversions (`inv(cholesky(Symmetric(M)))` with fallbacks) and log-determinant calculations.
-   **Line Search Algorithm:** Ensure the backtracking line search behaves correctly and aids convergence without excessive iterations.
-   **Log-Likelihood Values:** Check for correctness and consistency of REML log-likelihood values.

### 3.2. Prediction Accuracy (`prediction.jl`)
-   **`genomic_prediction` Function:**
    -   Requires fully functional `compute_grm_cross!` and `compute_epistatic_grm_cross!`. These cross-GRM functions themselves need unit tests for correctness (e.g., using reference allele frequencies).
    -   Test prediction accuracy (correlation between true and predicted BVs) in various cross-validation scenarios (e.g., across unrelated families, different generations).
    -   Validate against known results if possible (e.g., simple scenarios from textbooks or other validated software).
    -   Test handling of fixed effects during prediction.
-   **`cross_validation` Function:**
    -   Ensure correct partitioning of data and aggregation of results.
    -   Compare CV accuracy with theoretical expectations or results from other packages on benchmark datasets.

### 3.3. GPU Kernels (`gpu_kernels.jl`)
-   **Numerical Equivalence:** Compare outputs of GPU kernels against their CPU counterparts (where available) or reference implementations for a range of inputs.
-   **Performance Benchmarks:** Conduct detailed profiling of key kernels (GRM, MME steps, etc.) using `BenchmarkTools.jl` with varying data sizes to assess scaling and identify bottlenecks. Aim for high occupancy and memory throughput.
-   **Edge Cases:** Test with empty inputs, single individual/SNP inputs, etc.

### 3.4. Advanced Algorithm Stubs
For modules/functions currently marked as stubs or having placeholder implementations:
-   **Distance Correlation (`sparse_epistasis.jl`, `gpu_kernels.jl`):** The full O(N²) or O(N²logN) dCor calculation needs implementation and validation. The current `dcor_pairs_kernel!` and helpers are non-functional stubs.
-   **Tensor Core Kernels (`gpu_kernels.jl/tensor_epistasis_kernel_placeholder!`)**: Requires detailed WMMA/PTX implementation and validation against non-tensor core versions for numerical consistency and speedup. The assumed mathematical operation (`G_aa = M*M'` where `M_ik = W_ik^2`) needs confirmation against the intended epistatic GRM formulation for tensor cores (which should be `Σ(k<l) W[i,k]W[i,l]W[j,k]W[j,l] / N_pairs`).
-   **Dynamic Parallelism (`gpu_kernels.jl/parent_kernel_dynamic!`, `child_kernel_dynamic!`)**: The mechanism for dynamic kernel launching from within KernelAbstractions.jl kernels (or transitioning to raw CUDA C kernels for this feature) needs to be finalized and tested. Correctness of the atomic counter and output buffers under concurrency is critical.
-   **Other Sparse Epistasis Methods:** Full implementation and validation of MI, HSIC, advanced LASSO variants (Adaptive, Group), SMET, Knockoff filters.
-   **Distributed Computing:** All functions in `distributed_computing.jl` are high-level stubs and require full implementation and testing in a multi-node/multi-GPU environment.
-   **Multivariate Analysis & Breeding Optimization:** These modules are largely conceptual stubs and will require significant implementation and specialized testing.

## 4. Suggested Benchmark Datasets / Scenarios
-   Simulated datasets from `Notes.docx` (Mongolian sheep parameters) should be a primary validation target.
-   Standard public datasets for genomic prediction if available and suitable (e.g., small example datasets from QMToolBox, R packages like `rrBLUP`, or specific livestock datasets if permissible).
-   Scenarios with varying levels of heritability, QTL numbers, epistatic contributions, and population structures (e.g., family vs. unrelated individuals).

## 5. Reporting
-   Test results should clearly indicate pass/fail status.
-   For numerical comparisons, appropriate tolerances must be used and justified.
-   Performance benchmarks should report timing, memory allocation, and GPU utilization metrics.

This testing plan will be updated as development progresses and new features are implemented.
