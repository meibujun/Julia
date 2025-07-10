# ===== src/epistasis_core.jl =====
"""
    DynamicEpistasisGBLUP.EpistasisCore

This module provides the core function `orthogonal_epistasis_gblup` for fitting
Genomic Best Linear Unbiased Prediction (GBLUP) models that can incorporate
additive and epistatic (additive-by-additive) genetic effects.
It emphasizes an orthogonal parameterization of effects, often aligned with the
NOIA (Natural and Orthogonal Interactions) framework principles, to ensure
independent estimation of variance components.

The main function orchestrates GRM computation, REML estimation of variance
components (by calling specialized REML routines), and prediction of genetic values.
"""

using CUDA
using LinearAlgebra # For I in MME
using Statistics
using KernelAbstractions
# Assuming types.jl (PopulationData, GenotypeMatrix, OrthogonalGBLUP, VarianceComponents)
# and grm_computation.jl (compute_grm!, compute_epistatic_grm!)
# and reml.jl (fit_reml_aic criterion, or similar for variance component updates)
# and utils.jl (update_allele_frequencies!) are included and accessible.

# Helper to get Float type
_Float() = DynamicEpistasisGBLUP.Float

"""
    orthogonal_epistasis_gblup(population::PopulationData{T}; include_epistasis::Bool=true, update_frequencies_per_generation::Bool=true, reml_max_iterations::Int=100, reml_convergence_tol::T=T(1e-6)) where T <: AbstractFloat -> Tuple{OrthogonalGBLUP{T}, Vector{T}}

Fits a GBLUP model that can include orthogonally decomposed additive and epistatic
(additive-by-additive) effects.

The function performs the following main steps:
1.  Optionally updates allele frequencies based on the provided `population` data if `update_frequencies_per_generation` is true. This is crucial for dynamic modeling across generations.
2.  Computes the additive Genomic Relationship Matrix (GRM).
3.  If `include_epistasis` is true, computes the epistatic GRM (typically additive-by-additive).
4.  Initializes variance components (additive, epistatic, residual).
5.  Calls a REML routine (e.g., AI-REML, from `reml.jl` or `augmented_aireml.jl`) to iteratively estimate variance components.
6.  Predicts total genetic values (GEBVs) based on the final model parameters and MME solutions.

# Arguments
- `population::PopulationData{T}`: Struct containing genotype and phenotype data for the current population.
- `include_epistasis::Bool = true`: If `true`, includes epistatic effects (and G_aa) in the model. Otherwise, fits an additive-only GBLUP.
- `update_frequencies_per_generation::Bool = true`: If `true`, `genotypes.allele_freq` is updated from `genotypes.data` before GRM computation. Set to `false` if using static, pre-computed allele frequencies.
- `reml_max_iterations::Int = 100`: Maximum number of iterations for the REML algorithm.
- `reml_convergence_tol::T = T(1e-6)`: Convergence tolerance for the REML algorithm (e.g., change in log-likelihood).

# Returns
- `Tuple{OrthogonalGBLUP{T}, Vector{T}}`:
    - `OrthogonalGBLUP{T}`: A struct containing the fitted model (GRMs, estimated variance components, fixed effects estimates, generation).
    - `Vector{T}`: A vector of total Genomic Estimated Breeding Values (GEBVs) for individuals in the input `population`.

# Dependencies
- Relies on `update_allele_frequencies!` (from `utils.jl`).
- Relies on `compute_grm!`, `compute_epistatic_grm!` (from `grm_computation.jl`).
- Relies on `initialize_variance_components` and `estimate_variance_components_reml!` (from `reml.jl` or `augmented_aireml.jl`).
"""
function orthogonal_epistasis_gblup(
    population::PopulationData{T};
    include_epistasis::Bool = true,
    update_frequencies_per_generation::Bool = true,
    reml_max_iterations::Int = 100,
    reml_convergence_tol::T = _Float()(1e-6) # Use package's Float type for default
) where T <: AbstractFloat

    genotypes = population.genotypes
    # Ensure phenotypes are a Vector for internal calculations
    phenotypes_vector::Vector{T} = population.phenotypes.values # Assuming .values is the Vector{T}

    if update_frequencies_per_generation
        # This implies that `genotypes.allele_freq` should reflect the current generation.
        # `update_allele_frequencies!` modifies `genotypes.allele_freq` in-place.
        update_allele_frequencies!(genotypes)
    end

    # 1. Compute Genomic Relationship Matrices (GRMs)
    # Additive GRM (G)
    # println("Computing additive GRM...") # Logging can be conditional
    G_add = compute_grm!(genotypes, method=:vanraden, use_gpu=true) # Assuming GPU usage

    # Epistatic GRM (G_aa) - only if requested
    G_epi = nothing # Initialize as Nothing
    if include_epistasis
        # println("Computing epistatic GRM...")
        G_epi = compute_epistatic_grm!(genotypes, method=:hadamard, use_gpu=true) # Assuming Hadamard method
    end

    # 2. Initialize Variance Components
    # A common starting point: use phenotypic variance to guess initial components.
    # `initialize_variance_components` is defined in `reml.jl` (as per original structure).
    # This function needs to be accessible.
    initial_var_comps = initialize_variance_components(phenotypes_vector, include_epistasis) # Pass include_epistasis hint

    # 3. Fit Mixed Model using REML to estimate variance components
    # This is a complex step, often iterative (e.g., AI-REML).
    # The original `fit_orthogonal_gblup` function contained REML logic.
    # This should now call a dedicated REML fitting function, likely from `reml.jl` or `augmented_aireml.jl`.
    # Let's assume a function `estimate_variance_components_reml!` exists.

    # println("Fitting mixed model with REML...")
    # `final_var_comps` will be the REML estimates.
    # `mme_solver_results` might contain BLUPs or other intermediate results if REML func provides them.
    final_var_comps, mme_results_cache = estimate_variance_components_reml!(
        phenotypes_vector,
        G_add, # Additive GRM
        G_epi; # Epistatic GRM (can be Nothing)
        initial_variance_components = initial_var_comps,
        max_iterations = reml_max_iterations,
        convergence_tol = reml_convergence_tol
        # Potentially pass fixed effects matrix X if any: population.phenotypes.fixed_effects
    )

    # 4. Predict Genetic Values (BLUPs) using estimated variance components
    # BLUPs = Cov(genetic_effects, phenotypes) * Var(phenotypes)^-1 * (phenotypes - X*beta_hat)
    # For GBLUP: g_hat = G * Z' * V_inv * (y - X*beta) when G is part of V.
    # Or more directly using MME solutions if `mme_results_cache` provides them.
    # The original `fit_orthogonal_gblup` solved MME and updated vars iteratively.
    # Here, we assume `estimate_variance_components_reml!` provides final BLUPs or enough to compute them.

    # If `mme_results_cache` contains `u_add_final` and `u_epi_final` (genetic effects):
    u_add_final = mme_results_cache.u_additive # Vector of additive BLUPs
    u_epi_final = include_epistasis ? mme_results_cache.u_epistatic : CUDA.zeros(T, length(phenotypes_vector)) # Vector of epistatic BLUPs or zeros

    total_gebv = Array(u_add_final) # Move to CPU for final result vector
    if include_epistasis && u_epi_final !== nothing
        total_gebv .+= Array(u_epi_final)
    end

    # Fixed effects estimates (e.g., intercept) would also come from MME solution.
    # Assuming intercept is the only fixed effect for now.
    beta_hat = mme_results_cache.beta # Fixed effects estimates

    # Construct the OrthogonalGBLUP model object to return
    # Need to ensure G_add and G_epi are on GPU if stored as CuArray in struct.
    # The struct definition has them as CuArray.
    # The `compute_grm!` and `compute_epistatic_grm!` already return CuArrays if use_gpu=true.

    # The fixed_effects field in OrthogonalGBLUP struct expects Matrix{T} or Nothing.
    # `beta_hat` is likely a vector if only intercept. Reshape or adjust struct.
    # For now, assuming beta_hat is correctly formatted or OrthogonalGBLUP handles it.
    # If beta_hat is from GPU, convert to Array.
    fixed_effects_estimates_matrix = beta_hat isa CuArray ? Array(beta_hat) : beta_hat
    if fixed_effects_estimates_matrix isa Vector # If it's a vector (e.g. intercept only)
        fixed_effects_estimates_matrix = reshape(fixed_effects_estimates_matrix, :, 1)
    end


    fitted_model = OrthogonalGBLUP(
        G_add,       # Additive GRM (CuArray)
        G_epi,       # Epistatic GRM (CuArray or Nothing)
        final_var_comps,
        fixed_effects_estimates_matrix, # Fixed effects estimates (CPU Matrix)
        population.generation # Current generation
    )

    return fitted_model, total_gebv # Return the model object and the total GEBVs
end


# The original `fit_orthogonal_gblup` was a large function that iteratively solved MME and updated variance components.
# That logic should now reside primarily in a dedicated REML module (e.g., `reml.jl` or `augmented_aireml.jl`).
# The functions `construct_coefficient_matrix_additive`, `construct_coefficient_matrix_full`,
# `solve_mme_gpu`, `solve_pcg_gpu`, `inverse_gpu` were part of that.
# These are now internal to the REML estimation process.

# For example, `estimate_variance_components_reml!` would be a new high-level function in `reml.jl`.
# Let's define a placeholder for it here to make `orthogonal_epistasis_gblup` complete,
# assuming its actual implementation is in `reml.jl`.

"""
Placeholder for the actual REML variance component estimation function.
Its real implementation would be in `reml.jl` or `augmented_aireml.jl`.
"""
function estimate_variance_components_reml!(
    phenotypes_vec::Vector{T},
    G_additive::CuArray{T,2},
    G_epistatic::Union{Nothing, CuArray{T,2}};
    fixed_effects_X::Union{Nothing, Matrix{T}} = nothing, # Optional design matrix for fixed effects
    initial_variance_components::VarianceComponents{T},
    max_iterations::Int,
    convergence_tol::T
) where T <: AbstractFloat

    # This function would implement the iterative AI-REML algorithm (or similar).
    # It would call MME solvers, update variance components, check convergence, etc.
    # For now, this is a conceptual placeholder.

    # Simulate some iterations and return plausible looking results.
    # In a real scenario, this would involve complex linear algebra.
    n_individuals = length(phenotypes_vec)
    current_var_comps = deepcopy(initial_variance_components)

    # Mock fixed effects matrix if not provided (intercept only)
    X_matrix_cpu = fixed_effects_X === nothing ? ones(T, n_individuals, 1) : fixed_effects_X
    X_matrix_gpu = CuArray(X_matrix_cpu)
    y_gpu = CuArray(phenotypes_vec)

    for iter in 1:max_iterations
        # Mock MME solution step
        # C * sol = rhs
        # sol = [beta_hat; u_add_hat; u_epi_hat (if G_epistatic)]
        # This is where `construct_coefficient_matrix...` and `solve_mme_gpu` would be used.

        # Simplified mock update of variance components
        # In reality, this involves derivatives of likelihood, AI matrix, etc.
        current_var_comps.σ²_a *= (iter % 2 == 0 ? T(1.05) : T(0.95)) # Mock oscillation
        if G_epistatic !== nothing
            current_var_comps.σ²_aa *= (iter % 2 == 0 ? T(1.03) : T(0.97))
        end
        current_var_comps.σ²_e *= (iter % 2 == 0 ? T(0.98) : T(1.02))

        # Clamp to prevent negative variances
        current_var_comps.σ²_a = max(current_var_comps.σ²_a, eps(T))
        current_var_comps.σ²_aa = max(current_var_comps.σ²_aa, eps(T))
        current_var_comps.σ²_e = max(current_var_comps.σ²_e, eps(T))

        current_var_comps.σ²_p = current_var_comps.σ²_a + current_var_comps.σ²_aa + current_var_comps.σ²_e
        if current_var_comps.σ²_p > eps(T)
            current_var_comps.h² = current_var_comps.σ²_a / current_var_comps.σ²_p
            current_var_comps.H² = (current_var_comps.σ²_a + current_var_comps.σ²_aa) / current_var_comps.σ²_p
        else
            current_var_comps.h² = zero(T)
            current_var_comps.H² = zero(T)
        end

        # Mock convergence check
        if iter > 5 && rand() < 0.3 # Randomly "converge" for placeholder
            # println("Mock REML converged at iteration $iter.")
            break
        end
    end

    # Mock MME solution cache
    # These would be the actual BLUPs from the final MME solve.
    beta_hat_final = CUDA.zeros(T, size(X_matrix_gpu, 2)) # e.g., [mean_phenotype]
    u_add_final = CUDA.randn(T, n_individuals) .* sqrt(current_var_comps.σ²_a)
    u_epi_final = G_epistatic === nothing ? CUDA.zeros(T,n_individuals) : CUDA.randn(T, n_individuals) .* sqrt(current_var_comps.σ²_aa)

    mme_cache = (
        beta = beta_hat_final,
        u_additive = u_add_final,
        u_epistatic = u_epi_final
    )

    return current_var_comps, mme_cache
end


# Export functions if this file were a module
# export orthogonal_epistasis_gblup
