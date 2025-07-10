# ===== src/multivariate_extension.jl =====
"""
Multivariate trait analysis extension for DynamicEpistasisGBLUP.
Handles multiple traits simultaneously, estimates genetic correlations,
and can incorporate epistatic effects in a multivariate context.
"""

module MultivariateAnalysis

using CUDA
using LinearAlgebra # For I, inv, kron, cov, eigen, Diagonal, Symmetric, tr
using Statistics  # For mean, var, cov if used on CPU parts
using Distributions # If needed for simulating multivariate data or priors

# Assuming types.jl (GenotypeMatrix, VarianceComponents) and core modules are accessible.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export MultivariatePhenotypes, MultivariateGBLUPResult, # Renamed MultivariateGBLUP
       fit_multivariate_epistasis_gblup, # Renamed multivariate_epistasis_gblup
       analyze_genetic_correlations # Renamed genetic_correlation_analysis

"""
    MultivariatePhenotypes{T}

Structure to hold phenotype data for multiple traits.
`traits` is an individuals × n_traits matrix.
`correlation_matrix` is the phenotypic correlation matrix between traits.
"""
struct MultivariatePhenotypes{T<:AbstractFloat}
    traits_matrix::Matrix{T}  # Individuals × Number of traits (on CPU)
    trait_names::Vector{Symbol}
    missing_mask::Union{Nothing, BitMatrix} # Optional: Individuals × Traits mask for missing phenotypes
    # Phenotypic correlation_matrix can be computed or provided
    # phenotypic_correlation_matrix::Matrix{T}
end

"""
    MultivariateGBLUPResult{T}

Stores results from a multivariate GBLUP model fitting.
Includes estimated genetic and residual covariance matrices between traits,
and derived heritabilities and genetic correlations.
`G` and `G_aa` are references to the GRMs used (can be large).
"""
struct MultivariateGBLUPResult{T<:AbstractFloat}
    # GRM_additive_ref::CuArray{T,2} # Reference to the additive GRM used
    # GRM_epistatic_ref::Union{Nothing, CuArray{T,2}} # Reference to epistatic GRM

    # Covariance matrices (n_traits x n_traits)
    genetic_covariance_additive::Matrix{T}  # Additive genetic covariance matrix
    genetic_covariance_epistatic::Union{Nothing, Matrix{T}} # Epistatic genetic covariance
    # Other genetic components (e.g., dominance) could be added here.
    residual_covariance::Matrix{T} # Residual covariance matrix

    # Derived parameters
    # heritabilities_matrix::Matrix{T} # Rows: traits, Cols: components (h²_add, h²_epi, total H²)
    # genetic_correlations_additive::Matrix{T} # Between-trait additive genetic correlations
    # genetic_correlations_epistatic::Union{Nothing, Matrix{T}} # Epistatic genetic correlations
end

"""
    fit_multivariate_epistasis_gblup(...) -> MultivariateGBLUPResult{T}

Fits a multivariate GBLUP model, potentially including epistasis.
Estimates genetic and residual (co)variance components across multiple traits.
`genotypes_obj` provides GRMs. `multi_phenotypes` provides trait data.
"""
function fit_multivariate_epistasis_gblup(
    genotypes_obj::Main.DynamicEpistasisGBLUP.GenotypeMatrix{T}, # Provides access to compute_grm! if needed, or pass G, G_aa
    multi_phenotypes::MultivariatePhenotypes{T};
    G_additive_gpu::CuArray{T,2}, # Precomputed Additive GRM
    G_epistatic_gpu::Union{Nothing, CuArray{T,2}} = nothing, # Precomputed Epistatic GRM
    include_epistasis_in_model::Bool = G_epistatic_gpu !== nothing,
    # genetic_correlation_constraint::Union{Nothing, Symbol} = nothing, # e.g. :positive_definite, :diagonal
    max_reml_iterations::Int = 100,
    reml_tolerance::T = T(1e-6),
    verbose::Bool = false
) where T <: AbstractFloat

    n_individuals = genotypes_obj.n_individuals
    n_traits = size(multi_phenotypes.traits_matrix, 2)

    if n_individuals != size(multi_phenotypes.traits_matrix, 1)
        error("Number of individuals in genotype data and phenotype data do not match.")
    end

    # Initial estimates for covariance matrices (CPU side for now)
    # Additive genetic covariance G0
    initial_G0_add = initialize_multi_covariance_matrix(multi_phenotypes, n_traits, T(0.35)) # Approx 35% of Vp

    initial_G0_epi = nothing
    if include_epistasis_in_model && G_epistatic_gpu !== nothing
        initial_G0_epi = initialize_multi_covariance_matrix(multi_phenotypes, n_traits, T(0.10)) # Approx 10% of Vp
    else
        include_epistasis_in_model = false # Ensure consistency
    end

    # Residual covariance R0
    initial_R0 = initialize_multi_covariance_matrix(multi_phenotypes, n_traits, T(0.55)) # Approx 55% of Vp


    # Perform multivariate REML estimation
    # This is a complex procedure, often involving iterating solutions to MME for (co)variance components.
    # Standard approaches use Average Information (AI) REML or Expectation-Maximization (EM) REML.
    # For now, this is a placeholder for the actual M-REML algorithm.
    if verbose println("Starting Multivariate REML estimation for $n_traits traits...") end

    # Placeholder for actual M-REML iterative estimation:
    # converged_genetic_cov_add, converged_genetic_cov_epi, converged_residual_cov =
    #    perform_multivariate_reml_iterations(...)
    # This would be a loop similar to univariate REML but with matrix-valued parameters.
    # Equations involve Kronecker products. V = G_add ⊗ Σ_A + G_epi ⊗ Σ_AA + I ⊗ Σ_E
    # Likelihood derivatives and AI matrix are more complex.

    # For stubbing, let's just use initial estimates as "converged" ones.
    # THIS IS A MAJOR STUB.
    converged_genetic_cov_add = initial_G0_add
    converged_genetic_cov_epi = initial_G0_epi
    converged_residual_cov = initial_R0

    # Simulating some iterations (conceptual)
    for iter in 1:max_reml_iterations
        # M-step: Update Σ_A, Σ_AA, Σ_E based on current BLUPs of effects (u)
        # u_hat = (K ⊗ Σ_gen) * V_inv * vec(Y - Xβ)
        # Σ_new = (U' * (K_inv ⊗ I) * U) / n_obs (simplified)
        # E-step: Estimate fixed effects (β) and random effects (u)

        # Placeholder: mock updates
        if iter < 5 # Mock some changes
            converged_genetic_cov_add .*= T(0.9 + 0.2*rand())
            if converged_genetic_cov_epi !== nothing
                converged_genetic_cov_epi .*= T(0.9 + 0.2*rand())
            end
            converged_residual_cov .*= T(0.9 + 0.2*rand())

            # Ensure PSD
            converged_genetic_cov_add = make_psd_matrix(converged_genetic_cov_add)
            if converged_genetic_cov_epi !== nothing
                converged_genetic_cov_epi = make_psd_matrix(converged_genetic_cov_epi)
            end
            converged_residual_cov = make_psd_matrix(converged_residual_cov, min_diag_val=eps(T)*10)
        end
        if iter > 10 && rand() < 0.3 break end # Mock convergence
    end
    if verbose println("Multivariate REML (stub) completed.") end


    # Create and return the result structure
    result = MultivariateGBLUPResult(
        converged_genetic_cov_add,
        converged_genetic_cov_epi,
        converged_residual_cov
        # Derived parameters like heritabilities and genetic correlations would be computed from these.
    )

    return result
end


"""
    analyze_genetic_correlations(mv_gblup_result::MultivariateGBLUPResult{T}; ...)

Computes and returns genetic correlations between traits based on estimated
genetic covariance matrices from a fitted multivariate model.
Can also compute standard errors for these correlations (more advanced).
"""
function analyze_genetic_correlations(
    mv_gblup_result::MultivariateGBLUPResult{T};
    components_to_analyze::Vector{Symbol} = [:additive], # or [:additive, :epistatic]
    # compute_standard_errors::Bool = true # SE computation is complex
    verbose::Bool = false
) where T <: AbstractFloat

    genetic_correlations_output = Dict{Symbol, Matrix{T}}()
    # standard_errors_output = Dict{Symbol, Matrix{T}}() # If computing SEs

    for component_symbol in components_to_analyze
        cov_matrix_component = nothing
        if component_symbol == :additive
            cov_matrix_component = mv_gblup_result.genetic_covariance_additive
        elseif component_symbol == :epistatic && mv_gblup_result.genetic_covariance_epistatic !== nothing
            cov_matrix_component = mv_gblup_result.genetic_covariance_epistatic
        else
            if verbose println("Skipping genetic correlation analysis for component $component_symbol (not available or requested).") end
            continue
        end

        if cov_matrix_component !== nothing
            # Convert covariance matrix to correlation matrix: Corr(i,j) = Cov(i,j) / sqrt(Var(i)*Var(j))
            corr_matrix = cov_to_cor_matrix(cov_matrix_component)
            genetic_correlations_output[component_symbol] = corr_matrix

            # Standard error computation is complex, involves AI matrix inverse from REML.
            # if compute_standard_errors
            #    se_matrix = compute_genetic_correlation_standard_errors(cov_matrix_component, mv_gblup_result, component_symbol)
            #    standard_errors_output[component_symbol] = se_matrix
            # end
        end
    end

    # return genetic_correlations_output, standard_errors_output (if SEs computed)
    return genetic_correlations_output
end


# ===== Helper / Internal Functions =====

"""
Initialize a multivariate covariance matrix (n_traits x n_traits)
based on phenotypic (co)variances and an expected proportion.
"""
function initialize_multi_covariance_matrix(
    multi_phenos::MultivariatePhenotypes{T},
    n_traits::Int,
    expected_proportion_of_Vp::T # e.g., 0.3 for additive genetic part
) where T
    # Use observed phenotypic covariance as a base
    phenotypic_cov_matrix = cov(multi_phenos.traits_matrix, dims=1, corrected=true) # Individuals in rows

    # Scale it by the expected proportion
    initial_cov_matrix = phenotypic_cov_matrix .* expected_proportion_of_Vp

    # Ensure it's positive semi-definite and symmetric
    return make_psd_matrix(initial_cov_matrix)
end

"""
Ensure a matrix is positive semi-definite (PSD) by adjusting eigenvalues.
Also ensures symmetry.
"""
function make_psd_matrix(A::Matrix{T}; min_diag_val::T = eps(T)) where T
    A_symm = (A .+ A') ./ T(2) # Ensure symmetry

    # Eigen decomposition
    eigen_decomp = eigen(A_symm)
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors

    # Adjust small or negative eigenvalues
    # Threshold can be a small positive number like eps(T) or related to matrix norm.
    adjusted_eigenvalues = max.(eigenvalues, min_diag_val) # Ensure at least min_diag_val

    # Reconstruct the matrix: V * Diagonal(adjusted_λ) * V'
    return eigenvectors * Diagonal(adjusted_eigenvalues) * eigenvectors'
end

"""
Convert a covariance matrix to a correlation matrix.
Corr(i,j) = Cov(i,j) / (std_dev(i) * std_dev(j))
"""
function cov_to_cor_matrix(cov_matrix::Matrix{T}) where T
    n_traits = size(cov_matrix, 1)
    if n_traits == 0 return zeros(T,0,0) end

    std_devs = sqrt.(diag(cov_matrix)) # Standard deviations are sqrt of variances (diagonal elements)

    # Handle traits with zero variance to avoid division by zero -> correlation is undefined or zero.
    # Replace zero std_devs with a small number or handle resulting NaNs.
    std_devs_nozero = max.(std_devs, eps(T)) # Avoid division by zero

    cor_matrix = similar(cov_matrix)
    for i in 1:n_traits
        for j in 1:n_traits
            cor_matrix[i,j] = cov_matrix[i,j] / (std_devs_nozero[i] * std_devs_nozero[j])
        end
    end

    # Clamp values to [-1, 1] due to potential floating point inaccuracies
    cor_matrix = clamp.(cor_matrix, T(-1.0), T(1.0))
    # Ensure diagonal is exactly 1.0
    for i in 1:n_traits; cor_matrix[i,i] = one(T); end

    return cor_matrix
end


# Placeholder for the actual complex multivariate REML iteration logic.
# This would involve solving large Kronecker product structured MMEs.
# function perform_multivariate_reml_iterations(...)
#   ...
# end

# Placeholder for SE of genetic correlations (advanced topic)
# function compute_genetic_correlation_standard_errors(...)
#   ...
# end

# The original `test_pleiotropic_epistasis` is also advanced, likely involving
# likelihood ratio tests or score tests on constrained vs unconstrained models.
# This is stubbed for now.

# Export functions if this file were a module
# export MultivariatePhenotypes, MultivariateGBLUPResult, fit_multivariate_epistasis_gblup, analyze_genetic_correlations

end # module MultivariateAnalysis
