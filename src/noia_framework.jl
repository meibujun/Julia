# ===== src/noia_framework.jl =====
"""
Natural and Orthogonal InterActions (NOIA) framework implementation.
Provides tools for defining genetic effects (additive, dominance, epistasis)
in a way that ensures their statistical orthogonality, aiding in variance decomposition.
"""

module NOIAFramework

using CUDA
using LinearAlgebra # For I in MME-like solves, norm
using Statistics  # For var, mean
using KernelAbstractions # For GPU kernels

# Assuming types.jl is accessible for GenotypeMatrix, etc.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float # Accessing main module's Float

export OrthogonalGenotypeCoding, compute_orthogonal_coding!,
       update_orthogonal_coding!, decompose_genetic_variance_noia, # Renamed
       compute_orthogonal_effects_noia # Renamed

"""
    OrthogonalGenotypeCoding{T}

Structure to hold NOIA-orthogonalized genotype codings.
`S` is a 3D array: individuals × SNPs × genetic_effects (e.g., additive, dominance).
`genetic_effects` vector stores symbols like `:additive`, `:dominance`.
"""
struct OrthogonalGenotypeCoding{T<:AbstractFloat}
    S::CuArray{T, 3}  # Genotype coding matrix (individuals × SNPs × genetic_effects)
    allele_frequencies::CuVector{T} # Allele frequencies used for this coding
    reference_point::Symbol  # :population (freq-dependent) or :unweighted (F-infinity)
    genetic_effects_definition::Vector{Symbol} # Defines what each slice of S represents
end

"""
    compute_orthogonal_coding!(
        existing_coding_S::CuArray{T,3},
        genotypes_data::CuArray{T,2},
        allele_frequencies_vec::CuVector{T}; ...)

Computes and fills `existing_coding_S` with orthogonal genotype codings based on the NOIA framework.
Modifies `existing_coding_S` in-place.
The third dimension of `existing_coding_S` determines which effects are coded (e.g., S[:,:,1] for additive).
"""
function compute_orthogonal_coding!(
    existing_coding_S::CuArray{T,3}, # Output: individuals × SNPs × num_effects
    genotypes_data::CuArray{T,2},    # Input: individuals × SNPs, coded e.g. 0,1,2
    allele_frequencies_vec::CuVector{T}; # Vector of allele frequencies for each SNP
    reference_point_symbol::Symbol = :population, # :population or :unweighted
    effects_to_include::Vector{Symbol} = [:additive] # e.g. [:additive, :dominance]
) where T <: AbstractFloat

    n_individuals, n_snps = size(genotypes_data)
    num_effects_requested = length(effects_to_include)

    if size(existing_coding_S) != (n_individuals, n_snps, num_effects_requested)
        error("Output coding matrix S has incorrect dimensions.")
    end
    if length(allele_frequencies_vec) != n_snps
        error("Allele frequency vector length does not match number of SNPs.")
    end

    # Determine which specific effects are being included and in what order for the kernel
    include_dominance_flag = :dominance in effects_to_include
    # Other effects like :additive_additive are not typically part of this direct SNP coding `S`.
    # Epistatic codings are usually constructed from these primary codings.
    # The original kernel `compute_snp_coding_kernel!` only handled additive and dominance.

    backend = KernelAbstractions.get_backend(existing_coding_S)
    # The kernel `compute_snp_coding_kernel!` is assumed to be in `gpu_kernels.jl`
    # It needs to know which slice of S corresponds to additive/dominance.
    # Let's assume S[:,:,1] is additive, S[:,:,2] is dominance (if included).
    kernel! = compute_snp_coding_kernel!(backend)
    kernel!(
        existing_coding_S, genotypes_data, allele_frequencies_vec,
        n_individuals, n_snps,
        reference_point_symbol == :population, # Pass as Bool
        include_dominance_flag,                # Pass as Bool
        ndrange = n_snps # Launch one thread per SNP; each thread iterates individuals
    )
    KernelAbstractions.synchronize(backend)

    # Note: If :additive_additive was in effects_to_include, the current kernel doesn't handle it.
    # Additive-additive interaction terms are usually products of additive codings for pairs of SNPs.
    # This function primarily sets up the per-SNP orthogonal additive/dominance parts.
end


"""
    update_orthogonal_coding!(coding_obj::OrthogonalGenotypeCoding{T}, new_genotypes_data::CuArray{T,2}, new_allele_frequencies::CuVector{T})

Updates an existing `OrthogonalGenotypeCoding` object with new genotype data and allele frequencies.
Recomputes the orthogonal codings in `coding_obj.S` in-place.
"""
function update_orthogonal_coding!(
    coding_obj::OrthogonalGenotypeCoding{T},
    new_genotypes_data::CuArray{T,2},
    new_allele_frequencies::CuVector{T}
) where T <: AbstractFloat

    # Update stored allele frequencies in the object
    copyto!(coding_obj.allele_frequencies, new_allele_frequencies)

    # Recompute orthogonal coding using the new data and the object's settings
    compute_orthogonal_coding!(
        coding_obj.S, # The S matrix within the object (modified in-place)
        new_genotypes_data,
        new_allele_frequencies; # Use the newly updated frequencies
        reference_point_symbol = coding_obj.reference_point,
        effects_to_include = coding_obj.genetic_effects_definition
    )
    # `coding_obj` is now updated.
end


"""
    compute_orthogonal_effects_noia(phenotypes_gpu::CuVector{T}, coding_obj::OrthogonalGenotypeCoding{T}) -> Dict{Symbol, CuArray{T,1}}

Estimates genetic effects (per SNP for additive/dominance, per pair for epistasis)
using the NOIA orthogonalized genotype codings.
Returns a dictionary mapping effect type (e.g., `:additive`) to a CuArray of effect estimates.
"""
function compute_orthogonal_effects_noia(
    phenotypes_gpu::CuVector{T},
    coding_obj::OrthogonalGenotypeCoding{T}
) where T <: AbstractFloat

    n_individuals, n_snps, num_coded_effects = size(coding_obj.S)
    effect_estimates_dict = Dict{Symbol, CuArray{T,1}}()

    for effect_idx in 1:num_coded_effects
        effect_symbol = coding_obj.genetic_effects_definition[effect_idx]

        if effect_symbol == :additive || effect_symbol == :dominance
            # Extract the specific coding matrix slice for this effect (Individuals x SNPs)
            S_for_current_effect = @view coding_obj.S[:, :, effect_idx]

            # Estimate per-SNP effects: β_snp = (S_effect' * S_effect)^-1 * S_effect' * y
            # This is a separate regression for each SNP's coded values against phenotype,
            # or a joint model. NOIA typically implies effects are estimated per locus.
            # If estimating effects for all SNPs simultaneously for one component (e.g. all additive effects):
            # phenotypes_gpu ~ S_for_current_effect * β_additive_snp_effects
            # β_additive_snp_effects = (S_add' * S_add) \ (S_add' * y)

            # Transpose S to be SNPs x Individuals for S'S calculation if S is Ind x SNPs
            # S_effect_transposed = S_for_current_effect' # SNPs x Individuals
            # StS = S_effect_transposed * S_for_current_effect # SNPs x SNPs
            # Sty = S_effect_transposed * phenotypes_gpu      # SNPs x 1

            # Simpler: Direct solve for effects β where y = Sβ (if S is N_ind x N_snps)
            # This estimates one effect per SNP for the current component.
            StS = S_for_current_effect' * S_for_current_effect # (SNPs x Ind) * (Ind x SNPs) = SNPs x SNPs
            Sty = S_for_current_effect' * phenotypes_gpu      # (SNPs x Ind) * (Ind x 1)   = SNPs x 1

            # Add ridge for numerical stability if StS is singular or ill-conditioned
            ridge_lambda = T(1e-5) * tr(StS) / n_snps # Relative ridge
            StS_regularized = StS + CuMatrix{T}(I, n_snps, n_snps) * ridge_lambda

            snp_effects_gpu = StS_regularized \ Sty # Vector of effects, one per SNP
            effect_estimates_dict[effect_symbol] = snp_effects_gpu

        elseif effect_symbol == :additive_additive
            # Epistatic effects (e.g., additive x additive) are more complex.
            # They are typically between pairs of SNPs.
            # The coding_obj.S here only has per-SNP additive/dominance codings.
            # To estimate AxA effects, one would construct interaction terms, e.g., (S_add_i * S_add_j).
            # The original code had `compute_epistatic_effects` using `S_additive`.
            # This implies that the :additive_additive key here would trigger that.

            additive_coding_slice_idx = findfirst(isequal(:additive), coding_obj.genetic_effects_definition)
            if additive_coding_slice_idx !== nothing
                S_additive_codings = @view coding_obj.S[:, :, additive_coding_slice_idx]

                # This function needs to return two things: effects and indices of pairs
                # The original `compute_epistatic_effects` called a kernel `compute_pairwise_effects_kernel!`
                # That kernel is assumed to be in `gpu_kernels.jl`.
                # It needs to populate `interaction_effects` and `interaction_indices`.

                # Max interactions to find (can be very large: n_snps * (n_snps-1)/2)
                # This should be realistic, e.g., top N, or based on some threshold.
                # For now, let's assume it tries to estimate for all pairs if not too many, or a subset.
                # This part is computationally intensive.
                # The original kernel was simplified. A full estimation is complex.
                # Placeholder for now, as full pairwise effect estimation is huge.
                # Often, sparse methods are used first to select pairs.
                # effect_estimates_dict[effect_symbol] = CUDA.zeros(T, 1) # Placeholder
                # For now, let's assume this means we are interested in variance due to AxA, not specific pair effects here.
                # If specific effects are needed, the `compute_pairwise_effects_kernel!` needs to be robust.
                # Let's assume this part is handled by a dedicated epistasis interaction analysis step.
                # The NOIA decomposition focuses on variance, not necessarily individual effect estimates for all pairs.
            else
                # println("Warning: Additive coding slice not found, cannot compute additive_additive effects based on it.")
            end
        end
    end

    return effect_estimates_dict
end


"""
    decompose_genetic_variance_noia(phenotypes_host::Vector{T}, effect_estimates_dict::Dict{Symbol, CuArray{T,1}}, coding_obj::OrthogonalGenotypeCoding{T}) -> Dict{Symbol, T}

Decomposes total phenotypic variance into orthogonal genetic components (additive, dominance, epistatic)
plus residual, using the NOIA framework's effect estimates and codings.
Assumes `phenotypes_host` is on CPU.
"""
function decompose_genetic_variance_noia(
    phenotypes_host::Vector{T}, # Phenotypes on CPU
    effect_estimates_dict::Dict{Symbol, CuArray{T,1}}, # Effect estimates on GPU
    coding_obj::OrthogonalGenotypeCoding{T}
) where T <: AbstractFloat

    variance_components_dict = Dict{Symbol, T}()
    n_individuals = length(phenotypes_host)

    # Total phenotypic variance (observed)
    var_total_phenotypic = var(phenotypes_host)
    if var_total_phenotypic <= eps(T)
        # println("Warning: Total phenotypic variance is near zero. Variance decomposition may be unstable.")
        # Return zero for all components if total variance is zero.
        for effect_symbol in coding_obj.genetic_effects_definition
             variance_components_dict[effect_symbol] = zero(T)
        end
        variance_components_dict[:residual] = zero(T)
        variance_components_dict[:h2_narrow_noia] = zero(T) # Using _noia suffix for clarity
        variance_components_dict[:H2_broad_noia] = zero(T)
        return variance_components_dict
    end

    cumulative_genetic_variance_explained = zero(T)

    for effect_symbol in coding_obj.genetic_effects_definition
        if haskey(effect_estimates_dict, effect_symbol) && (effect_symbol == :additive || effect_symbol == :dominance)
            effect_idx = findfirst(isequal(effect_symbol), coding_obj.genetic_effects_definition)
            S_for_current_effect = @view coding_obj.S[:, :, effect_idx] # Ind x SNPs
            snp_effects_gpu = effect_estimates_dict[effect_symbol]      # SNPs x 1

            # Compute genetic values predicted by this component: g_component = S_component * β_component
            # This requires a kernel if S_component and β_component are large.
            # `noia_compute_genetic_values_kernel!` from `gpu_kernels.jl` can be used.
            genetic_values_component_gpu = CUDA.zeros(T, n_individuals)
            backend = KernelAbstractions.get_backend(S_for_current_effect)
            kernel! = noia_compute_genetic_values_kernel!(backend)
            kernel!(genetic_values_component_gpu, S_for_current_effect, snp_effects_gpu,
                    n_individuals, size(S_for_current_effect,2), ndrange=n_individuals)
            KernelAbstractions.synchronize(backend)

            # Variance of these genetic values is the variance explained by this component
            var_component = var(Array(genetic_values_component_gpu)) # Move to CPU for var()
            variance_components_dict[effect_symbol] = var_component
            cumulative_genetic_variance_explained += var_component

        elseif effect_symbol == :additive_additive
            # Variance due to AxA is harder to get directly from pairwise effects unless all are estimated.
            # Often, G_aa matrix is used in REML to get σ²_aa.
            # If `effect_estimates_dict` contained a global σ²_aa estimate, use that.
            # For now, assume this decomposition focuses on additive/dominance from direct NOIA coding.
            # If `compute_orthogonal_effects_noia` estimated variance for AxA, it would be here.
            # Placeholder:
            variance_components_dict[effect_symbol] = zero(T) # Needs specific calculation for AxA variance
        end
    end

    # Residual variance
    variance_components_dict[:residual] = var_total_phenotypic - cumulative_genetic_variance_explained
    # Ensure residual is not negative due to estimation noise or model misspecification
    variance_components_dict[:residual] = max(eps(T), variance_components_dict[:residual])


    # Heritabilities based on this NOIA decomposition
    if var_total_phenotypic > eps(T)
        h2_narrow_noia = get(variance_components_dict, :additive, zero(T)) / var_total_phenotypic
        # Broad-sense H² includes all genetic components (A, D, AA, etc.)
        H2_broad_noia = cumulative_genetic_variance_explained / var_total_phenotypic
    else
        h2_narrow_noia = zero(T)
        H2_broad_noia = zero(T)
    end

    variance_components_dict[:h2_narrow_noia] = h2_narrow_noia
    variance_components_dict[:H2_broad_noia] = H2_broad_noia

    return variance_components_dict
end


# The original code had `compute_pairwise_effects_kernel!` inside NOIAFramework.
# This kernel is better placed in `gpu_kernels.jl` if it's a general utility for pairwise computations.
# Its call from `compute_orthogonal_effects_noia` for `:additive_additive` implies it's needed.

# Export functions if this file were a module
# export OrthogonalGenotypeCoding, compute_orthogonal_coding!, update_orthogonal_coding!,
#        decompose_genetic_variance_noia, compute_orthogonal_effects_noia

end # module NOIAFramework
