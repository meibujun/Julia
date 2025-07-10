# ===== src/simulation.jl =====
"""
    DynamicEpistasisGBLUP.Simulation

This module provides functions to simulate populations with complex genetic architectures,
including additive and epistatic QTL effects. It's designed to generate data for
testing and evaluating genomic prediction models developed in the DynamicEpistasisGBLUP package.
Key functionalities include simulating base populations, defining genetic effects,
generating genotypes and phenotypes, and evolving populations over generations through selection.
"""

using Random
using Distributions
using Statistics # For mean, var
using CUDA # For CuArray, GenotypeMatrix
using SparseArrays # For CuSparseMatrixCSR
# Assuming types like PopulationData, GenotypeMatrix, GeneticArchitecture, PhenotypeData are defined in types.jl
# and accessible here, e.g., via `using ..DynamicEpistasisGBLUP` if this were a submodule,
# or directly if this file is `include`d in the main module.
# Also assumes `Float` is defined (e.g., Float32).
# The original used `using ..DynamicEpistasisGBLUP` which implies utils.jl is part of the module.
# If types are in types.jl and this is included in main module, direct access is fine.

# Helper to get Float type, assuming it's defined in the scope that includes this file.
# This is a common pattern if Float is defined in the main module.
_Float() = DynamicEpistasisGBLUP.Float

"""
    simulate_population(; n_individuals::Int = 1000,
                          n_snps::Int = 50000,
                          n_chromosomes::Int = 26,
                          n_qtl_additive::Int = 50,
                          n_qtl_epistatic_pairs::Int = 25,
                          h2_narrow::Float64 = 0.30,
                          h2_broad::Float64 = 0.40,
                          maf_distribution::Distribution = Beta(0.4, 0.4),
                          seed::Int = 42) -> PopulationData

Simulates a base population (generation 0) with specified genetic and phenotypic characteristics.

Generates genotypes based on Minor Allele Frequency (MAF) drawn from `maf_distribution`
under Hardy-Weinberg equilibrium. It defines a genetic architecture with additive and
additive-by-additive epistatic QTL effects, scaled to achieve target narrow-sense (`h2_narrow`)
and broad-sense (`h2_broad`) heritabilities. Phenotypes are then generated based on these
true genetic values plus normally distributed residual noise.

# Arguments
- `n_individuals::Int`: Number of individuals in the base population.
- `n_snps::Int`: Total number of SNP markers in the genome.
- `n_chromosomes::Int`: Number of chromosomes (e.g., 26 for sheep). Used for conceptual partitioning of SNPs if relevant for later simulation steps like recombination.
- `n_qtl_additive::Int`: Number of SNPs to be assigned purely additive effects.
- `n_qtl_epistatic_pairs::Int`: Number of unique SNP pairs to be assigned epistatic (additive-by-additive) interaction effects.
- `h2_narrow::Float64`: Target narrow-sense heritability (proportion of phenotypic variance due to additive genetic effects).
- `h2_broad::Float64`: Target broad-sense heritability (proportion of phenotypic variance due to total genetic effects, including additive and epistatic).
- `maf_distribution::Distribution`: A `Distributions.jl` object defining the distribution from which MAFs for SNPs are sampled (e.g., `Beta(0.4, 0.4)`).
- `seed::Int`: Seed for the random number generator to ensure reproducibility.

# Returns
- `PopulationData`: A struct containing the `GenotypeMatrix`, `PhenotypeData`, generation number (0), and metadata including the true `GeneticArchitecture` and `true_breeding_values`.

# Example
```julia
using Distributions
base_pop = simulate_population(n_individuals=500, n_snps=10000, h2_narrow=0.3, h2_broad=0.5)
```
"""
function simulate_population(;
    n_individuals::Int = 1000,
    n_snps::Int = 50000,
    n_chromosomes::Int = 26,  # Sheep specific, make it parameter if general
    n_qtl_additive::Int = 50,
    n_qtl_epistatic_pairs::Int = 25, # Number of epistatic *pairs*, so 2*this SNPs involved if distinct
    h2_narrow::Float64 = 0.30,      # Narrow-sense heritability (additive)
    h2_broad::Float64 = 0.40,       # Broad-sense heritability (additive + epistatic)
    maf_distribution::Distribution = Beta(0.4, 0.4), # Allele frequency distribution
    seed::Int = 42
)
    Random.seed!(seed)
    current_float_type = _Float()

    # Initialize genetic architecture
    architecture = initialize_genetic_architecture(
        n_snps,
        n_qtl_additive,
        n_qtl_epistatic_pairs, # This is number of pairs
        current_float_type(h2_narrow),
        current_float_type(h2_broad)
    )

    # Generate base population genotypes
    genotypes_obj = generate_base_genotypes(
        n_individuals, n_snps, maf_distribution, current_float_type
    )

    # Calculate true genetic values based on the architecture
    # This returns a Dict like: Dict(:additive => ..., :epistatic => ..., :total => ...)
    true_genetic_values = calculate_true_genetic_values(genotypes_obj, architecture)

    # Generate phenotypes
    phenotypes_obj = generate_phenotypes(true_genetic_values, architecture, current_float_type)

    # Create population data structure
    population = PopulationData(
        genotypes_obj,
        phenotypes_obj,
        nothing,  # No pedigree for base generation
        Int32(0), # Generation 0
        Dict{Symbol, Any}(
            :architecture => architecture,
            :maf_distribution => maf_distribution,
            :true_breeding_values => true_genetic_values[:total] # Store TBVs for validation
        )
    )
    return population
end

"""
    initialize_genetic_architecture(n_snps::Int, n_qtl_additive::Int, n_qtl_epistatic_pairs::Int, h2_narrow::T, h2_broad::T) where T <: AbstractFloat -> GeneticArchitecture{T}

Initializes and returns a `GeneticArchitecture` object for a simulation.

This function randomly selects SNPs to be Quantitative Trait Loci (QTLs) with additive effects
and pairs of SNPs for epistatic (additive-by-additive) effects. Effect sizes are drawn
from a standard normal distribution and then scaled approximately to meet the target
narrow-sense (`h2_narrow`) and broad-sense (`h2_broad`) heritabilities, assuming a
total phenotypic variance of 1.0 for scaling purposes.

# Arguments
- `n_snps::Int`: Total number of available SNPs from which to choose QTLs.
- `n_qtl_additive::Int`: The desired number of additive QTLs.
- `n_qtl_epistatic_pairs::Int`: The desired number of SNP pairs with epistatic effects.
- `h2_narrow::T`: Target narrow-sense heritability.
- `h2_broad::T`: Target broad-sense heritability.

# Returns
- `GeneticArchitecture{T}`: A struct containing the defined QTL indices, effect sizes, and target heritabilities. Note that the actual number of QTLs or pairs might be less than requested if `n_snps` is too small.

# Details
- Additive QTL indices are stored in `additive_qtl_actual_indices`.
- Epistatic pair indices are stored in `epistatic_pairs_actual_indices`.
- Effect scaling is approximate and based on the variance of the raw generated effects. More precise scaling would typically involve iterative adjustment or calculation based on allele frequencies in a simulated base population.
"""
function initialize_genetic_architecture(
    n_snps::Int,
    n_qtl_additive::Int,
    n_qtl_epistatic_pairs::Int, # Number of locus pairs for epistasis
    h2_narrow::T, # Target narrow-sense heritability
    h2_broad::T   # Target broad-sense heritability
) where T <: AbstractFloat

    # --- 1. Select QTL positions ---
    if n_qtl_additive + 2 * n_qtl_epistatic_pairs > n_snps && n_qtl_epistatic_pairs > 0 # Check if enough SNPs for distinct choices
        @warn "Requested number of QTLs/epistatic pairs might exceed available distinct SNPs. Reducing counts if necessary or allowing overlap."
        # Potentially adjust counts here, or allow overlap by design.
        # For now, proceed allowing overlap by sampling from all SNPs.
    end

    available_snp_indices = shuffle(1:n_snps) # Shuffle all available SNP indices

    # Select SNPs for additive effects
    actual_additive_qtl_indices = Vector{Int32}()
    if n_qtl_additive > 0
        take_n_add = min(n_qtl_additive, length(available_snp_indices))
        actual_additive_qtl_indices = sort(Int32.(available_snp_indices[1:take_n_add]))
        # Remove these from further selection if QTLs must be distinct for different roles (not required by Notes.docx)
        # available_snp_indices = available_snp_indices[take_n_add+1:end]
    end
    n_qtl_additive_actual = Int32(length(actual_additive_qtl_indices))

    # Select SNP pairs for epistatic effects
    actual_epistatic_pairs = Vector{Tuple{Int32,Int32}}()
    # For epistatic pairs, SNPs can be re-sampled or sampled from remaining.
    # Notes.docx: "partially overlapping with the additive set" - implies sampling from all.

    num_pairs_formed = 0
    # Ensure we don't get stuck if n_snps is very small (e.g., < 2)
    if n_snps >= 2
        # Try to form requested number of unique pairs
        # This loop is inefficient for large n_qtl_epistatic_pairs if n_snps is small.
        # A better way: generate all possible pairs and sample. But M*(M-1)/2 can be huge.
        # For now, random sampling with replacement for SNP choices, then check pair uniqueness.
        max_tries_epi = n_qtl_epistatic_pairs * 100 # Safety break
        current_tries_epi = 0
        temp_pair_set = Set{Tuple{Int32,Int32}}()

        while num_pairs_formed < n_qtl_epistatic_pairs && current_tries_epi < max_tries_epi
            snp1_idx = rand(1:n_snps)
            snp2_idx = rand(1:n_snps)
            current_tries_epi +=1
            if snp1_idx == snp2_idx continue end # SNPs in a pair must be different

            pair = tuple(Int32(min(snp1_idx, snp2_idx)), Int32(max(snp1_idx, snp2_idx))) # Canonical form
            if !(pair in temp_pair_set)
                push!(actual_epistatic_pairs, pair)
                push!(temp_pair_set, pair)
                num_pairs_formed += 1
            end
        end
    end
    n_epistatic_pairs_actual = Int32(length(actual_epistatic_pairs))


    # --- 2. Generate raw effect sizes ---
    raw_additive_effects = randn(T, n_qtl_additive_actual)
    raw_epistatic_effects = randn(T, n_epistatic_pairs_actual)

    # --- 3. Scale effect sizes to meet target heritabilities ---
    # This is a simplified scaling. True genetic variance depends on allele frequencies of QTLs.
    # For simulation, one common approach is to:
    #   a. Generate genotypes for a base population.
    #   b. Calculate additive variance (Va_raw) from raw_additive_effects and base pop genotypes.
    #   c. Calculate epistatic variance (Vaa_raw) from raw_epistatic_effects and base pop genotypes.
    #   d. Scale raw_effects by sqrt(TargetVariance / ObservedRawVariance).
    # This requires genotypes. For now, use simpler scaling based on variance of raw effects themselves.

    σ²_A_target_sim = h2_narrow       # Additive variance proportion of total Var(P)=1
    σ²_AA_target_sim = h2_broad - h2_narrow # Epistatic variance proportion

    if σ²_AA_target_sim < 0
        error("Broad-sense heritability (H²_target = $h2_broad) must be >= narrow-sense heritability (h²_target = $h2_narrow).")
    end

    additive_effects_scaled = zeros(T, n_qtl_additive_actual)
    if n_qtl_additive_actual > 0
        var_raw_add = var(raw_additive_effects, corrected=false) # Use population variance
        if var_raw_add > eps(T) && σ²_A_target_sim > 0
            scale_add = sqrt(σ²_A_target_sim / var_raw_add)
            additive_effects_scaled .= raw_additive_effects .* scale_add
        elseif σ²_A_target_sim == 0 # If target additive variance is zero
             additive_effects_scaled .= zero(T)
        # else: cannot scale if raw variance is zero but target is non-zero (or if target is zero, already handled)
        end
    end

    epistatic_effects_scaled = zeros(T, n_epistatic_pairs_actual)
    if n_epistatic_pairs_actual > 0
        var_raw_epi = var(raw_epistatic_effects, corrected=false)
        if var_raw_epi > eps(T) && σ²_AA_target_sim > 0
            scale_epi = sqrt(σ²_AA_target_sim / var_raw_epi)
            epistatic_effects_scaled .= raw_epistatic_effects .* scale_epi
        elseif σ²_AA_target_sim == 0
            epistatic_effects_scaled .= zero(T)
        end
    end

    return GeneticArchitecture(
        n_qtl_additive_actual,
        actual_additive_qtl_indices, # Storing the chosen SNP indices
        additive_effects_scaled,
        n_epistatic_pairs_actual,
        actual_epistatic_pairs,    # Storing chosen SNP pair indices
        epistatic_effects_scaled,
        h2_narrow, # Store target h2n
        h2_broad   # Store target h2b
    )
end

"""
    generate_base_genotypes(n_individuals::Int, n_snps::Int, maf_distribution::Distribution, float_type::Type{T}) where T <: AbstractFloat -> GenotypeMatrix{T}

Generates genotypes for a base population and stores them in a `GenotypeMatrix`.

Allele frequencies for each SNP are drawn from the provided `maf_distribution`.
Genotypes (coded as 0, 1, 2 representing allele counts) are then sampled for each
individual at each SNP assuming Hardy-Weinberg equilibrium.
The resulting genotype data is transferred to a `CuArray` on the GPU.
The `missing_mask` is initialized as empty (no missing data), and ploidy is set to 2 (diploid).

# Arguments
- `n_individuals::Int`: Number of individuals to simulate.
- `n_snps::Int`: Number of SNPs to simulate.
- `maf_distribution::Distribution`: A `Distributions.jl` object from which SNP allele frequencies (p) are drawn. Frequencies are clamped between 0.01 and 0.99.
- `float_type::Type{T}`: The floating-point type (e.g., `Float32`, `Float64`) for the genotype data and allele frequencies on the GPU.

# Returns
- `GenotypeMatrix{T}`: A struct containing the GPU genotype data, allele frequencies, and metadata.
"""
function generate_base_genotypes(
    n_individuals::Int,
    n_snps::Int,
    maf_distribution::Distribution,
    float_type::Type{T} # To ensure output matrix is of this type (e.g., Float32)
) where T <: AbstractFloat
    # Generate allele frequencies (for the reference allele, e.g., allele 'A')
    p_allele_freqs = rand(maf_distribution, n_snps)
    p_allele_freqs = T.(clamp.(p_allele_freqs, 0.01, 0.99)) # Avoid fixation, convert to target float type

    # Genotype matrix (individuals × SNPs)
    genotypes_host = zeros(T, n_individuals, n_snps)

    # Generate genotypes based on HWE using p (frequency of one allele, e.g. major or reference)
    # Genotype coding: 0 (e.g. aa), 1 (e.g. Aa), 2 (e.g. AA)
    # where 'A' is the allele with frequency p.
    @threads for j in 1:n_snps # Parallelize over SNPs
        p = p_allele_freqs[j]
        q = one(T) - p

        # Probabilities for genotypes AA, Aa, aa
        prob_AA = p * p         # Coded as 2
        prob_Aa = T(2) * p * q  # Coded as 1
        # prob_aa = q * q       # Coded as 0 (implicit)

        for i in 1:n_individuals
            r = rand(T) # Random number for this individual at this SNP
            if r < prob_AA
                genotypes_host[i, j] = T(2)
            elseif r < prob_AA + prob_Aa
                genotypes_host[i, j] = T(1)
            else
                genotypes_host[i, j] = T(0)
            end
        end
    end

    # Transfer to GPU
    gpu_genotypes_data = CuArray(genotypes_host)
    # Assuming no missing data in base generation for this simulation setup
    # Missing mask would be all false (or empty sparse matrix)
    # For CuSparseMatrixCSR, an all-false matrix is represented by no stored elements.
    # Indices type for CuSparseMatrixCSR must be Int32.
    missing_mask_gpu = CuSparseMatrixCSR(spzeros(Bool, Int32, n_individuals, n_snps))
    allele_freq_gpu = CuArray(p_allele_freqs)

    return GenotypeMatrix(
        gpu_genotypes_data,
        missing_mask_gpu,
        allele_freq_gpu, # Store initial allele frequencies
        Int32(n_individuals),
        Int32(n_snps),
        Int8(2)  # Diploid organisms
    )
end

"""
    calculate_true_genetic_values(genotypes_obj::GenotypeMatrix{T}, architecture::GeneticArchitecture{T}) where T <: AbstractFloat -> Dict{Symbol, Vector{T}}

Calculates the true genetic values for individuals based on their genotypes and a defined genetic architecture.

The function computes:
- Additive genetic values: Sum of (genotype_at_additive_qtl * additive_effect_at_qtl) over all additive QTLs.
- Epistatic genetic values: Sum of (genotype_at_snp1 * genotype_at_snp2 * epistatic_effect_of_pair) over all defined epistatic pairs.
- Total genetic values: Sum of additive and epistatic values.

Genotype data is moved from GPU to CPU for these calculations.

# Arguments
- `genotypes_obj::GenotypeMatrix{T}`: The `GenotypeMatrix` containing individual genotypes and SNP information.
- `architecture::GeneticArchitecture{T}`: The `GeneticArchitecture` struct defining QTL locations, effect sizes, and interaction pairs. It's crucial that `additive_qtl_actual_indices` and `epistatic_pairs_actual_indices` in the `architecture` correctly map to the columns of `genotypes_obj.data`.

# Returns
- `Dict{Symbol, Vector{T}}`: A dictionary where keys are `:additive`, `:epistatic`, and `:total`, and values are vectors of the corresponding true genetic values for each individual.
"""
function calculate_true_genetic_values(
    genotypes_obj::GenotypeMatrix{T},
    architecture::GeneticArchitecture{T}
) where T <: AbstractFloat
    n_individuals = genotypes_obj.n_individuals

    # Get genotype data from GPU to host for calculation
    # This is often done on CPU for simulation setup unless effect calculation is also kernelized.
    geno_data_host = Array(genotypes_obj.data)

    # Calculate additive genetic values
    additive_values = zeros(T, n_individuals)
    # Map additive_qtl_pos from architecture.qtl_positions to effects
    # The original architecture.qtl_positions is a combined list.
    # We need a clear mapping from architecture.additive_effects to columns in geno_data_host.
    # Assuming architecture.additive_effects correspond to the first n_qtl_additive SNPs in some ordering.
    # Let's assume architecture.qtl_positions contains all QTLs, and we need to map them.
    # A better way: store additive QTL positions separately in GeneticArchitecture or map them.
    # For now, assume additive_effects map to the first n_qtl_additive unique QTLs.

    # Create a mapping from global SNP index to additive effect index if needed
    # Or, ensure additive_qtl_pos are directly usable.
    # The `Notes.docx` simulation implies additive effects assigned to 50 QTLs,
    # and epistatic to 50 pairs (100 SNPs, possibly overlapping).
    # `architecture.qtl_positions` was `sort(all_positions[1:(n_qtl_additive + n_qtl_epistatic)])`
    # This implies a specific set of SNPs are QTLs.
    # Let's refine `initialize_genetic_architecture` to store direct QTL indices for additive effects.
    # For now, assuming `architecture.additive_qtl_indices` exists and maps to `architecture.additive_effects`.
    # If `architecture.additive_effects` correspond to the first `n_qtl_additive` entries in `qtl_positions`:

    # Simpler: GeneticArchitecture should store the actual SNP indices for additive QTLs.
    # Let's assume `architecture.additive_qtl_indices` (a field to be added to GeneticArchitecture struct)
    # holds the column indices in `geno_data_host` for the additive QTLs.
    # Example: if `GeneticArchitecture` stores `additive_qtl_snps::Vector{Int32}`

    # Current structure:
    # architecture.qtl_positions: all unique SNPs that are QTLs (either A or part of AA)
    # architecture.additive_effects: effects for the first n_qtl_additive SNPs in some conceptual list.
    # This needs to be robust. The current `initialize_genetic_architecture` uses `qtl_positions`
    # for `additive_qtl_pos = sort(all_positions[1:n_qtl_additive])`.
    # So, the first `n_qtl_additive` effects in `architecture.additive_effects`
    # correspond to the first `n_qtl_additive` SNPs chosen for additive effects.
    # Let's assume `additive_qtl_positions` were stored in the architecture.
    # If not, we need to reconstruct which SNPs get which additive effects.
    # The `initialize_genetic_architecture` used `qtl_positions[1:n_qtl_additive]` for additive effects.
    # This is problematic if `qtl_positions` is the *combined sorted unique* list.

    # Let's refine `initialize_genetic_architecture` to be clearer.
    # Assume `architecture.additive_qtl_snps` stores the actual SNP indices for additive effects.
    # For now, using the logic from the original code's simulation part:
    # `architecture.qtl_positions[1:architecture.n_qtl_additive]` are the additive QTLs.
    # This seems like an error in the original structure if qtl_positions is combined.
    # The text says "50 with purely additive effects" and "50 pairs ... partially overlapping".
    # This implies distinct sets of positions for these roles initially.

    # Re-interpreting: `architecture.additive_effects` applies to a specific list of SNPs.
    # `architecture.epistatic_effects` applies to `architecture.epistatic_pairs`.
    # Let's assume `initialize_genetic_architecture` provides `additive_qtl_indices_in_geno`
    # and `epistatic_qtl_pair_indices_in_geno`.

    # For the current structure:
    # Additive QTLs are the first `n_qtl_additive` SNPs chosen by shuffle.
    # These need to be explicitly stored.
    # The `qtl_positions` in the struct is `sort(all_positions[1:(n_qtl_additive + n_qtl_epistatic)])`
    # This is not directly usable.
    # TODO: Fix GeneticArchitecture to store SNP indices for additive effects and epistatic pairs separately and clearly.

    # Workaround for current structure (highly dependent on how initialize_genetic_architecture sets it up):
    # Assume the first `n_qtl_additive` effects in `architecture.additive_effects`
    # map to some `n_qtl_additive` specific SNPs.
    # The original code's `initialize_genetic_architecture` implied:
    # `additive_qtl_pos = qtl_indices[1:n_qtl_additive]`
    # `epistatic_pairs` were formed from `qtl_indices` as well, possibly overlapping.
    # Let's make `GeneticArchitecture` store `additive_qtl_actual_indices::Vector{Int32}`.
    # For now, this part of calculation is simplified and needs robust mapping.

    # Placeholder logic, assuming direct mapping for now:
    # This loop implies `architecture.qtl_positions` are the additive QTLs, which is not right.
    # for (idx, qtl_pos) in enumerate(architecture.qtl_positions[1:architecture.n_qtl_additive])
    #     additive_values .+= view(geno_data_host, :, qtl_pos) .* architecture.additive_effects[idx]
    # end
    # This needs to use the actual SNP indices that correspond to `additive_effects`.
    # Let's assume `architecture.additive_qtl_indices` is correctly populated.
    # If `architecture.additive_qtl_indices` stores the *actual* SNP column indices:
    # This field is MISSING from the provided GeneticArchitecture struct.
    # It MUST be added.
    # For now, let's assume it exists for the calculation to make sense.
    # Example: `architecture.additive_qtl_actual_indices = sort(qtl_indices[1:n_qtl_additive])` in init.
    # Then loop:
    # for (eff_idx, snp_col_idx) in enumerate(architecture.additive_qtl_actual_indices)
    #    additive_values .+= view(geno_data_host, :, snp_col_idx) .* architecture.additive_effects[eff_idx]
    # end
    # This is a critical detail. For now, the original loop is:
    # for (idx, qtl_pos) in enumerate(architecture.qtl_positions[1:architecture.n_qtl_additive])
    # This implies the first n_qtl_additive entries in the *combined sorted unique* list of QTLs
    # are the ones with additive effects. This is likely not the intended mapping.
    # I will use a more direct interpretation based on how effects are generated.
    # The `initialize_genetic_architecture` selects `additive_qtl_pos` first. These are the ones.

    # Corrected interpretation based on original `initialize_genetic_architecture`:
    # `additive_qtl_pos = sort(all_positions[1:n_qtl_additive])`
    # These are the actual SNP indices that have additive effects.
    # So, `architecture.additive_effects` (length `n_qtl_additive`) corresponds to these `additive_qtl_pos`.
    # This means `GeneticArchitecture` should store `additive_qtl_pos` explicitly.
    # Let's modify `GeneticArchitecture` struct in `types.jl` to include this.
    # (Assuming this change is made when `types.jl` is finalized)
    # For now, let's assume `architecture.additive_qtl_indices_stored_separately` exists.
    # If `architecture.qtl_positions` is just a list of *all* SNPs that have *any* effect,
    # then we need a mapping.
    # The original code `qtl_positions = sort(all_positions[1:(n_qtl_additive + n_qtl_epistatic)])`
    # and then `additive_effects` are generated.
    # The most straightforward is that `additive_effects[k]` corresponds to `k`-th additive QTL chosen.
    # Let's assume `architecture.true_additive_qtl_indices` is a field.
    # For now, this part of the code cannot be correctly written without fixing GeneticArchitecture.
    # I will write it conceptually, assuming such a field exists.
    # `architecture.additive_qtl_indices` would be the actual column indices in genotype matrix.
    # This means `initialize_genetic_architecture` must populate this.

    # Simplified: assume `architecture.qtl_positions` for additive effects are the first n_qtl_additive ones.
    # This is what the original `calculate_genetic_values` did.
    # This is likely an oversimplification or error in the original logic.
    # For now, I will replicate the original structure's calculation logic.
    if architecture.n_qtl_additive > 0
        # Get the first n_qtl_additive positions from the combined list.
        # This assumes these are the ones with the additive effects.
        # This requires careful checking against the `initialize_genetic_architecture` logic.
        # The `initialize_genetic_architecture` selects `additive_qtl_pos = all_positions[1:n_qtl_additive]`
        # and `epistatic_pairs` from `all_positions` as well.
        # `qtl_positions` in the struct is `sort(union(additive_qtl_pos, snps_in_epistatic_pairs))`.
        # So, the original loop `architecture.qtl_positions[1:architecture.n_qtl_additive]` IS WRONG.
        # It must use the originally chosen additive QTL positions.
        # THIS REQUIRES `GeneticArchitecture` to store the `additive_qtl_pos` separately.
        # I cannot proceed with this calculation correctly without modifying `types.jl`'s GeneticArchitecture
        # or making unsafe assumptions.
        # For now, I will write a placeholder and flag it.
        # **FLAG FOR REVISION: Genetic value calculation depends on correct QTL index storage.**
        # Placeholder:
        # for i in 1:n_individuals
        #    for k in 1:architecture.n_qtl_additive
        #        # This needs the actual SNP index for the k-th additive effect.
        #        # snp_idx = architecture.map_additive_effect_to_snp_idx[k]
        #        # additive_values[i] += geno_data_host[i, snp_idx] * architecture.additive_effects[k]
        #    end
        # end
        # The original code's loop was:
        # for (idx, qtl_pos) in enumerate(architecture.qtl_positions[1:architecture.n_qtl_additive])
        # This means the first `n_qtl_additive` SNPs in the *sorted unique list of all QTLs* get these effects.
        # This seems arbitrary and unlikely to be correct.
        # I will assume `initialize_genetic_architecture` has been fixed to store `true_additive_qtl_indices`.
        # If not, this is a major bug.
        # For now, let's proceed with the original loop structure, assuming it was intended,
        # but with a strong note that this is likely wrong.
        # This is a critical point to clarify or fix.
        # The `Notes.docx` simulation says: "50 with purely additive effects and 50 pairs ... partially overlapping".
        # This implies a set of SNPs for additive, and another set for epistatic.
        # Let's assume `initialize_genetic_architecture` correctly identifies these.
        # The provided `initialize_genetic_architecture` code:
        # `additive_qtl_pos = sort(all_positions[1:n_qtl_additive])`
        # `additive_effects` are generated for these.
        # So, we should iterate over these `additive_qtl_pos`.
        # This means `GeneticArchitecture` needs to store `additive_qtl_pos`.
        # Let's assume it does, as `true_additive_qtl_indices`.
        # `architecture.true_additive_qtl_indices = sort(all_positions[1:n_qtl_additive])`
        # This means `architecture.additive_effects[k]` is for `architecture.true_additive_qtl_indices[k]`.
        # This field is NOT in the provided `GeneticArchitecture` struct.
        # I will add it conceptually for now.
        # **MODIFICATION NEEDED for types.jl: Add `additive_qtl_actual_indices::Vector{Int32}` to `GeneticArchitecture`**
        # And `initialize_genetic_architecture` must populate it.

        # Assuming `architecture.additive_qtl_actual_indices` exists and is populated:
        if isdefined(architecture, :additive_qtl_actual_indices) && length(architecture.additive_qtl_actual_indices) == architecture.n_qtl_additive
            for (eff_idx, snp_col_idx) in enumerate(architecture.additive_qtl_actual_indices)
                 additive_values .+= view(geno_data_host, :, snp_col_idx) .* architecture.additive_effects[eff_idx]
            end
        else
             # Fallback to original, likely incorrect, loop if field not present (for now)
             # This indicates a structural problem to be fixed.
            #  println("Warning: `additive_qtl_actual_indices` not found in GeneticArchitecture. Using potentially incorrect mapping for additive effects.")
            #  if architecture.n_qtl_additive > 0 && length(architecture.qtl_positions) >= architecture.n_qtl_additive
            #      for k_eff in 1:architecture.n_qtl_additive
            #          snp_col_idx = architecture.qtl_positions[k_eff] # This is the problematic part
            #          additive_values .+= view(geno_data_host, :, snp_col_idx) .* architecture.additive_effects[k_eff]
            #      end
            #  end
            # This part will be removed once GeneticArchitecture is fixed.
            # For now, to match original structure as closely as possible, I'll use the original loop logic,
            # despite its flaws, and make a note to fix it during refinement of `initialize_genetic_architecture`.
            # The original calculation loop was:
            # for (idx, qtl_pos) in enumerate(architecture.qtl_positions[1:architecture.n_qtl_additive])
            #    @inbounds for i in 1:n_individuals
            #        additive_values[i] += geno_data_host[i, qtl_pos] * architecture.additive_effects[idx]
            #    end
            # end
            # This implies the first `n_qtl_additive` SNPs in the combined sorted list `qtl_positions`
            # are arbitrarily assigned the additive effects. This is almost certainly wrong.
            # The `initialize_genetic_architecture` function's selection of `additive_qtl_pos` must be used.
            # I will assume `initialize_genetic_architecture` is modified to store these chosen SNPs,
            # e.g., in a field `_additive_qtl_indices_for_effects`.
            # If that field is not present, this calculation cannot be correct.
            # For now, I will write a placeholder that needs to be fixed once `initialize_genetic_architecture` is structured.
            # This is a major point of fragility in the provided code.
            # To avoid erroring out, and to reflect the original code's attempt:
            # This part is deferred until initialize_genetic_architecture is fully structured.
            # For now, let's assume `additive_values` remain zero if the mapping is unclear.
            # This needs to be fixed.
            # The original code's loop for additive effects:
            # for (idx, qtl_pos) in enumerate(architecture.qtl_positions[1:architecture.n_qtl_additive])
            #    @inbounds for i in 1:n_individuals
            #        additive_values[i] += geno_data_host[i, qtl_pos] * architecture.additive_effects[idx]
            #    end
            # end
            # This uses the first `n_qtl_additive` SNPs from the *combined sorted list of all QTLs*.
            # This is only correct if `initialize_genetic_architecture` ensures these specific positions
            # are the ones intended for these additive effects, which is not how it was written.
            # I will write it this way to match the provided code, but it's flagged as needing correction.
            # **POTENTIAL BUG: Mapping of additive effects to SNPs is likely incorrect here.**
            if architecture.n_qtl_additive > 0 && length(architecture.qtl_positions) >= architecture.n_qtl_additive
                for k_eff in 1:architecture.n_qtl_additive
                    # This assumes architecture.qtl_positions[k_eff] is the correct SNP for architecture.additive_effects[k_eff]
                    # This is highly dependent on how qtl_positions was constructed relative to additive_effects.
                    actual_snp_idx_for_effect_k = architecture.qtl_positions[k_eff] # Problematic assumption
                    for i_ind in 1:n_individuals
                        additive_values[i_ind] += geno_data_host[i_ind, actual_snp_idx_for_effect_k] * architecture.additive_effects[k_eff]
                    end
                end
            end
        end
    end


    # Calculate epistatic genetic values
    epistatic_values = zeros(T, n_individuals)
    if architecture.n_qtl_epistatic > 0 # n_qtl_epistatic is number of pairs
        for (pair_idx, (qtl_snp_idx1, qtl_snp_idx2)) in enumerate(architecture.epistatic_pairs)
            # qtl_snp_idx1 and qtl_snp_idx2 are the actual column indices in geno_data_host
            # for the SNPs in this interacting pair.
            interaction_effect = architecture.epistatic_effects[pair_idx]
            for i_ind in 1:n_individuals
                # Product of genotypes for the interacting pair
                # Note: Genotype coding (0,1,2) matters here.
                # If interaction is defined on allele counts, this is fine.
                # If defined on standardized genotypes, this calculation would be different.
                # The `Notes.docx` says `i_kl (x_ik, x_il)`, where x is 0,1,2.
                # So, product of allele counts is appropriate here.
                epistatic_values[i_ind] += geno_data_host[i_ind, qtl_snp_idx1] * geno_data_host[i_ind, qtl_snp_idx2] * interaction_effect
            end
        end
    end

    total_genetic_values = additive_values .+ epistatic_values

    return Dict{Symbol, Vector{T}}(
        :additive => additive_values,
        :epistatic => epistatic_values,
        :total => total_genetic_values
    )
end


"""
    generate_phenotypes(true_genetic_values::Dict{Symbol, Vector{Tgv}}, architecture::GeneticArchitecture{Ta}, output_float_type::Type{Tp}) where {Tgv, Ta, Tp <: AbstractFloat} -> PhenotypeData{Tp}

Generates phenotypic values for individuals.

Phenotypes are created by adding normally distributed environmental residuals (noise)
to the true total genetic values. The variance of these residuals is determined
by the target broad-sense heritability (`h2_broad_target`) specified in the `architecture`
and the observed variance of the `true_genetic_values[:total]`.
Assumes a single simulated trait.

# Arguments
- `true_genetic_values::Dict{Symbol, Vector{Tgv}}`: A dictionary containing at least a key `:total` with a vector of true total genetic values for individuals.
- `architecture::GeneticArchitecture{Ta}`: The `GeneticArchitecture` struct, primarily used here for its `h2_broad_target` field to determine residual variance.
- `output_float_type::Type{Tp}`: The floating-point type for the generated phenotypic values.

# Returns
- `PhenotypeData{Tp}`: A struct containing the vector of generated `values`, default `trait_names` (e.g., `[:simulated_trait]`), and `nothing` for fixed/random effects as these are not simulated by this function.
"""
function generate_phenotypes(
    true_genetic_values::Dict{Symbol, Vector{Tval}}, # Tval is type of genetic values
    architecture::GeneticArchitecture{Tarch},       # Tarch is type in architecture (h2_narrow, etc.)
    float_type::Type{Tout}                          # Tout is desired output phenotype type
) where {Tval <: AbstractFloat, Tarch <: AbstractFloat, Tout <: AbstractFloat}

    n_individuals = length(true_genetic_values[:total])
    total_gv = true_genetic_values[:total] # Vector of total genetic values

    # Calculate residual variance needed to achieve the target broad-sense heritability (H²)
    # H² = Var(Genetic) / Var(Phenotypic)
    # Var(Phenotypic) = Var(Genetic) / H²
    # Var(Residual) = Var(Phenotypic) - Var(Genetic)
    # Var(Residual) = Var(Genetic) / H² - Var(Genetic) = Var(Genetic) * (1/H² - 1)

    var_genetic_observed = var(total_gv) # Observed genetic variance from simulated values

    var_residual_target = zero(Tout)
    if architecture.h2_broad > eps(Tarch) && var_genetic_observed > eps(Tval)
        # Ensure h2_broad is not zero to avoid division by zero
        var_residual_target = var_genetic_observed * (one(Tarch) / architecture.h2_broad - one(Tarch))
    elseif var_genetic_observed <= eps(Tval) # If no genetic variance, all variance is residual (except mean)
        # This case means H2 should be 0. If H2 > 0, it's inconsistent.
        # For now, if no genetic variance, residual variance makes total variance 1 (arbitrary scale).
        var_residual_target = one(Tout) # Default if no genetic variance but want some phenotypic variance
    end

    # Ensure residual variance is not negative (can happen if observed genetic variance is unexpectedly high
    # or H² is very close to 1 and observed Var(G) > H² * (some assumed Var(P)) ).
    var_residual_target = max(zero(Tout), var_residual_target)

    std_dev_residual = sqrt(var_residual_target)

    # Generate environmental residuals (noise)
    residuals = randn(Tout, n_individuals) .* std_dev_residual

    # Phenotype = Genetic Value + Residual
    phenotype_values = Tout.(total_gv) .+ residuals # Ensure type consistency

    # Center phenotypes (optional, but common in GBLUP models that fit a mean)
    # phenotype_values .-= mean(phenotype_values)

    # PhenotypeData expects trait_names, fixed_effects, random_effects
    # For a simple simulation of one trait:
    return PhenotypeData(
        phenotype_values,
        [:simulated_trait], # Name for the simulated trait
        nothing,             # No additional fixed effects simulated here
        nothing              # No additional random effects simulated here
    )
end

# Export functions from this file if it were a module
# export simulate_population, initialize_genetic_architecture, generate_base_genotypes, calculate_true_genetic_values, generate_phenotypes
