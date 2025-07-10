# ===== src/breeding_optimization.jl =====
"""
Breeding program optimization functionalities for DynamicEpistasisGBLUP.
Includes tools for optimal mating plans considering epistatic effects,
managing genetic gain, variance, and inbreeding.
Requires optimization solvers like Gurobi (via JuMP.jl).
"""

module BreedingOptimization

using JuMP
# using Gurobi # Gurobi is a specific optimizer; can be made generic with OptimizerWithAttributes
using LinearAlgebra # For dot products, matrix operations
using Statistics  # For mean, var
using Random      # For random aspects if any in selection/mating
using Combinatorics # If needed for enumerating pairs etc.
using CUDA          # For CuArray types if GRMs are on GPU

# Assuming types.jl (PopulationData, OrthogonalGBLUP) and other core modules are accessible.
# Helper to get Float type
_Float() = Main.DynamicEpistasisGBLUP.Float

export OptimalMatingPlanResult, # Renamed from OptimalMatingPlan
       optimize_breeding_program_matings, # Renamed
       allocate_mates_epistasis_aware, # Renamed
       simulate_long_term_genetic_gain # Renamed

"""
    OptimalMatingPlanResult{T}

Stores the results of an optimal mating plan computation.
Includes selected matings, expected genetic gain, variance, inbreeding, and epistatic value.
"""
struct OptimalMatingPlanResult{T<:AbstractFloat}
    matings::Vector{Tuple{Int, Int}}  # List of (parent1_idx, parent2_idx)
    expected_mean_genetic_gain_offspring::T # Average GEBV of offspring from these matings
    # expected_genetic_variance_offspring::T # Genetic variance among offspring (harder to predict simply)
    expected_mean_inbreeding_offspring::T # Average inbreeding coefficient of offspring
    # expected_mean_epistatic_value_offspring::T # Average epistatic value of offspring
end

"""
    optimize_breeding_program_matings(...) -> OptimalMatingPlanResult{T}

Optimizes mating decisions for a breeding program.
Considers additive and epistatic values, relationship constraints (inbreeding),
and aims to maximize a defined criterion (e.g., genetic gain).
`population_obj` and `gblup_model` provide necessary inputs.
"""
function optimize_breeding_program_matings(
    population_obj::Main.DynamicEpistasisGBLUP.PopulationData{T},
    gblup_model::Main.DynamicEpistasisGBLUP.OrthogonalGBLUP{T},
    # gebvs_additive_train::Vector{T}, # Additive GEBVs of candidates (from model fitting)
    # gebvs_epistatic_train::Union{Nothing, Vector{T}} = nothing; # Epistatic GEBVs
    # The gblup_model itself does not store GEBVs. These must be passed or recomputed.
    # Let's assume they are passed or obtained from a full `fitted_gblup_output` which includes GEBVs.
    # For now, this function will be a high-level stub as it needs GEBVs.
    num_matings_to_plan::Int = 100,
    # offspring_per_mating::Int = 20, # Not directly used in optimizing matings, but for predicting gain
    # selection_intensity_for_parents::Float64 = 0.2, # Applied before optimization, or part of it
    optimization_criterion_symbol::Symbol = :genetic_gain, # :genetic_gain, :genetic_variance, :epistatic_value
    constraints_dict::Dict{Symbol, Any} = Dict(:max_inbreeding_rate => T(0.05), :max_parental_contribution => 5) # Example constraints
    # optimizer_to_use = Gurobi.Optimizer # Example
    optimizer_to_use::Any = nothing # Specify optimizer via JuMP syntax later
) where T <: AbstractFloat

    n_candidates = population_obj.genotypes.n_individuals

    if n_candidates < 2
        error("Not enough candidates for mating.")
    end
    if optimizer_to_use === nothing
        @warn "No optimizer specified for JuMP. Install Gurobi, CPLEX, or another solver and pass it."
        # Fallback to a simpler heuristic if no optimizer? Or error out.
        # For now, let's assume an optimizer will be provided by the user.
        # return OptimalMatingPlanResult([], T(0), T(0)) # Empty plan
        error("JuMP optimizer not specified. Please provide one for `optimize_breeding_program_matings`.")
    end

    # Placeholder: GEBVs need to be provided or computed.
    # For this stub, generate random GEBVs.
    # In real use, these come from `orthogonal_epistasis_gblup`.
    Random.seed!(123)
    gebvs_additive_candidates = randn(T, n_candidates)
    gebvs_epistatic_candidates = gblup_model.G_aa === nothing ? nothing : randn(T, n_candidates)


    # Create JuMP model
    opt_model = Model(optimizer_to_use)
    # set_optimizer_attribute(opt_model, "TimeLimit", 600) # Example: Time limit 10 mins
    # set_silent(opt_model) # Suppress solver output

    # Decision variables: x[i,j] = 1 if individual i is mated with j, 0 otherwise.
    # Considering i < j to represent unique matings.
    # Or, x[i,j] is number of times pair (i,j) is used if multiple matings per pair allowed.
    # For now, assume x[i,j] is proportion of total matings, or related to number of offspring.
    # Let's use x[i,j] = 1 if pair (i,j) is chosen for one unit of mating.
    # We need to select `num_matings_to_plan` pairs.

    # Define x_ij for i < j (unique pairs)
    # mating_pairs_indices = [(i,j) for i in 1:n_candidates for j in (i+1):n_candidates]
    # @variable(opt_model, x[mating_pairs_indices], Bin) # x[ (i,j) ] = 1 if pair (i,j) is made

    # Simpler: x[i,j] where i is male, j is female, if sexes are distinct.
    # Or, general pool: x[i,j] for i < j.
    # Let's assume a general pool and x[i,j] is binary for selection of the pair.
    # To handle symmetry and avoid selfing:
    @variable(opt_model, mating_decision[1:n_candidates, 1:n_candidates], Bin)

    # --- Constraints ---
    # Symmetric decisions: mating_decision[i,j] == mating_decision[j,i] (handled by summing i<j later)
    # No selfing: mating_decision[i,i] == 0
    for i in 1:n_candidates
        @constraint(opt_model, mating_decision[i,i] == 0)
    end

    # Total number of matings: sum_{i<j} mating_decision[i,j] == num_matings_to_plan
    @constraint(opt_model, sum(mating_decision[i,j] for i in 1:n_candidates, j in (i+1):n_candidates) == num_matings_to_plan)

    # Parental contribution limits (max number of matings per individual)
    if haskey(constraints_dict, :max_parental_contribution)
        max_contrib = constraints_dict[:max_parental_contribution]
        for i in 1:n_candidates
            # Sum over j where mating_decision[i,j] or mating_decision[j,i] is involved.
            @constraint(opt_model, sum(mating_decision[i,j] for j in 1:n_candidates if i!=j) <= max_contrib)
        end
    end

    # Inbreeding constraint: Average relationship of selected pairs <= threshold
    # Requires GRM (additive relationship matrix)
    G_add_cpu = Array(gblup_model.G) # Move to CPU
    if haskey(constraints_dict, :max_inbreeding_rate)
        max_avg_inbreeding = constraints_dict[:max_inbreeding_rate]
        # Inbreeding F_offspring = 0.5 * G_parent1_parent2 (if parents unrelated to each other beyond this pair)
        # Or, more generally, expected F_offspring = (F_sire + F_dam)/2 if G is pedigree A.
        # If G is genomic relationship, G_ij is relationship.
        # Avg relationship of parents in chosen matings:
        # sum_{i<j} (mating_decision[i,j] * G_add_cpu[i,j]) / num_matings_to_plan
        # This should be <= some target (e.g., related to 2*max_F).
        # For simplicity, let's target average parental relationship.
        # Target max relationship: if max_F = 0.05, target_G_avg might be 0.10.
        # This constraint is complex to formulate precisely without more assumptions.
        # Placeholder: sum_{i<j} G_ij * x_ij <= max_total_relationship
        # Let's use avg offspring inbreeding: sum (0.5 * G_ij * x_ij) / N_matings <= max_F
        @constraint(opt_model,
            sum(T(0.5) * G_add_cpu[i,j] * mating_decision[i,j] for i in 1:n_candidates, j in (i+1):n_candidates)
            <= num_matings_to_plan * max_avg_inbreeding
        )
    end

    # --- Objective Function ---
    # Maximize expected genetic gain in offspring (or other criteria)
    # Expected additive value of offspring from pair (i,j) = 0.5 * (GEBV_add_i + GEBV_add_j)
    obj_expr_terms = AffExpr[]
    for i in 1:n_candidates
        for j in (i+1):n_candidates # Iterate over unique pairs i < j
            term_value = zero(T)
            if optimization_criterion_symbol == :genetic_gain
                term_value = T(0.5) * (gebvs_additive_candidates[i] + gebvs_additive_candidates[j])
                # Add expected epistatic contribution if model and criterion include it
                if gebvs_epistatic_candidates !== nothing && gblup_model.G_aa !== nothing
                    # `compute_expected_epistatic_value_offspring` is a helper needed.
                    # This depends on how epistasis transmits / re-combines.
                    # Simple proxy: 0.5 * (GEBV_epi_i + GEBV_epi_j) * (1 + factor * G_aa_ij)
                    # For now, just use additive part for gain.
                end
            elseif optimization_criterion_symbol == :epistatic_value && gebvs_epistatic_candidates !== nothing
                # Maximize sum of parental epistatic values or expected offspring epistatic value.
                # This needs a clear definition of "epistatic value" to maximize.
                # Placeholder:
                # term_value = compute_expected_epistatic_value_offspring(i,j, gebvs_epistatic_candidates, Array(gblup_model.G_aa))
            end
            push!(obj_expr_terms, term_value * mating_decision[i,j])
        end
    end

    if !isempty(obj_expr_terms)
        @objective(opt_model, Max, sum(obj_expr_terms))
    else # Handle cases with no terms (e.g. wrong criterion)
        @objective(opt_model, Max, 0) # Null objective
    end

    # Solve the optimization problem
    # println("Optimizing mating plan with JuMP...")
    try
        optimize!(opt_model)
    catch e
        println("Error during JuMP optimization: $e")
        println("Ensure a valid optimizer (e.g., Gurobi, HiGHS) is installed and accessible in your Julia environment.")
        error("JuMP optimization failed.")
    end

    # Extract results
    planned_matings = Tuple{Int,Int}[]
    mean_offspring_gain_final = T(0)
    mean_offspring_inbreeding_final = T(0)

    if termination_status(opt_model) == MOI.OPTIMAL || termination_status(opt_model) == MOI.SOLUTION_LIMIT
        # println("Optimal solution found (or solution limit reached).")
        mating_decision_values = value.(mating_decision)

        actual_num_matings_made = 0
        sum_offspring_gain = zero(T)
        sum_offspring_inbreeding = zero(T)

        for i in 1:n_candidates
            for j in (i+1):n_candidates
                if mating_decision_values[i,j] > T(0.5) # If pair (i,j) is selected
                    push!(planned_matings, (i,j))
                    actual_num_matings_made += 1
                    sum_offspring_gain += T(0.5) * (gebvs_additive_candidates[i] + gebvs_additive_candidates[j])
                    sum_offspring_inbreeding += T(0.5) * G_add_cpu[i,j] # Approx inbreeding
                end
            end
        end

        if actual_num_matings_made > 0
            mean_offspring_gain_final = sum_offspring_gain / actual_num_matings_made
            mean_offspring_inbreeding_final = sum_offspring_inbreeding / actual_num_matings_made
        end
    else
        println("Optimization did not find an optimal solution. Status: ", termination_status(opt_model))
    end

    return OptimalMatingPlanResult(
        planned_matings,
        mean_offspring_gain_final,
        mean_offspring_inbreeding_final
        # Other stats like expected epistatic value could be added.
    )
end


"""
    allocate_mates_epistasis_aware(...) -> Vector{Tuple{Int,Int}}

Mate allocation strategy that specifically tries to create favorable epistatic combinations
or manage epistatic variance. This is more heuristic than formal optimization.
"""
function allocate_mates_epistasis_aware(
    population_obj::Main.DynamicEpistasisGBLUP.PopulationData{T},
    gblup_model::Main.DynamicEpistasisGBLUP.OrthogonalGBLUP{T};
    # Again, needs GEBVs or similar measures of epistatic potential.
    num_matings_to_plan::Int = 100,
    strategy_symbol::Symbol = :complementarity # :complementarity, :positive_assortative_epistasis
) where T <: AbstractFloat

    # This function is highly dependent on how epistatic effects/values are quantified
    # and what constitutes a "favorable" epistatic combination.
    # Placeholder for now.
    # println("Epistasis-aware mate allocation (strategy: $strategy_symbol) - STUB")

    # Example for :complementarity:
    # 1. Identify target epistatic interactions (e.g., from a sparse epistasis model).
    # 2. For each candidate pair (sire, dam), predict the probability of offspring
    #    having desired genotypes at these interacting loci.
    # 3. Score pairs based on this complementarity.
    # 4. Select top N pairs, possibly with diversity constraints.

    # For now, return random pairings as a stub.
    n_candidates = population_obj.genotypes.n_individuals
    if n_candidates < 2 return Tuple{Int,Int}[] end

    matings_stub = Vector{Tuple{Int,Int}}(undef, num_matings_to_plan)
    for i in 1:num_matings_to_plan
        p1 = rand(1:n_candidates)
        p2 = rand(1:n_candidates)
        while p1 == p2 p2 = rand(1:n_candidates) end # Avoid selfing
        matings_stub[i] = (min(p1,p2), max(p1,p2)) # Store canonical pair
    end

    return unique(matings_stub) # Return unique random pairs
end


"""
    simulate_long_term_genetic_gain(...) -> Tuple{Vector{PopulationData}, Vector{OrthogonalGBLUP}, Vector{T}}

Simulates a breeding program over multiple generations, applying selection and mating strategies
(potentially optimized using epistatic information) and tracks long-term genetic gain.
"""
function simulate_long_term_genetic_gain(
    initial_population_obj::Main.DynamicEpistasisGBLUP.PopulationData{T},
    # initial_gblup_model::Main.DynamicEpistasisGBLUP.OrthogonalGBLUP{T};
    # Model should be fitted per generation or as specified.
    num_generations_to_simulate::Int = 10,
    num_matings_per_gen::Int = 50,
    num_offspring_per_mating::Int = 4, # Total N per gen = num_matings * offspring_per_mating
    parent_selection_intensity::Float64 = 0.2, # Top X% selected as parents
    re_estimate_model_freq_gens::Int = 1, # How often to re-fit GBLUP model
    mating_optimization_optimizer::Any = nothing # JuMP optimizer
) where T <: AbstractFloat

    # Store history
    populations_history = [initial_population_obj]
    models_history = Main.DynamicEpistasisGBLUP.OrthogonalGBLUP{T}[] # Store fitted models
    mean_genetic_gains_per_gen = T[] # Store mean GEBV or phenotype change

    current_pop = initial_population_obj

    for gen_num in 1:num_generations_to_simulate
        # println("\nSimulating Generation $gen_num...")

        # 1. Fit/Update GBLUP model for current population (if due)
        local current_fitted_model
        local current_gebvs # Additive + Epistatic total GEBVs

        if gen_num == 1 || (gen_num - 1) % re_estimate_model_freq_gens == 0
            # println("  Fitting GBLUP model for current generation...")
            # orthogonal_epistasis_gblup returns (model_obj, gebv_vector)
            model_fit, gebvs_fit = Main.DynamicEpistasisGBLUP.orthogonal_epistasis_gblup(
                current_pop,
                include_epistasis = true, # Assuming full model
                update_frequencies_per_generation = true # Uses current pop's freqs
            )
            current_fitted_model = model_fit
            current_gebvs = gebvs_fit
            push!(models_history, current_fitted_model)
        else
            # Use previous model to predict GEBVs for current population for selection
            # println("  Predicting GEBVs using previous model...")
            # genomic_prediction needs the model and new genotypes.
            # This requires careful handling of what `genomic_prediction` expects.
            # Assuming it can predict on `current_pop.genotypes` using `models_history[end]`.
            current_gebvs = Main.DynamicEpistasisGBLUP.genomic_prediction(
                models_history[end], current_pop.genotypes #; reference_genotypes=... u_hats=...
            )
            current_fitted_model = models_history[end] # Use last fitted model for decisions
        end

        # 2. Select Parents based on GEBVs
        num_parents_to_select = round(Int, current_pop.genotypes.n_individuals * parent_selection_intensity)
        if isodd(num_parents_to_select) num_parents_to_select = max(2, num_parents_to_select-1) end # Ensure even
        if num_parents_to_select < 2
            # println("  Not enough parents to select. Stopping simulation.")
            break
        end
        selected_parent_indices = Main.DynamicEpistasisGBLUP.select_parents(current_gebvs, num_parents_to_select)

        # Create a sub-population of selected parents for mating optimization
        # This requires subsetting PopulationData.
        # parent_pop_obj = subset_population(current_pop, selected_parent_indices)
        # For now, assume optimize_breeding_program_matings can work with indices into current_pop.
        # It needs their GEBVs and GRM slice. This is getting complex.

        # Simpler: Optimize matings among all current_pop individuals, but only use selected parents.
        # Or, optimize among selected parents.
        # For now, assume `optimize_breeding_program_matings` gets the full `current_pop`
        # and GEBVs, and internally handles selection or works on pre-selected candidates.
        # This part needs refinement.
        # Let's assume mating optimization is done on the *selected parents*.
        # This means `optimize_breeding_program_matings` needs to handle a subset.
        # For now, this is a conceptual call.

        # 3. Optimize Mating Plan for selected parents
        # println("  Optimizing mating plan...")
        # This is a placeholder as `optimize_breeding_program_matings` is a complex stub.
        # mating_plan_result = optimize_breeding_program_matings(parent_pop_obj, current_fitted_model, ...)
        # For stub, just get random matings among selected parents:
        planned_matings_indices_local = Vector{Tuple{Int,Int}}(undef, num_matings_per_gen)
        n_sel_parents = length(selected_parent_indices)
        for i in 1:num_matings_per_gen
            p1_local_idx = rand(1:n_sel_parents)
            p2_local_idx = rand(1:n_sel_parents)
            while p1_local_idx == p2_local_idx p2_local_idx = rand(1:n_sel_parents) end
            # Convert local indices (within selected_parent_indices) to global indices in current_pop
            planned_matings_indices_local[i] = (selected_parent_indices[p1_local_idx], selected_parent_indices[p2_local_idx])
        end
        # `OptimalMatingPlanResult` struct has a `matings` field.
        mating_plan_obj = OptimalMatingPlanResult(planned_matings_indices_local, T(0), T(0))


        # 4. Generate Offspring Population
        # println("  Generating offspring...")
        # `generate_offspring_population` is a helper function needed.
        # It takes current_pop (for parent genotypes), mating_plan, and n_offspring.
        offspring_pop = generate_offspring_population(
            current_pop, mating_plan_obj.matings, num_offspring_per_mating, gen_num
        )

        # 5. Track Genetic Gain (e.g., change in mean phenotype or true BV if known)
        mean_pheno_offspring = mean(offspring_pop.phenotypes.values)
        mean_pheno_parents_gen = mean(current_pop.phenotypes.values[selected_parent_indices]) # Approx parent gen mean
        # Or use mean of current_pop as baseline for previous gen.
        gain_this_gen = mean_pheno_offspring - mean(current_pop.phenotypes.values)
        push!(mean_genetic_gains_per_gen, gain_this_gen)
        # println("    Mean phenotype of offspring: $mean_pheno_offspring (Gain: $gain_this_gen)")

        current_pop = offspring_pop # Advance to next generation
        push!(populations_history, current_pop)
    end

    return populations_history, models_history, mean_genetic_gains_per_gen
end


# Helper function to generate an offspring population (simplified)
function generate_offspring_population(
    parent_population::Main.DynamicEpistasisGBLUP.PopulationData{T},
    mating_pairs_indices::Vector{Tuple{Int,Int}}, # List of (global_idx_parent1, global_idx_parent2)
    num_offspring_per_pair::Int,
    current_generation_number::Int
) where T
    num_total_offspring = length(mating_pairs_indices) * num_offspring_per_pair
    n_snps = parent_population.genotypes.n_snps

    offspring_genotypes_host = zeros(T, num_total_offspring, n_snps)

    # Assumed parameters for `generate_single_offspring`
    # This needs to be robust, e.g. from `parent_population.metadata`
    n_chromosomes = 26 # Sheep default
    snps_per_chromosome = n_snps ÷ n_chromosomes # Approximate
    if snps_per_chromosome == 0 snps_per_chromosome = 1 end # Avoid division by zero if n_snps < n_chromosomes

    parent_geno_data_host = Array(parent_population.genotypes.data) # Get parent genotypes to CPU

    offspring_counter = 0
    for (p1_idx, p2_idx) in mating_pairs_indices
        parent1_geno_slice = @view parent_geno_data_host[p1_idx, :]
        parent2_geno_slice = @view parent_geno_data_host[p2_idx, :]
        for _ in 1:num_offspring_per_pair
            offspring_counter += 1
            offspring_genotypes_host[offspring_counter, :] = Main.DynamicEpistasisGBLUP.generate_single_offspring(
                parent1_geno_slice, parent2_geno_slice, snps_per_chromosome, n_chromosomes
            )
        end
    end

    # Create GenotypeMatrix for offspring (on GPU)
    offspring_geno_gpu = CuArray(offspring_genotypes_host)
    offspring_missing_mask = CuSparseMatrixCSR(spzeros(Bool, Int32, num_total_offspring, n_snps))
    # Allele freqs for offspring will be computed when model is next updated or `update_allele_frequencies!` is called.
    # For now, can initialize with parent freqs or empty. Let's use parent for consistency.
    offspring_allele_freqs_gpu = copy(parent_population.genotypes.allele_freq)

    offspring_genotype_obj = Main.DynamicEpistasisGBLUP.GenotypeMatrix(
        offspring_geno_gpu, offspring_missing_mask, offspring_allele_freqs_gpu,
        Int32(num_total_offspring), Int32(n_snps), parent_population.genotypes.ploidy
    )

    # Generate phenotypes for offspring based on their new genotypes and original architecture
    # This requires `calculate_true_genetic_values` and `generate_phenotypes` from simulation.jl.
    offspring_true_genetic_values = Main.DynamicEpistasisGBLUP.calculate_true_genetic_values(
        offspring_genotype_obj, parent_population.metadata[:architecture]
    )
    offspring_phenotype_obj = Main.DynamicEpistasisGBLUP.generate_phenotypes(
        offspring_true_genetic_values, parent_population.metadata[:architecture], T
    )

    return Main.DynamicEpistasisGBLUP.PopulationData(
        offspring_genotype_obj,
        offspring_phenotype_obj,
        nothing, # Pedigree
        Int32(current_generation_number),
        parent_population.metadata # Inherit metadata like genetic architecture
    )
end


# Other helper functions from original:
# - `compute_expected_epistatic_value_offspring`
# - `identify_beneficial_interactions` (from model)
# - `compute_complementarity_score`
# - `select_diverse_matings`
# - `execute_mating_plan` (similar to `generate_offspring_population` but using plan obj)
# - `compute_plan_statistics` (to populate OptimalMatingPlanResult)
# These are more detailed implementations within the optimization strategies.

# Export functions if this file were a module
# export OptimalMatingPlanResult, optimize_breeding_program_matings,
#        allocate_mates_epistasis_aware, simulate_long_term_genetic_gain

end # module BreedingOptimization
