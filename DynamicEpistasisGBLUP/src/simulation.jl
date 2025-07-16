module Simulation

using ..DynamicEpistasisGBLUP: Random, Distributions, StatsBase, XSim

export simulate_population

"""
    simulate_population(; ...)

Simulates a multi-generation livestock population with selection, based on the description
in the paper.

This function sets up a base population, defines a genetic architecture with both
additive and epistatic QTLs, and simulates selection over several generations. It
tracks genotypes, phenotypes, and true genetic values.

# Keyword Arguments
- `num_generations::Int=10`: The number of generations to simulate after the base generation.
- `base_population_size::Int=1000`: The number of individuals in the base population.
- `num_markers::Int=50000`: The total number of SNP markers on the genome.
- `num_chromosomes::Int=26`: The number of chromosomes.
- `num_additive_qtl::Int=50`: The number of QTLs with purely additive effects.
- `num_epistatic_qtl_pairs::Int=50`: The number of pairs of QTLs with epistatic interactions.
- `heritability_h2::Float64=0.3`: The narrow-sense heritability (additive variance).
- `heritability_H2::Float64=0.4`: The broad-sense heritability (additive + epistatic variance).
- `selection_intensity::Float64=0.2`: The proportion of individuals selected as parents for the next generation.

# Returns
- `Dict`: A dictionary where keys are generation numbers (0 to `num_generations`) and
  values are named tuples `(population, phenotypes)` containing the `XSim` population
  object and a `DataFrame` of phenotypes and true genetic values.
"""
function simulate_population(;
    num_generations::Int=10,
    base_population_size::Int=1000,
    num_markers::Int=50000,
    num_chromosomes::Int=26,
    num_additive_qtl::Int=50,
    num_epistatic_qtl_pairs::Int=50,
    heritability_h2::Float64=0.3,
    heritability_H2::Float64=0.4,
    selection_intensity::Float64=0.2
    )

    # --- 1. Base Population Generation (Generation 0) ---
    println("Generating base population (Generation 0)...")

    # Create a population with specified number of individuals
    pop = Population(base_population_size)

    # Define the genome structure
    genome = Genome(num_markers, num_chromosomes)

    # Assign the genome to the population
    pop.genome = genome

    # Sample allele frequencies from a Beta distribution to mimic high genetic diversity
    maf = rand(Distributions.Beta(0.4, 0.4), num_markers)
    set_maf!(pop, maf)

    # Randomly select markers to be QTLs
    add_qtl_indices = sample(1:num_markers, num_additive_qtl, replace=false)
    epi_qtl_indices = sample(1:num_markers, num_epistatic_qtl_pairs * 2, replace=false)
    epi_qtl_pairs = [(epi_qtl_indices[i], epi_qtl_indices[i+1]) for i in 1:2:length(epi_qtl_indices)]

    # Generate QTL effects from a standard normal distribution
    add_effects = rand(Normal(0, 1), num_additive_qtl)
    epi_effects = rand(Normal(0, 1), num_epistatic_qtl_pairs)

    # --- This section calculates and scales genetic effects to match target heritabilities ---
    # It's a key part of the simulation, ensuring the genetic architecture is as specified.

    # Initialize a DataFrame to store phenotype and genetic values
    pheno = DataFrame(ID=1:base_population_size)
    pheno.sex = rand(["M", "F"], base_population_size)
    pheno.TBV = zeros(base_population_size) # True Breeding Value (additive)
    pheno.epistatic_val = zeros(base_population_size) # Epistatic value

    # Get genotypes for the base population
    genotypes = get_genotypes(pop)

    # Calculate raw genetic values based on QTL effects
    for i in 1:base_population_size
        # Sum additive effects
        for (j, qtl_idx) in enumerate(add_qtl_indices)
            pheno.TBV[i] += genotypes[i, qtl_idx] * add_effects[j]
        end
        # Sum epistatic effects (as interaction between genotypes)
        for (j, (qtl1, qtl2)) in enumerate(epi_qtl_pairs)
            pheno.epistatic_val[i] += genotypes[i, qtl1] * genotypes[i, qtl2] * epi_effects[j]
        end
    end

    # Calculate empirical variances of the raw genetic values
    var_a = var(pheno.TBV)
    var_aa = var(pheno.epistatic_val)

    # Determine the required residual variance to meet the target heritabilities
    # Total phenotypic variance = Var(A) + Var(AA) + Var(E)
    # H2 = (Var(A) + Var(AA)) / Var(P) => Var(P) = (Var(A) + Var(AA)) / H2
    # h2 = Var(A) / Var(P)
    total_genetic_var = var_a + var_aa
    pheno_var = total_genetic_var / heritability_H2
    err_var = pheno_var - total_genetic_var

    # Scale the genetic effects to match the heritabilities precisely
    scaling_factor_a = sqrt((pheno_var * heritability_h2) / var_a)
    scaling_factor_aa = sqrt((pheno_var * (heritability_H2 - heritability_h2)) / var_aa)
    pheno.TBV .*= scaling_factor_a
    pheno.epistatic_val .*= scaling_factor_aa

    # Generate final phenotypes by adding scaled genetic values and residual error
    pheno.phenotype = pheno.TBV + pheno.epistatic_val + rand(Normal(0, sqrt(err_var)), base_population_size)

    # Store the results for the base generation
    results = Dict()
    results[0] = (population=pop, phenotypes=pheno)

    # --- 2. Selection and Mating for Subsequent Generations ---
    for gen in 1:num_generations
        println("Simulating Generation $gen...")

        # Select parents based on TBV
        parents = select_parents(results[gen-1].population, results[gen-1].phenotypes, selection_intensity)

        # Create next generation
        progeny = mate(parents, base_population_size)

        # Generate phenotypes for the new generation
        progeny_pheno = DataFrame(ID=(1:base_population_size) .+ (gen * base_population_size))
        progeny_pheno.sex = rand(["M", "F"], base_population_size)
        progeny_pheno.TBV = zeros(base_population_size)
        progeny_pheno.epistatic_val = zeros(base_population_size)

        progeny_genotypes = get_genotypes(progeny)
        for i in 1:base_population_size
            # Additive effects
            for (j, qtl_idx) in enumerate(add_qtl_indices)
                progeny_pheno.TBV[i] += progeny_genotypes[i, qtl_idx] * add_effects[j] * scaling_factor_a
            end
            # Epistatic effects
            for (j, (qtl1, qtl2)) in enumerate(epi_qtl_pairs)
                progeny_pheno.epistatic_val[i] += progeny_genotypes[i, qtl1] * progeny_genotypes[i, qtl2] * epi_effects[j] * scaling_factor_aa
            end
        end
        progeny_pheno.phenotype = progeny_pheno.TBV + progeny_pheno.epistatic_val + rand(Normal(0, sqrt(err_var)), base_population_size)

        results[gen] = (population=progeny, phenotypes=progeny_pheno)
    end

    return results
end

"""
    get_genotypes(pop::Population) -> Matrix{Int}

A helper function to extract genotypes from an XSim `Population` object.

NOTE: This is a placeholder function. A real implementation would use `XSim.jl`'s
API to get the actual genotype matrix for the individuals in the population.
For demonstration, it returns a random genotype matrix.
"""
function get_genotypes(pop::Population)
    # In a real implementation, you would use a function from XSim, e.g.:
    # return XSim.get_genotypes(pop)
    return rand(0:2, n_individuals(pop), n_markers(pop.genome))
end

"""
    n_individuals(pop::Population) -> Int

Helper function to get the number of individuals in a population.
(Placeholder for `XSim.n_individuals`).
"""
n_individuals(pop::Population) = pop.n_individuals

"""
    n_markers(genome::Genome) -> Int

Helper function to get the number of markers in a genome.
(Placeholder for `XSim.n_markers`).
"""
n_markers(genome::Genome) = genome.n_markers


"""
    select_parents(pop::Population, pheno::DataFrame, selection_intensity::Float64) -> Population

Selects parents for the next generation based on true breeding value (TBV).

NOTE: This is a placeholder. A real implementation would subset the `pop` object
based on the selected indices to create a new population of parents.
"""
function select_parents(pop::Population, pheno::DataFrame, selection_intensity::Float64)
    # Determine the number of parents to select based on intensity
    num_to_select = Int(ceil(n_individuals(pop) * selection_intensity))

    # Sort individuals by their True Breeding Value in descending order
    sorted_pheno = sort(pheno, :TBV, rev=true)
    # Get the IDs of the top individuals
    selected_indices = sorted_pheno.ID[1:num_to_select]

    # In a real scenario, you would use XSim to create a new population
    # containing only the selected parents, e.g.:
    # return XSim.subset(pop, selected_indices)
    return Population(n_individuals(pop)) # Placeholder
end

"""
    mate(parents::Population, num_progeny::Int) -> Population

Mates selected parents to create the next generation of progeny.

NOTE: This is a placeholder. A real implementation would use `XSim.jl`'s
mating and recombination functions to generate progeny.
"""
function mate(parents::Population, num_progeny::Int)
    # In a real scenario, you would use a function like:
    # return XSim.random_mate(parents, n_progeny=num_progeny)
    return Population(num_progeny) # Placeholder
end

end # module Simulation
