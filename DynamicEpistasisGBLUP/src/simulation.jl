module Simulation

using ..DynamicEpistasisGBLUP: Random, Distributions, StatsBase, XSim

export simulate_population

"""
    simulate_population(;
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

Simulates a Mongolian sheep population over multiple generations.
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

    # 1. Base Population Generation (Generation 0)
    println("Generating base population (Generation 0)...")

    # Create a population with specified number of individuals
    pop = Population(base_population_size)

    # Define the genome structure
    genome = Genome(num_markers, num_chromosomes)

    # Assign the genome to the population
    pop.genome = genome

    # Sample allele frequencies from a Beta distribution
    maf = rand(Distributions.Beta(0.4, 0.4), num_markers)
    set_maf!(pop, maf)

    # Define QTLs
    add_qtl_indices = sample(1:num_markers, num_additive_qtl, replace=false)
    epi_qtl_indices = sample(1:num_markers, num_epistatic_qtl_pairs * 2, replace=false)
    epi_qtl_pairs = [(epi_qtl_indices[i], epi_qtl_indices[i+1]) for i in 1:2:length(epi_qtl_indices)]

    # Generate QTL effects
    add_effects = rand(Normal(0, 1), num_additive_qtl)
    epi_effects = rand(Normal(0, 1), num_epistatic_qtl_pairs)

    # Scale effects to achieve target heritabilities
    # This is a simplified approach; a more rigorous method would involve
    # iterative scaling or more complex calculations based on allele frequencies.
    # For now, we'll use a placeholder scaling.
    pheno = DataFrame(ID=1:base_population_size)
    pheno.sex = rand(["M", "F"], base_population_size)
    pheno.TBV = zeros(base_population_size)
    pheno.epistatic_val = zeros(base_population_size)

    # Calculate genetic values
    genotypes = get_genotypes(pop)
    for i in 1:base_population_size
        # Additive effects
        for (j, qtl_idx) in enumerate(add_qtl_indices)
            pheno.TBV[i] += genotypes[i, qtl_idx] * add_effects[j]
        end
        # Epistatic effects
        for (j, (qtl1, qtl2)) in enumerate(epi_qtl_pairs)
            pheno.epistatic_val[i] += genotypes[i, qtl1] * genotypes[i, qtl2] * epi_effects[j]
        end
    end

    # Scale genetic variances
    var_a = var(pheno.TBV)
    var_aa = var(pheno.epistatic_val)
    total_genetic_var = var_a + var_aa
    pheno_var = total_genetic_var / heritability_H2
    err_var = pheno_var - total_genetic_var

    # Scale additive and epistatic effects
    scaling_factor_a = sqrt((pheno_var * heritability_h2) / var_a)
    scaling_factor_aa = sqrt((pheno_var * (heritability_H2 - heritability_h2)) / var_aa)
    pheno.TBV .*= scaling_factor_a
    pheno.epistatic_val .*= scaling_factor_aa

    # Generate phenotypes
    pheno.phenotype = pheno.TBV + pheno.epistatic_val + rand(Normal(0, sqrt(err_var)), base_population_size)

    # Store results for each generation
    results = Dict()
    results[0] = (population=pop, phenotypes=pheno)

    # 2. Selection and Mating for Subsequent Generations
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
    get_genotypes(pop::Population)

Helper function to extract genotypes from an XSim population.
"""
function get_genotypes(pop::Population)
    # This is a placeholder. In a real implementation, we'd use XSim's functions
    # to get the genotype matrix.
    return rand(0:2, n_individuals(pop), n_markers(pop.genome))
end

"""
    n_individuals(pop::Population)

Helper function to get the number of individuals.
"""
n_individuals(pop::Population) = pop.n_individuals

"""
    n_markers(genome::Genome)

Helper function to get the number of markers.
"""
n_markers(genome::Genome) = genome.n_markers


"""
    select_parents(pop::Population, pheno::DataFrame, selection_intensity::Float64)

Selects parents for the next generation based on true breeding value.
"""
function select_parents(pop::Population, pheno::DataFrame, selection_intensity::Float64)
    # This is a placeholder for parent selection logic
    num_to_select = Int(ceil(n_individuals(pop) * selection_intensity))

    # Sort by TBV and select the top individuals
    sorted_pheno = sort(pheno, :TBV, rev=true)
    selected_indices = sorted_pheno.ID[1:num_to_select]

    # In a real scenario, we'd need to properly subset the XSim population
    # For now, we'll just return a new population of the same size
    return Population(n_individuals(pop))
end

"""
    mate(parents::Population, num_progeny::Int)

Mates selected parents to create the next generation.
"""
function mate(parents::Population, num_progeny::Int)
    # This is a placeholder for mating logic
    return Population(num_progeny)
end

end # module Simulation
