# Population simulation module

module PopulationSimulation

using Random
using Distributions
using StatsBase
using DataFrames
using ProgressMeter
using ..CoreTypes
using ..OGBLUPModel # Needed for selection on GEBV

export simulate_population, simulate_generation

function simulate_population(params::SimulationParameters)

    @info "Simulating base population..."
    population = create_base_population(params)

    # Simulate multiple generations
    history = [population]
    @showprogress "Simulating generations..." for gen in 1:params.n_generations
        population = simulate_generation(population, params)
        push!(history, population)
    end

    return history
end

function create_base_population(params)
    # ... (code from previous implementation to create base pop)
    # This part is assumed to be complete for now.
    genotypes = zeros(Int8, params.n_individuals, params.n_markers)
    allele_freq = rand(params.maf_distribution, params.n_markers)
    for j in 1:params.n_markers
        p = allele_freq[j]
        for i in 1:params.n_individuals
            genotypes[i, j] = rand(Binomial(2, p))
        end
    end

    marker_info = DataFrame(ID=1:params.n_markers, CHR=rand(1:params.n_chromosomes, params.n_markers), POS=rand(1:10^8, params.n_markers))

    geno_data = GenotypeData(
        genotypes, allele_freq, 0.01, ones(params.n_markers),
        params.n_individuals, params.n_markers, marker_info,
        DataFrame(ID=1:params.n_individuals), 2
    )

    phenotypes = randn(params.n_individuals, 1) # Placeholder

    return Population(geno_data, phenotypes, zeros(params.n_individuals, 0), 0, nothing, nothing, Dict())
end


function simulate_generation(current_pop::Population, params::SimulationParameters)

    @info "Simulating generation \$(current_pop.generation + 1)"

    # 1. Selection
    if isa(params.selection_scheme, TruncationSelection)
        # Fit a model to get GEBVs for selection
        model = fit_ogblup(current_pop, include_dominance=false, include_epistasis=false)
        gebv = predict_gebv(model, components=:additive)

        n_selected = round(Int, params.n_individuals * params.selection_scheme.proportion)
        parents = sortperm(vec(gebv), rev=true)[1:n_selected]
    else # NoSelection
        parents = 1:current_pop.n_individuals
    end

    # 2. Mating
    n_offspring = params.n_individuals
    matings = []
    if isa(params.mating_system, RandomMating)
        for _ in 1:n_offspring
            p1 = rand(parents)
            p2 = rand(parents)
            push!(matings, (p1, p2))
        end
    end

    # 3. Recombination and Gamete generation
    offspring_genotypes = zeros(Int8, n_offspring, params.n_markers)

    # This is a simplified meiosis. A real implementation would use chromosome lengths and recombination rates.
    for i in 1:n_offspring
        p1_geno = current_pop.genotype_data.genotypes[matings[i][1], :]
        p2_geno = current_pop.genotype_data.genotypes[matings[i][2], :]

        gamete1 = [rand() < 0.5 ? (g > 0 ? 1 : 0) : (g == 2 ? 1 : 0) for g in p1_geno]
        gamete2 = [rand() < 0.5 ? (g > 0 ? 1 : 0) : (g == 2 ? 1 : 0) for g in p2_geno]

        offspring_genotypes[i, :] = gamete1 + gamete2
    end

    # 4. Create new population object
    # (Phenotype simulation would be more complex, tied to QTLs)

    new_geno_data = GenotypeData(
        offspring_genotypes,
        vec(mean(offspring_genotypes, dims=1)) ./ 2, # new allele freqs
        current_pop.genotype_data.maf_filter,
        ones(params.n_markers),
        n_offspring, params.n_markers,
        current_pop.genotype_data.marker_info,
        DataFrame(ID=1:n_offspring), 2
    )

    new_phenotypes = randn(n_offspring, 1) # Placeholder

    next_gen_pop = Population(
        new_geno_data, new_phenotypes, zeros(n_offspring, 0),
        current_pop.generation + 1, nothing, nothing, Dict()
    )

    return next_gen_pop
end

end
