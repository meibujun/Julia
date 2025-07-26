# Population simulation module

module PopulationSimulation

using Random
using Distributions
using StatsBase
using DataFrames
using ProgressMeter
using ..CoreTypes
using ..Constants

export simulate_population

function simulate_population(params::SimulationParameters)

    @info "Simulating population with complex architecture..."

    # --- Base Population ---
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

    # --- Genetic Architecture ---
    arch = params.architecture
    qtl_indices = sample(1:params.n_markers, arch.n_qtl, replace=false)

    # Additive effects
    add_effects = randn(arch.n_qtl)

    # Dominance effects
    dom_effects = randn(arch.n_qtl)

    # Epistatic effects
    epi_pairs = sample(qtl_indices, (arch.n_epistatic_pairs, 2), replace=false)
    epi_effects = randn(arch.n_epistatic_pairs)

    # --- Phenotype Simulation ---
    true_additive = zeros(params.n_individuals)
    true_dominance = zeros(params.n_individuals)
    true_epistatic = zeros(params.n_individuals)

    for (i, q_idx) in enumerate(qtl_indices)
        g = genotypes[:, q_idx]
        p = allele_freq[q_idx]
        true_additive .+= (g .- 2p) .* add_effects[i]
        true_dominance .+= (g .== 1) .* dom_effects[i]
    end

    for i in 1:arch.n_epistatic_pairs
        g1 = genotypes[:, epi_pairs[i, 1]]
        g2 = genotypes[:, epi_pairs[i, 2]]
        true_epistatic .+= (g1 .- 1) .* (g2 .- 1) .* epi_effects[i]
    end

    # Scale to target heritabilities
    if var(true_additive) > 0 true_additive .*= sqrt(arch.h2_additive / var(true_additive)) end
    if var(true_dominance) > 0 true_dominance .*= sqrt(arch.h2_dominance / var(true_dominance)) end
    if var(true_epistatic) > 0 true_epistatic .*= sqrt(arch.h2_epistasis_aa / var(true_epistatic)) end

    true_genetic_value = true_additive + true_dominance + true_epistatic

    # Residuals
    total_h2 = arch.h2_additive + arch.h2_dominance + arch.h2_epistasis_aa
    var_g = var(true_genetic_value)
    var_e = var_g * (1 - total_h2) / total_h2
    residuals = randn(params.n_individuals) * sqrt(var_e)

    phenotypes = true_genetic_value + residuals

    population = Population(
        geno_data,
        reshape(phenotypes, :, 1),
        zeros(params.n_individuals, 0), 0,
        nothing, # Pedigree
        (additive=true_additive, dominance=true_dominance, epistatic=true_epistatic), # True values
        Dict()
    )

    return population
end

end
