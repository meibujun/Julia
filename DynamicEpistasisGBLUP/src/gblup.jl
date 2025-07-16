module GBLUP

using ..DynamicEpistasisGBLUP: LinearAlgebra, StatsBase, JWAS, DataFrames

export run_gblup, run_epistasis_gblup, calculate_g_matrix, calculate_gaa_matrix

"""
    calculate_g_matrix(genotypes::Matrix)

Calculates the additive genomic relationship matrix (G) using the VanRaden (2008) method 1.

# Arguments
- `genotypes::Matrix`: A matrix of genotypes (n_individuals x n_markers), coded as 0, 1, 2.

# Returns
- `Matrix`: The additive genomic relationship matrix (G).
"""
function calculate_g_matrix(genotypes::Matrix)
    n_individuals, n_markers = size(genotypes)

    # Calculate allele frequencies (p_j) for each marker
    p = vec(mean(genotypes, dims=1) ./ 2)

    # Center the genotype matrix W = X - 2p
    W = genotypes .- 2 .* p'

    # Scale W by 2 * p_j * (1 - p_j)
    # The denominator is sum over j of 2 * p_j * (1 - p_j)
    denominator = sum(2 .* p .* (1 .- p))

    # Calculate G
    G = (W * W') / denominator

    return G
end

"""
    calculate_gaa_matrix(genotypes::Matrix)

Calculates the epistatic genomic relationship matrix (G_AA) using the Hadamard product
of the additive genomic relationship matrix with itself (G .* G).

# Arguments
- `genotypes::Matrix`: A matrix of genotypes (n_individuals x n_markers), coded as 0, 1, 2.

# Returns
- `Matrix`: The epistatic genomic relationship matrix (G_AA).
"""
function calculate_gaa_matrix(genotypes::Matrix)
    # First, calculate the additive G matrix
    G = calculate_g_matrix(genotypes)

    # The epistatic relationship matrix is the Hadamard product of G with itself
    G_AA = G .* G

    return G_AA
end

"""
    run_gblup(phenotypes::DataFrames.DataFrame, genotypes::Matrix)

Runs the standard additive GBLUP model using `JWAS.jl`.

This function takes phenotype and genotype data, calculates the additive genomic
relationship matrix (G), and then uses `JWAS.jl` to solve the mixed-model
equations to estimate genomic breeding values (GEBVs).

# Arguments
- `phenotypes::DataFrames.DataFrame`: A DataFrame with at least `:ID` and `:phenotype` columns.
- `genotypes::Matrix`: A matrix of genotypes (n_individuals x n_markers).

# Returns
- `Vector`: A vector of genomic estimated breeding values (GEBVs).
"""
function run_gblup(phenotypes::DataFrames.DataFrame, genotypes::Matrix)
    println("Running standard GBLUP with JWAS...")

    # 1. Prepare data for JWAS
    # The model equation defines the fixed and random effects.
    # "phenotype = 1 + animal" means phenotype is the response variable,
    # "1" is the overall mean (fixed effect), and "animal" is a random effect.
    model_equation = "phenotype = 1 + animal"
    data = phenotypes[:, [:ID, :phenotype]]
    data.animal = data.ID # Link the random effect to the individual IDs

    # 2. Calculate the G matrix
    G = calculate_g_matrix(genotypes)

    # 3. Set up the model in JWAS
    model = build_model(model_equation)
    # Associate the "animal" random effect with the G matrix
    set_random(model, "animal", G)

    # 4. Run the analysis using REML
    output = run_analysis(model, data, quiet=true)

    # 5. Extract the solutions (GEBVs) for the "animal" effect
    gebv = solutions(output)["animal"][!,:solution]

    return gebv
end

"""
    run_epistasis_gblup(phenotypes::DataFrames.DataFrame, genotypes::Matrix)

Runs the dynamic orthogonal epistasis GBLUP model using `JWAS.jl`.

This function models both additive and additive-by-additive epistatic effects.
It calculates both the additive (G) and epistatic (G_AA) relationship matrices
and fits them as separate random effects in a mixed model.

# Arguments
- `phenotypes::DataFrames.DataFrame`: A DataFrame with at least `:ID` and `:phenotype` columns.
- `genotypes::Matrix`: A matrix of genotypes (n_individuals x n_markers).

# Returns
- `Tuple{Vector, Vector}`: A tuple containing the additive GEBVs (g) and the epistatic values (u).
"""
function run_epistasis_gblup(phenotypes::DataFrames.DataFrame, genotypes::Matrix)
    println("Running orthogonal epistasis GBLUP with JWAS...")

    # 1. Prepare data for JWAS
    # The model includes two random effects: "additive" and "epistatic".
    model_equation = "phenotype = 1 + additive + epistatic"
    data = phenotypes[:, [:ID, :phenotype]]
    data.additive = data.ID # Link the additive effect to individuals
    data.epistatic = data.ID # Link the epistatic effect to individuals

    # 2. Calculate the G and G_AA matrices
    G = calculate_g_matrix(genotypes)
    G_AA = calculate_gaa_matrix(genotypes)

    # 3. Set up the model in JWAS
    model = build_model(model_equation)
    # Associate the "additive" effect with G and the "epistatic" effect with G_AA
    set_random(model, "additive", G)
    set_random(model, "epistatic", G_AA)

    # 4. Run the analysis
    output = run_analysis(model, data, quiet=true)

    # 5. Extract solutions for both random effects
    g = solutions(output)["additive"][!,:solution]
    u = solutions(output)["epistatic"][!,:solution]

    return g, u
end


end # module GBLUP
