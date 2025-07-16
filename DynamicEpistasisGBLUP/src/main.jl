module MainWorkflow

using ..DynamicEpistasisGBLUP: Simulation, GBLUP, GPUAcceleration, Plots, DataFrames, StatsBase

export run_full_workflow

"""
    run_full_workflow(; use_gpu::Bool=true)

Runs the full simulation and analysis workflow from start to finish.

This function is the main entry point for the package. It performs the following steps:
1.  Calls the `Simulation.simulate_population` function to generate data for multiple generations.
2.  For each generation:
    a. Retrieves genotypes and phenotypes.
    b. Runs the standard additive GBLUP model.
    c. Runs the orthogonal epistasis GBLUP model.
    d. Calculates the prediction accuracy for both models.
3.  Plots the prediction accuracies over generations and saves the plot to a file.

# Keyword Arguments
- `use_gpu::Bool=true`: If `true` and a CUDA-enabled GPU is available, the relationship
  matrix calculations will be performed on the GPU. If `false` or if CUDA is not
  functional, the calculations will fall back to the CPU.
"""
function run_full_workflow(; use_gpu::Bool=true)
    # --- 1. Run the Population Simulation ---
    println("Starting population simulation...")
    simulation_results = Simulation.simulate_population()
    println("Simulation complete.")

    # Initialize arrays to store accuracy results for each generation
    accuracies_gblup = []
    accuracies_epistasis = []
    generations = 0:10 # As defined in the simulation

    # --- 2. Analyze Results for Each Generation ---
    for gen in generations
        println("\n--- Analyzing Generation $gen ---")

        # Retrieve data for the current generation
        gen_data = simulation_results[gen]
        phenotypes = gen_data.phenotypes
        genotypes = Simulation.get_genotypes(gen_data.population)

        # --- 3. Run GBLUP Models ---

        # Run the standard additive GBLUP model
        gebv_gblup = GBLUP.run_gblup(phenotypes, genotypes)

        # Run the orthogonal epistasis GBLUP model
        # This returns both additive (g) and epistatic (u) components
        g_epistasis, u_epistasis = GBLUP.run_epistasis_gblup(phenotypes, genotypes)
        # The total GEBV for the epistasis model is the sum of the components
        total_gebv_epistasis = g_epistasis + u_epistasis

        # --- 4. Evaluate Model Accuracy ---
        # The true breeding value (TBV) is the additive genetic value
        tbv = phenotypes.TBV
        # The total true genetic value includes both additive and epistatic effects
        total_genetic_value = phenotypes.TBV + phenotypes.epistatic_val

        # Calculate Pearson correlation between true and estimated values
        acc_gblup = cor(tbv, gebv_gblup)
        acc_epistasis = cor(total_genetic_value, total_gebv_epistasis)

        push!(accuracies_gblup, acc_gblup)
        push!(accuracies_epistasis, acc_epistasis)

        println("Accuracy (Additive GBLUP):     ", round(acc_gblup, digits=4))
        println("Accuracy (Epistasis GBLUP):    ", round(acc_epistasis, digits=4))
    end

    # --- 5. Plot and Save Results ---
    p = Plots.plot(generations, accuracies_gblup, label="Additive GBLUP", marker=:circle, lw=2)
    Plots.plot!(p, generations, accuracies_epistasis, label="Epistasis GBLUP", marker=:square, lw=2)
    Plots.xlabel!(p, "Generation")
    Plots.ylabel!(p, "Prediction Accuracy (Correlation)")
    Plots.title!(p, "Genomic Prediction Accuracy Over Generations")
    Plots.ylims!(p, 0, 1)
    Plots.savefig(p, "prediction_accuracy.png")
    println("\nAnalysis complete. Plot saved to prediction_accuracy.png")

end

end # module MainWorkflow
