module MainWorkflow

using ..DynamicEpistasisGBLUP: Simulation, GBLUP, GPUAcceleration, Plots, DataFrames, StatsBase

export run_full_workflow

"""
    run_full_workflow()

Runs the full simulation and analysis workflow.
"""
function run_full_workflow()
    # 1. Run the simulation
    println("Starting simulation...")
    simulation_results = Simulation.simulate_population()
    println("Simulation complete.")

    # 2. Analyze results for each generation
    accuracies_gblup = []
    accuracies_epistasis = []

    for gen in 0:10
        println("\n--- Analyzing Generation $gen ---")

        gen_data = simulation_results[gen]
        phenotypes = gen_data.phenotypes

        # This is a placeholder for getting genotypes. In a real scenario,
        # we would get the genotypes from the XSim population object.
        genotypes = Simulation.get_genotypes(gen_data.population)

        # 3. Run GBLUP models
        # Use GPU-accelerated functions if available
        use_gpu = true # Set to false to use CPU
        if use_gpu
            try
                G = GPUAcceleration.calculate_g_matrix_gpu(genotypes)
                G_AA = GPUAcceleration.calculate_gaa_matrix_gpu(genotypes)
            catch e
                println("GPU acceleration failed, falling back to CPU. Error: $e")
                G = GBLUP.calculate_g_matrix(genotypes)
                G_AA = GBLUP.calculate_gaa_matrix(genotypes)
            end
        else
            G = GBLUP.calculate_g_matrix(genotypes)
            G_AA = GBLUP.calculate_gaa_matrix(genotypes)
        end

        # Run the models (using placeholder solvers for now)
        gebv_gblup = GBLUP.run_gblup(phenotypes, genotypes)
        gebv_epistasis, u_epistasis = GBLUP.run_epistasis_gblup(phenotypes, genotypes)
        total_gebv_epistasis = gebv_epistasis + u_epistasis

        # 4. Evaluate model accuracy
        tbv = phenotypes.TBV
        total_genetic_value = phenotypes.TBV + phenotypes.epistatic_val

        acc_gblup = cor(tbv, gebv_gblup)
        acc_epistasis = cor(total_genetic_value, total_gebv_epistasis)

        push!(accuracies_gblup, acc_gblup)
        push!(accuracies_epistasis, acc_epistasis)

        println("Accuracy (GBLUP): $acc_gblup")
        println("Accuracy (Epistasis GBLUP): $acc_epistasis")
    end

    # 5. Plot results
    p = Plots.plot(0:10, accuracies_gblup, label="Additive GBLUP", marker=:circle)
    Plots.plot!(p, 0:10, accuracies_epistasis, label="Epistasis GBLUP", marker=:square)
    Plots.xlabel!(p, "Generation")
    Plots.ylabel!(p, "Prediction Accuracy")
    Plots.title!(p, "Genomic Prediction Accuracy Over Generations")
    Plots.savefig(p, "prediction_accuracy.png")
    println("\nPlot saved to prediction_accuracy.png")

end

end # module MainWorkflow
