# examples/scripts/comprehensive_workflow.jl
# ============================================
#
# This script demonstrates a comprehensive, end-to-end workflow using the
# GenomicPrediction.jl package. It covers:
#   1. Loading sample data.
#   2. Initializing multiple models (GBLUP and BayesA).
#   3. Running k-fold cross-validation to compare model performance.
#   4. Performing a grid search to find the best hyperparameter for GBLUP.
#   5. Training a final model on all data.
#   6. Saving the final model for future use.

using GenomicPrediction
using DataFrames
using Random

function main()
    println("--- Starting Comprehensive Workflow Example ---")

    # --- 1. Load Data ---
    # For this example, we'll create some dummy data.
    println("\n[1/5] Creating sample data...")
    Random.seed!(123)
    geno_df = DataFrame(ID=1:100, rand(0:2, 100, 50), :auto)
    pheno_df = DataFrame(ID=1:100, y=rand(100))
    data = GenomicData(geno_df, pheno_df)
    println("Data created successfully.")

    # --- 2. Compare Models with Cross-Validation ---
    println("\n[2/5] Comparing GBLUP and BayesA with 3-fold cross-validation...")

    gblup_generator() = GBLUPModel(10.0)
    bayes_a_generator() = BayesAModel(iterations=500)

    cv_results_gblup = cross_validate(gblup_generator, data, 3, fit!, predict)
    cv_results_bayes_a = cross_validate(bayes_a_generator, data, 3, fit!, predict)

    println("\nCross-Validation Results:")
    println("  GBLUP Mean Accuracy: ", cv_results_gblup.mean_accuracy)
    println("  BayesA Mean Accuracy: ", cv_results_bayes_a.mean_accuracy)

    # --- 3. Find Best Hyperparameter with Grid Search ---
    println("\n[3/5] Performing grid search for GBLUP lambda...")

    gblup_hp_generator(params) = GBLUPModel(params[:lambda])
    hyperparameters = Dict(:lambda => [0.1, 1.0, 10.0, 100.0])

    search_result = grid_search(gblup_hp_generator, data, hyperparameters, cross_validate, k=3)

    println("\nGrid Search Results:")
    println("  Best lambda: ", search_result.best_params[:lambda])
    println("  Best validation accuracy: ", search_result.best_score)

    best_lambda = search_result.best_params[:lambda]

    # --- 4. Train Final Model ---
    println("\n[4/5] Training final GBLUP model with best lambda...")
    final_model = GBLUPModel(best_lambda)
    fit!(final_model, data)
    println("Final model training complete.")

    # --- 5. Save Final Model ---
    println("\n[5/5] Saving final model...")
    model_path = "final_gblup_model.bson"
    save_model(final_model, model_path)
    println("Model saved to: ", model_path)

    println("\n--- Workflow Finished ---")
end

main()
