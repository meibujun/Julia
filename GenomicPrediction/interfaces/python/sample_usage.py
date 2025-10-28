# interfaces/python/sample_usage.py
# -------------------------------------
#
# This script demonstrates how to call the GenomicPrediction.jl package from Python
# using the PyJulia library.

import julia
from julia import Pkg

def main():
    print("Initializing Julia runtime...")
    julia.Julia(compiled_modules=False)

    print("Activating Julia environment for GenomicPrediction.jl...")
    Pkg.activate("../../")

    print("Loading GenomicPrediction.jl package...")
    from julia import GenomicPrediction as GP

    # --- 1. Load Data ---
    print("\n--- Step 1: Loading Data ---")
    geno_path = "../sample_data/genotypes.csv"
    pheno_path = "../sample_data/phenotypes.csv"

    # Create dummy data for the example
    from julia import CSV, DataFrame
    CSV.write(geno_path, DataFrame(ID=range(1, 6), m1=[0,1,2,0,1], m2=[2,1,0,2,1]))
    CSV.write(pheno_path, DataFrame(ID=range(1, 6), y=[1.1, 1.9, 3.2, 1.2, 2.3]))

    mock_data = GP.load_csv(geno_path, pheno_path, header_geno=True, header_pheno=True)
    print("Data loaded successfully.")

    # --- 2. Initialize and Train a Model ---
    print("\n--- Step 2: Training GBLUP Model ---")
    model = GP.GBLUPModel(10.0)
    GP.fit_b(model, mock_data) # fit! is aliased to fit_b
    print("Model training complete.")

    # --- 3. Save and Load Model ---
    print("\n--- Step 3: Saving and Reloading Model ---")
    temp_path = "temp_model.bson"
    GP.save_model(model, temp_path)
    print(f"Model saved to {temp_path}")
    loaded_model = GP.load_model(temp_path)
    print("Model successfully reloaded.")

    # --- 4. Make Predictions ---
    print("\n--- Step 4: Making Predictions ---")
    genotypes_df = mock_data.genotypes
    predictions = GP.predict(loaded_model, genotypes_df)

    print("Predictions obtained.")

    # --- 5. Display Results ---
    print("\n--- Step 5: Results ---")
    predictions_list = list(predictions)
    print(f"Predicted values: {predictions_list}")

if __name__ == "__main__":
    main()
