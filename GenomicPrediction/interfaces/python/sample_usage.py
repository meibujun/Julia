# interfaces/python/sample_usage.py
# -------------------------------------
#
# This script demonstrates how to call the GenomicPrediction.jl package from Python
# using the PyJulia library.
#
# Installation:
# 1. Make sure you have Julia installed and accessible in your PATH.
# 2. Install PyJulia: `pip install julia`
# 3. From Python, configure PyJulia for your environment:
#    >>> import julia
#    >>> julia.install()

import julia
from julia import Pkg

def main():
    """
    A complete workflow demonstrating calling Julia from Python.
    """
    print("Initializing Julia runtime...")
    # Initialize Julia. This will take a moment as it loads the environment.
    # By default, it will use the environment in the parent directory of this script.
    julia.Julia(compiled_modules=False)

    print("Activating Julia environment for GenomicPrediction.jl...")
    # Assumes the script is run from the `GenomicPrediction/interfaces/python` directory
    Pkg.activate("../../")

    print("Loading GenomicPrediction.jl package...")
    from julia import GenomicPrediction as GP

    # --- 1. Load Data ---
    print("\n--- Step 1: Loading Data ---")
    geno_path = "../sample_data/genotypes.csv"
    pheno_path = "../sample_data/phenotypes.csv"

    # Call the Julia function `load_csv`
    mock_data = GP.load_csv(geno_path, pheno_path, header_geno=False, header_pheno=True)
    print("Data loaded successfully into a GenomicData object.")

    # --- 2. Initialize and Train a Model ---
    print("\n--- Step 2: Training GBLUP Model ---")
    # Initialize a GBLUPModel with a lambda of 10.0
    model = GP.GBLUPModel(10.0)

    # Train the model (this modifies the model in-place)
    GP.fit_b(model, mock_data) # fit! is aliased to fit_b in PyJulia
    print("Model training complete.")

    # --- 3. Make Predictions ---
    print("\n--- Step 3: Making Predictions ---")
    # We'll predict on the same data we trained on for this example
    genotypes_df = mock_data.genotypes
    predictions = GP.predict(model, genotypes_df)

    print("Predictions obtained.")

    # --- 4. Display Results ---
    print("\n--- Step 4: Results ---")
    # The result from Julia is a Julia vector. We can convert it to a Python list.
    predictions_list = list(predictions)

    print(f"Number of predictions: {len(predictions_list)}")
    print("Predicted values:")
    for i, p in enumerate(predictions_list):
        print(f"  Individual {i+1}: {p:.4f}")

if __name__ == "__main__":
    main()
