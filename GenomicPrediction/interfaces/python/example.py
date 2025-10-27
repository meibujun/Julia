# example.py: Demonstrates calling GenomicPrediction.jl from Python using PyJulia

import os
import julia

print("Initializing PyJulia...")
# It's recommended to specify the Julia executable path if it's not in the system's PATH
# For this environment, the path is known.
julia.install(julia="/tmp/julia-1.11.6/bin/julia")
from julia import Main

print("Setting up Julia environment...")

# Define the project path relative to this script
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Using project path: {project_path}")

# Activate the Julia project
Main.eval(f'using Pkg; Pkg.activate("{project_path}")')

# Import the GenomicPrediction module
print("Importing GenomicPrediction.jl...")
Main.eval("using GenomicPrediction")

# Check if we can access a function
print("Successfully accessed GenomicPrediction.jl.")
print("---")

# --- Example Workflow ---
# This part of the script demonstrates a basic workflow.
# It requires sample data to be present. We will use the test data.

try:
    print("Running a sample workflow...")

    # Define paths to sample data
    geno_path = os.path.join(project_path, "test", "sample_data", "genotypes.csv")
    pheno_path = os.path.join(project_path, "test", "sample_data", "phenotypes.csv")

    # Load data using the Julia function
    print("Loading data...")
    data = Main.load_csv(geno_path, pheno_path)
    print("Data loaded successfully.")

    # Create a GBLUP model instance
    print("Creating GBLUP model...")
    model = Main.GBLUPModel(lambda_val=10.0)

    # Train the model using the non-mutating 'fit' wrapper
    print("Training model...")
    trained_model = Main.fit(model, data)
    print("Model trained successfully.")

    # Get the genotype data for prediction
    new_data = data.genotypes

    # Make predictions
    print("Making predictions...")
    predictions = Main.predict(trained_model, new_data)

    print("\n--- Results ---")
    print("Predictions:")
    print(predictions)
    print(f"Successfully made {len(predictions)} predictions.")

except Exception as e:
    print("\n--- An error occurred during the workflow ---")
    print("This is expected if the environment or paths are not perfectly configured,")
    print("or if there are subtle incompatibilities with PyJulia.")
    print("The primary goal of this script is to demonstrate the setup.")
    print("\nError details:")
    print(e)

print("\nPython script finished.")
