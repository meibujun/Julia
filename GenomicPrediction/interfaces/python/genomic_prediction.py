# interfaces/python/genomic_prediction.py

import julia
from julia import Main
import pandas as pd
import numpy as np
import os

class GBLUP:
    """
    A Python wrapper for the GenomicPrediction.jl GBLUPModel, designed to
    behave like a scikit-learn estimator.
    """
    def __init__(self, lambda_val=10.0, julia_project_path=None):
        """
        Initializes the GBLUP wrapper.

        Args:
            lambda_val (float): The regularization parameter for the GBLUP model.
            julia_project_path (str, optional): The absolute path to the
                GenomicPrediction.jl project directory. If None, it will be
                inferred relative to this file's location.
        """
        print("Initializing Julia runtime...")
        # Ensure Julia is available
        try:
            from julia import Main
        except ImportError:
            raise ImportError("PyJulia is not installed or configured. Please run `pip install pyjulia` and `julia.install()`.")

        self.lambda_val = lambda_val
        self._trained_model = None

        if julia_project_path is None:
            # Infer the project path as the root of the Git repository
            self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        else:
            self.project_path = julia_project_path

        print(f"Activating Julia project at: {self.project_path}")
        Main.eval(f'using Pkg; Pkg.activate("{self.project_path}")')

        print("Loading GenomicPrediction.jl...")
        Main.eval("using GenomicPrediction")
        print("Wrapper initialized successfully.")

    def fit(self, X, y):
        """
        Trains the GBLUP model.

        Args:
            X (pd.DataFrame): Genotype data (n_samples, n_features).
                              The first column must be the individual ID.
            y (pd.DataFrame): Phenotype data (n_samples, 2).
                              The columns must be 'ID' and 'y'.
        """
        print("Fitting model...")

        # Convert pandas DataFrames to Julia DataFrames
        Main.jl_geno_df = X
        Main.jl_pheno_df = y

        # Create GenomicData object in Julia
        Main.eval("py_data = GenomicData(jl_geno_df, jl_pheno_df)")

        # Create and train the model
        Main.lambda_val = self.lambda_val
        Main.eval("py_model = GBLUPModel(lambda=lambda_val)")

        print("Calling Julia's `fit` function...")
        self._trained_model = Main.fit(Main.py_model, Main.py_data)

        print("Model fitting complete.")
        return self

    def predict(self, X):
        """
        Makes predictions using the trained GBLUP model.

        Args:
            X (pd.DataFrame): Genotype data for prediction.

        Returns:
            np.ndarray: A numpy array of predicted values.
        """
        if self._trained_model is None:
            raise RuntimeError("The model has not been trained yet. Please call 'fit' first.")

        print("Making predictions...")

        Main.jl_new_geno_df = X

        predictions_jl = Main.predict(self._trained_model, Main.jl_new_geno_df)

        # Convert the Julia vector to a NumPy array
        return np.array(predictions_jl)

if __name__ == '__main__':
    print("--- Running GBLUP Wrapper Example ---")

    try:
        # 1. Initialize the wrapper
        gblup = GBLUP(lambda_val=50.0)

        # 2. Load sample data using pandas
        project_root = gblup.project_path
        geno_path = os.path.join(project_root, "test", "sample_data", "genotypes.csv")
        pheno_path = os.path.join(project_root, "test", "sample_data", "phenotypes.csv")

        geno_df = pd.read_csv(geno_path)
        pheno_df = pd.read_csv(pheno_path)
        # Ensure phenotype df has the correct column names for this example
        pheno_df.columns = ['ID', 'y', 'trait2']
        pheno_df = pheno_df[['ID', 'y']]

        # 3. Fit the model
        gblup.fit(geno_df, pheno_df)

        # 4. Make predictions
        predictions = gblup.predict(geno_df)

        print("\n--- Results ---")
        print(f"Successfully generated {len(predictions)} predictions.")
        print("First 5 predictions:", predictions[:5])

    except Exception as e:
        print("\nAn error occurred during the example workflow:")
        import traceback
        traceback.print_exc()
