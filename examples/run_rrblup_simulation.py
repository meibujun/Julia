import numpy as np
import pandas as pd
import sys
import os

# Adjust path to import pyjwas (if running example from outside project root)
# This assumes the example is run from a context where pyjwas is discoverable
# For example, from the root directory of the project: python examples/run_rrblup_simulation.py
# Or if pyjwas is installed.
try:
    from pyjwas.core import MixedModelEquations, ModelTerm, VarianceCovariance, MCMCInfo, GenotypesData
    from pyjwas.core.model_builder import build_model, get_mme_components # For full MME setup
    from pyjwas.genotypes_io import get_genotypes_data # If reading from file/df
    from pyjwas.mcmc import run_mcmc
except ImportError:
    # Simple fallback for running directly from examples directory relative to pyjwas
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from pyjwas.core import MixedModelEquations, ModelTerm, VarianceCovariance, MCMCInfo, GenotypesData
    from pyjwas.core.model_builder import build_model, get_mme_components
    from pyjwas.genotypes_io import get_genotypes_data # If using it
    from pyjwas.mcmc import run_mcmc


def simulate_data(n_obs=200, n_markers=100, intercept_true=10.0,
                  marker_var_true=0.05, residual_var_true=1.0):
    """Simulates simple phenotypic and genotypic data."""
    print(f"Simulating data: {n_obs} observations, {n_markers} markers...")
    # Genotypes (already centered and scaled for simplicity in this example)
    sim_Z = np.random.randn(n_obs, n_markers)
    # Normalize Z columns (optional, but helps in interpretation of marker variance)
    sim_Z_std = sim_Z.std(axis=0)
    sim_Z_std[sim_Z_std == 0] = 1.0 # Avoid division by zero for non-varying markers
    sim_Z = (sim_Z - sim_Z.mean(axis=0)) / sim_Z_std
    sim_Z = np.nan_to_num(sim_Z, nan=0.0)


    # True marker effects
    true_alphas = np.random.randn(n_markers) * np.sqrt(marker_var_true)

    # Genetic values
    y_genetic = sim_Z @ true_alphas

    # Phenotypes
    phenotypes = y_genetic + np.random.randn(n_obs) * np.sqrt(residual_var_true) + intercept_true

    pheno_df = pd.DataFrame({'pheno_y': phenotypes})
    # Add observation IDs (simple string IDs)
    pheno_df['obs_id'] = [f"obs_{i}" for i in range(n_obs)]
    pheno_df.set_index('obs_id', inplace=True) # Important for MME setup if IDs are from index

    # Genotype IDs
    geno_obs_ids = pheno_df.index.tolist()
    marker_ids = [f"mkr_{j}" for j in range(n_markers)]

    print("Data simulation complete.")
    return pheno_df, sim_Z, geno_obs_ids, marker_ids, \
           intercept_true, residual_var_true, marker_var_true, true_alphas


def main():
    print("--- PyJWAS RR-BLUP Simulation Example ---")

    # 1. Simulate Data
    n_obs, n_markers = 300, 200
    pheno_df, Z_geno_matrix, geno_obs_ids, marker_ids, \
    intercept_true, res_var_true, marker_var_true, true_alphas = \
        simulate_data(n_obs, n_markers)

    # 2. Setup GenotypesData object
    # For RR-BLUP, we provide the genotype matrix directly.
    # get_genotypes_data could be used if reading from file, but here we have the matrix.
    rrblup_geno_data = GenotypesData(
        name="sim_markers_rrblup",
        method="RR-BLUP", # or "BayesC0"
        genotype_matrix=Z_geno_matrix,
        obs_ids=geno_obs_ids,
        marker_ids=marker_ids,
        is_centered=True # Assuming our simulated Z is already centered
    )
    # Set up prior for common marker variance (sigma_g^2)
    # Value is initial guess, scale is S0^2 for prior ScaledInvChi2(df, S0^2)
    rrblup_geno_data.marker_effect_variance = VarianceCovariance(
        value=0.01, df=4.0, scale=0.005, estimate_variance=True
    )
    print(f"GenotypesData '{rrblup_geno_data.name}' configured for RR-BLUP.")

    # 3. Build the Model using pyjwas.core.model_builder
    model_equation = "pheno_y = intercept + sim_markers_rrblup"

    # Initial guess for residual variance (R) and its prior
    # Value is initial, scale is S0^2 for its prior ScaledInvChi2(df, S0^2)
    R_vc_obj = VarianceCovariance(value=1.0, df=4.0, scale=0.5, estimate_variance=True)

    mme = build_model(
        model_equations_str=model_equation,
        R_value=R_vc_obj.value, # Pass initial R value
        R_df=R_vc_obj.df,       # Pass R prior df
        genotypes_data_list=[rrblup_geno_data] # Associate the GenotypesData
    )
    # Ensure the full R_vc object is on mme for prior scale etc.
    mme.R_variance = R_vc_obj

    print(f"Model built: {mme.model_equations_str}")
    print(f"  Fixed effects terms: {[mt.trm_str for mt in mme.model_terms]}")
    print(f"  Genotype sets: {[gd.name for gd in mme.genotypes_data_list]}")


    # 4. Set MCMC Parameters
    mme.mcmc_info = MCMCInfo(
        chain_length=5000,  # Advise more for real run, e.g., 20k-50k
        burnin=1000,        # e.g., 5k-10k
        output_samples_frequency=100, # How often to save full samples (if implemented)
        printout_frequency=500,     # How often to print progress summary
        seed=12345
    )
    print(f"MCMC parameters set: Chain={mme.mcmc_info.chain_length}, Burn-in={mme.mcmc_info.burnin}")

    # 5. Construct MME Components (X, y, LHS, RHS)
    # This step uses the phenotype data (df_pheno) and model definition in mme.
    # It populates mme.X, mme.y_sparse, mme.mme_LHS, mme.mme_RHS, etc.
    # The `get_mme_components` function expects the MME object to have its
    # `model_terms` (for fixed/iid random) and `covariate_variables` set up.
    # It also needs `mme.R_variance.value` for initial R_inv.

    # In this example, 'sim_markers_rrblup' is a genotype term, not a covariate.
    # 'intercept' is the only fixed effect.

    # get_mme_components will build X for 'intercept'.
    # The initial mme.mme_LHS and mme.mme_RHS will be based on this X and initial R_inv.
    # The marker effects are handled separately by the marker sampler using Z_geno_matrix.
    try:
        print("Building MME components...")
        get_mme_components(mme, pheno_df) # df_pheno here is the source of 'pheno_y'
        print(f"  MME LHS shape: {mme.mme_LHS.shape if mme.mme_LHS is not None else 'None'}")
        print(f"  MME RHS shape: {mme.mme_RHS.shape if mme.mme_RHS is not None else 'None'}")
    except Exception as e:
        print(f"Error during MME component building: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Run MCMC
    print("\nStarting MCMC run...")
    try:
        results = run_mcmc(mme, pheno_df) # df_pheno passed for _make_Ri_matrix if MT R changes
        print("\nMCMC run completed.")

        # 7. Analyze Results
        print("\n--- Results ---")
        mean_intercept_est = results.get("mean_solutions")[0] if results.get("mean_solutions") is not None else None
        print(f"  True Intercept: {intercept_true:.4f}")
        print(f"  Estimated Mean Intercept: {mean_intercept_est:.4f}")

        mean_res_var_est = results.get("mean_residual_variance")
        print(f"  True Residual Variance (sigma_e^2): {res_var_true:.4f}")
        print(f"  Estimated Mean Residual Variance: {mean_res_var_est:.4f}")

        if results.get("genotypes_results"):
            geno_res = results["genotypes_results"][0] # Assuming one genotype set
            mean_marker_var_est = geno_res.get("mean_marker_variance")
            print(f"  True Marker Variance (sigma_g^2): {marker_var_true:.4f}")
            print(f"  Estimated Mean Marker Variance ({geno_res['name']}): {mean_marker_var_est:.4f}")

            mean_alpha_est = geno_res.get("mean_alpha")[0] # Single trait
            if mean_alpha_est is not None and len(true_alphas) == len(mean_alpha_est):
                correlation_alpha = np.corrcoef(true_alphas, mean_alpha_est)[0,1]
                print(f"  Correlation (True Alpha vs. Estimated Mean Alpha): {correlation_alpha:.4f}")
            else:
                print("  Could not calculate alpha correlation.")
        else:
            print("  No genotype results found.")

    except Exception as e:
        print(f"An error occurred during MCMC run or result processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```
