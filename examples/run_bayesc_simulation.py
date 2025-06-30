import numpy as np
import pandas as pd
import sys
import os

# Adjust path to import pyjwas
try:
    from pyjwas.core import MixedModelEquations, VarianceCovariance, MCMCInfo, GenotypesData
    from pyjwas.core.model_builder import build_model, get_mme_components
    from pyjwas.mcmc import run_mcmc
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from pyjwas.core import MixedModelEquations, VarianceCovariance, MCMCInfo, GenotypesData
    from pyjwas.core.model_builder import build_model, get_mme_components
    from pyjwas.mcmc import run_mcmc


def simulate_bayesc_data(n_obs=300, n_markers=500, intercept_true=5.0,
                         true_pi_param=0.1, # P(effect != 0)
                         common_marker_var_true=0.02,
                         residual_var_true=1.0):
    """Simulates data for a BayesC model."""
    print(f"Simulating data for BayesC: {n_obs} obs, {n_markers} markers, true Pi={true_pi_param}...")
    sim_Z = np.random.randn(n_obs, n_markers)
    sim_Z_std = sim_Z.std(axis=0); sim_Z_std[sim_Z_std == 0] = 1.0
    sim_Z = (sim_Z - sim_Z.mean(axis=0)) / sim_Z_std
    sim_Z = np.nan_to_num(sim_Z, nan=0.0)

    # True delta (inclusion indicators)
    true_delta = (np.random.rand(n_markers) < true_pi_param).astype(float)
    n_true_qtl = int(np.sum(true_delta))
    print(f"  Number of true QTLs (non-zero effects): {n_true_qtl} out of {n_markers}")

    # True marker effects (only for included markers)
    true_alphas_underlying = np.random.randn(n_markers) * np.sqrt(common_marker_var_true)
    true_alphas = true_alphas_underlying * true_delta

    y_genetic = sim_Z @ true_alphas
    phenotypes = y_genetic + np.random.randn(n_obs) * np.sqrt(residual_var_true) + intercept_true

    pheno_df = pd.DataFrame({'pheno_y': phenotypes})
    pheno_df['obs_id'] = [f"obs_{i}" for i in range(n_obs)]
    pheno_df.set_index('obs_id', inplace=True)

    geno_obs_ids = pheno_df.index.tolist()
    marker_ids = [f"mkr_{j}" for j in range(n_markers)]

    print("BayesC data simulation complete.")
    return pheno_df, sim_Z, geno_obs_ids, marker_ids, \
           intercept_true, residual_var_true, common_marker_var_true, true_pi_param, true_alphas, true_delta


def main():
    print("--- PyJWAS BayesC Simulation Example ---")

    # 1. Simulate Data
    n_obs, n_markers = 400, 1000
    pheno_df, Z_geno_matrix, geno_obs_ids, marker_ids, \
    intercept_true, res_var_true, marker_var_true, true_pi, true_alphas, true_delta = \
        simulate_bayesc_data(n_obs, n_markers, true_pi_param=0.05, common_marker_var_true=0.01)

    # 2. Setup GenotypesData
    bayesc_geno_data = GenotypesData(
        name="sim_markers_bayesc",
        method="BayesC",
        genotype_matrix=Z_geno_matrix,
        obs_ids=geno_obs_ids,
        marker_ids=marker_ids,
        is_centered=True,
        pi_value=0.05, # Initial guess for P(effect != 0)
        estimate_pi=True
    )
    bayesc_geno_data.marker_effect_variance = VarianceCovariance(
        value=0.005, df=4.0, scale=0.002, estimate_variance=True # Prior for common sigma_g^2
    )
    # Priors for pi (Beta distribution for P(effect!=0) ) can be set on geno_data if attributes exist
    # e.g., bayesc_geno_data.pi_prior_alpha = 1.0; bayesc_geno_data.pi_prior_beta = 1.0 (for Beta(1,1) uniform)
    # The sampler _sample_marker_effects_bayesc_st currently uses default (1,1) if not passed.
    print(f"GenotypesData '{bayesc_geno_data.name}' configured for BayesC.")

    # 3. Build Model
    model_equation = "pheno_y = intercept + sim_markers_bayesc"
    R_vc_obj = VarianceCovariance(value=1.0, df=4.0, scale=0.5, estimate_variance=True)
    mme = build_model(
        model_equations_str=model_equation, R_value=R_vc_obj.value, R_df=R_vc_obj.df,
        genotypes_data_list=[bayesc_geno_data])
    mme.R_variance = R_vc_obj
    print(f"Model built: {mme.model_equations_str}")

    # 4. Set MCMC Parameters
    mme.mcmc_info = MCMCInfo(
        chain_length=8000, burnin=2000,
        output_samples_frequency=100, printout_frequency=1000, seed=67890
    )
    print(f"MCMC parameters set: Chain={mme.mcmc_info.chain_length}, Burn-in={mme.mcmc_info.burnin}")

    # 5. Construct MME Components
    try:
        print("Building MME components...")
        get_mme_components(mme, pheno_df)
    except Exception as e:
        print(f"Error during MME component building: {e}"); traceback.print_exc(); return

    # 6. Run MCMC
    print("\nStarting BayesC MCMC run...")
    try:
        results = run_mcmc(mme, pheno_df)
        print("\nBayesC MCMC run completed.")

        # 7. Analyze Results
        print("\n--- BayesC Results ---")
        mean_intercept_est = results.get("mean_solutions")[0] if results.get("mean_solutions") is not None else None
        print(f"  True Intercept: {intercept_true:.4f}")
        print(f"  Estimated Mean Intercept: {mean_intercept_est:.4f}")

        mean_res_var_est = results.get("mean_residual_variance")
        print(f"  True Residual Variance (sigma_e^2): {res_var_true:.4f}")
        print(f"  Estimated Mean Residual Variance: {mean_res_var_est:.4f}")

        if results.get("genotypes_results"):
            geno_res = results["genotypes_results"][0]
            mean_marker_var_est = geno_res.get("mean_marker_variance")
            print(f"  True Common Marker Variance (sigma_g^2): {marker_var_true:.4f}")
            print(f"  Estimated Mean Common Marker Variance ({geno_res['name']}): {mean_marker_var_est:.4f}")

            mean_pi_est = geno_res.get("mean_pi")
            print(f"  True Pi (P(effect!=0)): {true_pi:.4f}")
            print(f"  Estimated Mean Pi: {mean_pi_est:.4f}")

            mean_alpha_est = geno_res.get("mean_alpha")[0] # Single trait
            if mean_alpha_est is not None and len(true_alphas) == len(mean_alpha_est):
                correlation_alpha = np.corrcoef(true_alphas, mean_alpha_est)[0,1]
                print(f"  Correlation (True Alpha vs. Estimated Mean Alpha): {correlation_alpha:.4f}")

            mean_delta_est = geno_res.get("mean_delta")[0] # Single trait, posterior prob of inclusion
            if mean_delta_est is not None:
                prop_incl_true = np.mean(true_delta)
                prop_incl_est = np.mean(mean_delta_est > 0.5) # Proportion of markers with P(incl) > 0.5
                avg_ppi = np.mean(mean_delta_est)
                print(f"  True proportion of included markers: {prop_incl_true:.4f}")
                print(f"  Estimated proportion (PPI > 0.5): {prop_incl_est:.4f}")
                print(f"  Average Posterior Prob Inclusion (PPI): {avg_ppi:.4f}")
        else:
            print("  No genotype results found.")

    except Exception as e:
        print(f"An error occurred during BayesC MCMC run or result processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import traceback # To see full trace for errors in __main__
    main()
```
