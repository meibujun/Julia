import unittest
import numpy as np
import pandas as pd
from scipy.sparse import diags, csc_matrix

from pyjwas.core import MixedModelEquations, ModelTerm, VarianceCovariance, MCMCInfo, GenotypesData
from pyjwas.core.model_builder import build_model, set_covariate, get_mme_components # For setup
from pyjwas.mcmc.engine import run_mcmc

class TestMCMCEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345) # For reproducibility

    def _setup_simple_mme_intercept_only(self, n_obs=100, intercept_true=5.0, residual_var_true=2.0):
        """Helper to set up a very simple MME: y = intercept."""
        test_df = pd.DataFrame({'y': np.random.randn(n_obs) * np.sqrt(residual_var_true) + intercept_true})

        R_vc = VarianceCovariance(value=1.0, df=4.0, scale=1.0, estimate_variance=True)
        mme_obj = build_model("y = intercept", R_value=R_vc.value, R_df=R_vc.df) # Pass initial R value
        mme_obj.R_variance = R_vc # Assign the full VC object

        mme_obj.mcmc_info = MCMCInfo(chain_length=200, burnin=50, output_samples_frequency=10, printout_frequency=500, seed=123)

        # Manually build MME components as get_mme_components would
        mme_obj.obs_ids = [str(i) for i in range(n_obs)]
        mme_obj.inverse_weights = np.ones(n_obs)

        # Populate terms (simplified from get_mme_components)
        intercept_term = mme_obj.model_term_dict["y:intercept"]
        intercept_term.data = ["intercept"] * n_obs
        intercept_term.val = np.ones(n_obs)
        intercept_term.names = ["intercept"]
        intercept_term.n_levels = 1
        intercept_term.start_pos = 0
        mme_obj.mme_pos_counter = 1

        X_intercept = np.ones((n_obs, 1))
        intercept_term.X = csc_matrix(X_intercept)
        mme_obj.X = intercept_term.X

        mme_obj.y_sparse = test_df['y'].values.reshape(-1,1)

        if mme_obj.X.shape[1] > 0 and mme_obj.R_variance.value is not None and float(mme_obj.R_variance.value) > 0:
            D_initial = diags(mme_obj.inverse_weights / float(mme_obj.R_variance.value), format="csc")
            mme_obj.mme_LHS = mme_obj.X.T @ D_initial @ mme_obj.X
            mme_obj.mme_RHS = mme_obj.X.T @ D_initial @ mme_obj.y_sparse
            # Store base X'R_invX for _update_mme_lhs_for_solver if it needs to reconstruct Lambda part
            mme_obj.mme_LHS_base_X_Rinv_X = mme_obj.mme_LHS.copy()
        else:
            mme_obj.mme_LHS = csc_matrix((1,1)); mme_obj.mme_RHS = np.zeros((1,1))
            mme_obj.mme_LHS_base_X_Rinv_X = mme_obj.mme_LHS.copy()

        mme_obj.solutions = np.zeros(mme_obj.mme_LHS.shape[0])
        mme_obj.mean_solutions = np.zeros_like(mme_obj.solutions)
        mme_obj.mean_solutions_sq = np.zeros_like(mme_obj.solutions)
        if mme_obj.R_variance.value is not None:
            mme_obj.mean_residual_variance = 0.0
            mme_obj.mean_residual_variance_sq = 0.0

        return mme_obj, test_df, intercept_true, residual_var_true

    def test_run_mcmc_intercept_only_model(self):
        """Test MCMC for a very simple 'y = intercept' model."""
        mme, df, intercept_true, res_var_true = self._setup_simple_mme_intercept_only(n_obs=100)
        mme.mcmc_info.chain_length = 2000
        mme.mcmc_info.burnin = 500
        mme.mcmc_info.printout_frequency = 2001 # Suppress print during test

        results = run_mcmc(mme, df)

        self.assertIn("mean_solutions", results)
        self.assertIn("mean_residual_variance", results)

        # Check if intercept is recovered reasonably
        # Posterior mean of intercept should be close to sample mean of y
        # which should be close to intercept_true
        self.assertAlmostEqual(results["mean_solutions"][0], np.mean(df['y']), delta=0.5)
        self.assertAlmostEqual(results["mean_solutions"][0], intercept_true, delta=0.7) # Wider delta due to sampling noise + prior effect

        # Check if residual variance is recovered reasonably
        self.assertGreater(results["mean_residual_variance"], 0)
        self.assertLess(abs(results["mean_residual_variance"] - res_var_true) / res_var_true, 0.7, "Residual variance not recovered well.")


    def _setup_rrblup_model(self, n_obs=100, n_markers=50, intercept_true=2.0, marker_var_true=0.05, res_var_true=1.0):
        """Helper to set up MME for y = intercept + markers (RR-BLUP)."""
        sim_Z = np.random.randn(n_obs, n_markers) # Already somewhat centered/scaled
        true_alphas = np.random.randn(n_markers) * np.sqrt(marker_var_true)
        y_genetic = sim_Z @ true_alphas
        test_df = pd.DataFrame({'y': y_genetic + np.random.randn(n_obs) * np.sqrt(res_var_true) + intercept_true})

        R_vc = VarianceCovariance(value=res_var_true*0.8, df=4.0, scale=res_var_true*0.8, estimate_variance=True) # Start R near true
        mme_obj = build_model("y = intercept + all_markers", R_value=R_vc.value, R_df=R_vc.df)
        mme_obj.R_variance = R_vc

        geno1 = GenotypesData(name="all_markers", method="RR-BLUP")
        geno1.genotypes = sim_Z
        geno1.n_markers = n_markers
        geno1.n_obs = n_obs
        geno1.obs_ids = [f"id_{i}" for i in range(n_obs)]
        geno1.marker_ids = [f"m_{j}" for j in range(n_markers)]
        geno1.marker_effect_variance = VarianceCovariance(value=marker_var_true*0.5, df=4.0, scale=marker_var_true*0.5*0.5, estimate_variance=True)
        geno1.initialize_mcmc_storage(mme_obj.n_models, n_markers) # Initialize alpha_samples etc.

        mme_obj.genotypes_data_list.append(geno1)

        mme_obj.mcmc_info = MCMCInfo(chain_length=1000, burnin=200, output_samples_frequency=100, printout_frequency=1001, seed=42)

        # Build MME components (simplified)
        mme_obj.obs_ids = [str(i) for i in range(n_obs)]
        mme_obj.inverse_weights = np.ones(n_obs)

        intercept_term = mme_obj.model_term_dict["y:intercept"] # Only intercept in fixed part
        from pyjwas.core.model_builder import _get_data_for_term, _get_incidence_matrix_for_term
        _get_data_for_term(intercept_term, test_df, mme_obj)
        _get_incidence_matrix_for_term(intercept_term, mme_obj, n_obs)

        mme_obj.X = intercept_term.X
        mme_obj.y_sparse = test_df['y'].values.reshape(-1,1)

        if mme_obj.X.shape[1] > 0 and mme_obj.R_variance.value is not None and float(mme_obj.R_variance.value) > 0:
            D_initial = diags(mme_obj.inverse_weights / float(mme_obj.R_variance.value), format="csc")
            mme_obj.mme_LHS_base_X_Rinv_X = mme_obj.X.T @ D_initial @ mme_obj.X
            mme_obj.mme_LHS = mme_obj.mme_LHS_base_X_Rinv_X.copy()
            mme_obj.mme_RHS = mme_obj.X.T @ D_initial @ mme_obj.y_sparse
        else:
            mme_obj.mme_LHS = csc_matrix((1,1)); mme_obj.mme_RHS = np.zeros((1,1))
            mme_obj.mme_LHS_base_X_Rinv_X = mme_obj.mme_LHS.copy()

        mme_obj.solutions = np.zeros(mme_obj.mme_LHS.shape[0])
        # Initialize mean storage
        mme_obj.mean_solutions = np.zeros_like(mme_obj.solutions)
        mme_obj.mean_solutions_sq = np.zeros_like(mme_obj.solutions)
        mme_obj.mean_residual_variance = 0.0
        mme_obj.mean_residual_variance_sq = 0.0
        # Geno data mean storage already init by initialize_mcmc_storage

        return mme_obj, test_df, intercept_true, res_var_true, marker_var_true, true_alphas


    def test_run_mcmc_rrblup_model(self):
        """Test MCMC for a simple RR-BLUP model."""
        mme, df, intercept_true, res_var_true, marker_var_true, true_alphas = self._setup_rrblup_model(n_obs=200, n_markers=100)
        mme.mcmc_info.chain_length=3000 # Longer chain for better estimates
        mme.mcmc_info.burnin=1000
        mme.mcmc_info.printout_frequency = 3001 # Suppress print

        results = run_mcmc(mme, df)

        self.assertIn("mean_solutions", results)
        self.assertIn("mean_residual_variance", results)
        self.assertTrue(len(results["genotypes_results"]) > 0)
        geno_res = results["genotypes_results"][0]
        self.assertIn("mean_alpha", geno_res)
        self.assertIn("mean_marker_variance", geno_res)

        # Check intercept
        self.assertAlmostEqual(results["mean_solutions"][0], intercept_true, delta=0.8, msg="Intercept not recovered well.")

        # Check residual variance
        self.assertGreater(results["mean_residual_variance"], 0)
        self.assertLess(abs(results["mean_residual_variance"] - res_var_true) / res_var_true, 0.8, "Residual variance not recovered well.")

        # Check marker variance
        self.assertGreater(geno_res["mean_marker_variance"], 0)
        self.assertLess(abs(geno_res["mean_marker_variance"] - marker_var_true) / marker_var_true, 0.9, "Marker variance not recovered well.") # Wider delta for marker variance

        # Check correlation of estimated alpha with true alpha
        if geno_res["mean_alpha"] and len(geno_res["mean_alpha"]) > 0 and geno_res["mean_alpha"][0] is not None:
            mean_alpha_sampled = geno_res["mean_alpha"][0]
            if len(true_alphas) == len(mean_alpha_sampled):
                 correlation = np.corrcoef(true_alphas, mean_alpha_sampled)[0,1]
                 self.assertGreater(correlation, 0.3, "Correlation between true and estimated alpha is too low for RR-BLUP.") # Expect some positive correlation
            else:
                self.fail("Mismatch in length of true_alphas and sampled mean_alpha.")
        else:
            self.fail("Mean alpha not found in results.")

if __name__ == '__main__':
    unittest.main()
```
