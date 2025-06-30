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


    def test_run_mcmc_animal_model_simple(self):
        """Test MCMC for a simple animal model: y = mu + animal + e."""
        from pyjwas.pedigree import read_pedigree, calculate_A_inverse # Local import for test setup

        # 1. Setup Pedigree
        ped_content = """
        A1,0,0
        A2,0,0
        A3,A1,A2
        A4,A1,A2
        A5,A3,0
        A6,A3,A4
        """
        # Create a dummy pedigree file
        dummy_ped_path = "test_animal_model_ped.csv"
        with open(dummy_ped_path, "w") as f: f.write(ped_content.strip())

        ped_data = read_pedigree(dummy_ped_path)
        A_inv = calculate_A_inverse(ped_data)
        animal_ids_ordered = ped_data.get_ordered_ids_str()
        n_animals = len(animal_ids_ordered)

        # 2. Simulate Data
        intercept_true = 10.0
        sigma_a_true = 1.5  # Polygenic variance
        sigma_e_true = 3.0  # Residual variance

        # Simulate animal effects: u ~ N(0, A * sigma_a_true)
        # Need A matrix for simulation. A_inv is for estimation.
        # For simplicity, let's assume A = I for simulation of u, so u ~ N(0, I*sigma_a_true)
        # This is a simplification; true u should use full A.
        # A more correct simulation would sample from MVN(0, A*sigma_a_true).
        # For now, iid animal effects for test simplicity.
        true_animal_effects_map = {aid: np.random.randn() * np.sqrt(sigma_a_true) for aid in animal_ids_ordered}

        n_obs = n_animals # One obs per animal for simplicity
        pheno_df = pd.DataFrame({
            'obs_id': animal_ids_ordered,
            'y': [intercept_true + true_animal_effects_map[aid] + np.random.randn() * np.sqrt(sigma_e_true) for aid in animal_ids_ordered]
        })
        pheno_df.set_index('obs_id', inplace=True)

        # 3. Build Model
        model_eq = "y = intercept + animal"
        R_vc = VarianceCovariance(value=sigma_e_true*0.8, df=4.0, scale=sigma_e_true*0.4, estimate_variance=True)
        mme = build_model(model_eq, R_value=R_vc.value, R_df=R_vc.df)
        mme.R_variance = R_vc

        # Add structured random effect for "animal"
        G0_prior = VarianceCovariance(value=sigma_a_true*0.8, df=4.0, scale=sigma_a_true*0.4, estimate_variance=True)
        mme.add_structured_random_effect(
            base_term_name="animal",
            V_inv_matrix=A_inv,
            level_ids=animal_ids_ordered,
            prior_vc_info=G0_prior,
            random_type_code="A"
        )

        # 4. MCMC Info
        mme.mcmc_info = MCMCInfo(chain_length=3000, burnin=500, printout_frequency=3001, seed=777)

        # 5. Get MME Components
        get_mme_components(mme, pheno_df) # df_pheno provides 'y'

        # 6. Run MCMC
        results = run_mcmc(mme, pheno_df)

        # 7. Check Results
        mean_intercept_est = results.get("mean_solutions")[mme.model_term_dict["y:intercept"].start_pos]
        self.assertAlmostEqual(mean_intercept_est, intercept_true, delta=1.0, msg="Animal model intercept not recovered well.")

        mean_res_var_est = results.get("mean_residual_variance")
        self.assertGreater(mean_res_var_est, 0)
        self.assertLess(abs(mean_res_var_est - sigma_e_true) / sigma_e_true, 0.8, "Animal model residual variance not recovered well.")

        # Check polygenic variance (G0)
        # This is stored in mme.pedigree_inv_covariance.value (which is G0_inv)
        # Or in the RandomEffectTerm's Gi.value
        animal_re_term = next(rt for rt in mme.random_effect_terms if rt.random_type=="A")
        # The stored value is G0_inv. We need to invert it to get G0, then take mean if matrix.
        # For ST animal model, G0 is scalar sigma_a^2. Gi.value is 1/sigma_a^2.
        # So, 1.0 / animal_re_term.Gi.value (or mean of this if it's from MCMC) is estimate of sigma_a^2
        # The results dict doesn't explicitly store mean G0 yet.
        # We need to access the mean of the sampled G0_inv from the RandomEffectTerm's accumulation storage.
        # This requires adding accumulation for rt.Gi.value / rt.Gi_new.value in engine.py
        # For now, check if the final sampled value in animal_re_term.Gi.value is reasonable.
        final_G0_inv_est = animal_re_term.Gi.value
        if isinstance(final_G0_inv_est, (float, np.floating)): # Scalar for ST
            final_G0_est = 1.0 / final_G0_inv_est if final_G0_inv_est != 0 else np.inf
            self.assertGreater(final_G0_est, 0)
            self.assertLess(abs(final_G0_est - sigma_a_true) / sigma_a_true, 0.9, "Animal model polygenic variance not recovered well.")
        else: # Matrix for MT
            self.fail("Polygenic variance G0 for ST animal model should be scalar-like.")

        if os.path.exists(dummy_ped_path): os.remove(dummy_ped_path)

    def test_run_mcmc_ssgblup_simple(self):
        """Test MCMC for a simple ssGBLUP animal model."""
        from pyjwas.pedigree import PedigreeData, read_pedigree # For test setup
        from pyjwas.single_step import calculate_H_inverse # For direct call if needed for parts

        # 1. Setup Pedigree (same as test_animal_model_simple)
        ped_content = """A1,0,0\nA2,0,0\nA3,A1,A2\nA4,A1,A2\nA5,A3,0\nA6,A3,A4"""
        dummy_ped_path = "test_ssgblup_ped.csv"
        with open(dummy_ped_path, "w") as f: f.write(ped_content.strip())
        ped_data = read_pedigree(dummy_ped_path)

        # 2. Define Genotyped Animals and create a GRM for them
        # Let's say A3, A4, A5, A6 are genotyped.
        # Original IDs before reordering by set_genotyped_animals in calculate_H_inverse
        # The order of IDs for GRM must match the GRM rows/cols.
        grm_ids = ["A3", "A4", "A5", "A6"]
        n_g = len(grm_ids)
        # Simple GRM: scaled identity + small noise to make it PD
        G_matrix = np.eye(n_g) * 0.9 + np.random.rand(n_g, n_g) * 0.05
        G_matrix = (G_matrix + G_matrix.T) / 2.0 # Symmetrize
        G_matrix += np.eye(n_g) * 0.05 # Ensure PD

        geno_data_grm = GenotypesData(name="grm_main", obs_ids=grm_ids, genotypes=G_matrix, is_grm=True)

        # 3. Simulate Data
        all_animal_ids_in_ped = ped_data.get_ordered_ids_str() # IDs in initial pedigree order
        n_total_animals = len(all_animal_ids_in_ped)

        intercept_true = 20.0
        sigma_a_true = 2.5  # Polygenic variance (relative to H)
        sigma_e_true = 5.0  # Residual variance

        # Simulate animal effects u ~ N(0, H * sigma_a_true) - this is complex as H is not easily available.
        # For simplicity, simulate u ~ N(0, I * sigma_a_true) for all animals in pedigree for now.
        # This is a simplification for testing the MCMC mechanics, not for validating H itself.
        true_animal_effects_map = {aid: np.random.randn() * np.sqrt(sigma_a_true) for aid in all_animal_ids_in_ped}

        pheno_df = pd.DataFrame({
            'obs_id': all_animal_ids_in_ped,
            'y': [intercept_true + true_animal_effects_map[aid] + np.random.randn() * np.sqrt(sigma_e_true)
                  for aid in all_animal_ids_in_ped]
        })
        pheno_df.set_index('obs_id', inplace=True)

        # 4. Build Model using setup_single_step_animal_model
        model_eq = "y = intercept + animal" # 'animal' is the base name
        R_vc = VarianceCovariance(value=sigma_e_true, df=4.0, scale=sigma_e_true/2, estimate_variance=True)
        mme = build_model(model_eq, R_value=R_vc.value, R_df=R_vc.df)
        mme.R_variance = R_vc

        G0_prior = VarianceCovariance(value=sigma_a_true, df=4.0, scale=sigma_a_true/2, estimate_variance=True)

        # This call will internally calculate H_inv and set up the "animal" RandomEffectTerm
        mme.setup_single_step_animal_model(
            pedigree_data=ped_data, # Will be modified (reordered)
            geno_data_for_grm=geno_data_grm,
            polygenic_variance_prior=G0_prior,
            base_animal_effect_name="animal",
            weight_G_for_Hinv=0.95
        )

        # 5. MCMC Info
        mme.mcmc_info = MCMCInfo(chain_length=2500, burnin=500, printout_frequency=2501, seed=888)

        # 6. Get MME Components
        # Phenotype data DF must use the same IDs as used in pedigree/GRM for alignment.
        # The `setup_single_step_animal_model` reorders pedigree_data.ordered_nodes.
        # If pheno_df is for *all* animals in H_ids order, it's fine.
        # The current pheno_df is based on initial ped order. This needs care.
        # `get_mme_components` expects df.index to match `mme.obs_ids`.
        # `mme.obs_ids` should be set to `ordered_ids_for_H` by `setup_single_step_animal_model`
        # or by `get_mme_components` using `mme.pedigree_data.get_ordered_ids_str()`.
        # For now, let's reindex pheno_df to match H_ids if they differ.
        H_ids_from_setup = mme.pedigree_data.get_ordered_ids_str()
        mme.obs_ids = H_ids_from_setup # Ensure MME knows its obs_ids order
        pheno_df_ordered = pheno_df.reindex(H_ids_from_setup)

        get_mme_components(mme, pheno_df_ordered)

        # 7. Run MCMC
        results = run_mcmc(mme, pheno_df_ordered)

        # 8. Check Results
        # Intercept is the first element in solutions if "intercept" is the first term
        intercept_model_term = mme.model_term_dict.get("y:intercept")
        self.assertIsNotNone(intercept_model_term, "Intercept term not found in MME.")
        mean_intercept_est = results.get("mean_solutions")[intercept_model_term.start_pos]
        self.assertAlmostEqual(mean_intercept_est, intercept_true, delta=1.5,
                               msg=f"ssGBLUP Intercept not recovered well (est={mean_intercept_est:.3f}, true={intercept_true:.3f}).")

        mean_res_var_est = results.get("mean_residual_variance")
        self.assertGreater(mean_res_var_est, 0)
        self.assertLess(abs(mean_res_var_est - sigma_e_true) / sigma_e_true, 0.9, # Wider tolerance for VCs
                        f"ssGBLUP Residual variance not recovered well (est={mean_res_var_est:.3f}, true={sigma_e_true:.3f}).")

        # Check polygenic variance (sigma_a^2)
        animal_re_term = next((rt for rt in mme.random_effect_terms if rt.random_type=="A_Hinv"), None)
        self.assertIsNotNone(animal_re_term, "Animal RandomEffectTerm not found after ssGBLUP setup.")

        # Need to accumulate posterior mean of G0 (sigma_a^2 for ST) in MCMC engine
        # For now, check the last sampled value if accumulation isn't in results dict yet
        # The results dict does not yet contain mean VCs for general random effects.
        # Accessing mme.pedigree_inv_covariance (which stores G0_inv)
        final_G0_inv_est = mme.pedigree_inv_covariance.value
        if isinstance(final_G0_inv_est, (float, np.floating)): # Scalar for ST
            final_G0_est = 1.0 / final_G0_inv_est if final_G0_inv_est > 1e-9 else np.inf
            self.assertGreater(final_G0_est, 0)
            self.assertLess(abs(final_G0_est - sigma_a_true) / sigma_a_true, 0.95, # Very wide tolerance
                            f"ssGBLUP Polygenic variance not recovered well (est={final_G0_est:.3f}, true={sigma_a_true:.3f}).")
        else:
            self.fail("Polygenic variance G0 for ST ssGBLUP model should be scalar-like.")

        if os.path.exists(dummy_ped_path): os.remove(dummy_ped_path)


if __name__ == '__main__':
    unittest.main()
```
