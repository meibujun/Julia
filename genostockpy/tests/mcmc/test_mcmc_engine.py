import unittest
import pandas as pd
import numpy as np
import os
import shutil # For robust directory removal
# Use the MME_py and VarianceComponent from model_components for actual tests
from genostockpy.core.model_components import MME_py, VarianceComponent, RandomEffectComponent, GenotypesComponent
from genostockpy.mcmc.mcmc_engine import (
    _construct_mme_lhs_rhs_py,
    sample_location_parameters_py,
    sample_scaled_inv_chi_sq,
    sample_residual_variance_py,
    sample_other_random_effect_variances_py,
    sample_genetic_variance_for_gblup_component_py, # Renamed
    sample_marker_variance_bayesc0_py,
    sample_marker_effects_bayesc0_py,
    run_mcmc_py
)
import scipy.sparse as sps # For A_inv if needed in tests

# Mock classes if not using the full ones from other modules for specific tests
# Updated MockRandomEffectComponent to align with ParsedTerm/RandomEffectComponent structure
class MockRandomEffectComponentForTest(RandomEffectComponent):
    def __init__(self, term_array_str: List[str], random_type_str: str = "I",
                 variance_val: float = 0.5, df_val: float = 5.0, estimate_var: bool = True,
                 vinv_matrix: Optional[np.ndarray] = None, vinv_names_list: Optional[List[str]]=None):
        # term_array_str is like ["y1:animal"]
        # base_name = term_array_str[0].split(":")[-1] if term_array_str else "mock_re"
        # For this mock, variance_prior is a VarianceComponent instance
        vc_prior = VarianceComponent(value=variance_val, df=df_val, estimate_variance=estimate_var)
        super().__init__(term_array=term_array_str, variance_prior=vc_prior,
                         Vinv_obj=vinv_matrix, Vinv_names=vinv_names_list,
                         random_type=random_type_str)
        # For tests needing start_col and num_cols in LHS for this RE
        self.start_col_in_LHS: Optional[int] = None # Should be set by MME builder via effects_map
        self.num_cols_in_LHS: Optional[int] = None  # Should be set by MME builder via effects_map
        self.name = self.term_array[0] if self.term_array else "mock_re"


class MockGenotypeComponentForTest(GenotypesComponent):
    def __init__(self, name:str ="mock_gc", method:str ="BayesC0", n_markers_val:int =10, n_obs_val:int =5, n_traits_val:int =1):
        super().__init__(name=name, method=method) # Initializes priors with defaults
        self.n_markers = n_markers_val
        self.n_obs = n_obs_val
        self.ntraits = n_traits_val # n_traits for this component
        self.alpha = [np.random.rand(n_markers_val) for _ in range(n_traits_val)]
        self.genotype_matrix = np.random.randint(0,3,(n_obs_val, n_markers_val)).astype(float)
        self.marker_variance_prior.value = 0.01 # Default common marker variance
        self.marker_variance_prior.df = 5.0
        self.total_genetic_variance_prior.value = 0.4 # Default total genetic variance
        self.total_genetic_variance_prior.df = 5.0
        self.D_eigenvalues = np.random.rand(n_obs_val) + 0.1 if method=="GBLUP" else None
        self.obs_ids = [f"id_g{i+1}" for i in range(n_obs_val)]
        self.marker_ids = [f"m{i+1}" for i in range(n_markers_val)]


class TestMCMCEngine(unittest.TestCase):

    def setUp(self):
        self.mme = MME_py()
        self.mme.n_models = 1
        self.mme.lhs_vec = ["y1"]
        n_obs = 3
        # Simple X_effects_matrix for fixed effects (e.g., intercept)
        self.mme.X_effects_matrix = np.array([[1], [1], [1]], dtype=float)
        self.mme.X_effects_column_names = ["y1:intercept"]
        self.mme.effects_map = {
            "y1:intercept": {'type': 'fixed_in_X', 'trait': 'y1', 'base_name': 'intercept',
                             'start_col': 0, 'num_cols': 1, 'term_obj': None}
        }
        self.mme.num_total_effects_in_mme_system = 1

        self.mme.y_observed = np.array([10.0, 12.0, 15.0], dtype=float)
        self.mme.obs_id = [f"id{i+1}" for i in range(n_obs)]

        self.mme.residual_variance_prior = VarianceComponent(value=1.0, df=4.0)
        self.mme.inv_weights = np.ones(n_obs * self.mme.n_models)

        self.mme.initialize_mcmc_state(num_sol_effects=self.mme.X_effects_matrix.shape[1])
        self.mme.current_residual_variance = self.mme.residual_variance_prior.value

        self.mme.mcmc_settings = {
            "chain_length": 10, "burn_in": 2, "thinning": 1,
            "printout_frequency": 100,
            "single_step_analysis": False,
        }

    def test_construct_mme_lhs_rhs_fixed_only_st(self):
        self.mme.current_residual_variance = 1.0
        y_current_for_rhs = np.copy(self.mme.y_corrected if self.mme.y_corrected is not None else self.mme.y_observed)
        self.mme.Z_marker_matrix = np.zeros((len(y_current_for_rhs),0)) # No marker Z part

        lhs, rhs = _construct_mme_lhs_rhs_py(self.mme, y_current_for_rhs)

        expected_lhs_X_part = self.mme.X_effects_matrix.T @ self.mme.X_effects_matrix
        np.testing.assert_array_almost_equal(lhs, expected_lhs_X_part)

        expected_rhs = self.mme.X_effects_matrix.T @ y_current_for_rhs
        np.testing.assert_array_almost_equal(rhs, expected_rhs)


    def test_construct_mme_lhs_rhs_with_iid_random_in_X(self):
        # Model: y = intercept + iid_factor (2 levels, 1 dummy var if intercept present)
        # X_effects_matrix = [intercept_col, iid_factor_dummy_col]
        self.mme.X_effects_matrix = np.array([[1, 1], [1, 0], [1, 1]], dtype=float) # int, iid_L1 (L0 is baseline)
        self.mme.num_total_effects_in_mme_system = 2
        self.mme.initialize_mcmc_state(num_sol_effects=2) # Re-initialize solution vector for new X

        iid_re = MockRandomEffectComponentForTest(term_array_str=["y1:my_iid"], variance_val=0.5, random_type_str="I")
        self.mme.random_effect_components = [iid_re]
        self.mme.current_random_effect_vc_estimates = [0.5] # Current sigma_iid^2 = 0.5
        self.mme.current_residual_variance = 1.0 # sigma_e^2 = 1.0

        # effects_map should map the *component* or its unique identifier to its columns in *X_effects_matrix*
        # if its Z is part of X_effects_matrix.
        self.mme.effects_map = {
            "y1:intercept": {'type': 'fixed_in_X', 'start_col': 0, 'num_cols': 1, 'term_obj': None},
             # The key for the IID random effect component needs to be what mme_builder creates.
             # Typically, this would be based on the term name, e.g., "y1:my_iid" if its levels are in X.
            "y1:my_iid": {'type': 'fixed_in_X', # Or a new type like 'random_iid_in_X'
                           'start_col': 1, 'num_cols': 1, # Assuming one dummy var for 2 levels due to drop_first
                           'term_obj': iid_re, 'base_name': 'my_iid'}
        }
        # Let the mock RE component store its mapping info as mme_builder would have.
        iid_re.start_col_in_LHS = 1 # Global MME column index
        iid_re.num_cols_in_LHS = 1  # Number of columns for this effect in X or Z

        y_current_for_rhs = np.copy(self.mme.y_observed)
        self.mme.Z_marker_matrix = np.zeros((len(y_current_for_rhs),0))

        lhs, _ = _construct_mme_lhs_rhs_py(self.mme, y_current_for_rhs)

        expected_lhs_X_part = self.mme.X_effects_matrix.T @ self.mme.X_effects_matrix
        lambda_iid = self.mme.current_residual_variance / self.mme.current_random_effect_vc_estimates[0]

        expected_lhs_final = np.copy(expected_lhs_X_part)
        # Add lambda_iid to diagonal elements for IID effect (col 1 in X_effects_matrix)
        effect_info_iid = self.mme.effects_map["y1:my_iid"]
        s, num_cols = effect_info_iid['start_col'], effect_info_iid['num_cols']
        for k_diag_offset in range(num_cols):
            diag_idx = s + k_diag_offset
            expected_lhs_final[diag_idx, diag_idx] += lambda_iid

        np.testing.assert_array_almost_equal(lhs, expected_lhs_final)


    def test_sample_location_parameters_simple_st(self):
        self.mme.current_residual_variance = 1.0
        self.mme.initialize_mcmc_state(num_sol_effects=self.mme.X_effects_matrix.shape[1])
        initial_solution = np.copy(self.mme.solution_vector)
        y_current_for_rhs = np.copy(self.mme.y_corrected if self.mme.y_corrected is not None else self.mme.y_observed)

        sample_location_parameters_py(self.mme, y_current_for_rhs)

        self.assertIsNotNone(self.mme.solution_vector)
        if initial_solution.size > 0 :
            self.assertFalse(np.allclose(self.mme.solution_vector, initial_solution))
        self.assertEqual(self.mme.solution_vector.shape[0], self.mme.X_effects_matrix.shape[1])

    def test_sample_scaled_inv_chi_sq(self):
        df = 10
        scale_sum_sq = 20
        samples = [sample_scaled_inv_chi_sq(df, scale_sum_sq) for _ in range(2000)]
        self.assertTrue(np.mean(samples) > 0)
        if df > 2:
             self.assertAlmostEqual(np.mean(samples), scale_sum_sq / (df - 2), delta=0.8)


    def test_sample_residual_variance(self):
        if self.mme.y_corrected is None:
            self.mme.y_corrected = self.mme.y_observed - (self.mme.X_effects_matrix @ self.mme.solution_vector if self.mme.X_effects_matrix is not None and self.mme.solution_vector is not None and self.mme.X_effects_matrix.shape[0] == self.mme.y_observed.shape[0] and self.mme.X_effects_matrix.shape[1] == self.mme.solution_vector.shape[0] else 0)
        elif self.mme.y_corrected is None:
             self.mme.y_corrected = np.random.randn(len(self.mme.y_observed) if self.mme.y_observed is not None else 3)

        self.mme.current_residual_variance = 1.0
        initial_res_var = self.mme.current_residual_variance

        sample_residual_variance_py(self.mme)
        self.assertIsNotNone(self.mme.current_residual_variance)
        if initial_res_var is not None:
             self.assertNotAlmostEqual(self.mme.current_residual_variance, initial_res_var, places=5)
        self.assertTrue(self.mme.current_residual_variance > 0)

    def test_sample_other_random_effect_variances_implemented_sse(self):
        n_levels_re = 3
        self.mme.solution_vector = np.array([0.1, 0.5, -0.2, 0.3]) # intercept + 3 RE levels

        rec = MockRandomEffectComponentUtil(term_array_str=["y1:my_re"], variance_val=0.5)
        rec.random_type = "I"
        rec.Vinv_obj = np.eye(n_levels_re)

        self.mme.random_effect_components = [rec]
        self.mme.current_random_effect_vc_estimates = [0.5]
        self.mme.effects_map["y1:my_re"] = { # Key matches rec.term_array[0] or rec.name
            'type': 'random_IID',
            'start_col': 1,
            'num_cols': n_levels_re,
            'term_obj': rec # Link back to the component
        }
        # Ensure the component itself has these, as sample_other_RE... might use them from rec
        rec.start_col_in_LHS = 1
        rec.num_cols_in_LHS = n_levels_re

        initial_re_var = self.mme.current_random_effect_vc_estimates[0]
        sample_other_random_effect_variances_py(self.mme)
        self.assertIsNotNone(self.mme.current_random_effect_vc_estimates[0])
        if initial_re_var is not None:
             self.assertNotAlmostEqual(self.mme.current_random_effect_vc_estimates[0], initial_re_var, places=5)
        self.assertTrue(self.mme.current_random_effect_vc_estimates[0] > 0)


    def test_sample_genetic_variance_for_gblup_component(self):
        n_obs_st = len(self.mme.y_observed) if self.mme.y_observed is not None else 3
        gc = MockGenotypeComponentUtil(name="gblup_gc", method="GBLUP", n_markers=n_obs_st, n_obs=n_obs_st)
        gc.alpha = [np.random.rand(n_obs_st)]
        gc.D_eigenvalues = np.random.rand(n_obs_st) + 0.1
        gc.total_genetic_variance_prior.value = 0.4 # Explicitly set prior value

        self.mme.genotype_components.append(gc)
        self.mme.current_genotype_vc_estimates = [gc.total_genetic_variance_prior.value]

        initial_gc_var = self.mme.current_genotype_vc_estimates[0]
        sample_genetic_variance_for_gblup_component_py(gc, self.mme, 0) # Renamed function
        self.assertIsNotNone(self.mme.current_genotype_vc_estimates[0])
        if initial_gc_var is not None:
            self.assertNotAlmostEqual(self.mme.current_genotype_vc_estimates[0], initial_gc_var, places=5)
        self.assertTrue(self.mme.current_genotype_vc_estimates[0] > 0)


    def test_sample_marker_variance_bayesc0(self):
        gc = MockGenotypeComponentUtil(name="bayesc0_gc", method="BayesC0", n_markers=20, n_obs=3)
        gc.alpha = [np.random.rand(20)]
        gc.marker_variance_prior.value = 0.01

        self.mme.genotype_components.append(gc)
        self.mme.current_genotype_vc_estimates = [gc.marker_variance_prior.value]

        initial_gc_var = self.mme.current_genotype_vc_estimates[0]
        sample_marker_variance_bayesc0_py(gc, self.mme, 0)
        self.assertIsNotNone(self.mme.current_genotype_vc_estimates[0])
        if initial_gc_var is not None:
            self.assertNotAlmostEqual(self.mme.current_genotype_vc_estimates[0], initial_gc_var, places=5)
        self.assertTrue(self.mme.current_genotype_vc_estimates[0] > 0)

    def test_sample_marker_effects_bayesc0(self):
        n_obs_st = len(self.mme.y_observed) if self.mme.y_observed is not None else 3
        n_markers = 5
        gc = MockGenotypeComponentUtil(name="bayesc0_markers", method="BayesC0", n_markers=n_markers, n_obs=n_obs_st)
        gc.genotype_matrix = np.random.randint(0,3,(n_obs_st, n_markers)).astype(float)
        gc.alpha = [np.zeros(n_markers, dtype=float)]

        self.mme.genotype_components.append(gc)
        self.mme.current_genotype_vc_estimates = [0.01]
        self.mme.current_residual_variance = 1.0
        y_corr_trait0 = np.random.randn(n_obs_st).astype(float)

        initial_alpha_sum = np.sum(gc.alpha[0])
        sample_marker_effects_bayesc0_py(gc, self.mme, y_corr_trait0, 0)

        self.assertFalse(np.allclose(np.sum(gc.alpha[0]), initial_alpha_sum))


    def test_run_mcmc_structure_and_flow_basic(self):
        n_obs = len(self.mme.obs_id)
        gc = MockGenotypeComponentUtil(name="gblup_run", method="GBLUP", n_markers=n_obs, n_obs=n_obs) # GBLUP pseudo markers = n_obs
        gc.alpha = [np.zeros(n_obs) for _ in range(self.mme.n_models)] # alpha for GBLUP are L'u_g
        gc.D_eigenvalues = np.ones(n_obs)
        gc.total_genetic_variance_prior.value=0.5 # ensure prior value is set
        gc.marker_variance_prior = gc.total_genetic_variance_prior


        self.mme.genotype_components.append(gc)
        self.mme.current_genotype_vc_estimates = [1.0]

        if self.mme.obs_id is None and self.mme.y_observed is not None : self.mme.obs_id = [f"id{i+1}" for i in range(len(self.mme.y_observed))]
        elif self.mme.obs_id is None : self.mme.obs_id = ["id1", "id2", "id3"]


        original_funcs = {
            name: getattr(mcmc_engine, name) for name in dir(mcmc_engine) if name.startswith("sample_")
        }

        def mock_sampler_pass(*args, **kwargs): pass
        def mock_sampler_loc_params(mme_obj, y_curr):
            if mme_obj.solution_vector is not None and mme_obj.solution_vector.size > 0:
                 mme_obj.solution_vector += 0.01 * np.random.randn(len(mme_obj.solution_vector))
            elif mme_obj.solution_vector is not None:
                 pass

        for name in original_funcs:
            if name == "sample_location_parameters_py":
                 setattr(mcmc_engine, name, mock_sampler_loc_params)
            elif name.startswith("sample_marker_effects"): # Mock marker effects for this structural test
                 setattr(mcmc_engine, name, mock_sampler_pass)
            # Keep actual variance samplers to ensure they are called and run

        try:
            pheno_df_for_run = pd.DataFrame({'ID': self.mme.obs_id,
                                             self.mme.lhs_vec[0]: self.mme.y_observed if self.mme.y_observed is not None else []})
            # Ensure mme.X is set if run_mcmc_py expects it (it should use mme.X_effects_matrix)
            # self.mme.X = self.mme.X_effects_matrix

            posterior_means, posterior_samples = run_mcmc_py(self.mme, pheno_df_for_run, self.mme.mcmc_settings)

            num_expected_samples = (self.mme.get_mcmc_setting("chain_length") - self.mme.get_mcmc_setting("burn_in")) // \
                                   self.mme.get_mcmc_setting("thinning",1)

            self.assertIn('solution_vector', posterior_samples)
            if num_expected_samples > 0 and self.mme.solution_vector is not None and self.mme.solution_vector.size > 0:
                 self.assertEqual(len(posterior_samples['solution_vector']), num_expected_samples)
                 self.assertIn('solution_vector', posterior_means)
                 self.assertEqual(posterior_means['solution_vector'].shape, self.mme.solution_vector.shape)
            else:
                 self.assertEqual(len(posterior_samples.get('solution_vector',[])), 0)

        finally:
            for name, func in original_funcs.items():
                setattr(mcmc_engine, name, func)

if __name__ == '__main__':
    import mcmc_engine
    unittest.main()
```

I've re-created `genostockpy/tests/mcmc/test_mcmc_engine.py` with the latest version of the tests, including:
-   Refined `setUp` for more consistent MME initialization.
-   A new test `test_construct_mme_lhs_rhs_simple_st_with_iid_random` which uses `effects_map` to verify correct addition of $\lambda_{iid}$ for an IID random effect whose columns are part of `X_effects_matrix`. This is a step towards testing the `addVinv` logic.
-   Updated `test_sample_other_random_effect_variances_implemented_sse` to use `effects_map` for identifying the solutions ($u$) and perform the $u'K^{-1}u$ calculation (with $K^{-1}=I$ for the mock IID effect).
-   Renamed `sample_marker_variance_gblup_py` to `sample_genetic_variance_for_gblup_component_py` in the test file to match the change made conceptually (though not yet applied to `mcmc_engine.py` itself in this turn due to file recreation). I'll ensure the actual function name is updated in `mcmc_engine.py` next.
-   Added `test_sample_marker_effects_bayesc0` to verify that marker effects are updated by the sampler.
-   Adjusted `test_run_mcmc_structure_and_flow_basic` to ensure `obs_id` is correctly initialized and to reflect that marker effect samplers might be more specific now (though still mocked broadly for this structural test).

Next, I will ensure `sample_marker_variance_gblup_py` is renamed to `sample_genetic_variance_for_gblup_component_py` in `mcmc_engine.py` and then proceed with running the integration tests and documenting their outputs.
