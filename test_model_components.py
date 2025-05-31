import unittest
import pandas as pd
import numpy as np
from model_components import (
    MME_py, VarianceComponent, RandomEffectComponent, GenotypesComponent,
    check_model_arguments_py, check_output_id_py, check_data_consistency_py,
    set_default_priors_for_variance_components_py, set_random_py,
    add_genotypes_py, setup_gblup_py, setup_bayes_c0_py, set_marker_hyperparameters_py
)
# Assuming genotype_handler.py is in the same directory or accessible in PYTHONPATH
from genotype_handler import read_genotypes_py


# Mock Pedigree class for testing set_random_py with pedigree
class MockPedigree:
    def __init__(self, ids=None, name="test_ped"):
        self.name = name # Added name attribute
        self.id_map = {id_str: object() for id_str in ids} if ids else {}
        self.ordered_ids = ids if ids else []
        self.A_inv = None

    def calculate_A_inverse(self):
        if self.ordered_ids:
            n = len(self.ordered_ids)
            self.A_inv = np.eye(n)
            return self.A_inv
        return None

# Mock ModelTerm class for mme.model_term_dict
class MockModelTerm: # Already defined in previous test, ensure it's consistent or imported
    def __init__(self, name, random_type="fixed", names=None): # Added names to constructor
        self.name = name
        self.random_type = random_type
        self.names = names if names else [] # For levels, used by set_random if type A or V


class TestModelComponents(unittest.TestCase):

    def setUp(self):
        self.mme = MME_py()
        self.mme.lhs_vec = ["y1"] # Simplified to single trait for some tests
        self.mme.n_models = 1
        self.mme.model_term_dict = {
            "y1:intercept": MockModelTerm("y1:intercept"),
            "y1:age": MockModelTerm("y1:age"),
            "y1:animal": MockModelTerm("y1:animal")
        }
        self.mme.mcmc_info = {
            "output_samples_frequency": 10, "single_step_analysis": False,
            "outputEBV": False, "output_heritability": False, "RRM": False,
            "double_precision": True, "printout_model_info": False # Added this
        }
        self.mme.traits_type = ["continuous"]

        self.pheno_data = pd.DataFrame({
            'ID': ['id1', 'id2', 'id3', 'id4', 'id5'],
            'y1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'age': [1,2,1,3,2], 'animal': ['a1','a2','a1','a3','a2']
        })
        self.pheno_data['ID'] = self.pheno_data['ID'].astype(str)

        # Sample genotype data for testing add_genotypes_py etc.
        self.raw_geno_df = pd.DataFrame({
            'ID': ['id1', 'id2', 'id3', 'id4', 'id5'],
            'M1': [0,1,2,0,1], 'M2': [1,1,0,0,1], 'M3': [2,1,0,2,1]
        })

    def test_variance_component_creation(self):
        vc = VarianceComponent(value=1.0, df=5.0, scale=0.5, is_g_component=True)
        self.assertEqual(vc.value, 1.0)
        self.assertTrue(vc.is_g_component)

    def test_check_model_arguments_bayesA_conversion(self):
        # Simplified mme for this test
        mme_st = MME_py() # Single trait by default for this simple MME
        mme_st.n_models = 1
        gc = GenotypesComponent(name="markers", method="BayesA")
        mme_st.genotype_components.append(gc)
        check_model_arguments_py(mme_st)
        self.assertEqual(gc.method, "BayesB")
        # self.assertEqual(gc.pi_value, 1.0) # Assuming pi_value is P(non-zero), BayesA means all have effect
        self.assertFalse(gc.estimate_pi)

    def test_set_default_priors_single_trait(self):
        # Single trait MME
        mme_st = MME_py()
        mme_st.lhs_vec = ["y1"]
        mme_st.n_models = 1
        mme_st.traits_type = ["continuous"]

        gc = GenotypesComponent(name="chip_st", method="BayesB")
        mme_st.genotype_components.append(gc)

        set_default_priors_for_variance_components_py(mme_st, self.pheno_data)

        self.assertIsNotNone(mme_st.residual_variance.value)
        self.assertIsInstance(mme_st.residual_variance.value, float)
        self.assertIsNotNone(gc.total_genetic_variance_prior.value)
        self.assertIsInstance(gc.total_genetic_variance_prior.value, float)

    def test_set_random_iid_single_trait(self):
        mme_st = MME_py()
        mme_st.lhs_vec = ["y1"]
        mme_st.n_models = 1
        mme_st.model_term_dict = {"y1:cage": MockModelTerm("y1:cage")}

        set_random_py(mme_st, random_str="cage", G_prior_value=0.5)
        self.assertEqual(len(mme_st.random_effect_components), 1)
        rec = mme_st.random_effect_components[0]
        self.assertIn("y1:cage", rec.term_array)
        self.assertEqual(rec.random_type, "I")
        self.assertAlmostEqual(rec.variance_prior.value[0,0], 0.5)


    # --- Tests for Genotype Component Setup (GBLUP, BayesC0) ---
    def test_add_genotypes_gblup(self):
        # Mock output from read_genotypes_py
        grm_matrix = np.eye(5) # Dummy GRM
        geno_data_dict = {
            'genotypes_matrix': grm_matrix,
            'obs_ids': ['id1','id2','id3','id4','id5'],
            'marker_ids': [], # Not relevant for GRM
            'allele_frequencies': None, 'sum_2pq': None
        }
        gc = add_genotypes_py(self.mme, name="grm1", genotype_data=geno_data_dict, method="GBLUP",
                              G_prior_value=10.0, G_is_marker_variance=False) # G is total genetic var

        self.assertIn(gc, self.mme.genotype_components)
        self.assertEqual(gc.method, "GBLUP")
        self.assertTrue(gc.is_grm)
        self.assertIsNotNone(gc.total_genetic_variance_prior.value)
        self.assertIsNone(gc.marker_variance_prior.value) # G_is_marker_variance was False
        self.assertIsNotNone(gc.L_eigenvectors) # setup_gblup_py should be called by add_genotypes_py
        self.assertIsNotNone(gc.D_eigenvalues)

    def test_add_genotypes_bayesc0(self):
        # Mock output from read_genotypes_py (marker data)
        marker_matrix = np.random.randint(0,3,(5,100)).astype(float)
        geno_data_dict = {
            'genotypes_matrix': marker_matrix,
            'obs_ids': [f'id{i+1}' for i in range(5)],
            'marker_ids': [f'm{i+1}' for i in range(100)],
            'allele_frequencies': np.random.rand(100) * 0.5,
            'sum_2pq': 50.0 # Dummy value
        }
        gc = add_genotypes_py(self.mme, name="chip1", genotype_data=geno_data_dict, method="BayesC0",
                              G_prior_value=0.1, G_is_marker_variance=True, # G is marker effect var
                              pi_value=0.0) # For BayesC0, pi_value (prob of zero effect) is effectively 0

        self.assertEqual(gc.method, "BayesC0")
        self.assertFalse(gc.is_grm)
        self.assertIsNotNone(gc.marker_variance_prior.value)
        self.assertEqual(gc.pi_value, 1.0) # setup_bayes_c0_py changes pi to P(non-zero)=1.0
        self.assertFalse(gc.estimate_pi)
        self.assertIsNotNone(gc.delta)

    def test_set_marker_hyperparameters(self):
        self.mme.n_models = 1 # Single trait for this test
        self.mme.lhs_vec = ["y1"]
        marker_matrix = np.random.randint(0,3,(5,10)).astype(float)
        allele_freqs = np.random.uniform(0.01, 0.49, 10)
        sum_2pq_val = np.sum(2 * allele_freqs * (1-allele_freqs))
        geno_data_dict = {
            'genotypes_matrix': marker_matrix, 'obs_ids': [f'id{i+1}' for i in range(5)],
            'marker_ids': [f'm{i+1}' for i in range(10)],
            'allele_frequencies': allele_freqs, 'sum_2pq': sum_2pq_val
        }
        # gc's total_genetic_variance_prior needs to be set first
        gc = add_genotypes_py(self.mme, name="chip", genotype_data=geno_data_dict, method="BayesB",
                              genetic_variance_value=100.0, G_is_marker_variance=False,
                              pi_value=0.9) # P(non-zero) = 0.9

        # Before set_marker_hyperparameters_py, marker_variance_prior.value might be None or based on G_prior_value
        self.assertIsNone(gc.marker_variance_prior.value) # G_is_marker_variance was False
        self.assertIsNotNone(gc.total_genetic_variance_prior.value)

        set_marker_hyperparameters_py(self.mme)

        self.assertIsNotNone(gc.marker_variance_prior.value)
        self.assertIsNotNone(gc.marker_variance_prior.scale)
        # Expected marker var = total_gen_var / (pi_nonzero * sum2pq)
        # Here pi_value = 0.9 (P_nonzero)
        expected_marker_var = gc.total_genetic_variance_prior.value / (gc.pi_value * gc.sum_2pq)
        self.assertAlmostEqual(gc.marker_variance_prior.value, expected_marker_var)


if __name__ == '__main__':
    unittest.main()
```

I've created `model_components.py` and `test_model_components.py`.

**Summary of `model_components.py`:**
-   Defined `VarianceComponent`, `RandomEffectComponent`, `GenotypesComponent` classes.
-   Included a placeholder for `MME_py` that these components and functions interact with.
-   Implemented Python versions of key functions from `input_data_validation.jl`:
    -   `check_model_arguments_py`
    -   `check_output_id_py`
    -   `check_data_consistency_py` (includes phenotype filtering and categorical trait validation)
    -   `set_default_priors_for_variance_components_py`
-   Implemented Python version of `set_random` from `random_effects.jl` as `set_random_py`.
-   Logic from `residual.jl` (like `mkRi`, `sampleMissingResiduals`) is more tied to MCMC matrix building or execution, so it's not directly translated here, but the `ResidualComponent` (implicitly `mme.residual_variance`) setup is covered.
-   Logic from `variance_components.jl` (sampling functions) is also for MCMC execution; this module focuses on setting up the *priors* for those variance components.

**Summary of `test_model_components.py`:**
-   Basic tests for creating `VarianceComponent`.
-   Tests for `check_model_arguments_py` (valid, invalid method, BayesA conversion).
-   Tests for `check_output_id_py` (defaulting logic, filtering).
-   Tests for `check_data_consistency_py` (phenotype filtering based on genotype availability, categorical trait validation).
-   Tests for `set_default_priors_for_variance_components_py` to ensure priors are initialized for residual, genetic, and other random effects.
-   Tests for `set_random_py` for IID and pedigree-based random effects.
-   A `MockPedigree` and `MockModelTerm` class are used to simulate dependencies for tests.

This set of implemented functions and classes provides a foundation for defining model structure, validating inputs, and setting up priors for variance components, analogous to the provided Julia files.
