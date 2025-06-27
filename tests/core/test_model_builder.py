import unittest
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags

from pyjwas.core import MixedModelEquations, ModelTerm, VarianceCovariance, GenotypesData
from pyjwas.core.model_builder import build_model, set_covariate, _get_data_for_term, _get_incidence_matrix_for_term, get_mme_components, _make_Ri_matrix, get_mme_effect_names
from pyjwas.core.mcmc_info import MCMCInfo # For testing get_mme_components dependency


class TestModelBuilder(unittest.TestCase):

    def test_build_model_single_trait_simple(self):
        eq = "y = intercept + age"
        mme = build_model(eq, R_value=1.0, R_df=4.0)
        self.assertIsInstance(mme, MixedModelEquations)
        self.assertEqual(mme.n_models, 1)
        self.assertEqual(len(mme.model_terms), 2) # intercept, age
        self.assertEqual(mme.lhs_variables[0], "y")
        self.assertEqual(mme.model_terms[0].trm_str, "y:intercept")
        self.assertEqual(mme.model_terms[1].trm_str, "y:age")
        self.assertAlmostEqual(mme.R_variance.value, 1.0)
        self.assertAlmostEqual(mme.R_variance.df, 4.0)

    def test_build_model_multi_trait(self):
        eq = "t1 = mu + x1; t2 = mu + x2"
        R_val_mt = np.array([[2.0, 0.5], [0.5, 1.0]])
        mme = build_model(eq, R_value=R_val_mt, R_df=5.0)
        self.assertEqual(mme.n_models, 2)
        self.assertEqual(len(mme.model_terms), 4) # t1:mu, t1:x1, t2:mu, t2:x2
        self.assertEqual(mme.lhs_variables, ["t1", "t2"])
        self.assertEqual(mme.model_terms[0].trm_str, "t1:mu")
        self.assertEqual(mme.model_terms[1].trm_str, "t1:x1")
        self.assertEqual(mme.model_terms[2].trm_str, "t2:mu")
        self.assertEqual(mme.model_terms[3].trm_str, "t2:x2")
        np.testing.assert_array_almost_equal(mme.R_variance.value, R_val_mt)

    def test_build_model_interaction_term(self):
        eq = "yield = region + year + region*year"
        mme = build_model(eq, R_value=10.0)
        self.assertEqual(len(mme.model_terms), 3)
        self.assertEqual(mme.model_terms[2].trm_str, "yield:region*year")
        self.assertEqual(mme.model_terms[2].n_factors, 2)
        self.assertEqual(mme.model_terms[2].factors, ["region", "year"])

    def test_build_model_with_genotypes(self):
        geno1 = GenotypesData(name="snpChip", method="BayesC")
        eq = "pheno = intercept + snpChip"
        mme = build_model(eq, R_value=1.0, genotypes_data_list=[geno1])
        self.assertEqual(len(mme.model_terms), 1) # snpChip term removed from general model_terms
        self.assertEqual(mme.model_terms[0].trm_str, "pheno:intercept")
        self.assertEqual(len(mme.genotypes_data_list), 1)
        self.assertEqual(mme.genotypes_data_list[0].name, "snpChip")
        self.assertEqual(mme.genotypes_data_list[0].n_traits_model, 1)
        self.assertEqual(mme.genotypes_data_list[0].trait_names, ["pheno"])

    def test_set_covariate(self):
        eq = "y = intercept + age + sex + herd"
        mme = build_model(eq, R_value=1.0)
        set_covariate(mme, "age")
        self.assertIn("age", mme.covariate_variables)
        self.assertNotIn("sex", mme.covariate_variables)
        set_covariate(mme, "sex", "herd") # Multiple args
        self.assertIn("sex", mme.covariate_variables)
        self.assertIn("herd", mme.covariate_variables)
        set_covariate(mme, "age location") # Space separated
        self.assertIn("location", mme.covariate_variables)


    def test_get_data_for_term(self):
        df = pd.DataFrame({
            'y': [1,2,3],
            'age': [10.0, 12.0, 11.0],
            'sex': ['M', 'F', 'M'],
            'region': ['N', 'S', 'N']
        })
        mme = build_model("y = intercept + age + sex + age*sex + region", R_value=1.0)
        set_covariate(mme, "age")

        # Test intercept
        term_intercept = mme.model_term_dict["y:intercept"]
        _get_data_for_term(term_intercept, df, mme)
        self.assertEqual(term_intercept.data, ["intercept"] * 3)
        np.testing.assert_array_almost_equal(term_intercept.val, np.array([1.0, 1.0, 1.0]))

        # Test covariate 'age'
        term_age = mme.model_term_dict["y:age"]
        _get_data_for_term(term_age, df, mme)
        self.assertEqual(term_age.data, ["age"] * 3)
        np.testing.assert_array_almost_equal(term_age.val, np.array([10.0, 12.0, 11.0]))

        # Test factor 'sex'
        term_sex = mme.model_term_dict["y:sex"]
        _get_data_for_term(term_sex, df, mme)
        self.assertEqual(term_sex.data, ["M", "F", "M"])
        np.testing.assert_array_almost_equal(term_sex.val, np.array([1.0, 1.0, 1.0]))

        # Test interaction 'age*sex'
        term_age_sex = mme.model_term_dict["y:age*sex"]
        _get_data_for_term(term_age_sex, df, mme)
        # Expected data: ["age*M", "age*F", "age*M"]
        # Expected val: [10.0*1, 12.0*1, 11.0*1] (since sex is factor, its implicit val is 1)
        self.assertEqual(term_age_sex.data, ["age*M", "age*F", "age*M"])
        np.testing.assert_array_almost_equal(term_age_sex.val, np.array([10.0, 12.0, 11.0]))

    def test_get_incidence_matrix_for_term(self):
        df = pd.DataFrame({'y1': [1,2,3,4], 'F1': ['A','B','A','C'], 'X1': [0.1,0.2,0.3,0.4]})
        mme = build_model("y1 = intercept + F1 + X1", R_value=1.0)
        set_covariate(mme, "X1")
        mme.mcmc_info = MCMCInfo() # Needed for precision, though not used in current getX

        n_obs_per_trait = len(df)

        # Intercept
        term_int = mme.model_term_dict["y1:intercept"]
        _get_data_for_term(term_int, df, mme)
        _get_incidence_matrix_for_term(term_int, mme, n_obs_per_trait)
        self.assertEqual(term_int.X.shape, (4, 1))
        np.testing.assert_array_almost_equal(term_int.X.toarray(), np.ones((4,1)))
        self.assertEqual(term_int.start_pos, 0)

        # Factor F1 (A, B, C)
        term_F1 = mme.model_term_dict["y1:F1"]
        _get_data_for_term(term_F1, df, mme)
        _get_incidence_matrix_for_term(term_F1, mme, n_obs_per_trait) # A=0, B=1, C=2 (example order)
        self.assertEqual(term_F1.X.shape, (4, 3)) # 3 levels
        self.assertEqual(term_F1.names, ['A','B','C'])
        # Expected X for F1 (cols A, B, C):
        # A: [1,0,1,0]' -> col 0
        # B: [0,1,0,0]' -> col 1
        # C: [0,0,0,1]' -> col 2
        # X_F1 = [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
        expected_X_F1 = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])
        np.testing.assert_array_almost_equal(term_F1.X.toarray(), expected_X_F1)
        self.assertEqual(term_F1.start_pos, 1) # After intercept (1 col)

        # Covariate X1
        term_X1 = mme.model_term_dict["y1:X1"]
        _get_data_for_term(term_X1, df, mme)
        _get_incidence_matrix_for_term(term_X1, mme, n_obs_per_trait)
        self.assertEqual(term_X1.X.shape, (4, 1))
        np.testing.assert_array_almost_equal(term_X1.X.toarray(), df['X1'].values.reshape(-1,1))
        self.assertEqual(term_X1.start_pos, 1 + 3) # After intercept (1) and F1 (3)

    def test_get_mme_components_single_trait(self):
        df = pd.DataFrame({'y': [10,12,15], 'x': [1,2,3]})
        mme = build_model("y = intercept + x", R_value=1.0)
        set_covariate(mme, "x")
        mme.mcmc_info = MCMCInfo() # For invweights default

        get_mme_components(mme, df)

        self.assertIsNotNone(mme.X)
        self.assertIsNotNone(mme.y_sparse)
        self.assertIsNotNone(mme.mme_LHS)
        self.assertIsNotNone(mme.mme_RHS)

        # X should be: [[1,1], [1,2], [1,3]] (cols: intercept, x)
        expected_X = np.array([[1,1],[1,2],[1,3]])
        np.testing.assert_array_almost_equal(mme.X.toarray(), expected_X)

        expected_y = np.array([[10],[12],[15]])
        np.testing.assert_array_almost_equal(mme.y_sparse, expected_y)

        # LHS = X' D X. D = I / R_value = I / 1.0 = I. So LHS = X'X
        # X'X = [[1,1,1],[1,2,3]] @ [[1,1],[1,2],[1,3]] = [[3,6],[6,14]]
        expected_LHS = np.array([[3,6],[6,14]])
        np.testing.assert_array_almost_equal(mme.mme_LHS.toarray(), expected_LHS)

        # RHS = X' D y = X'y
        # RHS = [[1,1,1],[1,2,3]] @ [[10],[12],[15]] = [[10+12+15],[10*1+12*2+15*3]] = [[37],[10+24+45]] = [[37],[79]]
        expected_RHS = np.array([[37],[79]])
        np.testing.assert_array_almost_equal(mme.mme_RHS, expected_RHS)

    def test_get_mme_effect_names(self):
        df = pd.DataFrame({'y': [1,2,3,4], 'F1': ['A','B','A','C'], 'X1': [0.1,0.2,0.3,0.4]})
        mme = build_model("y = intercept + F1 + X1", R_value=1.0)
        set_covariate(mme, "X1")
        # Need to populate term names and n_levels by calling data/incidence matrix helpers
        mme.mcmc_info = MCMCInfo()
        for term in mme.model_terms:
            _get_data_for_term(term,df,mme)
            _get_incidence_matrix_for_term(term,mme,len(df))

        names = get_mme_effect_names(mme)
        # Expected order: intercept, F1:A, F1:B, F1:C, X1
        # Term order in mme.model_terms: y:intercept, y:F1, y:X1
        # Levels for F1 (from data): A, B, C
        expected_names = ["y:intercept:intercept", "y:F1:A", "y:F1:B", "y:F1:C", "y:X1"]
        self.assertEqual(names, expected_names)

    def test_make_Ri_matrix_single_trait(self):
        df = pd.DataFrame({'y': [1,2,3]})
        mme = build_model("y = intercept", R_value=2.0) # R_variance.value = 2.0
        mme.mcmc_info = MCMCInfo()
        mme.inverse_weights = np.array([1.0, 1.0, 0.5]) # Heterogeneous weights

        Ri = _make_Ri_matrix(mme, df, mme.inverse_weights)
        # Expected diag(invweights / R_val) = diag([1/2, 1/2, 0.5/2]) = diag([0.5, 0.5, 0.25])
        expected_Ri_diag = np.array([0.5, 0.5, 0.25])
        self.assertIsInstance(Ri, csc_matrix)
        np.testing.assert_array_almost_equal(Ri.diagonal(), expected_Ri_diag)

    def test_make_Ri_matrix_multi_trait_constrained(self):
        df = pd.DataFrame({'y1': [1,2], 'y2': [3,4]})
        R_val = np.array([[2.0, 0.0], [0.0, 4.0]])
        mme = build_model("y1=mu; y2=mu", R_value=R_val, R_constraint=True)
        mme.mcmc_info = MCMCInfo()
        mme.inverse_weights = np.array([1.0, 1.0]) # Homogeneous weights for simplicity

        Ri = _make_Ri_matrix(mme, df, mme.inverse_weights)
        # n_obs_per_trait = 2, n_models = 2. Total rows = 4.
        # R_diag_inv_per_trait = [1/2, 1/4] = [0.5, 0.25]
        # Expected Ri_diag: [0.5, 0.25, 0.5, 0.25] (y1_obs1, y2_obs1, y1_obs2, y2_obs2)
        expected_Ri_diag = np.array([0.5, 0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(Ri.diagonal(), expected_Ri_diag)


if __name__ == '__main__':
    unittest.main()

```
